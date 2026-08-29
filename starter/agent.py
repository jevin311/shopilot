from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

try:
    import anthropic
except ImportError:  # pragma: no cover - offline environments
    anthropic = None

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

ATTRIBUTE_PRIORITY = [
    "material", "color", "feature", "style", "budget", "use_case", "size", "other",
]

NON_INFO_RE = re.compile(
    r"(not quite right yet|don't have (a|an additional) preference|please use your judgment)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.IGNORECASE)
JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# How many BM25 candidates to hand the LLM to choose/reorder from.
# Must be > top_k or there's nothing for the LLM to actually rerank.
RERANK_POOL_SIZE = 30
# Hard safety cap so a slow/hanging API call can't stall a whole eval run.
LLM_TIMEOUT_SECONDS = 15.0


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class _SessionState:
    __slots__ = ("category_terms", "history_terms", "asked", "turn_count")

    def __init__(self) -> None:
        self.category_terms: list[str] = []
        self.history_terms: list[str] = []
        self.asked: set[str] = set()
        self.turn_count = 0


class Agent:
    """v4: adds an optional LLM re-ranking pass on top of the v3 BM25 +
    clarification agent.

    Network/offline behavior (per docs/submission_rules.md's disclosure
    requirement):
    - Requires the `anthropic` package and an ANTHROPIC_API_KEY environment
      variable to activate. Neither is required to run the agent.
    - If the package is missing, the key is unset, or any API call fails
      or times out, the agent silently falls back to plain BM25 ordering
      (identical to v3's behavior) rather than raising or degrading output.
    - Can be force-disabled regardless of key presence by setting
      SHOPILOT_ENABLE_LLM=0 (useful to keep local eval runs fast/free).
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, _SessionState] = {}
        self.products: dict[str, dict[str, str]] = {}
        self._build_index()

        self.llm_client = None
        self.llm_model = os.environ.get("SHOPILOT_LLM_MODEL", "claude-sonnet-4-5-20250929")
        llm_enabled = os.environ.get("SHOPILOT_ENABLE_LLM", "1") != "0"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if llm_enabled and anthropic is not None and api_key:
            try:
                self.llm_client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
            except Exception:
                self.llm_client = None

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                features = _text(product.get("features"))
                # Keep only a short snippet per product for rerank prompts --
                # 50,000 full records in memory would be wasteful, we only
                # need enough text for the LLM to judge relevance.
                self.products[asin] = {
                    "title": title[:120],
                    "snippet": features[:160],
                }
                batch.append(
                    (
                        asin,
                        title,
                        _text(product.get("categories")),
                        features,
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = _SessionState()
        tags = user_profile.get("preference_tags") or []
        state.history_terms.extend(_terms(" ".join(str(t) for t in tags)))
        self._sessions[session_id] = state

    def _next_attribute(self, state: _SessionState) -> str | None:
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in state.asked:
                return attr
        return None

    def _bm25_search(self, terms: list[str], limit: int) -> list[str]:
        expression = " OR ".join(f'"{term}"' for term in terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _llm_rerank(
        self, query_terms: list[str], candidates: list[str], top_k: int
    ) -> tuple[list[str], dict[str, int]]:
        no_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self.llm_client is None or len(candidates) <= 1:
            return candidates[:top_k], no_usage

        lines = []
        for asin in candidates:
            info = self.products.get(asin, {})
            lines.append(f"{asin}: {info.get('title', '')} | {info.get('snippet', '')}")
        prompt = (
            "A shopping customer has expressed these requirements so far: "
            + " ".join(query_terms[:40])
            + "\n\nCandidate products (id: title | features):\n"
            + "\n".join(lines)
            + "\n\nReturn ONLY a JSON array of the product ids, ordered best "
            "match first, most likely to satisfy every stated requirement. "
            "Include every id exactly once. No other text."
        )
        try:
            response = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            match = JSON_ARRAY_RE.search(raw_text)
            if not match:
                return candidates[:top_k], no_usage
            parsed = json.loads(match.group(0))
            candidate_set = set(candidates)
            ranked = [str(item) for item in parsed if str(item) in candidate_set]
            seen = set(ranked)
            ranked.extend(asin for asin in candidates if asin not in seen)  # keep any dropped ids
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
            }
            return ranked[:top_k], usage
        except Exception:
            # Any failure (timeout, malformed JSON, API error, rate limit) --
            # fall back to the BM25 order rather than breaking the session.
            return candidates[:top_k], no_usage

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn_count += 1

        if state.turn_count == 1:
            state.category_terms = _terms(user_message)[:8]
            state.history_terms.extend(state.category_terms)
        elif OVERRIDE_RE.search(user_message):
            new_part = user_message.split(":", 1)[-1]
            state.history_terms = list(state.category_terms) + _terms(new_part) * 2
        elif not NON_INFO_RE.search(user_message):
            state.history_terms.extend(_terms(user_message))

        unique_terms = list(dict.fromkeys(state.history_terms))[:60]
        pool = self._bm25_search(unique_terms, max(RERANK_POOL_SIZE, top_k))
        ranked_ids, usage = self._llm_rerank(unique_terms, pool, top_k)
        recommendations = [{"parent_asin": asin} for asin in ranked_ids]

        next_attr = self._next_attribute(state) if turn < 10 else None
        if next_attr is not None:
            state.asked.add(next_attr)
            message = f"Here are some options so far -- do you have a {next_attr.replace('_', ' ')} preference?"
        else:
            message = "Here are the closest matches I found."

        return {
            "message": message,
            "ask_attribute": next_attr,
            "recommendations": recommendations,
            "usage": usage,
        }