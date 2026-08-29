from __future__ import annotations

"""TechJam Conversational Search — hybrid shopping agent.

Design summary (maps to the four pillars in the problem statement):

  I.   Core Architecture — IntentRouter splits traffic into a "buying" track
       (precise BM25 keyword filtering) and a "browsing" track (category-
       level dense expansion for cross-category discovery). HybridRetriever
       fuses both before handing a bounded candidate pool to the optional
       LLM reranker.
  II.  Dialog Strategy — SessionState tracks per-attribute slots with
       incremental accumulation and explicit override handling (slot
       erasure + rewrite). ClarificationPolicy triggers a retrieval cutoff
       once the candidate pool has converged (Over-Generality check) instead
       of blindly asking through a fixed attribute list every turn.
  III. Self-Evolution — ContextDistiller blends long-term profile tags with
       short-term session slots, decaying stale evidence when the user
       overrides an earlier answer, and re-weighting terms each turn.
  IV.  Evaluation — the agent never asks a question on the organizer's final
       allowed turn, and always returns its best-effort ranked list, since
       HitRate@10 (0.50) and MRR (0.30) dominate the TechnicalScore and a
       missed recommendation costs far more than one extra clarifying turn.

Network/offline behavior (per docs/submission_rules.md's disclosure
requirement):
- Requires the `anthropic` package and an ANTHROPIC_API_KEY environment
  variable to activate LLM reranking. Neither is required to run the agent.
- If the package is missing, the key is unset, or any API call fails or
  times out, the agent silently falls back to the hybrid BM25/dense order
  (identical output shape, no exception raised).
- Can be force-disabled regardless of key presence by setting
  SHOPILOT_ENABLE_LLM=0 (keeps local eval runs fast/free).

This file intentionally keeps the public Agent contract — __init__,
reset(session_id, user_profile), respond(session_id, user_message, turn,
top_k) -> dict — identical to the one published in
docs/agent_api_contract.json / the repo README, so it drops into the
existing evaluator without changes.
"""

import json
import math
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - offline environments
    anthropic = None

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "do", "have", "any", "just", "like", "need", "im", "ive", "get", "got",
}

# The eleven values the evaluator accepts for ask_attribute (10 named
# fields + null). The original starter only asked about six of these and
# silently never asked about "category" or "brand" — both are cheap,
# high-signal questions for narrowing a 50k-item apparel/jewelry catalog,
# so they're included here.
ALL_ATTRIBUTES = [
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
]

# Attribute questions are ordered differently depending on detected intent:
# a buyer with a specific item in mind converges fastest on hard filters
# (size/color/budget); a browser benefits more from scope-narrowing
# questions (category/use_case/style) before fine detail.
BUYING_ATTRIBUTE_ORDER = [
    "category", "size", "color", "budget", "material", "brand", "feature", "style", "use_case", "other",
]
BROWSING_ATTRIBUTE_ORDER = [
    "use_case", "category", "style", "material", "color", "budget", "feature", "brand", "size", "other",
]

NON_INFO_RE = re.compile(
    r"(not quite right yet|don't have (a|an additional) preference|please use your judgment|"
    r"no preference|doesn't matter|any (?:is|are) fine|whatever works)",
    re.IGNORECASE,
)

# An earlier version matched only "ignore my earlier preference" verbatim.
# Real override language is more varied, so this is a small bank of patterns
# rather than one fixed phrase. Still intentionally simple regex (per the
# allowed assumptions, inputs are pre-cleaned text with no ASR/typo noise),
# not an ML classifier — a state-machine slot-rewrite doesn't need one.
OVERRIDE_PATTERNS = [
    re.compile(r"ignore my earlier preference", re.IGNORECASE),
    re.compile(r"actually,?\s+i\s+(?:want|need|meant|prefer)", re.IGNORECASE),
    re.compile(r"never\s?mind (?:that|about|what i said)", re.IGNORECASE),
    re.compile(r"change (?:that|my mind|it)", re.IGNORECASE),
    re.compile(r"scratch that", re.IGNORECASE),
    re.compile(r"forget (?:that|what i said)", re.IGNORECASE),
    re.compile(r"on second thought", re.IGNORECASE),
    re.compile(r"instead of that", re.IGNORECASE),
]

BUYING_CUES = {
    "buy", "buying", "purchase", "order", "need", "asap", "today", "checkout",
    "gift", "budget", "under", "size", "stock", "price", "cheapest", "fast",
    "delivery", "exact", "specific", "brand",
}
BROWSING_CUES = {
    "looking", "browsing", "ideas", "inspire", "explore", "something",
    "options", "maybe", "thinking", "suggest", "recommend", "curious",
    "trend", "trending", "style", "not sure", "open to",
}

JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

# How many BM25 candidates to hand the LLM to choose/reorder from.
# Must be > top_k or there's nothing for the LLM to actually rerank.
RERANK_POOL_SIZE = 30
# Hard safety cap so a slow/hanging API call can't stall a whole eval run.
LLM_TIMEOUT_SECONDS = 15.0
# Max turns the organizer's evaluator allows per session (see README).
MAX_TURNS = 10
# Once the merged candidate pool is at most this many multiples of top_k,
# it's considered "converged" — asking another question is more likely to
# cost efficiency (MTTC) than it is to help precision (MRR).
CONVERGENCE_FACTOR = 3
# How many categories the dense/browsing track pulls into the candidate
# pool, and how many representative products per matched category.
DENSE_TOP_CATEGORIES = 5
DENSE_PER_CATEGORY = 12


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


def _vector(terms: list[str]) -> Counter:
    """Bag-of-words term-frequency vector used for cosine scoring.

    Deliberately not full TF-IDF over the whole 50k-doc corpus at query
    time — that would mean an O(catalog) scan per turn in pure Python,
    which doesn't fit the "light execution, no heavy vector DB" constraint.
    Instead, IDF weighting is pre-baked once at index time (see
    CatalogIndex._category_idf) and applied only to the bounded candidate
    pool a query actually touches.
    """
    return Counter(terms)


def _cosine(vec_a: Counter, vec_b: Counter, idf: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = vec_a.keys() & vec_b.keys()
    if not shared:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] * idf.get(t, 1.0) ** 2 for t in shared)
    norm_a = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in vec_a.items()))
    norm_b = math.sqrt(sum((v * idf.get(t, 1.0)) ** 2 for t, v in vec_b.items()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    terms: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class SessionState:
    intent: str = "browsing"          # "buying" | "browsing", sticky once confident
    intent_confidence: float = 0.0
    slots: dict[str, Slot] = field(default_factory=dict)
    long_term_terms: list[str] = field(default_factory=list)
    asked: set[str] = field(default_factory=set)
    pending_attribute: Optional[str] = None   # attribute the last question targeted
    category_terms: list[str] = field(default_factory=list)
    turn_count: int = 0
    last_pool_size: int = 0

    def weighted_terms(self, cap: int = 60) -> list[str]:
        """Recency- and confidence-weighted term list used for retrieval.

        Long-term profile terms contribute a light, constant background
        signal; slot terms are repeated proportionally to their weight so
        BM25's OR-expression and the dense cosine score both lean toward
        whatever the user most recently and most confidently told us.
        """
        weighted: list[str] = list(self.long_term_terms)
        for slot in self.slots.values():
            repeats = max(1, round(slot.weight * 2))
            weighted.extend(slot.terms * repeats)
        # dict.fromkeys preserves first-seen order while deduping, then we
        # still want the *weighted* copies for the term frequency signal,
        # so we only dedupe for the capped OR-expression, not for the
        # vector count.
        ordered_unique = list(dict.fromkeys(weighted))
        return ordered_unique[:cap]

    def term_vector(self) -> Counter:
        counts: Counter = Counter(self.long_term_terms)
        for slot in self.slots.values():
            counts.update({t: slot.weight for t in slot.terms})
        return counts


# ---------------------------------------------------------------------------
# Intent routing (Pillar I: Dual-Track Routing)
# ---------------------------------------------------------------------------

class IntentRouter:
    """Classifies each turn as "buying" or "browsing" and keeps the
    session's intent sticky so a single ambiguous message mid-conversation
    doesn't flip the retrieval strategy back and forth.
    """

    STICKY_THRESHOLD = 0.6

    def classify(self, message: str, state: SessionState) -> str:
        lowered = message.lower()
        buy_score = sum(1 for cue in BUYING_CUES if cue in lowered)
        browse_score = sum(1 for cue in BROWSING_CUES if cue in lowered)
        has_number = bool(re.search(r"\$?\d+(\.\d+)?", lowered))
        if has_number:
            buy_score += 1

        total = buy_score + browse_score
        if total == 0:
            # No fresh signal this turn — keep whatever intent the session
            # already committed to instead of resetting to the default.
            return state.intent

        confidence = abs(buy_score - browse_score) / total
        new_intent = "buying" if buy_score >= browse_score else "browsing"

        if confidence >= self.STICKY_THRESHOLD or state.turn_count <= 1:
            state.intent_confidence = confidence
            return new_intent

        # Weak, conflicting signal on a mature session: don't thrash.
        return state.intent


# ---------------------------------------------------------------------------
# Catalog index: BM25 (buying track) + category-dense expansion (browsing)
# ---------------------------------------------------------------------------

class CatalogIndex:
    def __init__(self, catalog_path: Path) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict[str, str]] = {}
        # category token -> list of asins in that category (browsing track)
        self.category_asins: dict[str, list[str]] = {}
        # category token -> aggregated term-frequency vector for that
        # category's product titles (browsing track similarity signal)
        self.category_vectors: dict[str, Counter] = {}
        self._category_idf: dict[str, float] = {}
        self._build(catalog_path)

    def _build(self, catalog_path: Path) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        category_doc_freq: Counter = Counter()

        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                title = _text(product.get("title"))
                features = _text(product.get("features"))
                categories_raw = _text(product.get("categories"))

                # Short snippet kept per product for LLM rerank prompts and
                # for the bounded dense cosine pass — full records for all
                # 50k products would be wasteful; this is enough text for
                # both to judge relevance.
                snippet_terms = _terms(title)[:20] + _terms(features)[:10]
                self.products[asin] = {
                    "title": title[:120],
                    "snippet": features[:160],
                    "vector": Counter(snippet_terms),
                }

                for cat_term in dict.fromkeys(_terms(categories_raw)):
                    self.category_asins.setdefault(cat_term, []).append(asin)
                    self.category_vectors.setdefault(cat_term, Counter()).update(
                        _terms(title)
                    )
                    category_doc_freq[cat_term] += 1

                batch.append(
                    (
                        asin, title, categories_raw, features,
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

        total_categories = max(1, len(self.category_vectors))
        for term, count in category_doc_freq.items():
            self._category_idf[term] = math.log(1 + total_categories / count)

    def bm25_search(self, terms: list[str], limit: int) -> list[str]:
        expression = " OR ".join(f'"{term}"' for term in terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def dense_category_expand(self, terms: list[str]) -> list[str]:
        """Cross-category discovery for open-ended browsing: match the
        query vector against category-level centroids (cheap — a few
        thousand categories, not 50k products) and pull representative
        products from the best-matching categories, including ones the
        raw BM25 keyword pass would never surface.
        """
        if not terms:
            return []
        query_vec = _vector(terms)
        scored = [
            (cat, _cosine(query_vec, vec, self._category_idf))
            for cat, vec in self.category_vectors.items()
            if cat in query_vec  # cheap prefilter before the cosine call
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        expanded: list[str] = []
        for cat, score in scored[:DENSE_TOP_CATEGORIES]:
            if score <= 0:
                continue
            expanded.extend(self.category_asins.get(cat, [])[:DENSE_PER_CATEGORY])
        return list(dict.fromkeys(expanded))

    def product_vector(self, asin: str) -> Counter:
        return self.products.get(asin, {}).get("vector", Counter())


# ---------------------------------------------------------------------------
# Hybrid retrieval (fuses both tracks; Pillar I "Multi-Route Retrieval")
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(self, index: CatalogIndex) -> None:
        self.index = index

    def retrieve(self, state: SessionState, terms: list[str], pool_size: int) -> tuple[list[str], int]:
        bm25_ids = self.index.bm25_search(terms, max(RERANK_POOL_SIZE, pool_size))

        dense_ids: list[str] = []
        # Run the dense/cross-category track whenever the session is
        # browsing, or whenever BM25 alone came back thin (a buying query
        # with an unusual keyword combination still benefits from category
        # expansion rather than returning an under-filled pool).
        if state.intent == "browsing" or len(bm25_ids) < pool_size:
            dense_ids = self.index.dense_category_expand(terms)

        merged = list(dict.fromkeys(bm25_ids + dense_ids))
        pool_size_before_rerank = len(merged)

        # Score fusion: reward BM25 rank position (buying signal) and
        # cosine similarity against the query vector (browsing / semantic
        # signal), weighted by detected intent so a buying session still
        # trusts exact keyword matches most.
        bm25_rank = {asin: i for i, asin in enumerate(bm25_ids)}
        query_vec = _vector(terms)
        bm25_weight, dense_weight = (0.75, 0.25) if state.intent == "buying" else (0.45, 0.55)

        def fused_score(asin: str) -> float:
            rank_score = 1.0 / (1 + bm25_rank.get(asin, RERANK_POOL_SIZE))
            cos_score = _cosine(query_vec, self.index.product_vector(asin), self.index._category_idf)
            return bm25_weight * rank_score + dense_weight * cos_score

        merged.sort(key=fused_score, reverse=True)
        return merged[:RERANK_POOL_SIZE], pool_size_before_rerank


# ---------------------------------------------------------------------------
# Clarification policy (Pillar II: Proactive Guidance / Over-Generality)
# ---------------------------------------------------------------------------

class ClarificationPolicy:
    def next_attribute(self, state: SessionState, pool_size: int, top_k: int, turn: int) -> Optional[str]:
        # Never ask on the organizer's final allowed turn — a question the
        # evaluator can't act on only costs efficiency for free.
        if turn >= MAX_TURNS - 1:
            return None

        order = BUYING_ATTRIBUTE_ORDER if state.intent == "buying" else BROWSING_ATTRIBUTE_ORDER
        remaining = [attr for attr in order if attr not in state.asked]
        if not remaining:
            return None

        converged = pool_size <= max(top_k * CONVERGENCE_FACTOR, top_k + 3)
        if converged:
            # Over-Generality check has already resolved itself. Only ask
            # a single cold-start question (helps intent-ambiguous first
            # turns) rather than continuing to interrogate a user whose
            # candidate pool is already small enough to just show them.
            if state.turn_count <= 1 and not state.asked:
                return remaining[0]
            return None

        return remaining[0]


# ---------------------------------------------------------------------------
# Context distillation (Pillar III: Self-Evolution)
# ---------------------------------------------------------------------------

class ContextDistiller:
    """Folds a turn's message into session state: either as new evidence
    for whichever slot the previous question targeted, as a full override
    (slot erasure + rewrite) when the user contradicts themselves, or as
    general free-text evidence otherwise.
    """

    def apply_override(self, state: SessionState, message: str) -> None:
        new_part = message.split(":", 1)[-1] if ":" in message else message
        new_terms = _terms(new_part)
        # An override rewrites whichever slot was most recently in focus;
        # if none was pending, it resets every slot so the next answer
        # starts the profile fresh rather than blending old and new intent.
        if state.pending_attribute and state.pending_attribute in state.slots:
            state.slots[state.pending_attribute] = Slot(terms=new_terms, weight=1.4)
        else:
            # No specific slot was in focus: wipe accumulated detail but
            # keep the turn-1 category intent (e.g. "hiking boots") so the
            # override doesn't lose what the product even *is* -- only the
            # detail layered on top of it.
            base = list(state.category_terms)
            state.slots.clear()
            if base:
                state.slots["other"] = Slot(terms=base, weight=1.0)
            override_terms = [t for t in new_terms if t not in base]
            if override_terms:
                state.slots["override"] = Slot(terms=override_terms, weight=1.6)

    def apply_answer(self, state: SessionState, message: str) -> None:
        terms = _terms(message)
        if not terms:
            return
        target = state.pending_attribute or "other"
        if target in state.slots:
            existing = state.slots[target]
            existing.terms = list(dict.fromkeys(existing.terms + terms))
            existing.weight = min(2.0, existing.weight + 0.3)
        else:
            state.slots[target] = Slot(terms=terms, weight=1.0)


# ---------------------------------------------------------------------------
# Optional LLM reranking pass
# ---------------------------------------------------------------------------

class LLMReranker:
    def __init__(self) -> None:
        self.client = None
        self.model = os.environ.get("SHOPILOT_LLM_MODEL", "claude-sonnet-4-5-20250929")
        llm_enabled = os.environ.get("SHOPILOT_ENABLE_LLM", "1") != "0"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if llm_enabled and anthropic is not None and api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
            except Exception:
                self.client = None

    def rerank(
        self,
        state: SessionState,
        query_terms: list[str],
        candidates: list[str],
        products: dict[str, dict[str, str]],
        top_k: int,
    ) -> tuple[list[str], dict[str, int]]:
        no_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if self.client is None or len(candidates) <= 1:
            return candidates[:top_k], no_usage

        lines = []
        for asin in candidates:
            info = products.get(asin, {})
            lines.append(f"{asin}: {info.get('title', '')} | {info.get('snippet', '')}")

        slot_summary = "; ".join(
            f"{attr}: {' '.join(slot.terms)}" for attr, slot in state.slots.items() if slot.terms
        )
        prompt = (
            f"A shopping customer is in '{state.intent}' mode "
            f"(browsing broadly vs. buying something specific).\n"
            f"Confirmed preferences so far: {slot_summary or 'none yet'}\n"
            "Requirement keywords: " + " ".join(query_terms[:40]) + "\n\n"
            "Candidate products (id: title | features):\n" + "\n".join(lines) + "\n\n"
            "Return ONLY a JSON array of the product ids, ordered best match "
            "first, most likely to satisfy every stated requirement. Include "
            "every id exactly once. No other text."
        )
        try:
            response = self.client.messages.create(
                model=self.model,
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
            ranked.extend(asin for asin in candidates if asin not in seen)
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
            }
            return ranked[:top_k], usage
        except Exception:
            # Any failure (timeout, malformed JSON, API error, rate limit) --
            # fall back to the hybrid order rather than breaking the session.
            return candidates[:top_k], no_usage


# ---------------------------------------------------------------------------
# Agent (public contract — unchanged shape from docs/agent_api_contract.json)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.index = CatalogIndex(Path(catalog_path))
        self.retriever = HybridRetriever(self.index)
        self.intent_router = IntentRouter()
        self.clarifier = ClarificationPolicy()
        self.distiller = ContextDistiller()
        self.reranker = LLMReranker()
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = SessionState()
        tags = user_profile.get("preference_tags") or []
        state.long_term_terms = _terms(" ".join(str(t) for t in tags))
        self._sessions[session_id] = state

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn_count += 1

        is_override = any(pattern.search(user_message) for pattern in OVERRIDE_PATTERNS)
        is_non_info = bool(NON_INFO_RE.search(user_message))

        if state.turn_count == 1:
            state.category_terms = _terms(user_message)[:8]
            state.slots["other"] = Slot(terms=list(state.category_terms), weight=1.0)
        elif is_override:
            self.distiller.apply_override(state, user_message)
        elif not is_non_info:
            self.distiller.apply_answer(state, user_message)
        # else: user explicitly had no preference -- leave slots untouched,
        # but still mark the attribute asked so we don't loop on it.

        state.intent = self.intent_router.classify(user_message, state)

        query_terms = state.weighted_terms()
        candidates, pool_size = self.retriever.retrieve(state, query_terms, top_k)
        state.last_pool_size = pool_size

        ranked_ids, usage = self.reranker.rerank(state, query_terms, candidates, self.index.products, top_k)
        recommendations = [{"parent_asin": asin} for asin in ranked_ids]

        next_attr = self.clarifier.next_attribute(state, pool_size, top_k, turn)
        if next_attr is not None:
            state.asked.add(next_attr)
            state.pending_attribute = next_attr
            phrasing = next_attr.replace("_", " ")
            message = f"Here are some options so far -- do you have a {phrasing} preference?"
        else:
            state.pending_attribute = None
            message = "Here are the closest matches I found."

        return {
            "message": message,
            "ask_attribute": next_attr,
            "recommendations": recommendations,
            "usage": usage,
        }


# ---------------------------------------------------------------------------
# Smoke test — exercises the full pipeline against a tiny synthetic catalog
# so import/index/retrieve/clarify/rerank-fallback all run without needing
# the real 50k-row catalog or an ANTHROPIC_API_KEY. Not part of the graded
# interface; run directly with `python3 agent.py`.
# ---------------------------------------------------------------------------

def _write_fixture_catalog(path: Path) -> None:
    rows = [
        {"parent_asin": "B001", "title": "Men's Leather Chelsea Boots",
         "categories": ["Men", "Shoes", "Boots"], "features": ["genuine leather", "slip resistant"],
         "details": {"Color": "Brown"}, "store": "TrailForge", "description": "Durable everyday boots."},
        {"parent_asin": "B002", "title": "Women's Wool Winter Coat",
         "categories": ["Women", "Coats", "Outerwear"], "features": ["wool blend", "water resistant"],
         "details": {"Color": "Charcoal"}, "store": "Northline", "description": "Warm winter coat."},
        {"parent_asin": "B003", "title": "Silver Hoop Earrings",
         "categories": ["Jewelry", "Earrings"], "features": ["sterling silver", "hypoallergenic"],
         "details": {"Color": "Silver"}, "store": "Lumen", "description": "Everyday hoop earrings."},
        {"parent_asin": "B004", "title": "Men's Running Sneakers",
         "categories": ["Men", "Shoes", "Athletic"], "features": ["breathable mesh", "lightweight"],
         "details": {"Color": "Black"}, "store": "SprintCo", "description": "Lightweight running shoe."},
        {"parent_asin": "B005", "title": "Kids Rain Jacket",
         "categories": ["Kids", "Outerwear", "Rain Gear"], "features": ["waterproof", "reflective trim"],
         "details": {"Color": "Yellow"}, "store": "Northline", "description": "Bright waterproof jacket."},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _run_smoke_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        catalog_path = Path(tmp) / "catalog.jsonl"
        _write_fixture_catalog(catalog_path)

        agent = Agent(catalog_path=catalog_path)
        session_id = "smoke-1"
        agent.reset(session_id, {"preference_tags": ["outdoor", "durable"]})

        turns = [
            "Hi, I'm buying boots for hiking, need something durable.",
            "Brown leather please, budget around $120.",
            "Actually, ignore my earlier preference: I want black instead.",
        ]
        for i, message in enumerate(turns):
            result = agent.respond(session_id, message, turn=i, top_k=3)
            print(f"turn {i}: intent={agent._sessions[session_id].intent}")
            print(f"  message: {result['message']}")
            print(f"  ask_attribute: {result['ask_attribute']}")
            print(f"  recommendations: {result['recommendations']}")
            print(f"  usage: {result['usage']}")

        assert all("parent_asin" in rec for rec in result["recommendations"])
        assert result["ask_attribute"] in ALL_ATTRIBUTES + [None]
        print("\nsmoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()
