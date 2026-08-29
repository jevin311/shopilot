"""Interactive tester: chat with your own Shopilot agent from the terminal.

This is NOT part of the official scoring -- the real evaluator plays a fixed
scripted "customer" against your agent to keep scoring fair and repeatable.
This script exists purely so *you* can be the customer and see, turn by
turn, what your agent asks and recommends. Great for your demo video too.

Run it with:  python try_it.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from starter.agent import Agent

CATALOG_PATH = "data/catalog.jsonl"


def load_titles(limit_ids: set[str]) -> dict[str, str]:
    """Look up product titles for whatever we recommend, just for display."""
    titles: dict[str, str] = {}
    with open(CATALOG_PATH, encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            if asin in limit_ids:
                titles[asin] = str(product.get("title") or asin)[:70]
    return titles


def main() -> None:
    agent = Agent(CATALOG_PATH)
    session_id = f"local_{random.randint(1000, 9999)}"

    # A minimal fake profile -- feel free to edit these values.
    profile = {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": 4.0,
        "rating_style": "usually positive",
        "preference_tags": ["comfort", "durability"],
        "summary": "Prior purchases emphasize comfort and durability.",
    }
    agent.reset(session_id, profile)

    print("=" * 60)
    print("Shopilot -- interactive test (type 'quit' to stop)")
    print("=" * 60)
    print("You are the customer. Try: 'I'm looking for running shoes, need something waterproof'\n")

    for turn in range(1, 11):
        message = input(f"[Turn {turn}] You: ").strip()
        if message.lower() in {"quit", "exit"}:
            break

        response = agent.respond(session_id, message, turn, top_k=5)
        rec_ids = {item["parent_asin"] for item in response["recommendations"]}
        titles = load_titles(rec_ids)

        print(f"\n  Shopilot: {response['message']}")
        if response.get("ask_attribute"):
            print(f"  (internally flagged as asking about: {response['ask_attribute']})")
        if response["recommendations"]:
            print("  Top picks:")
            for i, item in enumerate(response["recommendations"], start=1):
                asin = item["parent_asin"]
                print(f"    {i}. {titles.get(asin, asin)}  [{asin}]")
        else:
            print("  (no matches yet)")
        print()


if __name__ == "__main__":
    main()