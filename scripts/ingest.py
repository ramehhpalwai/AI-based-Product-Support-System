#!/usr/bin/env python
"""Validate the raw ticket JSON and report duplicate stats."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.ingestion import load_tickets
from src.data.transforms import cleaned_doc_text, ticket_as_dict

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate support tickets JSON and report duplicates.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "support_tickets.json"),
        help="Path to tickets JSON (array or NDJSON)",
    )
    args = parser.parse_args()

    valid, invalid = load_tickets(args.input)
    ticket_ids = [t.ticket_id for t in valid]
    doc_keys = [cleaned_doc_text(ticket_as_dict(t)) for t in valid]
    unique_docs = len(set(doc_keys))
    dup_docs = sum(1 for _, n in Counter(doc_keys).items() if n > 1)

    print(f"input:               {args.input}")
    print(f"valid tickets:       {len(valid)}")
    print(f"invalid records:     {len(invalid)}")
    print(f"unique ticket_id:    {len(set(ticket_ids))}")
    print(f"unique cleaned docs: {unique_docs}")
    print(f"duplicated templates:{dup_docs}")
    if invalid:
        print(f"example invalid keys: {list(invalid[0].keys())[:10]}")


if __name__ == "__main__":
    main()
