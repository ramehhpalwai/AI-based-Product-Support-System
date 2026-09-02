#!/usr/bin/env python
"""Index deduplicated tickets into Milvus for hybrid dense + BM25 search."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.ingestion import load_tickets
from src.data.transforms import dedup_tickets_for_index
from src.rag.milvus_store import MilvusTicketStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Milvus support_tickets collection.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "support_tickets.json"),
        help="Path to tickets JSON",
    )
    parser.add_argument("--uri", default="http://127.0.0.1:19530", help="Milvus URI")
    parser.add_argument("--collection", default="support_tickets", help="Collection name")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max unique rows to insert after dedup (0 = no cap)",
    )
    parser.add_argument("--drop", action="store_true", help="Drop existing collection before insert")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    valid, invalid = load_tickets(args.input)
    print(f"loaded {len(valid)} valid tickets ({len(invalid)} invalid)")

    unique_rows, total = dedup_tickets_for_index(valid)
    n_unique = len(unique_rows)
    if args.limit and args.limit > 0:
        unique_rows = unique_rows[: args.limit]

    print(f"{total} -> {n_unique} unique -> {len(unique_rows)} inserted")

    store = MilvusTicketStore(uri=args.uri, collection_name=args.collection)
    store.connect()
    print("connected", store.list_collections())
    store.create_collection(drop_if_exists=args.drop)
    store.insert_tickets(unique_rows, batch_size=args.batch_size)
    print(f"indexed {len(unique_rows)} tickets into {args.collection}")


if __name__ == "__main__":
    main()
