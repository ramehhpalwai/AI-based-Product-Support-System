#!/usr/bin/env python
"""Build and pickle the Product ↔ Issue ↔ Solution ↔ Ticket graph (full dataset, no dedup)."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.graph_rag_ingest import build_and_save_graph
from src.data.ingestion import load_tickets

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the support Graph-RAG pickle from tickets JSON.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "support_tickets.json"),
        help="Path to tickets JSON",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "artifacts" / "support_graph.pkl"),
        help="Output pickle path",
    )
    args = parser.parse_args()

    valid, invalid = load_tickets(args.input)
    print(f"loaded {len(valid)} valid tickets ({len(invalid)} invalid)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    graph = build_and_save_graph(valid, args.output)
    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
