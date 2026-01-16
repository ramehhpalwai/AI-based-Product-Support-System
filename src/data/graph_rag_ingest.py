
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pickle
import networkx as nx
from typing import Iterable, Any

# Example matches: ERROR_DISK_FULL, ERROR_SERVER_500, ERROR_RATELIMIT_429
ERROR_CODE_PATTERN = re.compile(r"\bERROR_[A-Z0-9_]+\b")


def extract_error_codes(text: str) -> List[str]:
    """Extract unique error codes from text."""
    if not text:
        return []
    return list(set(ERROR_CODE_PATTERN.findall(text)))


def get_solution_identifier(ticket_record: Dict[str, Any]) -> str:
    """
    Prefer stable solution identifiers if present.
    Fallback to a short hash of the resolution text.
    """
    stable_id = ticket_record.get("resolution_code") or ticket_record.get("resolution_template_used")
    if stable_id:
        return str(stable_id)

    resolution_text = (ticket_record.get("resolution") or "").strip()
    if not resolution_text:
        return "UNKNOWN_SOLUTION"

    digest = hashlib.md5(resolution_text.encode("utf-8")).hexdigest()[:10]
    return f"RES_HASH_{digest}"


def build_support_graph(tickets: Iterable[Any]) -> nx.Graph:
    """
    Build a Graph-RAG graph that maps:
      Product ↔ Issue ↔ Solution ↔ Ticket (+ ErrorCode nodes)

    tickets can be:
      - Pydantic Ticket objects (has model_dump)
      - dicts
      - any dict-like record
    """
    graph = nx.Graph()

    def add_or_increment_edge(
        source_node: str,
        target_node: str,
        relationship: str,
        increment_count: int = 1,
        increment_helpful: int = 0,
    ) -> None:
        if graph.has_edge(source_node, target_node):
            graph[source_node][target_node]["count"] = graph[source_node][target_node].get("count", 0) + increment_count
            if increment_helpful:
                graph[source_node][target_node]["helpful_count"] = (
                    graph[source_node][target_node].get("helpful_count", 0) + increment_helpful
                )
        else:
            edge_attrs = {"rel": relationship, "count": increment_count}
            if increment_helpful:
                edge_attrs["helpful_count"] = increment_helpful
            graph.add_edge(source_node, target_node, **edge_attrs)

    for ticket in tickets:
        ticket_record: Dict[str, Any]
        if hasattr(ticket, "model_dump"):
            ticket_record = ticket.model_dump()
        elif isinstance(ticket, dict):
            ticket_record = ticket
        else:
            ticket_record = dict(ticket)

        ticket_id = str(ticket_record.get("ticket_id", "")).strip()
        if not ticket_id:
            continue

        # ---- Node IDs ----
        ticket_node = f"ticket:{ticket_id}"

        product_name = str(ticket_record.get("product", "")).strip()
        product_node = f"product:{product_name}" if product_name else None

        module_name = str(ticket_record.get("product_module", "")).strip()
        module_node = f"module:{module_name}" if module_name else None

        category = str(ticket_record.get("category", "")).strip()
        subcategory = str(ticket_record.get("subcategory", "")).strip()
        issue_node = f"issue:{category}|{subcategory}" if category and subcategory else None

        solution_identifier = get_solution_identifier(ticket_record)
        solution_node = f"solution:{solution_identifier}"

        resolution_helpful = bool(ticket_record.get("resolution_helpful", False))

        combined_text = (
            f"{ticket_record.get('subject','')} "
            f"{ticket_record.get('description','')} "
            f"{ticket_record.get('error_logs','')}"
        )
        error_codes = extract_error_codes(combined_text)

        # ---- Add nodes ----
        graph.add_node(ticket_node, kind="ticket")

        # Ticket → Product
        if product_node:
            graph.add_node(product_node, kind="product", name=product_name)
            graph.add_edge(ticket_node, product_node, rel="for_product")

        # Ticket → Module
        if module_node:
            graph.add_node(module_node, kind="module", name=module_name)
            graph.add_edge(ticket_node, module_node, rel="in_module")

        # Ticket → Issue
        if issue_node:
            graph.add_node(issue_node, kind="issue", category=category, subcategory=subcategory)
            graph.add_edge(ticket_node, issue_node, rel="about_issue")

        # Ticket → Solution (always)
        graph.add_node(solution_node, kind="solution", solution_id=solution_identifier)
        graph.add_edge(ticket_node, solution_node, rel="resolved_by", helpful=resolution_helpful)

        # ---- Aggregated edges: Product ↔ Issue ↔ Solution ----
        if product_node and issue_node:
            add_or_increment_edge(product_node, issue_node, relationship="product_sees_issue", increment_count=1)

        if issue_node and solution_node:
            add_or_increment_edge(
                issue_node,
                solution_node,
                relationship="issue_has_solution",
                increment_count=1,
                increment_helpful=int(resolution_helpful),
            )

        # ---- Error code links ----
        for error_code in error_codes:
            error_node = f"error:{error_code}"
            graph.add_node(error_node, kind="error_code", code=error_code)

            # Ticket ↔ ErrorCode
            graph.add_edge(ticket_node, error_node, rel="mentions_error")

            # Issue ↔ ErrorCode (aggregated)
            if issue_node:
                add_or_increment_edge(issue_node, error_node, relationship="issue_mentions_error", increment_count=1)

    return graph



def build_and_save_graph(tickets: Iterable[Any], path: str) -> nx.Graph:
    g = build_support_graph(tickets)
    with open(path, "wb") as f:
        pickle.dump(g, f)
    return g

def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ =="__main__":
    from src.data.ingestion import load_tickets
    json_data = load_tickets("data/raw/support_tickets.json")
    Data_tickets = json_data[0]
    path = "data/artifacts/support_graph.pkl"
    build_and_save_graph(Data_tickets,path)

# # ---- Example usage ----
# support_graph = build_support_graph(tickets)

# print("nodes:", support_graph.number_of_nodes())
# print("edges:", support_graph.number_of_edges())

# print(top_solutions_for_issue(support_graph, "Account Management", "Upgrade", top_k=5))

