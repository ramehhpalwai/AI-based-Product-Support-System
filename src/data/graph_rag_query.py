from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import re

import networkx as nx
import numpy as np


# ---------- Query object ----------
@dataclass
class GraphQuery:
    text: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    product: Optional[str] = None
    product_module: Optional[str] = None


# ---------- Node id helpers ----------
def get_issue_node(category: str, subcategory: str) -> str:
    return f"issue:{category}|{subcategory}"

def get_error_node(error_code: str) -> str:
    return f"error:{error_code}"


# ---------- Issue -> Solution (prior) ----------
def top_solutions_for_issue(
    graph: nx.Graph,
    category: str,
    subcategory: str,
    top_k: int = 10,
) -> List[Tuple[str, float, int]]:
    """
    Returns: [(solution_node, helpful_rate, count), ...]
    solution_node includes prefix, e.g. "solution:FEATURE_ADDED"
    """
    issue_node = get_issue_node(category, subcategory)
    if not graph.has_node(issue_node):
        return []

    candidates: List[Tuple[str, float, int]] = []
    for neighbor in graph.neighbors(issue_node):
        if not str(neighbor).startswith("solution:"):
            continue

        edge = graph[issue_node][neighbor]
        if edge.get("rel") != "issue_has_solution":
            continue

        count = int(edge.get("count", 0))
        helpful_count = int(edge.get("helpful_count", 0))
        helpful_rate = helpful_count / count if count else 0.0

        candidates.append((str(neighbor), helpful_rate, count))

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[:top_k]


# ---------- Error code extraction ----------
ERROR_CODE_PATTERN = re.compile(r"\bERROR_[A-Z0-9_]+\b")

def extract_error_codes(text: str) -> List[str]:
    if not text:
        return []
    return list(set(ERROR_CODE_PATTERN.findall(text)))


# ---------- Error -> Ticket -> Solution (precision boost) ----------
def solutions_from_error_codes(
    graph: nx.Graph,
    text: str,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Returns: [(solution_node, score), ...]
    score here is just frequency count across linked tickets (simple + effective).
    """
    error_codes = extract_error_codes(text)
    if not error_codes:
        return []

    solution_scores: Dict[str, float] = {}

    for code in error_codes:
        error_node = get_error_node(code)
        if not graph.has_node(error_node):
            continue

        # tickets connected to this error
        for ticket_node in graph.neighbors(error_node):
            if not str(ticket_node).startswith("ticket:"):
                continue
            if graph[error_node][ticket_node].get("rel") != "mentions_error":
                continue

            # solutions connected to those tickets
            for neighbor in graph.neighbors(ticket_node):
                if not str(neighbor).startswith("solution:"):
                    continue
                if graph[ticket_node][neighbor].get("rel") != "resolved_by":
                    continue

                solution_scores[str(neighbor)] = solution_scores.get(str(neighbor), 0.0) + 1.0

    ranked = sorted(solution_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------- Issue -> Tickets (citations) ----------
def tickets_for_issue(
    graph: nx.Graph,
    category: str,
    subcategory: str,
    top_k: int = 50,
) -> List[str]:
    issue_node = get_issue_node(category, subcategory)
    if not graph.has_node(issue_node):
        return []

    ticket_nodes: List[str] = []
    for neighbor in graph.neighbors(issue_node):
        if str(neighbor).startswith("ticket:") and graph[issue_node][neighbor].get("rel") == "about_issue":
            ticket_nodes.append(str(neighbor))

    return ticket_nodes[:top_k]


# ---------- Combined output ----------
@dataclass
class GraphCandidates:
    solution_nodes: List[str]              # e.g. ["solution:FEATURE_ADDED", ...]
    ticket_nodes: List[str]                # e.g. ["ticket:TK-2024-...", ...]
    solution_priors: Dict[str, float]      # solution_node -> prior score


def graph_rag_candidates(
    graph: nx.Graph,
    query: GraphQuery,
    top_solutions: int = 10,
    top_error_solutions: int = 10,
    top_tickets: int = 50,
) -> GraphCandidates:

    solution_priors: Dict[str, float] = {}

    # 1) issue-level solutions (strong prior)
    if query.category and query.subcategory:
        issue_solutions = top_solutions_for_issue(graph, query.category, query.subcategory, top_k=top_solutions)
        for solution_node, helpful_rate, count in issue_solutions:
            prior = 0.7 * helpful_rate + 0.3 * (np.log1p(count) / 5.0)
            solution_priors[solution_node] = max(solution_priors.get(solution_node, 0.0), float(prior))

    # 2) error-code solutions (precision boost)
    for solution_node, score in solutions_from_error_codes(graph, query.text, top_k=top_error_solutions):
        solution_priors[solution_node] = max(solution_priors.get(solution_node, 0.0), float(0.2 + 0.05 * score))

    # 3) tickets for citations
    ticket_nodes: List[str] = []
    if query.category and query.subcategory:
        ticket_nodes = tickets_for_issue(graph, query.category, query.subcategory, top_k=top_tickets)

    # final ordering
    solution_nodes = sorted(solution_priors.keys(), key=lambda s: solution_priors[s], reverse=True)

    return GraphCandidates(solution_nodes=solution_nodes, ticket_nodes=ticket_nodes, solution_priors=solution_priors)
