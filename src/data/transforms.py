from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def ticket_as_dict(ticket: Any) -> Dict[str, Any]:
    if hasattr(ticket, "model_dump"):
        return ticket.model_dump()
    if isinstance(ticket, dict):
        return ticket
    return dict(ticket)


def cleaned_doc_text(ticket_row: Dict[str, Any]) -> str:
    """Subject + description + error logs with timestamps/IDs stripped.

    Matches the text used for classification / Milvus embedding, minus log noise
    that otherwise defeats near-duplicate detection.
    """
    from src.classification.data_processing import clean_error_logs_preserve_codes

    cleaned_logs = clean_error_logs_preserve_codes(ticket_row.get("error_logs", "") or "")
    return (
        f"{ticket_row.get('subject', '')}\n"
        f"{ticket_row.get('description', '')}\n"
        f"{cleaned_logs}"
    ).strip()


def _helpful(row: Dict[str, Any]) -> int:
    return 1 if row.get("resolution_helpful") else 0


def _satisfaction(row: Dict[str, Any]) -> float:
    score = row.get("satisfaction_score")
    try:
        return float(score) if score is not None else -1.0
    except (TypeError, ValueError):
        return -1.0


def dedup_tickets_for_index(tickets: Iterable[Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Keep one representative per cleaned doc_text.

    Prefer ``resolution_helpful=True``, then highest ``satisfaction_score``.
    Returns ``(unique_rows, original_count)``.
    """
    best: Dict[str, Dict[str, Any]] = {}
    original = 0
    for ticket in tickets:
        original += 1
        row = ticket_as_dict(ticket)
        key = cleaned_doc_text(row) or (row.get("ticket_id") or str(original))
        prev = best.get(key)
        if prev is None or (_helpful(row), _satisfaction(row)) > (_helpful(prev), _satisfaction(prev)):
            best[key] = row
    return list(best.values()), original
