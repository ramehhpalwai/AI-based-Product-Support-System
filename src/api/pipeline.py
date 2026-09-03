from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.api.schemas import GraphInfo, IncomingTicket, ProcessResponse, SolutionHit
from src.api.state import AppState
from src.data.graph_rag_query import GraphQuery, graph_rag_candidates
from src.rag.milvus_retrieve import milvus_hybrid_retrieve
from src.rag.milvus_store import split_doc_text

logger = logging.getLogger(__name__)


def query_text(ticket: IncomingTicket) -> str:
    return f"{ticket.subject} {ticket.description} {ticket.error_logs}".strip()


def classify(state: AppState, ticket: IncomingTicket) -> str:
    if state.predictor is None:
        raise RuntimeError("Category model is not loaded")
    labels = state.predictor.predict([ticket.model_dump()])
    return labels[0]


def _wrap_milvus_hits(rows: List[dict]) -> List[dict]:
    """Adapt retriever dicts to the nested {score, entity} shape the reranker expects."""
    wrapped = []
    for row in rows:
        score = row.get("score", row.get("distance", 0.0))
        nested = row.get("entity")
        if isinstance(nested, dict) and any(
            k in nested for k in ("ticket_id", "doc_text", "category", "resolution_code")
        ):
            fields = dict(nested)
        else:
            fields = {k: v for k, v in row.items() if k not in ("id", "distance", "score", "entity")}
        wrapped.append({"score": score, "entity": fields})
    return wrapped


def _hits_to_solutions(reranked: List[dict]) -> List[SolutionHit]:
    out: List[SolutionHit] = []
    for hit in reranked:
        fields = hit.get("fields") or {}
        subject, description, error_logs = split_doc_text(fields.get("doc_text") or "")
        out.append(
            SolutionHit(
                ticket_id=hit.get("ticket_id") or fields.get("ticket_id"),
                subject=subject or None,
                description=description or None,
                error_logs=error_logs or None,
                final_score=float(hit.get("final_score") or 0.0),
                resolution_code=fields.get("resolution_code"),
                category=fields.get("category"),
                product=fields.get("product"),
                graph_prior=float(hit.get("graph_prior") or 0.0),
                milvus_score=float(hit.get("milvus_score") or 0.0),
            )
        )
    return out


def retrieve(
    state: AppState,
    ticket: IncomingTicket,
    category: str,
    subcategory: Optional[str] = None,
    top_k: int = 20,
    top_n: int = 10,
) -> Tuple[List[SolutionHit], GraphInfo, bool]:
    if state.graph is None:
        raise RuntimeError("Support graph is not loaded")

    text = query_text(ticket)
    gq = GraphQuery(
        text=text,
        category=category,
        subcategory=subcategory,
        product=ticket.product or None,
        product_module=ticket.product_module or None,
    )
    candidates = graph_rag_candidates(state.graph, gq)

    degraded = False
    milvus_rows: List[dict] = []
    if state.milvus_client is None:
        degraded = True
    else:
        try:
            milvus_rows = milvus_hybrid_retrieve(
                client=state.milvus_client,
                collection_name=state.settings.milvus_collection,
                query_text=text,
                category=category,
                subcategory=subcategory,
                top_k=top_k,
                embedder=state.embedder,
            )
        except Exception:
            logger.exception("Milvus hybrid retrieve failed; returning graph-only results")
            degraded = True

    reranked = state.reranker.rerank(
        _wrap_milvus_hits(milvus_rows),
        graph_solution_priors=candidates.solution_priors,
        pred_category=category,
        pred_subcategory=subcategory,
        top_n=top_n,
    )
    graph_info = GraphInfo(
        solution_nodes=list(candidates.solution_nodes),
        ticket_nodes=list(candidates.ticket_nodes),
    )
    return _hits_to_solutions(reranked), graph_info, degraded


def process_ticket(
    state: AppState,
    ticket: IncomingTicket,
    *,
    predict_category: bool = True,
    top_k: int = 20,
    top_n: int = 10,
) -> ProcessResponse:
    if predict_category or not ticket.category:
        category = classify(state, ticket)
    else:
        category = ticket.category
    subcategory = ticket.subcategory
    solutions, graph_info, degraded = retrieve(
        state,
        ticket,
        category=category,
        subcategory=subcategory,
        top_k=top_k,
        top_n=top_n,
    )
    return ProcessResponse(
        category=category,
        subcategory=subcategory,
        solutions=solutions,
        graph=graph_info,
        degraded=degraded,
    )
