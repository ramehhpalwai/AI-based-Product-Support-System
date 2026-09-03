import time
from typing import List, Optional, Dict, Any

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


def _quote_expr_value(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter_expr(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    candidate_ticket_ids: Optional[List[str]] = None,
) -> Optional[str]:
    expr_parts: List[str] = []
    if category:
        expr_parts.append(f"category == {_quote_expr_value(category)}")
    if subcategory:
        expr_parts.append(f"subcategory == {_quote_expr_value(subcategory)}")
    if candidate_ticket_ids:
        ids = [tid.replace("ticket:", "") for tid in candidate_ticket_ids]
        quoted = ",".join(_quote_expr_value(x) for x in ids)
        expr_parts.append(f"ticket_id in [{quoted}]")
    return " and ".join(expr_parts) if expr_parts else None


def connect_milvus(
    url: str = "http://127.0.0.1:19530",
    retries: int = 15,
    sleep: float = 1.0,
) -> MilvusClient:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            client = MilvusClient(uri=url)
            client.list_collections()  # force real call
            return client
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError("Milvus not ready / cannot connect") from last_err


def milvus_hybrid_retrieve(
    client: MilvusClient,
    collection_name: str,
    query_text: str,
    candidate_ticket_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    top_k: int = 20,
    dense_search_params: Optional[Dict[str, Any]] = None,
    embedder: Optional[Any] = None,
):
    expr = build_filter_expr(
        category=category,
        subcategory=subcategory,
        candidate_ticket_ids=candidate_ticket_ids,
    )

    # -------- dense embedding ----------
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    dense_q = embedder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()

    # If you built HNSW index, use ef; if IVF, use nprobe.
    # Default here assumes HNSW 
    if dense_search_params is None:
        dense_search_params = {"metric_type": "IP", "params": {"ef": 64}}

    dense_req = AnnSearchRequest(
        data=[dense_q],
        anns_field="dense_vec",
        param=dense_search_params,
        limit=top_k,
        expr=expr,
    )

    # -------- sparse (BM25) ----------
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vec",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=expr,
    )

    # -------- hybrid search ----------
    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=[sparse_req, dense_req],
        ranker=RRFRanker(),  # you can tune k if needed
        limit=top_k,
        output_fields=[
            "ticket_id",
            "doc_text",  # optional but very useful
            "category",
            "subcategory",
            "product",
            "product_module",
            "resolution_code",
            "resolution_helpful",
        ],
    )

    # -------- normalize result ----------
    hits = res[0] if res else []
    return [_hit_to_row(h) for h in hits]


def _hit_to_row(h: Any) -> dict:
    """Return a flat field dict plus score, regardless of pymilvus hit shape."""
    payload = h
    if not isinstance(h, dict):
        if hasattr(h, "to_dict"):
            payload = h.to_dict()
        else:
            payload = {
                "score": getattr(h, "score", getattr(h, "distance", 0.0)),
                "entity": getattr(h, "entity", None),
                "id": getattr(h, "id", None),
                "distance": getattr(h, "distance", None),
            }

    score = payload.get("score", payload.get("distance", 0.0))
    inner = payload.get("entity")
    if isinstance(inner, dict) and any(k in inner for k in ("ticket_id", "doc_text", "category", "resolution_code")):
        row = dict(inner)
    else:
        row = {k: v for k, v in payload.items() if k not in ("id", "distance", "score", "entity")}
        if not row and isinstance(inner, dict):
            row = dict(inner)
    row["score"] = float(score or 0.0)
    return row


# ---------------- example usage ----------------
if __name__ == "__main__":
    client = connect_milvus()
    results = milvus_hybrid_retrieve(
        client=client,
        collection_name="support_tickets",
        query_text="App crashes when I open dashboard after login",
        top_k=10,
    )
    for r in results[:5]:
        print(r.get("ticket_id"), r["score"], r.get("category"), r.get("resolution_code"))
