from __future__ import annotations

from typing import List, Optional
from pymilvus import AnnSearchRequest, RRFRanker, MilvusClient
from sentence_transformers import SentenceTransformer

def milvus_hybrid_retrieve(
    client: MilvusClient,
    collection_name: str,
    query_text: str,
    candidate_ticket_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    top_k: int = 50,
):
    # Build filter expression
    expr_parts = []
    if category:
        expr_parts.append(f'category == "{category}"')
    if subcategory:
        expr_parts.append(f'subcategory == "{subcategory}"')
    if candidate_ticket_ids:
        # strip "ticket:" if present
        ids = [tid.replace("ticket:", "") for tid in candidate_ticket_ids]
        quoted = ",".join([f'"{x}"' for x in ids])
        expr_parts.append(f"ticket_id in [{quoted}]")

    expr = " and ".join(expr_parts) if expr_parts else None

    # Dense query embedding
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dense_q = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

    dense_req = AnnSearchRequest(
        data=[dense_q],
        anns_field="dense_vec",
        param={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
    )

    # Sparse/BM25 request: Milvus uses raw text query for BM25 sparse search
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vec",
        param={"metric_type": "BM25"},
        limit=top_k,
        expr=expr,
    )

    res = client.hybrid_search(
        collection_name=collection_name,
        reqs=[sparse_req, dense_req],
        ranker=RRFRanker(),
        limit=top_k,
        output_fields=["ticket_id", "category", "subcategory", "product", "product_module", "resolution_code", "resolution_helpful"],
    )
    return res
