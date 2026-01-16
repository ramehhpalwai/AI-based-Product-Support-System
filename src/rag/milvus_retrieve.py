import time
from typing import List, Optional, Dict, Any

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import SentenceTransformer


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
    dense_search_params: Optional[Dict[str, Any]] = None,  # allow override
):
    # -------- filter expression ----------
    expr_parts = []
    if category:
        expr_parts.append(f'category == "{category}"')
    if subcategory:
        expr_parts.append(f'subcategory == "{subcategory}"')
    if candidate_ticket_ids:
        ids = [tid.replace("ticket:", "") for tid in candidate_ticket_ids]
        quoted = ",".join([f'"{x}"' for x in ids])
        expr_parts.append(f"ticket_id in [{quoted}]")
    expr = " and ".join(expr_parts) if expr_parts else None

    # -------- dense embedding ----------
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device='cpu')
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
    # res is typically: List[List[Hit]] (one list per query vector; you have 1 query)
    hits = res[0] if res else []
    out = []
    for h in hits:
        row = dict(h.entity) if hasattr(h, "entity") else {}
        row["score"] = float(getattr(h, "score", 0.0))
        out.append(row)

    return out


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
        print(r['entity']["ticket_id"], r["score"], r['entity'].get("category"), r['entity'].get("resolution_code"))
