from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable

from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
    Function,
    FunctionType,
)

from sentence_transformers import SentenceTransformer


# -----------------------------
# Helpers
# -----------------------------
def ticket_to_dict(ticket: Any) -> Dict[str, Any]:
    if hasattr(ticket, "model_dump"):  # pydantic
        return ticket.model_dump()
    if isinstance(ticket, dict):
        return ticket
    return dict(ticket)


def make_doc_text(ticket_row: Dict[str, Any]) -> str:
    return (
        f"{ticket_row.get('subject','')}\n"
        f"{ticket_row.get('description','')}\n"
        f"{ticket_row.get('error_logs','')}"
    ).strip()


def ensure_ticket_id(row: Dict[str, Any]) -> str:
    tid = (row.get("ticket_id") or "").strip()
    if tid:
        return tid
    # generate stable-ish id for this insert run
    return f"gen_{uuid.uuid4().hex}"


# -----------------------------
# Milvus Store (Hybrid Dense + BM25)
# -----------------------------
class MilvusTicketStore:
    def __init__(
        self,
        uri: str = "http://127.0.0.1:19530",
        collection_name: str = "support_tickets",
        dense_dim: int = 384,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        analyzer: str = "english",
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.embed_model_name = embed_model_name
        self.analyzer = analyzer

        self.client: MilvusClient | None = None
        self._embedder: SentenceTransformer | None = None

    def connect(self, retries: int = 15, sleep: float = 1.0) -> MilvusClient:
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                client = MilvusClient(uri=self.uri)
                client.list_collections()  # forces real call
                self.client = client
                return client
            except Exception as e:
                last_err = e
                time.sleep(sleep)
        raise RuntimeError("Milvus not ready / cannot connect") from last_err

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embed_model_name,device='cpu')
        return self._embedder

    def create_collection(self, drop_if_exists: bool = False) -> None:
        if self.client is None:
            raise RuntimeError("Call connect() first")

        c = self.collection_name
        if self.client.has_collection(c):
            if drop_if_exists:
                self.client.drop_collection(c)
            else:
                # already exists, just ensure loaded
                self.client.load_collection(c)
                return

        fields = [
            FieldSchema(name="ticket_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),

            FieldSchema(
                name="doc_text",
                dtype=DataType.VARCHAR,
                max_length=8192,
                enable_analyzer=True,
                analyzer_params={"type": self.analyzer},
                enable_match=True,
            ),

            FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),

            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="subcategory", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="product_module", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="resolution_code", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="resolution_helpful", dtype=DataType.BOOL),
        ]

        bm25_fn = Function(
            name="bm25_doc_text",
            function_type=FunctionType.BM25,
            input_field_names=["doc_text"],
            output_field_names=["sparse_vec"],
        )

        schema = CollectionSchema(
            fields=fields,
            description="Support tickets for hybrid (dense + BM25 sparse) search",
            functions=[bm25_fn],
        )

        self.client.create_collection(collection_name=c, schema=schema)

        index_params = self.client.prepare_index_params()

        # Dense vector index
        index_params.add_index(
            field_name="dense_vec",
            index_type="HNSW",
            metric_type="IP",  # keep IP if normalize_embeddings=True
            params={"M": 16, "efConstruction": 200},
        )

        # Sparse BM25 index
        index_params.add_index(
            field_name="sparse_vec",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={},  # usually empty is fine
        )
        self.client.create_index(collection_name=c, index_params=index_params)
        self.client.load_collection(c)

    def insert_tickets(self, tickets: Iterable[Any], batch_size: int = 256) -> None:
        if self.client is None:
            raise RuntimeError("Call connect() first")

        # Materialize once (we need texts for embedding)
        rows = [ticket_to_dict(t) for t in tickets]
        if not rows:
            return

        doc_texts = [make_doc_text(r) for r in rows]

        dense_vecs = self.embedder.encode(
            doc_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        data_batch = []
        for r, doc_text, dense in zip(rows, doc_texts, dense_vecs):
            data_batch.append(
                {
                    "ticket_id": ensure_ticket_id(r),
                    "doc_text": doc_text,
                    "dense_vec": dense.tolist(),

                    "category": (r.get("category") or ""),
                    "subcategory": (r.get("subcategory") or ""),
                    "product": (r.get("product") or ""),
                    "product_module": (r.get("product_module") or ""),
                    "resolution_code": (r.get("resolution_code") or ""),
                    "resolution_helpful": bool(r.get("resolution_helpful", False)),
                }
            )

            # flush batches
            if len(data_batch) >= batch_size:
                self.client.insert(collection_name=self.collection_name, data=data_batch)
                data_batch = []

        if data_batch:
            self.client.insert(collection_name=self.collection_name, data=data_batch)

    def list_collections(self) -> list[str]:
        if self.client is None:
            raise RuntimeError("Call connect() first")
        return self.client.list_collections()


# -----------------------------
# Local run (optional)
# -----------------------------
if __name__ == "__main__":
    from src.data.ingestion import load_tickets

    json_data = load_tickets(
        "/home/ramesh/Personal_projects/AI-based-Product-Support-System/data/raw/support_tickets.json"
    )
    tickets = json_data[0]  # your current structure

    store = MilvusTicketStore(
        uri="http://127.0.0.1:19530",
        collection_name="support_tickets",
        dense_dim=384,
    )

    store.connect()
    print("connected", store.list_collections())

    store.create_collection(drop_if_exists=True)  # dev
    store.insert_tickets(tickets[:10000])

    print("inserted  tickets")
