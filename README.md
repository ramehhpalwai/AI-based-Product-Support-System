# AI-based Product Support System

End-to-end ticket classification and solution retrieval: XGBoost category model, NetworkX Graph-RAG, and Milvus hybrid (dense + BM25) search, exposed as a FastAPI service.

Architecture (diagrams, data flows, technology choices): [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup

```bash
uv sync --extra dev
docker compose up -d etcd minio milvus
```

Milvus is at `http://127.0.0.1:19530`. Build artifacts (once, or after data changes):

```bash
uv run python scripts/ingest.py            # validate JSON + duplicate stats
uv run python scripts/build_graph.py       # full dataset -> data/artifacts/support_graph.pkl
uv run python scripts/build_milvus.py --drop   # dedup ~110k -> ~1,950 unique, then index
```

`build_graph.py` uses **all** tickets (graph priors need full counts). `build_milvus.py` deduplicates on cleaned ticket text (timestamps/IDs stripped) and keeps one representative per template (`resolution_helpful`, then highest `satisfaction_score`). `--limit 10000` applies after dedup (default 10000; use `--limit 0` for no cap).

## Run the API

```bash
uv run uvicorn src.api.main:app --reload --port 8000
# or: uv run serve
```

Or run the API next to Milvus (graph pickle is mounted from `data/artifacts/`, not baked into the image):

```bash
docker compose up -d --build
```

Docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Process is up |
| GET | `/ready` | Model + graph loaded and Milvus reachable (503 otherwise) |
| POST | `/v1/tickets/classify` | Predict category |
| POST | `/v1/tickets/solutions` | Retrieve/rerank solutions (classifies unless `category` is supplied) |
| POST | `/v1/tickets/process` | Classify then retrieve (primary pipeline) |
| POST | `/v1/feedback` | Append agent correction / satisfaction JSONL (no retrain) |

If Milvus is down, `/process` and `/solutions` still return graph candidates with `"degraded": true`. Missing model or graph is a 503.

### Sample: process a ticket

```bash
curl -s http://127.0.0.1:8000/v1/tickets/process \
  -H 'Content-Type: application/json' \
  -d '{
    "subject": "Database sync failing with timeout error",
    "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets.",
    "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
    "product": "DataSync Pro",
    "product_module": "sync_engine",
    "priority": "high",
    "channel": "email",
    "customer_tier": "enterprise"
  }'
```

Example response:

```json
{
  "category": "Technical Issue",
  "subcategory": null,
  "solutions": [
    {
      "ticket_id": "TK-2024-001234",
      "subject": "Database sync failing with timeout error",
      "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets.",
      "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
      "final_score": 0.82,
      "resolution_code": "CONFIG_CHANGE",
      "category": "Technical Issue",
      "product": "DataSync Pro",
      "graph_prior": 0.61,
      "milvus_score": 0.74
    }
  ],
  "graph": {
    "solution_nodes": ["solution:CONFIG_CHANGE"],
    "ticket_nodes": ["ticket:TK-2024-001230"]
  },
  "degraded": false
}
```

Optional `subcategory` on the request is a client hint (no subcategory model is shipped). Graph issue-level priors are stronger when both category and subcategory are present.

### Feedback

```bash
curl -s http://127.0.0.1:8000/v1/feedback \
  -H 'Content-Type: application/json' \
  -d '{"ticket_id": "TK-2024-001234", "predicted_category": "Technical Issue", "corrected_category": "Data Issue", "resolution_helpful": true}'
```

Appends to `data/artifacts/feedback.jsonl`.

## Tests

```bash
uv run pytest
```
