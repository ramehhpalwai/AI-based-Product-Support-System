from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.pipeline import _hits_to_solutions, _wrap_milvus_hits
from src.api.state import AppState
from src.config import Settings
from src.rag.milvus_retrieve import build_filter_expr
from src.rag.reranker import GraphMilvusReranker

TICKET = {
    "subject": "Database sync failing with timeout error",
    "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets.",
    "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
    "product": "DataSync Pro",
    "product_module": "sync_engine",
    "priority": "high",
    "channel": "email",
    "customer_tier": "enterprise",
}


class StubPredictor:
    def predict(self, tickets):
        return ["Technical Issue"]


class StubMilvus:
    def list_collections(self):
        return ["support_tickets"]

    def hybrid_search(self, **kwargs):
        raise AssertionError("retrieve tests should patch milvus_hybrid_retrieve, not call the client")


def _state(tmp_path: Path, *, predictor=None, graph=None, milvus_client=None) -> AppState:
    return AppState(
        settings=Settings(feedback_path=tmp_path / "feedback.jsonl"),
        predictor=StubPredictor() if predictor is None else predictor,
        graph=nx.Graph() if graph is None else graph,
        milvus_client=milvus_client,
        embedder=None,
        reranker=GraphMilvusReranker(),
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(state=_state(tmp_path))) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classify(client: TestClient) -> None:
    response = client.post("/v1/tickets/classify", json=TICKET)
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Technical Issue"
    assert body["subcategory"] is None


def test_classify_missing_subject(client: TestClient) -> None:
    payload = {k: v for k, v in TICKET.items() if k != "subject"}
    response = client.post("/v1/tickets/classify", json=payload)
    assert response.status_code == 422


def test_process_rejects_unbounded_top_k(client: TestClient) -> None:
    response = client.post("/v1/tickets/process", json=TICKET, params={"top_k": 10_000})
    assert response.status_code == 422


def test_process_degraded_without_milvus(client: TestClient) -> None:
    response = client.post("/v1/tickets/process", json=TICKET)
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Technical Issue"
    assert body["degraded"] is True
    assert body["solutions"] == []


def test_solutions_without_predictor_when_category_supplied(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.predictor = None
    with TestClient(create_app(state=state)) as client:
        response = client.post(
            "/v1/tickets/solutions",
            json={**TICKET, "category": "Technical Issue"},
        )
        assert response.status_code == 200
        assert response.json()["category"] == "Technical Issue"
        assert response.json()["degraded"] is True


def test_solutions_requires_predictor_when_category_missing(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.predictor = None
    with TestClient(create_app(state=state)) as client:
        response = client.post("/v1/tickets/solutions", json=TICKET)
        assert response.status_code == 503


def test_feedback_appends_jsonl(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/v1/feedback",
        json={"corrected_category": "Data Issue", "resolution_helpful": True},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    feedback_file = tmp_path / "feedback.jsonl"
    assert feedback_file.exists()
    line = feedback_file.read_text(encoding="utf-8").strip()
    assert "Data Issue" in line


def test_wrap_and_rerank_preserves_ticket_fields() -> None:
    flat_rows = [
        {
            "ticket_id": "TK-1",
            "score": 0.2,
            "category": "Technical Issue",
            "product": "DataSync Pro",
            "resolution_code": "CONFIG_CHANGE",
            "resolution_helpful": True,
        },
        {
            "ticket_id": "TK-2",
            "score": 0.9,
            "category": "Technical Issue",
            "product": "DataSync Pro",
            "resolution_code": "WORKAROUND",
            "resolution_helpful": False,
        },
    ]
    reranked = GraphMilvusReranker().rerank(
        _wrap_milvus_hits(flat_rows),
        graph_solution_priors={"solution:CONFIG_CHANGE": 1.0, "solution:WORKAROUND": 0.1},
        pred_category="Technical Issue",
        top_n=2,
    )
    solutions = _hits_to_solutions(reranked)
    assert [s.ticket_id for s in solutions] == ["TK-1", "TK-2"]
    assert solutions[0].resolution_code == "CONFIG_CHANGE"
    assert solutions[0].product == "DataSync Pro"
    assert solutions[0].category == "Technical Issue"


def test_build_filter_expr_escapes_quotes() -> None:
    expr = build_filter_expr(category='Technical" Issue', subcategory='Upgrade" or x=="y')
    assert expr == 'category == "Technical\\" Issue" and subcategory == "Upgrade\\" or x==\\"y"'


def test_process_uses_wrapped_milvus_hits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.api import pipeline as pipeline_mod

    def fake_retrieve(**kwargs):
        return [
            {
                "ticket_id": "TK-99",
                "score": 0.4,
                "category": "Technical Issue",
                "product": "DataSync Pro",
                "resolution_code": "CONFIG_CHANGE",
                "resolution_helpful": True,
            }
        ]

    monkeypatch.setattr(pipeline_mod, "milvus_hybrid_retrieve", fake_retrieve)
    state = _state(tmp_path, milvus_client=StubMilvus())
    with TestClient(create_app(state=state)) as client:
        response = client.post("/v1/tickets/process", json=TICKET)
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is False
    assert body["solutions"][0]["ticket_id"] == "TK-99"
    assert body["solutions"][0]["resolution_code"] == "CONFIG_CHANGE"
