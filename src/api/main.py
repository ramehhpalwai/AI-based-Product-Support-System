from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.api.pipeline import classify, process_ticket
from src.api.schemas import (
    ClassifyResponse,
    FeedbackIn,
    FeedbackResponse,
    HealthResponse,
    IncomingTicket,
    ProcessResponse,
    ReadyResponse,
)
from src.api.state import AppState, load_runtime
from src.config import get_settings

logger = logging.getLogger(__name__)

TopK = Annotated[int, Query(ge=1, le=100)]
TopN = Annotated[int, Query(ge=1, le=100)]


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_state(request: Request) -> AppState:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Application state is not initialized")
    return runtime


def require_predictor(state: AppState = Depends(get_state)) -> AppState:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Category model is not loaded")
    return state


def require_graph(state: AppState = Depends(get_state)) -> AppState:
    if state.graph is None:
        raise HTTPException(status_code=503, detail="Support graph is not loaded")
    return state


def require_pipeline(state: AppState = Depends(get_state)) -> AppState:
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Category model is not loaded")
    if state.graph is None:
        raise HTTPException(status_code=503, detail="Support graph is not loaded")
    return state


def create_app(state: Optional[AppState] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.runtime = state if state is not None else load_runtime()
        yield

    app = FastAPI(
        title="AI Product Support API",
        description="Classify support tickets and retrieve solutions via Graph-RAG + Milvus hybrid search.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse, tags=["health"])
    def ready(runtime: AppState = Depends(get_state)) -> ReadyResponse:
        milvus_ok = False
        if runtime.milvus_client is not None:
            try:
                runtime.milvus_client.list_collections()
                milvus_ok = True
            except Exception:
                logger.exception("Milvus readiness check failed")
                milvus_ok = False

        predictor_ok = runtime.predictor is not None
        graph_ok = runtime.graph is not None
        if not (predictor_ok and graph_ok and milvus_ok):
            return JSONResponse(
                status_code=503,
                content=ReadyResponse(
                    status="not_ready",
                    predictor=predictor_ok,
                    graph=graph_ok,
                    milvus=milvus_ok,
                ).model_dump(),
            )
        return ReadyResponse(status="ready", predictor=True, graph=True, milvus=True)

    @app.post("/v1/tickets/classify", response_model=ClassifyResponse, tags=["tickets"])
    def classify_ticket(
        ticket: IncomingTicket,
        runtime: AppState = Depends(require_predictor),
    ) -> ClassifyResponse:
        category = classify(runtime, ticket)
        return ClassifyResponse(category=category, subcategory=ticket.subcategory)

    @app.post("/v1/tickets/solutions", response_model=ProcessResponse, tags=["tickets"])
    def ticket_solutions(
        ticket: IncomingTicket,
        runtime: AppState = Depends(require_graph),
        top_k: TopK = 20,
        top_n: TopN = 10,
    ) -> ProcessResponse:
        if ticket.category is None and runtime.predictor is None:
            raise HTTPException(status_code=503, detail="Category model is not loaded")
        return process_ticket(
            runtime,
            ticket,
            predict_category=ticket.category is None,
            top_k=top_k,
            top_n=top_n,
        )

    @app.post("/v1/tickets/process", response_model=ProcessResponse, tags=["tickets"])
    def ticket_process(
        ticket: IncomingTicket,
        runtime: AppState = Depends(require_pipeline),
        top_k: TopK = 20,
        top_n: TopN = 10,
    ) -> ProcessResponse:
        return process_ticket(
            runtime,
            ticket,
            predict_category=True,
            top_k=top_k,
            top_n=top_n,
        )

    @app.post("/v1/feedback", response_model=FeedbackResponse, tags=["feedback"])
    def submit_feedback(
        payload: FeedbackIn,
        runtime: AppState = Depends(get_state),
    ) -> FeedbackResponse:
        path = Path(runtime.settings.feedback_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = payload.model_dump()
        record["recorded_at"] = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return FeedbackResponse()

    return app


app = create_app()


def run() -> None:
    import uvicorn

    configure_logging()
    settings = get_settings()
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
