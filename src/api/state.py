from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.classification.predictions import TicketCategoryPredictor, load_category_predictor
from src.config import Settings, get_settings
from src.data.graph_rag_ingest import load_graph
from src.rag.reranker import GraphMilvusReranker

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    predictor: Optional[TicketCategoryPredictor] = None
    graph: Any = None
    milvus_client: Any = None
    embedder: Any = None
    reranker: GraphMilvusReranker = field(default_factory=GraphMilvusReranker)


def load_runtime(settings: Optional[Settings] = None) -> AppState:
    settings = settings or get_settings()
    state = AppState(settings=settings)

    # Unpickle XGBoost before importing sentence-transformers (OpenMP clash otherwise).
    try:
        state.predictor = load_category_predictor(str(settings.category_model_path))
        logger.info("Loaded category model from %s", settings.category_model_path)
    except Exception:
        logger.exception("Failed to load category model from %s", settings.category_model_path)

    try:
        state.graph = load_graph(str(settings.graph_path))
        logger.info("Loaded support graph from %s", settings.graph_path)
    except Exception:
        logger.exception("Failed to load support graph from %s", settings.graph_path)

    try:
        from sentence_transformers import SentenceTransformer

        state.embedder = SentenceTransformer(settings.embed_model_name, device="cpu")
        logger.info("Loaded embedder %s", settings.embed_model_name)
    except Exception:
        logger.exception("Failed to load embedder %s", settings.embed_model_name)

    try:
        from src.rag.milvus_retrieve import connect_milvus

        state.milvus_client = connect_milvus(url=settings.milvus_uri, retries=3, sleep=1.0)
        logger.info("Connected to Milvus at %s", settings.milvus_uri)
    except Exception:
        logger.warning("Milvus unavailable at %s; retrieval will be degraded", settings.milvus_uri)

    return state
