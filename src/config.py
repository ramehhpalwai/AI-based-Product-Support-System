from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "support_tickets"
    category_model_path: Path = Path("trained_models/xgb_ticket_model_category.joblib")
    graph_path: Path = Path("data/artifacts/support_graph.pkl")
    feedback_path: Path = Path("data/artifacts/feedback.jsonl")
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
