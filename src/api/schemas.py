from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class IncomingTicket(BaseModel):
    """Inbound ticket for classify / retrieve. Not the historical resolved Ticket schema."""

    subject: str
    description: str
    error_logs: str = ""
    product: str = ""
    product_module: str = ""
    priority: str = ""
    channel: str = ""
    customer_tier: str = ""
    category: Optional[str] = None
    subcategory: Optional[str] = None


class ClassifyResponse(BaseModel):
    category: str
    subcategory: Optional[str] = None


class SolutionHit(BaseModel):
    ticket_id: Optional[str] = None
    subject: Optional[str] = None
    description: Optional[str] = None
    error_logs: Optional[str] = None
    final_score: float
    resolution_code: Optional[str] = None
    category: Optional[str] = None
    product: Optional[str] = None
    graph_prior: float = 0.0
    milvus_score: float = 0.0


class GraphInfo(BaseModel):
    solution_nodes: List[str] = Field(default_factory=list)
    ticket_nodes: List[str] = Field(default_factory=list)


class ProcessResponse(BaseModel):
    category: str
    subcategory: Optional[str] = None
    solutions: List[SolutionHit] = Field(default_factory=list)
    graph: GraphInfo = Field(default_factory=GraphInfo)
    degraded: bool = False


class FeedbackIn(BaseModel):
    ticket_id: Optional[str] = None
    predicted_category: Optional[str] = None
    corrected_category: Optional[str] = None
    resolution_helpful: Optional[bool] = None
    satisfaction_score: Optional[int] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    ok: bool = True
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    status: str
    predictor: bool
    graph: bool
    milvus: bool
