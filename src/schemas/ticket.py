from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class Ticket(BaseModel):
    # Ignore unknown fields so schema can evolve
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    created_at: datetime
    updated_at: datetime

    customer_id: str
    customer_tier: str
    organization_id: str

    product: str
    product_version: str
    product_module: str

    category: str
    subcategory: str
    priority: str
    severity: str
    channel: str

    subject: str
    description: str

    error_logs: str = ""
    stack_trace: str = ""

    customer_sentiment: Optional[str] = None
    previous_tickets: int = 0

    resolution: Optional[str] = None
    resolution_code: Optional[str] = None
    resolved_at: Optional[datetime] = None

    agent_id: Optional[str] = None
    agent_actions: List[str] = Field(default_factory=list)

    escalated: bool = False
    transferred_count: int = 0

    satisfaction_score: Optional[int] = None
    resolution_helpful: Optional[bool] = None

    tags: List[str] = Field(default_factory=list)

    environment: Optional[str] = None
    business_impact: Optional[str] = None

    affected_users: int = 0
    language: str = "en"
    region: str = "NA"

    @field_validator("created_at", "updated_at", "resolved_at", mode="before")
    @classmethod
    def parse_z_datetime(cls, v):
        # accepts "2023-11-02T12:30:10Z"
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v
