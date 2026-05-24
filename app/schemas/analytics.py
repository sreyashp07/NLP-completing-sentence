"""
Pydantic schemas for analytics endpoints.

Defines request and response models for:
- Session analytics
- Intent distribution
- Priority breakdown
- Department routing summary
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class IntentCount(BaseModel):
    intent: str
    display_label: str
    count: int
    percentage: float


class PriorityCount(BaseModel):
    priority: str
    count: int
    percentage: float


class DepartmentSummary(BaseModel):
    department: str
    ticket_count: int
    critical_count: int
    avg_confidence: float


class SessionAnalytics(BaseModel):
    total_tickets: int
    session_start: datetime
    avg_confidence: float
    intent_distribution: List[IntentCount]
    priority_distribution: List[PriorityCount]
    department_summary: List[DepartmentSummary]
    top_intent: Optional[str] = None
    critical_rate: float = Field(
        description="Percentage of critical priority tickets"
    )


class AnalyticsResponse(BaseModel):
    success: bool = True
    analytics: SessionAnalytics
    generated_at: datetime = Field(default_factory=datetime.utcnow)
