"""
Shared Pydantic schemas used across all routers.
"""
from typing import Any, Dict, Generic, List, Optional, TypeVar
from datetime import datetime

from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageResponse(BaseModel):
    message: str
    count: int = 0

class ErrorResponse(BaseModel):
    detail: str

class PaginatedResponse(BaseModel):
    """Offset-based paginated response for master data."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int

class CursorPaginatedResponse(BaseModel):
    """Cursor-based paginated response for telemetry."""
    items: List[Any]
    next_cursor: Optional[int] = None
    has_more: bool = False
    limit: int
