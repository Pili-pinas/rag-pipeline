"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    max_results: int = Field(5, ge=1, le=20, description="Number of source documents to retrieve")


class Source(BaseModel):
    content: str
    title: str
    source: str
    type: str
    date: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
