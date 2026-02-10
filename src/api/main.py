"""FastAPI application for the RAG pipeline."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import QueryRequest, QueryResponse
from src.rag.chain import RAGQueryEngine

load_dotenv()

app = FastAPI(
    title="Philippine Politician RAG API",
    description="RAG pipeline for Philippine government documents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialize the RAG engine on first request
_rag_engine: RAGQueryEngine | None = None


def get_rag_engine() -> RAGQueryEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGQueryEngine()
    return _rag_engine


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        engine = get_rag_engine()
        result = engine.query(request.question, request.max_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
