"""
deploy/app.py
=============
FastAPI deployment server for SafeLang-1M.
Runs CertifiedVerifier (ONNX) + full pipeline via HF API.

Endpoints:
  POST /classify         → SafetyResult JSON
  POST /classify/batch   → List[SafetyResult]
  GET  /health           → status + version
  GET  /docs             → Swagger UI
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8192, description="Prompt to classify")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)

class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., max_items=32)
    threshold: Optional[float] = 0.5

class ClassifyResponse(BaseModel):
    text: str
    verdict: str
    p_malicious: float
    cert_radius: float
    intent: str
    threat_score: float
    entities: List[str]
    analyst_conclusion: str
    analyst_confidence: float
    proof_string: str
    is_certified: bool
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    model_mode: str
    uptime_s: float


# ---------------------------------------------------------------------------
# App lifespan: load pipeline on startup
# ---------------------------------------------------------------------------
_START_TIME = time.time()
_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline import SafeLangPipeline
    _pipeline = SafeLangPipeline.from_api()
    print("✅ SafeLang-1M API ready")
    yield
    _pipeline = None


app = FastAPI(
    title="SafeLang-1M API",
    description="Certified neurosymbolic LLM safety classification with Z3 proofs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        model_mode="api",
        uptime_s=round(time.time() - _START_TIME, 1),
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = _pipeline.classify(request.text)
        return ClassifyResponse(**result.to_dict(), text=result.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=List[ClassifyResponse])
async def classify_batch(request: BatchClassifyRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        results = _pipeline.classify_batch(request.texts)
        return [ClassifyResponse(**r.to_dict(), text=r.text) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SAFELANG_API_PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
