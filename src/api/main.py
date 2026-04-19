"""
FastAPI service for the Hybrid Recommendation System.

Endpoints
─────────
  GET  /health                       → service & model status
  GET  /recommend/{user_id}          → personalised recommendations
  GET  /similar/{product_id}         → content-similar products
  GET  /products                     → paginated product catalogue
  GET  /products/{product_id}        → single product detail + Bayesian rating

Usage
─────
  # After running train.py:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Environment variables
─────────────────────
  MODEL_PATH   Path to the saved_models directory (default: "saved_models")
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.recommender import HybridRecommender

MODEL_PATH = os.getenv("MODEL_PATH", "saved_models")

# ── global model instance ─────────────────────────────────────────────────────
recommender: Optional[HybridRecommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    model_dir = Path(MODEL_PATH)
    if model_dir.exists() and (model_dir / "hybrid_config.pkl").exists():
        recommender = HybridRecommender()
        recommender.load(MODEL_PATH)
        print(f"Models loaded from '{MODEL_PATH}'  ({len(recommender.all_product_ids)} products)")
    else:
        print(f"WARNING: '{MODEL_PATH}' not found or incomplete. Run `python train.py` first.")
    yield
    # shutdown – nothing to clean up


# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hybrid Recommendation System API",
    description=(
        "3-layer hybrid RS:\n"
        "- **Layer 1** Neural CF (NeuMF) – candidate generation\n"
        "- **Layer 2** Content-Based Filtering (TF-IDF) – personalisation\n"
        "- **Layer 3** Bayesian Average – quality / social proof"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schema models ─────────────────────────────────────────────────────────────

class ScoreBreakdown(BaseModel):
    ncf:      float = Field(..., description="Normalised Neural CF score [0,1]")
    cbf:      float = Field(..., description="Normalised CBF cosine similarity [0,1]")
    bayesian: float = Field(..., description="Normalised Bayesian average [0,1]")
    hybrid:   float = Field(..., description="Weighted fusion score [0,1]")


class RecommendedItem(BaseModel):
    rank:           int
    product_id:     str
    name:           Optional[str]  = None
    category:       Optional[str]  = None
    subcategory:    Optional[str]  = None
    price:          Optional[float] = None
    n_ratings:      Optional[int]  = None
    bayesian_score: Optional[float] = None
    scores:         Optional[ScoreBreakdown] = None


class RecommendationResponse(BaseModel):
    user_id:         str
    top_n:           int
    recommendations: List[RecommendedItem]


class SimilarItem(BaseModel):
    rank:             int
    product_id:       str
    similarity_score: float
    name:             Optional[str]  = None
    category:         Optional[str]  = None
    price:            Optional[float] = None


class SimilarProductsResponse(BaseModel):
    product_id:      str
    similar_products: List[SimilarItem]


class HealthResponse(BaseModel):
    status:       str
    models_loaded: bool
    n_products:   Optional[int] = None
    n_users:      Optional[int] = None


# ── helper ────────────────────────────────────────────────────────────────────

def _require_model():
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run `python train.py` then restart the server."
        )


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check service health and whether models are loaded."""
    loaded = recommender is not None
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        n_products=len(recommender.all_product_ids) if loaded else None,
        n_users=len(recommender.ncf.user_to_idx)   if loaded else None,
    )


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
async def recommend(
    user_id: str,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations"),
    show_scores: bool = Query(default=False, description="Include per-layer score breakdown"),
    candidate_k: int  = Query(default=100, ge=10, le=500,
                              description="NCF candidate pool size before re-ranking"),
):
    """
    Return personalised product recommendations for a user.

    - Known users receive NCF + CBF + Bayesian recommendations.
    - Unknown (cold-start) users fall back to CBF + Bayesian popularity ranking.
    """
    _require_model()

    raw = recommender.recommend(
        user_id=user_id,
        top_n=n,
        candidate_k=candidate_k,
        return_scores=show_scores,
    )

    items = []
    for r in raw:
        scores = None
        if show_scores and "scores" in r:
            scores = ScoreBreakdown(**r["scores"])
        items.append(RecommendedItem(
            rank=r["rank"],
            product_id=r["product_id"],
            name=r.get("name"),
            category=r.get("category"),
            subcategory=r.get("subcategory"),
            price=r.get("price"),
            n_ratings=r.get("n_ratings"),
            bayesian_score=r.get("bayesian_score"),
            scores=scores,
        ))

    return RecommendationResponse(user_id=user_id, top_n=n, recommendations=items)


@app.get(
    "/similar/{product_id}",
    response_model=SimilarProductsResponse,
    tags=["Recommendations"],
)
async def similar_products(
    product_id: str,
    n: int = Query(default=10, ge=1, le=50, description="Number of similar products"),
):
    """Return products most similar to a given product (content-based)."""
    _require_model()

    if product_id not in recommender.all_product_ids:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    raw = recommender.similar_products(product_id, top_k=n)
    items = [
        SimilarItem(
            rank=r["rank"],
            product_id=r["product_id"],
            similarity_score=r["similarity_score"],
            name=r.get("name"),
            category=r.get("category"),
            price=r.get("price"),
        )
        for r in raw
    ]
    return SimilarProductsResponse(product_id=product_id, similar_products=items)


@app.get("/products", tags=["Products"])
async def list_products(
    category: Optional[str] = Query(default=None, description="Filter by category name"),
    limit:    int = Query(default=50, ge=1,  le=500),
    offset:   int = Query(default=0,  ge=0),
) -> Dict[str, Any]:
    """List products with optional category filter and pagination."""
    _require_model()

    df = recommender.products_df.copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    total = len(df)
    page  = df.iloc[offset: offset + limit]
    return {
        "total":    total,
        "offset":   offset,
        "limit":    limit,
        "products": page.where(pd.notna(page), None).to_dict(orient="records"),
    }


@app.get("/products/{product_id}", tags=["Products"])
async def get_product(product_id: str) -> Dict[str, Any]:
    """Get full details for a single product including Bayesian rating stats."""
    _require_model()

    df  = recommender.products_df
    row = df[df["product_id"] == product_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    product = row.iloc[0].where(pd.notna(row.iloc[0]), None).to_dict()
    stats   = recommender.bayesian.get_stats(product_id)
    if stats:
        product.update(stats)
    return product


@app.get("/categories", tags=["Products"])
async def list_categories() -> Dict[str, Any]:
    """List all product categories and their item counts."""
    _require_model()
    counts = (
        recommender.products_df
        .groupby("category")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    return {"categories": counts}
