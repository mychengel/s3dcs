"""
Hybrid Recommendation Pipeline – combines all three layers.

Flow
────
  1. Layer 1 (NCF)      → candidate pool of top-K items per user
  2. Layer 2 (CBF)      → re-score candidates with content similarity
  3. Layer 3 (Bayesian) → adjust scores with quality/social-proof signal
  4. Weighted sum        → final ranking, return top-N

Score fusion (all sub-scores are normalised to [0,1] before weighting):
    hybrid = w_ncf × s_ncf  +  w_cbf × s_cbf  +  w_bay × s_bayesian
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.neural_cf       import NCFRecommender
from src.models.content_based   import CBFRecommender
from src.models.review_analysis import BayesianAverageScorer


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


class HybridRecommender:
    """
    3-layer hybrid recommendation system.

    Parameters
    ----------
    ncf_weight, cbf_weight, bayesian_weight
        Fusion weights (must sum to 1.0).
    candidate_pool_size
        How many NCF candidates to consider before CBF/Bayesian re-ranking.
    """

    def __init__(
        self,
        ncf_weight:          float = 0.50,
        cbf_weight:          float = 0.30,
        bayesian_weight:     float = 0.20,
        candidate_pool_size: int   = 100,
    ):
        total = ncf_weight + cbf_weight + bayesian_weight
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

        self.ncf_weight          = ncf_weight
        self.cbf_weight          = cbf_weight
        self.bayesian_weight     = bayesian_weight
        self.candidate_pool_size = candidate_pool_size

        self.ncf      = NCFRecommender()
        self.cbf      = CBFRecommender()
        self.bayesian = BayesianAverageScorer()

        self.all_product_ids: List[str] = []
        self.products_df: Optional[pd.DataFrame] = None

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        transactions_df: pd.DataFrame,
        products_df:     pd.DataFrame,
        reviews_df:      pd.DataFrame,
        ncf_epochs:      int  = 20,
        verbose:         bool = True,
    ) -> "HybridRecommender":
        # Normalise IDs
        def _str_ids(df, *cols):
            df = df.copy()
            for c in cols:
                if c in df.columns:
                    df[c] = df[c].astype(str)
            return df

        transactions_df = _str_ids(transactions_df, "user_id", "product_id")
        products_df     = _str_ids(products_df,     "product_id")
        reviews_df      = _str_ids(reviews_df,      "user_id", "product_id")

        self.all_product_ids = products_df["product_id"].tolist()
        self.products_df     = products_df.reset_index(drop=True)

        if verbose:
            print("=" * 50)
            print("Layer 1 — Neural Collaborative Filtering (NeuMF)")
            print("=" * 50)
        self.ncf.num_epochs = ncf_epochs
        self.ncf.fit(transactions_df, verbose=verbose)

        if verbose:
            print("\n" + "=" * 50)
            print("Layer 2 — Content-Based Filtering (TF-IDF)")
            print("=" * 50)
        self.cbf.fit(products_df, transactions_df, verbose=verbose)

        if verbose:
            print("\n" + "=" * 50)
            print("Layer 3 — Bayesian Average (Quality Signal)")
            print("=" * 50)
        self.bayesian.fit(reviews_df, verbose=verbose)

        return self

    # ── inference ────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id:       str,
        top_n:         int            = 10,
        candidate_k:   Optional[int]  = None,
        return_scores: bool           = False,
    ) -> List[Dict[str, Any]]:
        """
        Return top-N recommended products for a user.

        Parameters
        ----------
        user_id       : Target user (string ID).
        top_n         : Number of final recommendations to return.
        candidate_k   : Override the candidate pool size.
        return_scores : Whether to include per-layer score breakdown.
        """
        user_id = str(user_id)
        k = candidate_k or self.candidate_pool_size

        # ── Layer 1: candidate generation ───────────────────────────────────
        ncf_pairs = self.ncf.recommend(user_id, self.all_product_ids, top_k=k)

        if ncf_pairs:
            candidate_ids = [iid for iid, _ in ncf_pairs]
            ncf_raw       = np.array([s for _, s in ncf_pairs])
        else:
            # Cold-start fallback: use all products, uniform NCF score
            candidate_ids = self.all_product_ids[:k]
            ncf_raw       = np.zeros(len(candidate_ids))

        # ── Layer 2: CBF re-scoring ──────────────────────────────────────────
        cbf_raw = self.cbf.score(user_id, candidate_ids)

        # ── Layer 3: Bayesian quality signal ────────────────────────────────
        bay_raw = self.bayesian.score(candidate_ids)

        # ── Normalise each layer to [0, 1] and fuse ─────────────────────────
        s_ncf = _minmax(ncf_raw)
        s_cbf = _minmax(cbf_raw)
        s_bay = _minmax(bay_raw)

        hybrid = (
            self.ncf_weight      * s_ncf +
            self.cbf_weight      * s_cbf +
            self.bayesian_weight * s_bay
        )

        # Sort descending, take top-N
        top_idx = np.argsort(hybrid)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            pid  = candidate_ids[idx]
            item = {"rank": rank, "product_id": pid}
            item.update(self._product_info(pid))

            if return_scores:
                item["scores"] = {
                    "ncf":      round(float(s_ncf[idx]), 4),
                    "cbf":      round(float(s_cbf[idx]), 4),
                    "bayesian": round(float(s_bay[idx]), 4),
                    "hybrid":   round(float(hybrid[idx]), 4),
                }
            results.append(item)

        return results

    def similar_products(self, product_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Content-similarity based similar-products lookup (CBF)."""
        pairs = self.cbf.similar_items(str(product_id), top_k=top_k)
        results = []
        for rank, (pid, sim) in enumerate(pairs, start=1):
            item = {"rank": rank, "product_id": pid, "similarity_score": round(sim, 4)}
            item.update(self._product_info(pid))
            results.append(item)
        return results

    # ── helpers ──────────────────────────────────────────────────────────────

    def _product_info(self, product_id: str) -> Dict[str, Any]:
        if self.products_df is None:
            return {}
        row = self.products_df[self.products_df["product_id"] == product_id]
        if row.empty:
            return {}
        info: Dict[str, Any] = {}
        for col in ("name", "category", "subcategory", "price", "description"):
            if col in row.columns:
                val = row.iloc[0][col]
                info[col] = None if pd.isna(val) else val
        stats = self.bayesian.get_stats(product_id)
        if stats:
            info["n_ratings"]      = stats["n_ratings"]
            info["bayesian_score"] = stats["bayesian_score"]
        return info

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        self.ncf.save(str(d / "ncf"))
        self.cbf.save(str(d / "cbf"))
        self.bayesian.save(str(d / "bayesian"))

        with open(d / "hybrid_config.pkl", "wb") as f:
            pickle.dump({
                "ncf_weight":          self.ncf_weight,
                "cbf_weight":          self.cbf_weight,
                "bayesian_weight":     self.bayesian_weight,
                "candidate_pool_size": self.candidate_pool_size,
                "all_product_ids":     self.all_product_ids,
                "products_df":         self.products_df,
            }, f)

        print(f"\nHybrid recommender saved → {d}")

    def load(self, path: str) -> "HybridRecommender":
        d = Path(path)

        with open(d / "hybrid_config.pkl", "rb") as f:
            cfg = pickle.load(f)

        self.ncf_weight          = cfg["ncf_weight"]
        self.cbf_weight          = cfg["cbf_weight"]
        self.bayesian_weight     = cfg["bayesian_weight"]
        self.candidate_pool_size = cfg["candidate_pool_size"]
        self.all_product_ids     = cfg["all_product_ids"]
        self.products_df         = cfg["products_df"]

        self.ncf.load(str(d / "ncf"))
        self.cbf.load(str(d / "cbf"))
        self.bayesian.load(str(d / "bayesian"))

        print(f"Hybrid recommender loaded ← {d}")
        return self
