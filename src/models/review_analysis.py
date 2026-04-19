"""
Layer 3 – Quality Assurance / Social Proof: Bayesian Average Rating.

Bayesian Average formula
────────────────────────
    BA(p) = (C × m  +  Σ_i r_i) / (C + n)

Where:
    C  = confidence weight (= min_count, the "prior strength")
    m  = global mean rating across all products
    n  = number of ratings for product p
    Σr = sum of ratings for product p

Effect:
  • Products with few ratings are pulled toward the global mean (conservative).
  • Products with many ratings converge to their true empirical mean.
  • This prevents low-volume items with a single 5-star review from dominating.

The final score() method returns values normalised to [0, 1] assuming a 1–5 scale,
ready for linear combination with the NCF and CBF scores.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BayesianAverageScorer:
    """Compute per-product Bayesian average scores from review ratings."""

    def __init__(self, min_count: int = 10, rating_min: float = 1.0, rating_max: float = 5.0):
        self.min_count   = min_count      # C in the formula
        self.rating_min  = rating_min
        self.rating_max  = rating_max

        self.global_mean:    float = 3.0
        self.product_scores: Dict[str, float] = {}   # product_id → BA score (raw, 1–5)
        self.product_stats:  Dict[str, dict]  = {}   # for introspection / API

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, reviews_df: pd.DataFrame, verbose: bool = True) -> "BayesianAverageScorer":
        ratings = pd.to_numeric(reviews_df["rating"], errors="coerce").dropna()
        self.global_mean = float(ratings.mean())

        agg = reviews_df.groupby("product_id")["rating"].agg(
            n="count", sum_r="sum", mean_r="mean"
        )

        for pid, row in agg.iterrows():
            n     = int(row["n"])
            sum_r = float(row["sum_r"])
            ba    = (self.min_count * self.global_mean + sum_r) / (self.min_count + n)

            self.product_scores[str(pid)] = ba
            self.product_stats[str(pid)] = {
                "n_ratings":      n,
                "mean_rating":    round(float(row["mean_r"]), 3),
                "bayesian_score": round(ba, 3),
            }

        if verbose:
            print(f"    Bayesian scores for {len(self.product_scores)} products  "
                  f"(global mean={self.global_mean:.3f}, C={self.min_count})")

        return self

    def score(self, item_ids: List[str]) -> np.ndarray:
        """
        Return Bayesian scores normalised to [0, 1] for each item.
        Products without reviews receive the global mean (after normalisation).
        """
        scale = self.rating_max - self.rating_min
        raw = np.array([
            self.product_scores.get(str(iid), self.global_mean)
            for iid in item_ids
        ])
        return (raw - self.rating_min) / scale   # → [0, 1]

    def get_stats(self, product_id: str) -> Optional[dict]:
        return self.product_stats.get(str(product_id))

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "bayesian.pkl", "wb") as f:
            pickle.dump({
                "min_count":      self.min_count,
                "rating_min":     self.rating_min,
                "rating_max":     self.rating_max,
                "global_mean":    self.global_mean,
                "product_scores": self.product_scores,
                "product_stats":  self.product_stats,
            }, f)
        print(f"  Bayesian scorer saved → {d}")

    def load(self, path: str) -> "BayesianAverageScorer":
        with open(Path(path) / "bayesian.pkl", "rb") as f:
            data = pickle.load(f)
        self.min_count      = data["min_count"]
        self.rating_min     = data["rating_min"]
        self.rating_max     = data["rating_max"]
        self.global_mean    = data["global_mean"]
        self.product_scores = data["product_scores"]
        self.product_stats  = data["product_stats"]
        print(f"  Bayesian scorer loaded ← {Path(path)}")
        return self
