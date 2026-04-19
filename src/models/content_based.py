"""
Layer 2 – Personalization: Content-Based Filtering (CBF).

Products are embedded with TF-IDF over concatenated text features
(name, category, subcategory, description).  Category tokens are repeated 3×
to increase their influence.

User profiles are built as recency-weighted averages of the TF-IDF vectors of
products the user has purchased.  Scoring is cosine similarity between the
user profile and each candidate item vector.

To upgrade to dense embeddings, replace the TF-IDF block in `fit()` with a
sentence-transformers call:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    self.product_matrix = normalize(model.encode(texts))
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class CBFRecommender:
    """Content-Based Filtering recommender using TF-IDF text embeddings."""

    def __init__(
        self,
        text_fields: Optional[List[str]] = None,
        max_features: int = 8000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.text_fields  = text_fields or ["name", "category", "subcategory", "description"]
        self.max_features = max_features
        self.ngram_range  = ngram_range

        self.vectorizer    = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.product_matrix: Optional[np.ndarray] = None   # (n_products, n_features)
        self.product_ids:    List[str] = []
        self.product_idx:    Dict[str, int] = {}
        self.user_profiles:  Dict[str, np.ndarray] = {}

    # ── private helpers ─────────────────────────────────────────────────────

    def _build_texts(self, products_df: pd.DataFrame) -> List[str]:
        texts = []
        for _, row in products_df.iterrows():
            parts = []
            for field in self.text_fields:
                if field in row and pd.notna(row[field]) and str(row[field]).strip():
                    # Repeat category/subcategory tokens for higher TF-IDF weight
                    reps = 3 if field in ("category", "subcategory") else 1
                    parts.extend([str(row[field])] * reps)
            texts.append(" ".join(parts))
        return texts

    def _build_user_profiles(self, transactions_df: pd.DataFrame):
        has_ts = "timestamp" in transactions_df.columns

        for uid, grp in transactions_df.groupby("user_id"):
            if has_ts:
                grp = grp.sort_values("timestamp")

            n = len(grp)
            recency = np.linspace(0.5, 1.0, n) if has_ts else np.ones(n)

            indices, weights = [], []
            for i, (_, row) in enumerate(grp.iterrows()):
                idx = self.product_idx.get(row["product_id"])
                if idx is not None:
                    indices.append(idx)
                    weights.append(recency[i])

            if not indices:
                continue

            vecs     = self.product_matrix[indices]               # (k, f)
            w        = np.array(weights).reshape(-1, 1)
            profile  = (vecs * w).sum(axis=0)
            norm     = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
            self.user_profiles[str(uid)] = profile

    # ── public API ───────────────────────────────────────────────────────────

    def fit(
        self,
        products_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        verbose: bool = True,
    ) -> "CBFRecommender":
        self.product_ids = products_df["product_id"].astype(str).tolist()
        self.product_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        texts = self._build_texts(products_df)
        tfidf = self.vectorizer.fit_transform(texts)
        self.product_matrix = normalize(tfidf.toarray(), norm="l2")

        if verbose:
            print(f"    Product matrix: {self.product_matrix.shape}")

        self._build_user_profiles(transactions_df)

        if verbose:
            print(f"    User profiles built: {len(self.user_profiles)}")

        return self

    def score(self, user_id: str, item_ids: List[str]) -> np.ndarray:
        """Cosine similarity between user profile and each candidate item."""
        profile = self.user_profiles.get(str(user_id))
        if profile is None:
            return np.zeros(len(item_ids))   # cold-start: all zeros

        scores = np.zeros(len(item_ids))
        for i, iid in enumerate(item_ids):
            idx = self.product_idx.get(iid)
            if idx is not None:
                scores[i] = float(np.dot(profile, self.product_matrix[idx]))
        return scores

    def similar_items(self, product_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve the most content-similar products."""
        idx = self.product_idx.get(str(product_id))
        if idx is None:
            return []

        vec  = self.product_matrix[idx].reshape(1, -1)
        sims = cosine_similarity(vec, self.product_matrix)[0]
        sims[idx] = -1.0   # exclude self

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.product_ids[i], float(sims[i])) for i in top_idx]

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "cbf.pkl", "wb") as f:
            pickle.dump({
                "vectorizer":     self.vectorizer,
                "product_matrix": self.product_matrix,
                "product_ids":    self.product_ids,
                "product_idx":    self.product_idx,
                "user_profiles":  self.user_profiles,
                "text_fields":    self.text_fields,
            }, f)
        print(f"  CBF saved → {d}")

    def load(self, path: str) -> "CBFRecommender":
        with open(Path(path) / "cbf.pkl", "rb") as f:
            data = pickle.load(f)
        self.vectorizer     = data["vectorizer"]
        self.product_matrix = data["product_matrix"]
        self.product_ids    = data["product_ids"]
        self.product_idx    = data["product_idx"]
        self.user_profiles  = data["user_profiles"]
        self.text_fields    = data["text_fields"]
        print(f"  CBF loaded ← {Path(path)}")
        return self
