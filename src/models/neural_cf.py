"""
Layer 1 – Candidate Generation: Neural Matrix Factorization (NeuMF).

Architecture: GMF path + MLP path combined at the output layer.
  - GMF (element-wise product of user/item embeddings) captures linear interactions.
  - MLP (concat embeddings → dense layers) captures non-linear interactions.
  - Both paths are concatenated before the final sigmoid output.

Training uses implicit feedback from transactions with negative sampling (4:1 ratio).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch components
# ──────────────────────────────────────────────────────────────────────────────

class _InteractionDataset(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray):
        self.users  = torch.LongTensor(users)
        self.items  = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NeuMF(nn.Module):
    """Neural Matrix Factorization combining GMF and MLP paths."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 64,
        mlp_layers: List[int] = [256, 128, 64],
        dropout: float = 0.1,
    ):
        super().__init__()

        # GMF embeddings
        self.gmf_user = nn.Embedding(num_users, embed_dim)
        self.gmf_item = nn.Embedding(num_items, embed_dim)

        # MLP embeddings (separate embedding tables for each path)
        self.mlp_user = nn.Embedding(num_users, embed_dim)
        self.mlp_item = nn.Embedding(num_items, embed_dim)

        # MLP tower: [embed_dim*2] → mlp_layers
        dims = [embed_dim * 2] + mlp_layers
        blocks = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(in_d, out_d), nn.BatchNorm1d(out_d), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp_tower = nn.Sequential(*blocks)

        # Final output: concat(GMF output, last MLP layer) → scalar
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity="sigmoid")

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_out  = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_in   = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out  = self.mlp_tower(mlp_in)
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return torch.sigmoid(self.output_layer(combined)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Recommender wrapper
# ──────────────────────────────────────────────────────────────────────────────

class NCFRecommender:
    """Wraps NeuMF with training, inference, and serialisation helpers."""

    def __init__(
        self,
        embed_dim: int = 64,
        mlp_layers: Optional[List[int]] = None,
        num_neg_samples: int = 4,
        batch_size: int = 1024,
        num_epochs: int = 20,
        lr: float = 0.001,
        device: Optional[str] = None,
    ):
        self.embed_dim       = embed_dim
        self.mlp_layers      = mlp_layers or [256, 128, 64]
        self.num_neg_samples = num_neg_samples
        self.batch_size      = batch_size
        self.num_epochs      = num_epochs
        self.lr              = lr
        self.device          = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[NeuMF] = None
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
        self.user_interactions: Dict[int, set] = {}  # user_idx → set of item_idx

    # ── private helpers ─────────────────────────────────────────────────────

    def _encode(self, transactions_df: pd.DataFrame):
        users = sorted(transactions_df["user_id"].unique().tolist())
        items = sorted(transactions_df["product_id"].unique().tolist())
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

    def _build_interactions(self, transactions_df: pd.DataFrame):
        self.user_interactions = {}
        for _, row in transactions_df.iterrows():
            u = self.user_to_idx.get(row["user_id"])
            it = self.item_to_idx.get(row["product_id"])
            if u is not None and it is not None:
                self.user_interactions.setdefault(u, set()).add(it)

    def _make_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_items = len(self.item_to_idx)
        users, items, labels = [], [], []

        for u_idx, pos_set in self.user_interactions.items():
            for it_idx in pos_set:
                users.append(u_idx); items.append(it_idx); labels.append(1.0)

            need = len(pos_set) * self.num_neg_samples
            neg = []
            while len(neg) < need:
                cands = np.random.randint(0, n_items, need * 2)
                for c in cands:
                    if c not in pos_set:
                        neg.append(c)
                    if len(neg) >= need:
                        break
            for it_idx in neg:
                users.append(u_idx); items.append(it_idx); labels.append(0.0)

        return np.array(users), np.array(items), np.array(labels)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, transactions_df: pd.DataFrame, verbose: bool = True) -> "NCFRecommender":
        self._encode(transactions_df)
        self._build_interactions(transactions_df)

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        self.model = NeuMF(n_users, n_items, self.embed_dim, self.mlp_layers).to(self.device)

        users, items, labels = self._make_training_data()
        loader = DataLoader(
            _InteractionDataset(users, items, labels),
            batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        optimizer  = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion  = nn.BCELoss()
        scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss = 0.0
            for b_users, b_items, b_labels in loader:
                b_users  = b_users.to(self.device)
                b_items  = b_items.to(self.device)
                b_labels = b_labels.to(self.device)
                preds    = self.model(b_users, b_items)
                loss     = criterion(preds, b_labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if verbose and epoch % 5 == 0:
                print(f"    Epoch {epoch:>3}/{self.num_epochs}  loss={total_loss/len(loader):.4f}")

        self.model.eval()
        return self

    def score(self, user_id: str, item_ids: List[str]) -> np.ndarray:
        """Return NCF scores in [0,1] for each item in item_ids."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")

        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            return np.full(len(item_ids), 0.5)  # cold-start: neutral score

        valid_it_idx, valid_pos = [], []
        for pos, iid in enumerate(item_ids):
            idx = self.item_to_idx.get(iid)
            if idx is not None:
                valid_it_idx.append(idx)
                valid_pos.append(pos)

        scores = np.zeros(len(item_ids))
        if not valid_it_idx:
            return scores

        self.model.eval()
        with torch.no_grad():
            u_t = torch.LongTensor([u_idx] * len(valid_it_idx)).to(self.device)
            i_t = torch.LongTensor(valid_it_idx).to(self.device)
            preds = self.model(u_t, i_t).cpu().numpy()

        for i, pos in enumerate(valid_pos):
            scores[pos] = preds[i]

        return scores

    def recommend(
        self,
        user_id: str,
        all_items: List[str],
        top_k: int = 100,
        exclude_seen: bool = True,
    ) -> List[Tuple[str, float]]:
        """Return top-K (item_id, score) pairs for a user."""
        scores = self.score(user_id, all_items)

        if exclude_seen:
            u_idx = self.user_to_idx.get(user_id)
            if u_idx is not None:
                seen = self.user_interactions.get(u_idx, set())
                for i, iid in enumerate(all_items):
                    if self.item_to_idx.get(iid) in seen:
                        scores[i] = -1.0

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(all_items[i], float(scores[i])) for i in top_idx if scores[i] >= 0]

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str):
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), d / "model.pt")
        with open(d / "meta.pkl", "wb") as f:
            pickle.dump({
                "embed_dim":          self.embed_dim,
                "mlp_layers":         self.mlp_layers,
                "user_to_idx":        self.user_to_idx,
                "item_to_idx":        self.item_to_idx,
                "idx_to_item":        self.idx_to_item,
                "user_interactions":  {k: list(v) for k, v in self.user_interactions.items()},
                "num_users":          len(self.user_to_idx),
                "num_items":          len(self.item_to_idx),
            }, f)
        print(f"  NCF saved → {d}")

    def load(self, path: str) -> "NCFRecommender":
        d = Path(path)
        with open(d / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        self.embed_dim          = meta["embed_dim"]
        self.mlp_layers         = meta["mlp_layers"]
        self.user_to_idx        = meta["user_to_idx"]
        self.item_to_idx        = meta["item_to_idx"]
        self.idx_to_item        = meta["idx_to_item"]
        self.user_interactions  = {k: set(v) for k, v in meta["user_interactions"].items()}

        self.model = NeuMF(
            meta["num_users"], meta["num_items"], self.embed_dim, self.mlp_layers
        ).to(self.device)
        self.model.load_state_dict(torch.load(d / "model.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"  NCF loaded ← {d}")
        return self
