"""
Training script for the Hybrid Recommendation System.

Steps
─────
  1. (Optional) Generate synthetic data if CSVs don't exist yet.
  2. Load transactions.csv, products.csv, reviews.csv from data/.
  3. Fit all three layers (NCF → CBF → Bayesian).
  4. Save the trained recommender to saved_models/.
  5. Run a quick sanity check.

Usage
─────
  # First time – generate sample data then train:
  python data/generate_sample_data.py
  python train.py

  # With your own data – just run:
  python train.py

  # Then start the API:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader  import load_datasets, dataset_stats
from src.recommender  import HybridRecommender


def parse_args():
    p = argparse.ArgumentParser(description="Train the hybrid RS.")
    p.add_argument("--data-dir",    default="data",         help="Directory with CSV files")
    p.add_argument("--model-dir",   default="saved_models", help="Where to save trained models")
    p.add_argument("--ncf-epochs",  type=int, default=20,   help="NeuMF training epochs")
    p.add_argument("--ncf-weight",  type=float, default=0.50)
    p.add_argument("--cbf-weight",  type=float, default=0.30)
    p.add_argument("--bay-weight",  type=float, default=0.20)
    p.add_argument("--pool-size",   type=int,   default=100, help="NCF candidate pool size")
    p.add_argument("--top-n-check", type=int,   default=5,   help="Items shown in sanity check")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    transactions, products, reviews = load_datasets(args.data_dir)
    stats = dataset_stats(transactions, products, reviews)
    print(f"  users        : {stats['n_users']}")
    print(f"  products     : {stats['n_products']}")
    print(f"  transactions : {stats['n_transactions']}")
    print(f"  reviews      : {stats['n_reviews']}  (avg rating {stats['avg_rating']})")
    print(f"  categories   : {', '.join(stats['categories'])}")

    # ── 2. Build & train ────────────────────────────────────────────────────
    recommender = HybridRecommender(
        ncf_weight          = args.ncf_weight,
        cbf_weight          = args.cbf_weight,
        bayesian_weight     = args.bay_weight,
        candidate_pool_size = args.pool_size,
    )

    recommender.fit(
        transactions_df = transactions,
        products_df     = products,
        reviews_df      = reviews,
        ncf_epochs      = args.ncf_epochs,
        verbose         = True,
    )

    # ── 3. Save ─────────────────────────────────────────────────────────────
    recommender.save(args.model_dir)

    # ── 4. Sanity check ─────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("Sanity check")
    print("─" * 50)

    sample_user = str(transactions["user_id"].iloc[0])
    recs = recommender.recommend(sample_user, top_n=args.top_n_check, return_scores=True)

    print(f"\nTop-{args.top_n_check} recommendations for user '{sample_user}':\n")
    for r in recs:
        name  = r.get("name", r["product_id"])
        cat   = r.get("category", "?")
        price = r.get("price", 0)
        s     = r.get("scores", {})
        print(f"  {r['rank']:>2}. [{cat}] {name:<50}  ${price:>7.2f}"
              f"  hybrid={s.get('hybrid', 0):.3f}"
              f"  ncf={s.get('ncf', 0):.3f}"
              f"  cbf={s.get('cbf', 0):.3f}"
              f"  bay={s.get('bayesian', 0):.3f}")

    print("\nSimilar products to the top recommendation:")
    if recs:
        sims = recommender.similar_products(recs[0]["product_id"], top_k=3)
        for s in sims:
            print(f"  {s['rank']}. [{s.get('category','?')}] {s.get('name', s['product_id'])}"
                  f"  (similarity={s['similarity_score']:.3f})")

    print("\n✓ Training complete!")
    print(f"  Models saved to  : {args.model_dir}/")
    print("  Start the API    : uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
    print("  Interactive docs : http://localhost:8000/docs")


if __name__ == "__main__":
    main()
