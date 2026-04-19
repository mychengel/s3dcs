"""Data loading and validation utilities."""

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {
    "transactions": {"user_id", "product_id"},
    "products":     {"product_id", "name", "category"},
    "reviews":      {"product_id", "rating"},
}


def load_datasets(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate all three CSV datasets."""
    base = Path(data_dir)

    transactions = _load_csv(base / "transactions.csv", "transactions")
    products     = _load_csv(base / "products.csv",     "products")
    reviews      = _load_csv(base / "reviews.csv",      "reviews")

    # Normalise ID types to string
    for df, col in [(transactions, "user_id"), (transactions, "product_id"),
                    (products,     "product_id"),
                    (reviews,      "product_id")]:
        df[col] = df[col].astype(str)

    if "user_id" in reviews.columns:
        reviews["user_id"] = reviews["user_id"].astype(str)

    # Fill missing text fields
    for col in ["description", "subcategory"]:
        if col not in products.columns:
            products[col] = ""
        else:
            products[col] = products[col].fillna("")

    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce").fillna(3.0)

    return transactions, products, reviews


def _load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_sample_data.py` to create sample data."
        )
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def dataset_stats(transactions: pd.DataFrame, products: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    return {
        "n_users":        transactions["user_id"].nunique(),
        "n_products":     products["product_id"].nunique(),
        "n_transactions": len(transactions),
        "n_reviews":      len(reviews),
        "avg_rating":     round(float(reviews["rating"].mean()), 3),
        "categories":     sorted(products["category"].unique().tolist()),
    }
