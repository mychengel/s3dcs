"""
Generate synthetic e-commerce data for the hybrid recommendation system.

Replace the output CSVs with your real datasets — just keep the same column names:
  transactions.csv : transaction_id, user_id, product_id, timestamp, quantity, price
  products.csv     : product_id, name, category, subcategory, description, price
  reviews.csv      : review_id, user_id, product_id, rating, review_text, timestamp
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

NUM_USERS = 500
NUM_PRODUCTS = 200
NUM_TRANSACTIONS = 6000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

CATEGORIES = {
    "Electronics":    {"subcategories": ["Smartphones", "Laptops", "Headphones", "Cameras", "Tablets"],   "price_range": (100, 2000)},
    "Books":          {"subcategories": ["Fiction", "Non-fiction", "Science", "History", "Programming"],   "price_range": (10, 80)},
    "Clothing":       {"subcategories": ["T-Shirts", "Jeans", "Jackets", "Shoes", "Accessories"],         "price_range": (20, 300)},
    "Home & Kitchen": {"subcategories": ["Cookware", "Storage", "Decor", "Lighting", "Furniture"],        "price_range": (15, 500)},
    "Sports":         {"subcategories": ["Running", "Cycling", "Yoga", "Swimming", "Gym"],                "price_range": (20, 400)},
    "Beauty":         {"subcategories": ["Skincare", "Haircare", "Makeup", "Fragrance", "Tools"],         "price_range": (10, 150)},
    "Toys & Games":   {"subcategories": ["Board Games", "Action Figures", "Puzzles", "Educational", "Outdoor"], "price_range": (10, 100)},
    "Food & Grocery": {"subcategories": ["Snacks", "Beverages", "Organic", "Coffee", "Supplements"],      "price_range": (5, 60)},
    "Music":          {"subcategories": ["Guitars", "Keyboards", "Drums", "DJ Equipment", "Accessories"], "price_range": (30, 1500)},
    "Gaming":         {"subcategories": ["Consoles", "PC Games", "Controllers", "VR", "Gaming Chairs"],   "price_range": (20, 600)},
}

ADJECTIVES = ["Premium", "Ultra", "Pro", "Elite", "Classic", "Smart", "Eco", "Deluxe", "Compact", "Advanced"]

WORD_SUFFIX = {
    "Electronics": ["device", "unit", "system", "module", "gadget"],
    "Books": ["guide", "handbook", "manual", "anthology", "collection"],
    "Clothing": ["set", "collection", "wear", "style", "edition"],
    "Home & Kitchen": ["set", "kit", "organizer", "system", "bundle"],
    "Sports": ["gear", "equipment", "kit", "trainer", "set"],
    "Beauty": ["kit", "set", "collection", "formula", "treatment"],
    "Toys & Games": ["set", "kit", "collection", "edition", "pack"],
    "Food & Grocery": ["pack", "bundle", "mix", "blend", "selection"],
    "Music": ["kit", "bundle", "set", "package", "edition"],
    "Gaming": ["bundle", "edition", "pack", "collection", "kit"],
}

DESC_TEMPLATES = [
    "High-quality {category} product designed for maximum performance and durability.",
    "Experience the best in {subcategory} with this premium {category} item.",
    "Perfect for enthusiasts and professionals. This {subcategory} product delivers exceptional results.",
    "Innovative design meets functionality in this top-rated {category} product.",
    "Built with premium materials, this {subcategory} product exceeds expectations.",
    "Trusted by thousands of customers, this {category} item offers great value.",
    "Award-winning {subcategory} product with cutting-edge features and reliable performance.",
    "The ultimate {category} solution for those who demand the best quality.",
]


def generate_products() -> pd.DataFrame:
    rows, pid = [], 1
    per_cat = NUM_PRODUCTS // len(CATEGORIES)

    for category, details in CATEGORIES.items():
        subcats = details["subcategories"]
        lo, hi = details["price_range"]
        suffixes = WORD_SUFFIX[category]

        for i in range(per_cat):
            subcat = subcats[i % len(subcats)]
            name = f"{random.choice(ADJECTIVES)} {subcat} {random.choice(suffixes).title()} {chr(65 + i % 26)}"
            description = random.choice(DESC_TEMPLATES).format(category=category, subcategory=subcat)
            rows.append({
                "product_id":  f"P{pid:04d}",
                "name":        name,
                "category":    category,
                "subcategory": subcat,
                "description": description,
                "price":       round(random.uniform(lo, hi), 2),
            })
            pid += 1

    return pd.DataFrame(rows)


def generate_transactions(products_df: pd.DataFrame) -> pd.DataFrame:
    categories = list(CATEGORIES.keys())
    # Each user prefers 2-4 categories (simulates real browsing patterns)
    user_prefs = {u: random.sample(categories, random.randint(2, 4)) for u in range(1, NUM_USERS + 1)}
    date_range = (END_DATE - START_DATE).days
    rows = []

    for tid in range(1, NUM_TRANSACTIONS + 1):
        user_id = random.randint(1, NUM_USERS)
        if random.random() < 0.70:
            cat = random.choice(user_prefs[user_id])
            pool = products_df[products_df["category"] == cat]
            product = pool.sample(1).iloc[0] if len(pool) > 0 else products_df.sample(1).iloc[0]
        else:
            product = products_df.sample(1).iloc[0]

        ts = START_DATE + timedelta(days=random.randint(0, date_range),
                                    hours=random.randint(0, 23),
                                    minutes=random.randint(0, 59))
        rows.append({
            "transaction_id": f"T{tid:06d}",
            "user_id":        f"U{user_id:04d}",
            "product_id":     product["product_id"],
            "timestamp":      ts.strftime("%Y-%m-%d %H:%M:%S"),
            "quantity":       random.choices([1, 2, 3, 4, 5], weights=[0.60, 0.20, 0.10, 0.07, 0.03])[0],
            "price":          product["price"],
        })

    return pd.DataFrame(rows)


def generate_reviews(transactions_df: pd.DataFrame) -> pd.DataFrame:
    pos = [
        "Excellent product! Really happy with this purchase.",
        "Absolutely love it. Will definitely buy again.",
        "Great quality and fast delivery. Highly recommend!",
        "Exceeded my expectations. Perfect for my needs.",
        "Best purchase I've made this year. Five stars!",
    ]
    neu = [
        "Decent product. Does what it's supposed to do.",
        "Average quality, but okay for the price.",
        "It's fine. Nothing special but gets the job done.",
        "Arrived on time, product is as described.",
        "Nothing special but meets basic requirements.",
    ]
    neg = [
        "Not what I expected. A bit disappointed.",
        "Quality could be better for this price.",
        "Had some issues. Not very satisfied with the product.",
        "Packaging was damaged and quality below average.",
        "Doesn't match the description. Would not recommend.",
    ]

    sampled = transactions_df.sample(frac=0.60, random_state=42).reset_index(drop=True)
    rows = []
    for i, tx in sampled.iterrows():
        rating = int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30]))
        text = random.choice(pos if rating >= 4 else (neu if rating == 3 else neg))
        rows.append({
            "review_id":   f"R{i+1:06d}",
            "user_id":     tx["user_id"],
            "product_id":  tx["product_id"],
            "rating":      rating,
            "review_text": text,
            "timestamp":   tx["timestamp"],
        })

    return pd.DataFrame(rows)


def main():
    out = Path(__file__).parent
    print("Generating synthetic e-commerce datasets...")

    products_df      = generate_products()
    transactions_df  = generate_transactions(products_df)
    reviews_df       = generate_reviews(transactions_df)

    products_df.to_csv(out / "products.csv", index=False)
    transactions_df.to_csv(out / "transactions.csv", index=False)
    reviews_df.to_csv(out / "reviews.csv", index=False)

    print(f"  products.csv      : {len(products_df):>5} rows | {products_df['category'].nunique()} categories")
    print(f"  transactions.csv  : {len(transactions_df):>5} rows | {transactions_df['user_id'].nunique()} users")
    print(f"  reviews.csv       : {len(reviews_df):>5} rows | avg rating {reviews_df['rating'].mean():.2f}")
    print(f"\nFiles saved to {out}/")
    print("Replace these CSVs with your real data (keep the same column names).")


if __name__ == "__main__":
    main()
