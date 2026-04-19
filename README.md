# s3dcs – Hybrid Recommendation System

3-layer hybrid recommendation system for e-commerce, from model to API service.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 – Candidate Generation (Neural CF / NeuMF)     │
│  • GMF path  : user_embed ⊙ item_embed                  │
│  • MLP path  : concat(user_embed, item_embed) → FC×3    │
│  • Output    : top-K candidates per user                │
└────────────────────────┬────────────────────────────────┘
                         │ top-K items
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 – Personalization (Content-Based / TF-IDF)     │
│  • Product text  : name + category + description        │
│  • User profile  : recency-weighted avg of bought items │
│  • Scoring       : cosine similarity (profile, item)    │
└────────────────────────┬────────────────────────────────┘
                         │ re-scored candidates
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 – Quality Signal (Bayesian Average Rating)     │
│  BA(p) = (C × m  +  sum_r) / (C + n)                   │
│  C = min_count (prior), m = global mean, n = # ratings  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
         hybrid = 0.5 × NCF + 0.3 × CBF + 0.2 × Bayesian
                         │
                         ▼
                   Top-N results → API Response
```

---

## Project Structure

```
s3dcs/
├── data/
│   ├── generate_sample_data.py   # Generate synthetic CSVs
│   ├── transactions.csv
│   ├── products.csv
│   └── reviews.csv
├── src/
│   ├── data_loader.py
│   ├── models/
│   │   ├── neural_cf.py          # Layer 1: NeuMF
│   │   ├── content_based.py      # Layer 2: TF-IDF CBF
│   │   └── review_analysis.py    # Layer 3: Bayesian Average
│   ├── recommender.py            # HybridRecommender pipeline
│   └── api/
│       └── main.py               # FastAPI service
├── saved_models/                 # Created by train.py
├── train.py
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Generate sample data (skip if you have real CSVs)
python data/generate_sample_data.py

# 2b. Or drop your own CSVs into data/ — required columns:
#   transactions.csv : user_id, product_id
#   products.csv     : product_id, name, category
#   reviews.csv      : product_id, rating

# 3. Train all models (~2 min on CPU)
python train.py

# 4. Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. Interactive docs
open http://localhost:8000/docs
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health + model status |
| `GET` | `/recommend/{user_id}` | Top-N personalised recommendations |
| `GET` | `/similar/{product_id}` | Content-similar products |
| `GET` | `/products` | Paginated product catalogue |
| `GET` | `/products/{product_id}` | Single product + Bayesian rating |
| `GET` | `/categories` | Category list with counts |

### Example

```bash
curl "http://localhost:8000/recommend/U0001?n=10&show_scores=true"
```

```json
{
  "user_id": "U0001",
  "top_n": 10,
  "recommendations": [
    {
      "rank": 1,
      "product_id": "P0042",
      "name": "Pro Laptops Kit A",
      "category": "Electronics",
      "price": 1299.99,
      "n_ratings": 87,
      "bayesian_score": 4.21,
      "scores": {
        "ncf": 0.9120,
        "cbf": 0.7834,
        "bayesian": 0.8025,
        "hybrid": 0.8631
      }
    }
  ]
}
```

---

## Training Options

```bash
# Custom layer weights
python train.py --ncf-weight 0.6 --cbf-weight 0.25 --bay-weight 0.15

# More NCF epochs for higher accuracy
python train.py --ncf-epochs 50

# Custom paths
python train.py --data-dir /path/to/data --model-dir /path/to/models
```

---

## Upgrading Layer 2 to Dense Embeddings

```bash
pip install sentence-transformers
```

In `src/models/content_based.py`, replace the TF-IDF block in `fit()`:

```python
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

st_model = SentenceTransformer("all-MiniLM-L6-v2")
self.product_matrix = normalize(st_model.encode(texts))
```
