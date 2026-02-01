# 🏠 RentShield Germany

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RentShield Germany** is a full-stack Data Science application that detects overpriced and potentially fraudulent rental listings in the German rental market.

![RentShield Demo](https://img.icons8.com/fluency/200/home.png)

## 🌟 Features

- **Fair Rent Estimation** - ML-powered price prediction based on city, size, and features
- **Overpricing Detection** - Identifies listings priced above market rates
- **Scam Risk Scoring** - NLP + Anomaly Detection + Rule-based analysis
- **Explainable Outputs** - Transparent reasoning for every prediction
- **Bulk Analysis** - Process multiple listings via CSV upload
- **Market Insights** - Interactive visualizations of German rental market

## 🏗️ Architecture

```
┌────────────────────────────┐
│        FastAPI Backend      │
│ ──────────────────────────  │
│ • Load ML artifacts         │
│ • Feature preprocessing     │
│ • Price prediction          │
│ • Scam risk scoring         │
│ • REST endpoints            │
└─────────────┬──────────────┘
              │
  HTTP / JSON API calls
              │
┌─────────────▼──────────────┐
│        Streamlit UI         │
│ ──────────────────────────  │
│ • User input forms          │
│ • CSV upload                │
│ • Calls FastAPI endpoints   │
│ • Visualization & reports   │
└────────────────────────────┘
```

**Key Design Principle:** Streamlit is a pure frontend - all ML inference happens in FastAPI.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/yourusername/RentShield.git
cd RentShield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data & Train Models

```bash
# Generate synthetic training data
python training/build_dataset.py

# Train pricing model
python training/train_price.py

# Train scam detection models
python training/train_scam.py
```

### 3. Start the Backend (FastAPI)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 4. Start the Frontend (Streamlit)

In a new terminal:

```bash
cd frontend
streamlit run app.py
```

The UI will open at: http://localhost:8501

## 📁 Project Structure

```
RentShield/
├── README.md
├── requirements.txt              # All dependencies
│
├── backend/
│   ├── requirements.txt          # Backend-only deps
│   └── app/
│       ├── main.py               # FastAPI application
│       ├── schemas.py            # Pydantic models
│       ├── core/
│       │   ├── preprocess.py     # Feature engineering
│       │   ├── pricing.py        # Rent prediction
│       │   └── scam.py           # Scam detection
│       └── artifacts/            # ML model files
│           ├── rent_model.joblib
│           ├── tfidf.joblib
│           ├── scam_model.joblib
│           └── anomaly_model.joblib
│
├── frontend/
│   ├── requirements.txt          # Frontend-only deps
│   ├── app.py                    # Streamlit main
│   └── pages/
│       ├── 1_Single_Checker.py   # Individual listing analysis
│       ├── 2_Bulk_CSV.py         # Batch processing
│       └── 3_Market_Insights.py  # Market statistics
│
├── training/
│   ├── build_dataset.py          # Generate synthetic data
│   ├── train_price.py            # Train regression model
│   └── train_scam.py             # Train scam classifiers
│
└── data/
    ├── sample/
    │   └── sample_listings.csv   # Test data
    └── training/
        └── training_data.csv     # Generated training data
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check & model status |
| POST | `/predict/price` | Rent prediction only |
| POST | `/predict/scam` | Scam analysis only |
| POST | `/predict/full` | Complete listing analysis |
| POST | `/predict/bulk` | Batch analysis |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/full" \
     -H "Content-Type: application/json" \
     -d '{
       "city": "Berlin",
       "postcode": "10115",
       "living_space": 55,
       "rooms": 2,
       "rent": 950,
       "description": "Beautiful apartment, deposit required before viewing"
     }'
```

### Example Response

```json
{
  "predicted_rent": 825.0,
  "predicted_rent_per_sqm": 15.0,
  "actual_rent": 950.0,
  "actual_rent_per_sqm": 17.27,
  "overpricing_pct": 0.152,
  "is_overpriced": true,
  "scam_score": 0.45,
  "risk_level": "MEDIUM",
  "reasons": [
    "moderately overpriced (15% above market)",
    "caution phrase: 'deposit required'"
  ],
  "nlp_score": 0.35,
  "anomaly_score": 0.20,
  "rules_score": 0.50
}
```

## 🤖 ML Models

### Pricing Model (Regression)

- **Algorithm:** GradientBoostingRegressor
- **Target:** Rent per square meter (€/m²)
- **Features:** City tier, postcode, size, rooms, building age, floor

### Scam Detection (Hybrid)

| Component | Method | Weight |
|-----------|--------|--------|
| NLP Classifier | TF-IDF + Logistic Regression | 50% |
| Anomaly Detection | Isolation Forest | 30% |
| Rule Engine | Keyword-based red flags | 20% |

### Risk Levels

| Level | Score Range | Action |
|-------|-------------|--------|
| LOW | 0-25% | Appears legitimate |
| MEDIUM | 25-50% | Verify before proceeding |
| HIGH | 50-75% | Multiple red flags |
| CRITICAL | 75-100% | Avoid this listing |

## 🚀 Deployment

### Backend (Render/Fly.io)

1. Push code to GitHub
2. Connect repository to Render
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Streamlit Cloud)

1. Push code to GitHub
2. Go to share.streamlit.io
3. Deploy from `frontend/app.py`
4. Set environment variable: `RENTSHIELD_API_URL=https://your-api.onrender.com`

## 🧪 Testing

```bash
# Test API health
curl http://localhost:8000/health

# Run with sample data
python -c "
import requests
response = requests.post('http://localhost:8000/predict/full', json={
    'city': 'Berlin',
    'postcode': '10115',
    'living_space': 55,
    'rooms': 2,
    'rent': 950,
    'description': 'Test listing'
})
print(response.json())
"
```

## 📊 German Rental Market Data

The system uses city tier classification:

| Tier | Cities | Avg. Rent |
|------|--------|-----------|
| 1 | Munich, Frankfurt, Stuttgart | €16-22/m² |
| 2 | Berlin, Hamburg, Düsseldorf, Cologne | €13-15/m² |
| 3 | Hannover, Dresden, Leipzig, Nuremberg | €8-12/m² |
| 4 | Essen, Dortmund, Duisburg | €6-8/m² |

## 🧠 Data & Training Details

### 1. Data Generation (Synthetic)
Since real rental scraping faces legal restrictions, we use a **Synthetic Data Generator** that mathematically mimics the German rental market.
- **Source:** `training/build_dataset.py`
- **Logic:**
  - Uses realistic base rents per city tier (e.g., Munich €20/m², Leipzig €8/m²).
  - Injects **15% scam listings** using known fraud keywords (e.g., "Western Union", "deposit before viewing").
  - Injects **20% overpriced listings** (1.3x - 1.8x market rate) to train the pricing model.
  - Generates 5,000 samples for robust training.

### 2. Machine Learning Models
We use a hybrid ensemble of three distinct models:

| Component | Model Algorithm | Implementation | Purpose |
|-----------|-----------------|----------------|---------|
| **Pricing** | `GradientBoostingRegressor` | `scikit-learn` | Predicts fair market rent. Captures complex non-linear price tiers. |
| **Scam NLP** | `TF-IDF` + `LogisticRegression` | `scikit-learn` | Detects suspicious text patterns in descriptions. |
| **Anomalies** | `IsolationForest` | `scikit-learn` | Detects outliers (e.g., huge size but tiny price). |

### 3. Artifact Storage
Trained models are serialized using `joblib` and stored locally to ensure fast inference without retraining.
- **Location:** `backend/app/artifacts/`
- **Files:**
  - `rent_model.joblib`: The pricing brain.
  - `scam_model.joblib`: The fraud detector.
  - `tfidf.joblib`: The text vectorizer.
  - `anomaly_model.joblib`: The outlier detector.

## ⚠️ Disclaimer

RentShield provides risk assessments based on ML models and should be used as **one factor** in your decision-making process. Always:

- Verify listings through official channels
- Never send money without viewing the property
- Meet landlords in person at the actual apartment
- Check landlord identity and ownership documents

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

Made with ❤️ for safe renting in Germany
