# RedFlag AI

Real-time fraud detection API. Feed it a transaction, get back a fraud probability score and risk level instantly.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-green?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=flat-square)

---

## What it does

Takes 7 transaction features (amount, time, location patterns, merchant info) and runs them through a trained Random Forest classifier to return a fraud probability score between 0 and 1.

Built this to understand the full ML lifecycle — data generation, model training, evaluation, and serving predictions through a REST API with a UI on top.

---

## How it works

```
Transaction features → StandardScaler → Random Forest (100 trees)
                                               │
                              fraud_probability + risk_score + confidence
```

The model is trained on a synthetic dataset of 10,000 transactions (95% legit, 5% fraud) with class balancing to handle the imbalance. Key fraud signals: high amounts at odd hours, many transactions in one day, large distance from home, foreign merchants.

---

## Stack

- **scikit-learn** — Random Forest classifier
- **FastAPI** — REST API + web UI
- **joblib** — model serialization
- **pandas / numpy** — data processing

---

## Setup

```bash
git clone https://github.com/raghavendrapd/RedFlagAI.git
cd RedFlagAI

python -m venv venv
venv\Scripts\activate      # windows
source venv/bin/activate   # mac/linux

pip install scikit-learn pandas numpy fastapi uvicorn joblib
```

Train the model first:
```bash
python train.py
```

Then start the API:
```bash
uvicorn app:app --reload
```

Open `https://redflagai.onrender.com/`

---

## API

**POST `/predict`**
```json
{
  "amount": 1200.0,
  "hour_of_day": 2,
  "day_of_week": 6,
  "transactions_today": 8,
  "distance_from_home": 450.0,
  "is_foreign": 1,
  "is_new_merchant": 1
}
```

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.91,
  "risk_score": 91,
  "confidence": "Very High"
}
```

**GET `/health`** — returns model status

---

## Model performance

Evaluated on 2,000 held-out transactions:
- Precision on fraud class: ~0.88
- Recall on fraud class: ~0.85
- Handles class imbalance via `class_weight="balanced"`

---

## Project structure

```
RedFlagAI/
├── train.py       # generates data, trains model, saves to /models
├── app.py         # fastapi app + prediction endpoint + ui
├── models/        # saved model, scaler, feature list (auto-created)
└── .gitignore
```

---

## What I'd improve

- Replace synthetic data with a real dataset (e.g. Kaggle credit card fraud dataset)
- Add SHAP values to explain why a transaction was flagged
- Add a threshold tuning endpoint so teams can adjust sensitivity
- Docker + deploy to Render or Railway

---

## License

MIT
