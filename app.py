"""
RedFlag AI - app.py
FastAPI app that serves the fraud detection model.
Accepts transaction data and returns fraud probability.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="RedFlag AI", description="Real-time fraud detection API")

MODELS_DIR = "models"
model    = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))


class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    day_of_week: int
    transactions_today: int
    distance_from_home: float
    is_foreign: int
    is_new_merchant: int


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RedFlag AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #0f0f13; color: #e2e2e8; min-height: 100vh; }
    header { padding: 18px 32px; background: #16161d; border-bottom: 1px solid #2a2a35;
             display: flex; align-items: center; gap: 12px; }
    .logo { width: 32px; height: 32px; background: linear-gradient(135deg, #ff4444, #ff8800);
            border-radius: 8px; display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px; color: #fff; }
    header h1 { font-size: 18px; font-weight: 600; color: #fff; }
    header span { font-size: 12px; color: #6c6c80; margin-left: 4px; }
    .container { max-width: 560px; margin: 40px auto; padding: 0 16px; display: flex; flex-direction: column; gap: 20px; }
    .card { background: #16161d; border: 1px solid #2a2a35; border-radius: 12px; padding: 24px; }
    .card h2 { font-size: 15px; font-weight: 600; margin-bottom: 18px; color: #fff; }
    .field { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
    .field label { font-size: 12px; color: #9090a0; }
    .field input, .field select { padding: 10px 14px; background: #0f0f13; border: 1px solid #2a2a35;
                border-radius: 8px; color: #e2e2e8; font-size: 14px; outline: none; }
    .field input:focus, .field select:focus { border-color: #ff4444; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    #check-btn { width: 100%; padding: 13px; background: #ff4444; color: #fff; border: none;
                 border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background .2s; }
    #check-btn:hover { background: #cc3333; }
    #result { display: none; }
    .result-safe  { border-color: #22c55e; }
    .result-fraud { border-color: #ef4444; }
    .verdict { font-size: 28px; font-weight: 700; text-align: center; margin-bottom: 8px; }
    .verdict.safe  { color: #22c55e; }
    .verdict.fraud { color: #ef4444; }
    .prob { text-align: center; font-size: 13px; color: #9090a0; margin-bottom: 16px; }
    .bar-wrap { background: #0f0f13; border-radius: 999px; height: 8px; overflow: hidden; }
    .bar-fill { height: 8px; border-radius: 999px; transition: width .6s ease; }
    .bar-safe  { background: #22c55e; }
    .bar-fraud { background: #ef4444; }
    .details { margin-top: 16px; font-size: 12px; color: #6c6c80; text-align: center; }
  </style>
</head>
<body>
<header>
  <div class="logo">R</div>
  <h1>RedFlag AI <span>— Fraud Detection</span></h1>
</header>
<div class="container">
  <div class="card">
    <h2>Check a Transaction</h2>
    <div class="row">
      <div class="field"><label>Amount ($)</label><input type="number" id="amount" value="250" min="1"/></div>
      <div class="field"><label>Hour of Day (0-23)</label><input type="number" id="hour" value="14" min="0" max="23"/></div>
    </div>
    <div class="row">
      <div class="field"><label>Day of Week (0=Mon)</label><input type="number" id="dow" value="2" min="0" max="6"/></div>
      <div class="field"><label>Transactions Today</label><input type="number" id="txns" value="2" min="1"/></div>
    </div>
    <div class="row">
      <div class="field"><label>Distance from Home (km)</label><input type="number" id="dist" value="5" min="0"/></div>
      <div class="field"><label>Foreign Transaction</label>
        <select id="foreign"><option value="0">No</option><option value="1">Yes</option></select>
      </div>
    </div>
    <div class="field"><label>New Merchant</label>
      <select id="merchant"><option value="0">No</option><option value="1">Yes</option></select>
    </div>
    <button id="check-btn" onclick="checkFraud()">Analyze Transaction</button>
  </div>

  <div class="card" id="result">
    <div class="verdict" id="verdict-text"></div>
    <div class="prob" id="prob-text"></div>
    <div class="bar-wrap"><div class="bar-fill" id="bar"></div></div>
    <div class="details" id="details"></div>
  </div>
</div>
<script>
async function checkFraud() {
  const payload = {
    amount:             parseFloat(document.getElementById('amount').value),
    hour_of_day:        parseInt(document.getElementById('hour').value),
    day_of_week:        parseInt(document.getElementById('dow').value),
    transactions_today: parseInt(document.getElementById('txns').value),
    distance_from_home: parseFloat(document.getElementById('dist').value),
    is_foreign:         parseInt(document.getElementById('foreign').value),
    is_new_merchant:    parseInt(document.getElementById('merchant').value),
  };

  const btn = document.getElementById('check-btn');
  btn.textContent = 'Analyzing...';
  btn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    const result  = document.getElementById('result');
    const verdict = document.getElementById('verdict-text');
    const prob    = document.getElementById('prob-text');
    const bar     = document.getElementById('bar');
    const details = document.getElementById('details');

    result.style.display = 'block';
    const pct = (data.fraud_probability * 100).toFixed(1);

    if (data.prediction === 'FRAUD') {
      result.className  = 'card result-fraud';
      verdict.className = 'verdict fraud';
      verdict.textContent = 'Flagged as Fraud';
      bar.className = 'bar-fill bar-fraud';
    } else {
      result.className  = 'card result-safe';
      verdict.className = 'verdict safe';
      verdict.textContent = 'Legitimate Transaction';
      bar.className = 'bar-fill bar-safe';
    }

    prob.textContent = 'Fraud probability: ' + pct + '%';
    bar.style.width  = pct + '%';
    details.textContent = 'Risk score: ' + data.risk_score + ' | Confidence: ' + data.confidence;
  } catch(e) {
    alert('Error connecting to API');
  }

  btn.textContent = 'Analyze Transaction';
  btn.disabled = false;
}
</script>
</body>
</html>
"""


@app.post("/predict")
def predict(tx: Transaction):
    """Accept transaction data and return fraud prediction."""
    data = [[
        tx.amount, tx.hour_of_day, tx.day_of_week,
        tx.transactions_today, tx.distance_from_home,
        tx.is_foreign, tx.is_new_merchant
    ]]

    scaled     = scaler.transform(data)
    prediction = model.predict(scaled)[0]
    proba      = model.predict_proba(scaled)[0]
    fraud_prob = float(proba[1])

    risk_score = int(fraud_prob * 100)

    if fraud_prob > 0.85:
        confidence = "Very High"
    elif fraud_prob > 0.6:
        confidence = "High"
    elif fraud_prob > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "prediction":        "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability":  round(fraud_prob, 4),
        "risk_score":         risk_score,
        "confidence":         confidence,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "version": "1.0"}