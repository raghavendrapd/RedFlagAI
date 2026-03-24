"""
RedFlag AI - train.py
Trains a Random Forest classifier on credit card transaction data
and saves the model for serving via FastAPI.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_dataset(n_samples=10000):
    """
    Generate a realistic synthetic fraud dataset.
    In production this would be real transaction data.
    """
    np.random.seed(42)

    # Legitimate transactions (95%)
    n_legit = int(n_samples * 0.95)
    legit = pd.DataFrame({
        "amount":           np.random.normal(100, 50, n_legit).clip(1, 2000),
        "hour_of_day":      np.random.randint(8, 22, n_legit),       # normal hours
        "day_of_week":      np.random.randint(0, 7, n_legit),
        "transactions_today": np.random.randint(1, 5, n_legit),
        "distance_from_home": np.random.normal(10, 5, n_legit).clip(0, 100),
        "is_foreign":       np.random.choice([0, 1], n_legit, p=[0.95, 0.05]),
        "is_new_merchant":  np.random.choice([0, 1], n_legit, p=[0.85, 0.15]),
        "label": 0
    })

    # Fraudulent transactions (5%)
    n_fraud = n_samples - n_legit
    fraud = pd.DataFrame({
        "amount":           np.random.normal(800, 300, n_fraud).clip(100, 5000),  # higher amounts
        "hour_of_day":      np.random.choice([0,1,2,3,4,23], n_fraud),            # odd hours
        "day_of_week":      np.random.randint(0, 7, n_fraud),
        "transactions_today": np.random.randint(5, 20, n_fraud),                  # many txns
        "distance_from_home": np.random.normal(200, 100, n_fraud).clip(50, 1000), # far away
        "is_foreign":       np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),      # often foreign
        "is_new_merchant":  np.random.choice([0, 1], n_fraud, p=[0.2, 0.8]),      # new merchant
        "label": 1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    print(f"✅ Dataset generated: {len(df)} transactions ({n_fraud} fraud, {n_legit} legit)")
    return df


def train():
    print("\n🚀 RedFlag AI — Model Training Pipeline")
    print("=" * 45)

    # Step 1: Generate data
    df = generate_dataset()

    # Step 2: Split features and labels
    features = ["amount", "hour_of_day", "day_of_week", "transactions_today",
                "distance_from_home", "is_foreign", "is_new_merchant"]
    X = df[features]
    y = df["label"]

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Step 5: Train Random Forest
    print("\n⚙️  Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",   # handles class imbalance (95/5 split)
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Step 6: Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\n📊 Model Performance:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Step 7: Feature importance
    print("🔍 Feature Importance:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"   {feat:<25} {bar} {imp:.3f}")

    # Step 8: Save model + scaler
    joblib.dump(model,  os.path.join(MODELS_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "features.pkl"))

    print(f"\n✅ Model saved to {MODELS_DIR}/fraud_model.pkl")
    print("✅ Run app.py to start the API\n")


if __name__ == "__main__":
    train()