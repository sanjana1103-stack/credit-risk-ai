import pandas as pd
import numpy as np
import joblib
import json
import os

from preprocessing import preprocess

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

df = pd.read_csv('data/Loan_default.csv')
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)
df = preprocess(df)

X = df.drop('Default', axis=1)
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define models ──────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost":             XGBClassifier(
                               scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                               eval_metric='logloss',
                               random_state=42
                           ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics_store = {}

for name, model in models.items():
    print(f"\n{'='*50}\nTraining: {name}")

    # 5-fold cross-validated AUC
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"Test AUC-ROC: {auc:.4f}  |  Avg Precision: {ap:.4f}")

    metrics_store[name] = {
        "cv_auc_mean":   round(float(cv_auc.mean()), 4),
        "cv_auc_std":    round(float(cv_auc.std()),  4),
        "test_auc":      round(float(auc), 4),
        "avg_precision": round(float(ap),  4),
    }

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(models["Logistic Regression"], 'models/lr_model.pkl')
joblib.dump(models["Random Forest"],       'models/rf_model.pkl')
joblib.dump(models["XGBoost"],             'models/xgb_model.pkl')

with open('models/metrics.json', 'w') as f:
    json.dump(metrics_store, f, indent=2)

print("\n✅ Models saved to models/")
print("✅ Metrics saved to models/metrics.json")
