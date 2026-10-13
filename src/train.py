"""Emergency-weighted training.

- Triage model: TF-IDF + numeric → LogisticRegression (sparse-safe, <100ms).
  sample_weight: is_emergency rows x20 (matches scorer), L1 rows extra x2.
- Dept model: TF-IDF + numeric → LogisticRegression.
- Saves models/triage_model.joblib, models/dept_model.joblib,
  models/tfidf.joblib, models/scaler.joblib, models/meta.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import recall_score
from .features import row_text, numeric_matrix, build_tfidf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/")
    ap.add_argument("--weight", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data)
    texts = [row_text(r) for _, r in df.iterrows()]
    vec, Xt = build_tfidf(texts)
    Xn, cols = numeric_matrix(df)
    X = hstack([Xt, csr_matrix(Xn)]).tocsr()
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    y = df["triage_level"].astype(int).values
    w = np.ones(len(df))
    w[df["is_emergency"].values == 1] *= a.weight
    w[y == 1] *= 2.0
    clf = LogisticRegression(max_iter=2000, C=1.5, solver="saga")
    clf.fit(X, y, sample_weight=w)
    joblib.dump(clf, os.path.join(a.out, "triage_model.joblib"))
    joblib.dump(vec, os.path.join(a.out, "tfidf.joblib"))
    joblib.dump(scaler, os.path.join(a.out, "scaler.joblib"))
    dclf = LogisticRegression(max_iter=3000, C=4.0, solver="saga", class_weight="balanced")
    dclf.fit(X, df["department"].astype(str).values)
    joblib.dump(dclf, os.path.join(a.out, "dept_model.joblib"))
    # quick holdout recall check
    pred = clf.predict(X)
    rec = recall_score((y <= 2).astype(int), (pred <= 2).astype(int), zero_division=0)
    meta = {"numeric_cols": cols, "emergency_weight": a.weight, "train_emergency_recall": float(rec)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(f"trained on {len(df)} rows; train emergency-recall={rec:.3f}; saved to {a.out}")

if __name__ == "__main__":
    main()
