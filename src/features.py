"""Feature builder shared by train + inference (no leakage, fast).

Text: TF-IDF(word 1-2gram, 4000 feats) on normalised complaint+conditions.
Numeric: age, sbp, dbp, map, pulse, temp_c, spo2 + missingness indicators +
vitals severity flags + red-flag count + comorbidity flags.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .text_features import normalize_complaint
from .vitals import clean_vitals
from .red_flags import detect_red_flags

HIGH_RISK_CONDS = {"hypertension", "diabetes", "asthma", "copd", "ckd", "epilepsy"}

def row_text(df_row) -> str:
    c = normalize_complaint(df_row.get("complaint", ""))
    cond = str(df_row.get("conditions", "")).lower()
    med = str(df_row.get("medications", "")).lower()
    return f"{c} {cond} {med}".strip()

def numeric_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = []
    for _, r in df.iterrows():
        clean, _ = clean_vitals({"sbp": r.get("sbp"), "dbp": r.get("dbp"),
                                 "pulse": r.get("pulse"), "temp_c": r.get("temp_c"),
                                 "spo2": r.get("spo2")})
        try: age = float(r.get("age"))
        except (TypeError, ValueError): age = float("nan")
        sbp, dbp, pu, te, sp = clean["sbp"], clean["dbp"], clean["pulse"], clean["temp_c"], clean["spo2"]
        mape = (sbp + 2 * dbp) / 3 if not (isinstance(sbp, float) and math.isnan(sbp)) and not (isinstance(dbp, float) and math.isnan(dbp)) else float("nan")
        norm = normalize_complaint(r.get("complaint", ""))
        flags = detect_red_flags(norm, str(r.get("conditions", "")))
        def f(v): return 0.0 if isinstance(v, float) and math.isnan(v) else float(v)
        def m(v): return 1.0 if isinstance(v, float) and math.isnan(v) else 0.0
        cond = str(r.get("conditions", "")).lower()
        feats.append([f(age if not (isinstance(age, float) and math.isnan(age)) else 0.0),
                      1.0 if str(r.get("sex", "")).upper().startswith("F") else 0.0,
                      f(sbp), f(dbp), f(mape), f(pu), f(te), f(sp),
                      m(sbp), m(dbp), m(pu), m(te), m(sp),
                      float(len(flags)),
                      1.0 if any(c in cond for c in HIGH_RISK_CONDS) else 0.0,
                      1.0 if "pregnan" in cond else 0.0,
                      1.0 if (isinstance(te, float) and not math.isnan(te) and te >= 39) else 0.0,
                      1.0 if (isinstance(sp, float) and not math.isnan(sp) and sp < 94) else 0.0])
    cols = ["age", "is_f", "sbp", "dbp", "map", "pulse", "temp", "spo2",
            "m_sbp", "m_dbp", "m_pu", "m_te", "m_sp", "n_flags", "comorb", "preg", "hi_fever", "lo_spo2"]
    return np.array(feats, dtype=float), cols

def build_tfidf(texts: list[str], max_features: int = 4000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    return vec, X
