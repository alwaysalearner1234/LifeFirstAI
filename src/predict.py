"""TriageSystem: rules → model → abstention. Near-instant (<100ms).

predict_one(age, sex, sbp, dbp, pulse, temp_c, spo2, complaint, conditions,
medications) -> dict(triage_level, department, red_flags, escalation,
abstain, reasons, latency_ms, probs)
"""
from __future__ import annotations
import time, math, os
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from .vitals import clean_vitals
from .text_features import normalize_complaint
from .red_flags import detect_red_flags
from .escalation_rules import apply_escalation
from .departments import route_department
from .abstention import should_abstain, ABSTAIN_MESSAGE
from .features import numeric_matrix
import pandas as pd

class TriageSystem:
    def __init__(self, model_dir: str = "models", abstain_threshold: float = 0.45):
        self.clf = joblib.load(os.path.join(model_dir, "triage_model.joblib"))
        self.dclf = joblib.load(os.path.join(model_dir, "dept_model.joblib"))
        self.vec = joblib.load(os.path.join(model_dir, "tfidf.joblib"))
        try:
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        except Exception:
            self.scaler = None
        self.thr = abstain_threshold

    def _ml(self, rec: dict):
        df = pd.DataFrame([rec])
        from .features import row_text
        Xt = self.vec.transform([row_text(df.iloc[0])])
        Xn, _ = numeric_matrix(df)
        X = hstack([Xt, csr_matrix(Xn)]).tocsr()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        proba = self.clf.predict_proba(X)[0]
        classes = list(self.clf.classes_)
        i = int(np.argmax(proba))
        try:
            dept = str(self.dclf.predict(X)[0])
        except Exception:
            dept = "General Medicine"
        return int(classes[i]), float(proba[i]), {int(c): float(p) for c, p in zip(classes, proba)}, dept

    def predict_one(self, age=None, sex="M", sbp=None, dbp=None, pulse=None,
                    temp_c=None, spo2=None, complaint="", conditions="",
                    medications="") -> dict:
        t0 = time.perf_counter()
        rec = {"age": age, "sex": sex, "sbp": sbp, "dbp": dbp, "pulse": pulse,
               "temp_c": temp_c, "spo2": spo2, "complaint": complaint,
               "conditions": conditions, "medications": medications}
        clean, qflags = clean_vitals(rec)
        norm = normalize_complaint(complaint)
        flags = detect_red_flags(norm, conditions)
        forced, triggered = apply_escalation(clean, flags, age)
        ml_level, maxp, probs, ml_dept = self._ml(rec)
        rule_dept, rule_why = route_department(flags, age, sex, norm, conditions)
        dept = rule_dept if rule_why != "default OPD" else ml_dept
        # escalation overrides model (min = most urgent wins)
        if forced is not None:
            level = min(forced, ml_level)
            escalated = True
            if forced == 1:
                dept, rule_why = "Emergency", "L1 resuscitation override"
        else:
            level, escalated = ml_level, False
        n_missing = sum(1 for v in clean.values() if isinstance(v, float) and math.isnan(v))
        abstain, why = should_abstain(maxp, n_missing, len(flags), len(norm), qflags, self.thr)
        # critical overrides must NEVER abstain into a mild queue: abstain keeps safe level
        if abstain:
            out_level = min(level, 2)
            out = {"triage_level": out_level, "abstain": True, "abstain_message": ABSTAIN_MESSAGE,
                   "abstain_reason": why}
        else:
            out = {"triage_level": level, "abstain": False, "abstain_message": "", "abstain_reason": ""}
        out.update({"department": dept, "dept_reason": rule_why,
                    "red_flags": flags, "escalation": triggered, "escalated": escalated,
                    "model_level": ml_level, "model_confidence": round(maxp, 3),
                    "probs": probs, "vitals_clean": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in clean.items()},
                    "quality_flags": qflags,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return out
