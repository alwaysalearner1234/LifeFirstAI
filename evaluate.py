"""Official scorer — asymmetric safety evaluation.

Scoring (per case):
- abstain: 4.0 if true emergency (L1/L2), else 1.5   (abstention rewarded vs missing emergency)
- correct level: 0
- over-triage (pred < true, i.e. predicted MORE urgent): (true - pred) * 1.0
- under-triage non-emergency (pred > true, true>=3): (pred - true) * 3.0
- under-triage EMERGENCY (pred > true, true in {1,2}): (pred - true) * 20.0

Emergency = triage_level in {1,2} OR is_emergency==1 (uses is_emergency if present).

Usage:
  python evaluate.py --data data/intake_30k.csv --preds preds.csv
    preds.csv cols: triage_level, [department], [abstain:0/1]
  python evaluate.py --data data/intake_30k.csv --system models/ --split test
    (trains nothing; loads TriageSystem, 80/20 split by row order, reports + wait-time sim)

Prints EMERGENCY RECALL first and prominently.
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score

C_UNDER_EMERG = 20.0
C_UNDER_NON = 3.0
C_OVER = 1.0
C_ABSTAIN_EMERG = 4.0
C_ABSTAIN_NON = 1.5

def is_emerg_row(r) -> bool:
    if "is_emergency" in r and pd.notna(r["is_emergency"]):
        try:
            if int(r["is_emergency"]) == 1: return True
        except (ValueError, TypeError): pass
    return int(r["triage_level"]) <= 2

def case_cost(yt: int, yp: int, abstain: bool, emerg: bool) -> float:
    if abstain:
        return C_ABSTAIN_EMERG if emerg else C_ABSTAIN_NON
    if yp == yt: return 0.0
    if yp < yt: return (yt - yp) * C_OVER            # over-triage (safe direction)
    return (yp - yt) * (C_UNDER_EMERG if emerg else C_UNDER_NON)  # under-triage

def score(df: pd.DataFrame, pred_level, abstain, pred_dept=None) -> dict:
    costs, under_e, over = [], 0, 0
    n_emerg = 0
    y_true_bin, y_pred_bin = [], []
    for i, r in df.iterrows():
        yt = int(r["triage_level"]); yp = int(pred_level[i]); ab = bool(abstain[i])
        em = is_emerg_row(r)
        n_emerg += em
        costs.append(case_cost(yt, yp, ab, em))
        if em and (yp > 2 and not ab): under_e += 1
        if not ab and yp < yt: over += 1
        y_true_bin.append(1 if em else 0)
        # abstain counts as "caught" (safe) for recall
        y_pred_bin.append(1 if (yp <= 2 or ab) else 0)
    import numpy as np
    costs = np.array(costs)
    out = {
        "n": len(df), "n_emergency": n_emerg,
        "emergency_recall_sensitivity": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "emergency_under_triage_rate": round(under_e / max(n_emerg, 1), 4),
        "total_asymmetric_cost": round(float(costs.sum()), 2),
        "mean_asymmetric_cost": round(float(costs.mean()), 4),
        "accuracy": round(float(accuracy_score(df["triage_level"], [p if not a else t for p, a, t in zip(pred_level, abstain, df['triage_level'])] )) , 4),
        "over_triage_rate": round(over / len(df), 4),
        "abstention_rate": round(float(sum(abstain)) / len(df), 4),
    }
    if pred_dept is not None and "department" in df.columns:
        out["dept_accuracy"] = round(float((pd.Series(list(pred_dept)).values == df["department"].values).mean()), 4)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--system", default=None, help="model dir for self-eval")
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    if a.system:
        sys.path.insert(0, ".")
        from src.predict import TriageSystem
        from src.wait_time import simulate
        n = len(df); k = n if a.split == "all" else n - n // 5
        sub = df.iloc[k:].reset_index(drop=True) if a.split == "test" else df
        sys_ = TriageSystem(a.system)
        pl, ab, pdpt = [], [], []
        for _, r in sub.iterrows():
            o = sys_.predict_one(r.get("age"), r.get("sex"), r.get("sbp"), r.get("dbp"),
                                 r.get("pulse"), r.get("temp_c"), r.get("spo2"),
                                 r.get("complaint", ""), str(r.get("conditions", "")),
                                 str(r.get("medications", "")))
            pl.append(o["triage_level"] if not o["abstain"] else o["triage_level"])
            ab.append(o["abstain"]); pdpt.append(o["department"])
        res = score(sub, pl, ab, pdpt)
        # wait-time stretch: compare FCFS vs priority using predicted levels
        sim_df = sub.copy(); sim_df["triage_level"] = [p if not x else 2 for p, x in zip(pl, ab)]
        res["wait_time_simulation"] = simulate(sim_df)
        df, tag = sub, f"system({a.system}) split={a.split}"
    else:
        assert a.preds, "--preds required when --system not given"
        p = pd.read_csv(a.preds)
        pl = p["triage_level"].astype(int).tolist()
        ab = p["abstain"].astype(int).tolist() if "abstain" in p.columns else [0] * len(p)
        pdpt = p["department"].tolist() if "department" in p.columns else None
        assert len(p) == len(df), "preds/data row mismatch"
        res = score(df, pl, ab, pdpt); tag = a.preds
    print("=" * 64)
    print(f"  EMERGENCY RECALL (sensitivity on hidden ~4% set): {res['emergency_recall_sensitivity']:.3f}")
    print(f"  Emergency under-triage rate: {res['emergency_under_triage_rate']:.3f}  (target ~ 0)")
    print("=" * 64)
    print(f"[{tag}] n={res['n']} emergencies={res['n_emergency']}")
    print(f"total_asymmetric_cost={res['total_asymmetric_cost']}  mean={res['mean_asymmetric_cost']}")
    print(f"accuracy={res['accuracy']}  over_triage={res['over_triage_rate']}  abstention={res['abstention_rate']}"
          + (f"  dept_acc={res.get('dept_accuracy')}" if "dept_accuracy" in res else ""))
    if "wait_time_simulation" in res:
        print("wait-time FCFS->priority per level:", json.dumps(res["wait_time_simulation"]))
    print("NOTE: under-triaging a true emergency costs 20x over-triaging a mild case. Optimise recall, not accuracy.")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2); print(f"saved to {a.out}")

if __name__ == "__main__":
    main()
