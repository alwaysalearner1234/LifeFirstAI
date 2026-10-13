"""FCFS vs priority-queue wait simulation (stretch goal).

Model: 800 patients/day, 2 doctors, mean service 6 min (2 parallel servers).
FCFS: sort by arrival. Priority: sort key = (triage_level, arrival) with
L1 pre-empting; abstains treated as L2 (safe). Reports mean wait per level
and % reduction for L1/L2.
"""
from __future__ import annotations
import pandas as pd

SERVICE_MIN = 6.0
SERVERS = 2

def simulate(df: pd.DataFrame, level_col: str = "triage_level", arrival_col: str = "arrival_time") -> dict:
    d = df.copy()
    d[arrival_col] = pd.to_datetime(d[arrival_col])
    # FCFS
    d = d.sort_values(arrival_col).reset_index(drop=True)
    free = [0.0] * SERVERS
    waits_fcfs = []
    t0 = d[arrival_col].min()
    for _, r in d.iterrows():
        arr = (r[arrival_col] - t0).total_seconds() / 60
        s = min(range(SERVERS), key=lambda k: free[k])
        start = max(arr, free[s]); waits_fcfs.append(start - arr); free[s] = start + SERVICE_MIN
    d["wait_fcfs"] = waits_fcfs
    # Priority
    d = d.sort_values([level_col, arrival_col]).reset_index(drop=True)
    free = [0.0] * SERVERS
    waits_p = []
    for _, r in d.iterrows():
        arr = (r[arrival_col] - t0).total_seconds() / 60
        s = min(range(SERVERS), key=lambda k: free[k])
        start = max(arr, free[s]); waits_p.append(start - arr); free[s] = start + SERVICE_MIN
    d["wait_prio"] = waits_p
    out = {}
    for lvl in sorted(d[level_col].unique()):
        m = d[d[level_col] == lvl]
        out[int(lvl)] = {"n": int(len(m)), "fcfs_min": round(float(m.wait_fcfs.mean()), 1),
                         "prio_min": round(float(m.wait_prio.mean()), 1)}
        out[int(lvl)]["reduction_pct"] = round(100 * (out[int(lvl)]["fcfs_min"] - out[int(lvl)]["prio_min"]) / max(out[int(lvl)]["fcfs_min"], 1e-6), 1)
    return out
