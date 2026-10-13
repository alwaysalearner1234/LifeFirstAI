"""Generate intake CSV + arrival log."""
import argparse
from src.data_generator import generate
import pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="data/intake_30k.csv")
    ap.add_argument("--arrival-log", default="data/arrival_log.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    df.to_csv(a.out, index=False)
    df[["arrival_time", "triage_level"]].to_csv(a.arrival_log, index=False)
    print(f"wrote {len(df)} rows -> {a.out}; emergency_rate={df.is_emergency.mean():.3f}")
