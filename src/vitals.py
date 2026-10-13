"""Vitals cleaning: handles missing + mis-entered values.

Rules:
- 0 / negative for BP, pulse, spo2, temp -> treated as missing (NaN).
- Temp > 50 -> almost certainly Fahrenheit (e.g. 96, 98.6, 102) -> convert to C.
  Bare "96" therefore becomes 35.6 C, not 96 C. Temp still out of
  [-10, 50]C after conversion -> NaN (quarantined) + flag.
- BP bounds: SBP 50-300, DBP 20-200 else NaN. SBP<DBP swap if plausible.
- Pulse 20-250, SpO2 50-100 else NaN.
- Returns cleaned dict + list of quality flags (used by abstention).
"""
from __future__ import annotations
import math

def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or v == "" or v == 0 or v == "0"

def to_float(v):
    if v is None or v == "":
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f

def clean_temp(raw) -> tuple[float, list[str]]:
    flags: list[str] = []
    t = to_float(raw)
    if math.isnan(t) or t == 0:
        return float("nan"), ["temp_missing"]
    # Mis-entered Fahrenheit
    if t > 50:
        flags.append(f"temp_unit_fix_F_to_C:raw={t}")
        t = (t - 32.0) * 5.0 / 9.0
    if t < 25 or t > 44:
        flags.append(f"temp_implausible:{t:.1f}C_quarantined")
        return float("nan"), flags
    return round(t, 1), flags

def clean_bp(sbp_raw, dbp_raw):
    flags: list[str] = []
    s = to_float(sbp_raw); d = to_float(dbp_raw)
    if s == 0: s = float("nan")
    if d == 0: d = float("nan")
    if not math.isnan(s) and (s < 50 or s > 300):
        flags.append(f"sbp_implausible:{s}_quarantined"); s = float("nan")
    if not math.isnan(d) and (d < 20 or d > 200):
        flags.append(f"dbp_implausible:{d}_quarantined"); d = float("nan")
    if not math.isnan(s) and not math.isnan(d) and s < d and d - s < 60:
        # likely swapped entry
        s, d = d, s
        flags.append("bp_swapped_SBP_DBP")
    missing = []
    if math.isnan(s): missing.append("sbp_missing")
    if math.isnan(d): missing.append("dbp_missing")
    return s, d, flags + missing

def clean_pulse(raw):
    p = to_float(raw)
    if math.isnan(p) or p == 0:
        return float("nan"), ["pulse_missing"]
    if p < 20 or p > 250:
        return float("nan"), [f"pulse_implausible:{p}_quarantined"]
    return float(p), []

def clean_spo2(raw):
    s = to_float(raw)
    if math.isnan(s) or s == 0:
        return float("nan"), ["spo2_missing"]
    if s > 100 and s <= 1000:
        # e.g. entered as 985 instead of 98.5? quarantine
        return float("nan"), [f"spo2_implausible:{s}_quarantined"]
    if s < 50 or s > 100:
        return float("nan"), [f"spo2_implausible:{s}_quarantined"]
    return float(s), []

def clean_vitals(rec: dict) -> tuple[dict, list[str]]:
    """rec keys: sbp, dbp, pulse, temp_c, spo2 (also accepts temp, bp aliases)."""
    flags: list[str] = []
    sbp, dbp, f = clean_bp(rec.get("sbp", rec.get("sbp_mmHg")), rec.get("dbp", rec.get("dbp_mmHg")))
    flags += f
    pulse, f = clean_pulse(rec.get("pulse", rec.get("hr")))
    flags += f
    temp, f = clean_temp(rec.get("temp_c", rec.get("temp", rec.get("temperature"))))
    flags += f
    spo2, f = clean_spo2(rec.get("spo2", rec.get("o2sat")))
    flags += f
    missing_count = sum(1 for v in (sbp, dbp, pulse, temp, spo2) if isinstance(v, float) and math.isnan(v))
    if missing_count >= 3:
        flags.append(f"many_vitals_missing:{missing_count}/5")
    return {"sbp": sbp, "dbp": dbp, "pulse": pulse, "temp_c": temp, "spo2": spo2}, flags
