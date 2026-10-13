"""Synthetic 30k intake generator matching the problem spec.

Columns: age, sex, sbp, dbp, pulse, temp_c, spo2, complaint, conditions,
medications, arrival_time (+hidden: triage_level, department, is_emergency).

Design:
- ~4% true emergencies (L1/L2) via red-flag scenarios.
- Complaints in mixed styles: plain English, Hinglish transliteration, terse.
- Vitals noise: 8% missing each, 1.5% mis-entered (0 BP, 96/102 temp as F, typos).
- Under/over-reporting: 10% mild cases use dramatic words, 15% emergencies understate.
"""
from __future__ import annotations
import random, math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

COMPLAINT_POOL = {
    # (text variants, triage, department, red_flags implied, emergency?)
    "chest_emerg": (["chest pain with sweating and breathlessness since morning",
                     "seene me dard with pasina and saans phool raha",
                     "pressure in chest radiating to left arm, nausea",
                     "dil me dard, ghabrahat, ulti jaisa lag raha"], 2, "Cardiology", True),
    "stroke": (["face drooping on one side, slurred speech since 1 hour",
                "right hand kamzori, bolne me dikkat, muh tedha",
                "sudden vision loss + weakness left side"], 1, "Emergency", True),
    "severe_breath": (["cannot breathe, gasping, lips blue",
                       "saans nahi aa rahi, dam ghut raha hai"], 1, "Emergency", True),
    "bleed": (["heavy bleeding after cut, blood not stopping",
               "khoon zyada beh raha, chakkar aa raha"], 1, "Emergency", True),
    "fever_mild": (["mild fever cold cough since 2 days",
                    "halka bukhar sardi khansi",
                    "fever 99 with runny nose, no distress"], 5, "General Medicine", False),
    "fever_high": (["high fever with rigors 103F since 3 days",
                    "tez bukhar with thand lag, sar dard"], 3, "General Medicine", False),
    "child_fever": (["3 month baby fever, poor feeding", "infant bukhar, doodh nahi pi raha"], 3, "Pediatrics", False),
    "abd": (["severe stomach pain vomiting since night", "pet dard tez with ulti"], 3, "Gastroenterology", False),
    "injury": (["ankle sprain after fall, swelling", "road traffic accident, head injury, bleeding"], 4, "Orthopedics", False),
    "ob": (["pregnant 8 month severe headache swelling, BP high",
            "labour pain with bleeding"], 2, "OBGYN", True),
    "routine": (["BP sugar followup, need refill, no complaints",
                 "BP sugar followup, stable, need medicines"], 5, "General Medicine", False),
    "derm": (["skin rash itching since week", "khujli with red patches on arms"], 5, "Dermatology", False),
    "eye": (["eye redness watering since 2 days"], 5, "Ophthalmology", False),
}

CONDITIONS = ["hypertension", "diabetes", "asthma", "COPD", "CKD", "pregnant", "epilepsy", "none"]
MEDS = ["amlodipine", "metformin", "salbutamol inhaler", "aspirin", "none"]

def _vitals_for(level: int, rng: random.Random):
    if level == 1:
        sbp = rng.choice([rng.randint(60, 79), rng.randint(221, 240), rng.randint(70, 90)])
        dbp = rng.randint(40, 140); pulse = rng.choice([rng.randint(28, 34), rng.randint(151, 170)])
        spo2 = rng.randint(82, 89); temp = round(rng.uniform(37.5, 40.5), 1)
    elif level == 2:
        sbp = rng.randint(85, 225); dbp = rng.randint(55, 132); pulse = rng.randint(100, 145)
        spo2 = rng.randint(90, 93); temp = round(rng.uniform(37.0, 40.0), 1)
    elif level == 3:
        sbp = rng.randint(95, 160); dbp = rng.randint(60, 100); pulse = rng.randint(90, 120)
        spo2 = rng.randint(94, 97); temp = round(rng.uniform(38.0, 40.5), 1)
    elif level == 4:
        sbp = rng.randint(100, 150); dbp = rng.randint(60, 95); pulse = rng.randint(70, 110)
        spo2 = rng.randint(96, 99); temp = round(rng.uniform(36.5, 38.5), 1)
    else:
        sbp = rng.randint(105, 135); dbp = rng.randint(65, 88); pulse = rng.randint(65, 95)
        spo2 = rng.randint(97, 99); temp = round(rng.uniform(36.5, 37.8), 1)
    return sbp, dbp, pulse, temp, spo2

def generate(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    keys = list(COMPLAINT_POOL.keys())
    # force ~4% emergency keys
    emerg_keys = ["chest_emerg", "stroke", "severe_breath", "bleed", "ob"]
    non_emerg_keys = [k for k in keys if k not in emerg_keys]
    rows = []
    base = datetime(2026, 9, 1, 8, 0, 0)
    for i in range(n):
        r = rng.random()
        if r < 0.03:
            k = rng.choice(emerg_keys)
        else:
            k = rng.choice(non_emerg_keys)
        variants, level, dept, is_em = COMPLAINT_POOL[k]
        if k == "injury":
            # minor sprain common; major RTA rare (≈10% of injuries)
            complaint = variants[0] if rng.random() < 0.9 else variants[1]
        else:
            complaint = rng.choice(variants)
        # under/over-report noise
        if is_em and rng.random() < 0.15:
            complaint = complaint.replace("severe ", "").replace("heavy ", "")
        if not is_em and rng.random() < 0.10:
            complaint = "severe " + complaint
        # fix department for child/injury variants
        age = rng.randint(0, 90) if k != "child_fever" else rng.randint(0, 1)
        if k == "injury" and "head injury" in complaint:
            level, dept, is_em = 1, "Emergency", True
        # children route to Pediatrics (matches department router)
        if age < 12 and dept in ("General Medicine", "Gastroenterology", "Dermatology",
                                 "Ophthalmology", "Orthopedics") and not is_em:
            dept = "Pediatrics"
        sex = rng.choice(["M", "F"])
        sbp, dbp, pulse, temp, spo2 = _vitals_for(level, rng)
        # missing 8%
        for col in ["sbp", "dbp", "pulse", "temp", "spo2"]:
            if rng.random() < 0.08:
                if col == "sbp": sbp = 0
                elif col == "dbp": dbp = 0
                elif col == "pulse": pulse = 0
                elif col == "temp": temp = 0
                else: spo2 = 0
        # mis-entered 1.5%: temp as Fahrenheit bare number
        if rng.random() < 0.015 and temp != 0:
            temp = round(temp * 9 / 5 + 32, 1)  # e.g. 102.2 stored in temp_c column
        cond = rng.choice(CONDITIONS); med = rng.choice(MEDS)
        if dept == "OBGYN": cond = "pregnant"; sex = "F"
        arrival = base + timedelta(minutes=int(i * 0.6 + rng.expovariate(1 / 2)))
        rows.append({"age": age, "sex": sex, "sbp": sbp, "dbp": dbp, "pulse": pulse,
                     "temp_c": temp, "spo2": spo2, "complaint": complaint,
                     "conditions": cond, "medications": med,
                     "arrival_time": arrival.strftime("%Y-%m-%d %H:%M:%S"),
                     "triage_level": level, "department": dept, "is_emergency": int(is_em)})
    df = pd.DataFrame(rows)
    # ensure emergency rate ~4%: adjust threshold already; report
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="data/intake_30k.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}; emergency_rate={df.is_emergency.mean():.3f}")
