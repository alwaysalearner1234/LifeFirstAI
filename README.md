# LifeFirstAI — OPD Triage Assistant for Overloaded Government Hospitals

> **Safety-first ML triage:** from a 60-second intake form + free-text complaint → triage level (1–5), department, red flags — with hard escalation overrides, abstention, and 20× asymmetric safety scoring.

OPD reality: **800 walk-ins/day, 2 doctors, first-come-first-served.** A chest-pain patient waits behind 40 mild fevers. This system re-orders the queue so true emergencies (≈4%) are never missed.

---

## ⚡ 60-second nurse workflow

1. Enter age / sex / vitals (BP, pulse, temp, SpO₂) — missing allowed.
2. Type chief complaint as the patient speaks (Hindi / Hinglish / transliterated OK: `seene me dard`, `saans phool raha`).
3. Add known conditions + meds (optional).
4. Hit **Triage** → instant (<100 ms) color-coded card: **Level 1–5 + department + red flags + escalation reason**, or **"cannot assess — send to nurse now"** (abstention).

## 🛡️ Safety architecture (the core of this repo)

```
Intake → Vitals cleaning → Red-flag scan → ┌──────────────────────┐
                                           │ HARD ESCALATION RULES │── overrides model, never under-triage
                                           └──────────────────────┘
        → ML model (TF-IDF + vitals, emergency-weighted 20×) → Abstention gate → Final output
```

1. **Hard-coded escalation layer** (`src/escalation_rules.py`) — 12 auditable rules on critical vitals + red-flag phrases. Overrides the model. Shown verbatim in UI/API.
2. **Asymmetric cost** (`evaluate.py`) — under-triaging a true emergency costs **20×** over-triaging a mild case. Plain accuracy is misleading by design.
3. **Abstention** (`src/abstention.py`) — says `cannot assess — send to nurse now` on low confidence / too-much-missing; rewarded by the scorer (cost 4 on emergency vs 20–80 for missing it).
4. **Emergency recall reported separately and prominently** — the only metric that matters for lives.

## 📁 Repo layout

```
LifeFirstAI/
├── README.md
├── requirements.txt
├── config.yaml
├── evaluate.py                  # official scorer: 20× asymmetric cost + emergency recall + wait-time sim
├── domain_primer/
│   └── triage_rubric_ESI.md     # published 5-level rubric (ESI-style)
├── src/
│   ├── vitals.py                # missing/mis-entered vitals handling (0 BP, 96°C, °F→°C)
│   ├── text_features.py         # Hinglish/transliteration normalisation
│   ├── red_flags.py             # red-flag lexicon + detector
│   ├── escalation_rules.py      # MANDATORY hard-coded override layer (12 rules)
│   ├── departments.py           # department router
│   ├── features.py              # train/inference feature builder
│   ├── data_generator.py        # 30k synthetic intake generator (multilingual complaints)
│   ├── train.py                 # emergency-weighted training
│   ├── predict.py               # TriageSystem: rules → model → abstention
│   ├── abstention.py            # abstention gate
│   └── wait_time.py             # FCFS vs priority-queue wait simulation
├── app/
│   ├── main.py                  # FastAPI: POST /triage, <100ms
│   └── static/index.html        # 60-second nurse UI
├── scripts/
│   ├── make_data.py
│   └── demo.py
├── tests/
│   ├── test_escalation.py
│   ├── test_evaluate.py
│   └── test_predict.py
└── data/
    ├── README.md
    ├── sample_intake.csv        # 1k sample (generate full 30k via script)
    └── arrival_log_sample.csv
```

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (30k default, matches problem spec)
python scripts/make_data.py --n 30000 --out data/intake_30k.csv --arrival-log data/arrival_log.csv

# 2. Train (emergency-weighted, saves to models/)
python -m src.train --data data/intake_30k.csv --out models/

# 3. Score with asymmetric safety cost
python evaluate.py --data data/intake_30k.csv --preds preds.csv
# or self-evaluate the system:
python evaluate.py --data data/intake_30k.csv --system models/ --split test

# 4. Instant API + UI
uvicorn app.main:app --reload
# → http://127.0.0.1:8000  (nurse UI) | POST /triage | GET /health
```

## 📊 Metrics that matter

`evaluate.py` prints (and returns as JSON):

| Metric | Why it matters |
|---|---|
| **Total / mean asymmetric cost** (under-emergency = 20×) | Primary leaderboard score |
| **Emergency recall / sensitivity** (true L1–L2 caught as L1–L2 or abstain) | Lives saved — shown first |
| Under-triage rate on emergencies | Must be ≈0 |
| Over-triage rate | Tolerated |
| Abstention rate + abstention appropriateness | Rewarded when uncertain |
| Dept accuracy | Right doctor |
| Wait-time reduction vs FCFS (stretch) | OPD impact |

> Teams optimising plain accuracy will score badly. Optimise **recall on the hidden ~4% emergencies**.

## 🧷 Escalation rules (auditable, shown in UI)

See `src/escalation_rules.py` + `domain_primer/triage_rubric_ESI.md`. Summary:

| ID | Trigger | Forced level |
|---|---|---|
| R1 | SpO₂ < 90 | L1 |
| R2 | SpO₂ 90–92 + respiratory distress | L2 |
| R3 | SBP <80 or >220, DBP >130, MAP <55 | L1 |
| R4 | Pulse <35 or >150 | L1 |
| R5 | Chest pain + (sweat/breathlessness/radiation/jaw/arm) | L2 min |
| R6 | Stroke signs (face droop/slurred/weakness/vision) | L1 |
| R7 | Unresponsive / severe bleeding / anaphylaxis / status seizure / major trauma | L1 |
| R8 | OB emergency (labour+bleed / pregnant+headache+HTN) | L2 min |
| R9 | Infant <3 mo + fever ≥38 °C | L2 min |
| R10 | Temp >41 or <32 °C (post unit-fix) | L2 min |
| R11 | Suicidal / homicidal + plan | L2 min |
| R12 | Severe dehydration + hypotension/tachycardia | L2 min |

Vitals cleaning: `0` → missing; `Temp >50` → assume °F → °C; `Temp 96` (bare) → 96 °F = 35.6 °C path; impossible values quarantined → abstention, never silent.

## 🧪 Tests

```bash
pytest -q
```

## 📜 Data contract

Input columns: `age, sex, sbp, dbp, pulse, temp_c, spo2, complaint, conditions, medications, arrival_time`.
Labels (hidden at test): `triage_level (1–5), department, is_emergency (0/1)`.
No labs / imaging / ECG at triage — intake form only.

## ⚠️ Disclaimer

Decision-support only. Not a medical device. Escalation rules are conservative by design — when in doubt the system over-triages or abstains to a human nurse.
