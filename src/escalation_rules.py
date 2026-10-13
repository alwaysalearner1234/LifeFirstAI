"""MANDATORY hard-coded escalation rule layer.

These rules OVERRIDE the ML model. They are deliberately conservative:
critical vitals or red-flag phrases force triage to L1/L2 regardless of
what the statistical model says. Every override returns its rule id + reason
so nurses/auditors can see exactly why.

Display these rules in the UI (they are imported by app/main.py).
"""
from __future__ import annotations
import math

RULES_CARD = [
    ("R1", "SpO2 < 90% → L1 (severe hypoxia)", "spo2 < 90"),
    ("R2", "SpO2 90–92% + respiratory distress → L2", "spo2 90-92 + breathlessness flag"),
    ("R3", "SBP <80 or >220, DBP >130, MAP <55 → L1 (shock / hypertensive emergency)", "sbp/dbp/map"),
    ("R4", "Pulse <35 or >150 → L1 (unstable rhythm)", "pulse"),
    ("R5", "Chest pain + sweat/breathlessness/radiation/jaw/arm/nausea → min L2", "red-flag combo"),
    ("R6", "Any stroke sign (face droop / slurred speech / one-side weakness / vision loss) → L1", "red-flag"),
    ("R7", "Unresponsive / severe bleeding / anaphylaxis / active seizure / major trauma → L1", "red-flag"),
    ("R8", "OB emergency: labour+bleed OR pregnant+headache+HTN → min L2", "red-flag + vitals"),
    ("R9", "Infant <3 months + fever ≥38.0°C → min L2", "age + temp"),
    ("R10", "Temp >41°C or <32°C (post unit-fix) → min L2", "temp"),
    ("R11", "Suicidal/homicidal with plan → min L2 (psychiatry + emergency)", "red-flag"),
    ("R12", "Severe dehydration + (hypotension or tachycardia) → min L2", "red-flag + vitals"),
]

def _has(flag_ids: set[str], *names: str) -> bool:
    return any(n in flag_ids for n in names)

def apply_escalation(clean: dict, red_flags: list[dict], age) -> tuple[int | None, list[dict]]:
    """Return (forced_level or None, triggered_rules[{id, reason}]). Lower number = more urgent."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        a = float("nan")
    spo2, sbp, dbp, pulse, temp = clean["spo2"], clean["sbp"], clean["dbp"], clean["pulse"], clean["temp_c"]
    ids = {f["id"] for f in red_flags}
    triggered: list[dict] = []
    forced: int | None = None

    def force(level: int, rid: str, reason: str):
        nonlocal forced
        triggered.append({"id": rid, "reason": reason, "forces_level": level})
        if forced is None or level < forced:
            forced = level

    if not (isinstance(spo2, float) and math.isnan(spo2)) and spo2 < 90:
        force(1, "R1", f"SpO2 {spo2:.0f}% < 90% — severe hypoxia")
    if (not (isinstance(spo2, float) and math.isnan(spo2)) and 90 <= spo2 <= 92
            and _has(ids, "breathlessness", "breathlessness_severe")):
        force(2, "R2", f"SpO2 {spo2:.0f}% borderline with respiratory distress")
    if ((not math.isnan(sbp) and (sbp < 80 or sbp > 220))
            or (not math.isnan(dbp) and dbp > 130)):
        force(1, "R3", f"Extreme BP SBP={sbp:.0f} DBP={dbp:.0f}")
    if not math.isnan(sbp) and not math.isnan(dbp):
        m = (sbp + 2 * dbp) / 3
        if m < 55:
            force(1, "R3", f"MAP {m:.0f} < 55 — shock")
    if not math.isnan(pulse) and (pulse < 35 or pulse > 150):
        force(1, "R4", f"Pulse {pulse:.0f} — unstable rhythm range")
    if "chest_pain" in ids and _has(ids, "chest_pain_accompanied", "breathlessness",
                                    "breathlessness_severe", "palpitation_syncope"):
        force(2, "R5", "Chest pain with red-flag accompaniment")
    elif "chest_pain_accompanied" in ids:
        force(2, "R5", "Chest pain with red-flag accompaniment")
    if _has(ids, "stroke_face", "stroke_weakness", "stroke_vision_speech"):
        force(1, "R6", "Stroke sign detected — time-critical")
    if _has(ids, "unresponsive", "severe_bleeding", "anaphylaxis", "seizure_active", "trauma_major"):
        hit = sorted(ids & {"unresponsive", "severe_bleeding", "anaphylaxis", "seizure_active", "trauma_major"})
        force(1, "R7", f"Critical red flag: {','.join(hit)}")
    if _has(ids, "ob_bleed_labour", "ob_preeclampsia"):
        force(2, "R8", "Obstetric emergency pattern")
    if not math.isnan(a) and a < 0.25 and not math.isnan(temp) and temp >= 38.0:
        force(2, "R9", f"Infant {a*12:.0f} mo with fever {temp:.1f}°C")
    if not math.isnan(temp) and (temp > 41 or temp < 32):
        force(2, "R10", f"Extreme temperature {temp:.1f}°C")
    if "suicidal" in ids:
        force(2, "R11", "Suicidal/homicidal ideation")
    if "dehydration" in ids and ((not math.isnan(sbp) and sbp < 90) or (not math.isnan(pulse) and pulse > 120)):
        force(2, "R12", "Dehydration with haemodynamic compromise")
    return forced, triggered
