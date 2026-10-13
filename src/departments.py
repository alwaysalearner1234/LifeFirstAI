"""Department router: red-flag + demographics + vitals → department.

Rule-first, ML-second. Rules handle emergencies deterministically;
the ML department classifier (train.py) handles the long tail.
"""
from __future__ import annotations

DEPARTMENTS = ["Emergency", "Cardiology", "Pulmonology", "Neurology", "General Medicine",
               "Pediatrics", "OBGYN", "Orthopedics", "Surgery", "Gastroenterology",
               "ENT", "Ophthalmology", "Dermatology", "Psychiatry"]

def route_department(red_flags: list[dict], age, sex: str, complaint_norm: str, conditions: str = "") -> tuple[str, str]:
    ids = {f["id"] for f in red_flags}
    blob = f"{complaint_norm} {str(conditions or '').lower()}"
    try:
        a = float(age)
    except (TypeError, ValueError):
        a = float("nan")
    # Emergency first
    if ids & {"unresponsive", "severe_bleeding", "anaphylaxis", "seizure_active", "trauma_major",
              "stroke_face", "stroke_weakness", "stroke_vision_speech", "breathlessness_severe"}:
        return "Emergency", "critical red flag"
    if "chest_pain_accompanied" in ids or "chest_pain" in ids:
        return "Cardiology", "chest pain pathway"
    if ids & {"breathlessness", "breathlessness_severe"} or "asthma" in blob or "copd" in blob:
        return "Pulmonology", "respiratory pattern"
    if ids & {"stroke_face", "stroke_weakness", "stroke_vision_speech"}:
        return "Neurology", "neurological deficit"
    if ids & {"ob_bleed_labour", "ob_preeclampsia"} or "pregnan" in blob or "labour" in blob:
        return "OBGYN", "obstetric pattern"
    if not (a != a) and a < 12:
        return "Pediatrics", "child"
    if "fracture" in blob or "sprain" in blob or "joint" in blob or "back pain" in blob and "trauma" in blob:
        return "Orthopedics", "musculoskeletal trauma"
    if "suicid" in blob:
        return "Psychiatry", "self-harm"
    if any(w in blob for w in ["eye", "vision", "red eye"]):
        return "Ophthalmology", "eye"
    if any(w in blob for w in ["ear", "throat", "tonsil", "nose bleed"]):
        return "ENT", "ent"
    if any(w in blob for w in ["rash", "itch", "eczema", "skin"]):
        return "Dermatology", "skin"
    if any(w in blob for w in ["vomit blood", "abdomen", "diarrhoea", "jaundice", "pet dard", "abdominal"]):
        return "Gastroenterology", "abdominal"
    if any(w in blob for w in ["cut", "wound", "abscess", "appendix", "hernia"]):
        return "Surgery", "surgical"
    return "General Medicine", "default OPD"
