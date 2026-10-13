"""Red-flag lexicon + detector. Returns list of detected red-flag ids with spans.

Each flag: (id, label, patterns[], min_level_hint). Patterns run on
normalised complaint text (see text_features) + conditions field.
"""
from __future__ import annotations
import re

FLAGS = [
    ("chest_pain", "Chest pain", [r"chest pain", r"seene? .*dard", r"pressure in chest"], 2),
    ("chest_pain_accompanied", "Chest pain + sweat/breathlessness/radiation/nausea",
     [r"chest pain.*(sweat|breathless|radiat|left arm|jaw|nausea|vomit|palpitat)", r"(sweat|breathless).*chest pain"], 2),
    ("breathlessness_severe", "Severe breathing difficulty", [r"severe.*breath", r"cannot breathe", r"gasping", r"breathlessness.*severe", r"oxygen.*low"], 1),
    ("breathlessness", "Breathlessness", [r"breathless", r"shortness of breath", r"saans", r"wheez"], 2),
    ("stroke_face", "Stroke: face droop", [r"face.*droop", r"mouth.*deviat", r"slurr.*speech"], 1),
    ("stroke_weakness", "Stroke: one-side weakness", [r"one side.*weak", r"arm.*drift", r"paralysis.*sudden", r"lakwa"], 1),
    ("stroke_vision_speech", "Stroke: vision/speech loss", [r"sudden.*vision.*loss", r"cannot speak", r"slurred"], 1),
    ("unresponsive", "Unresponsive / fainted", [r"unconscious", r"unresponsive", r"behosh", r"fainted", r"not waking"], 1),
    ("severe_bleeding", "Severe bleeding", [r"heavy bleed", r"severe bleed", r"blood.*not stop", r"hematemesis", r"melena.*heavy", r"khoon.*zyada"], 1),
    ("trauma_major", "Major trauma / accident", [r"road traffic accident", r"major trauma", r"fracture.*open", r"head injury"], 1),
    ("anaphylaxis", "Anaphylaxis (swelling + breathlessness after bite/drug/food)",
     [r"lip.*swell.*breath", r"anaphylaxis", r"throat.*clos"], 1),
    ("seizure_active", "Active / prolonged seizure", [r"seizure.*now", r"fitting.*now", r"status epilepticus", r"continuous.*seizure"], 1),
    ("ob_bleed_labour", "Labour with bleeding / fluid leak", [r"labou?r.*bleed", r"water.*broke.*bleed", r"prasav.*khoon"], 2),
    ("ob_preeclampsia", "Pregnant + severe headache + high BP / swelling",
     [r"pregnan.*headache", r"pregnan.*swell.*bp", r"eclampsia"], 2),
    ("infant_fever", "Infant fever (handled with age in escalation)", [r"fever"], 2),
    ("suicidal", "Suicidal / homicidal ideation", [r"suicid", r"kill myself", r"hang myself"], 2),
    ("palpitation_syncope", "Palpitation with syncope / dizziness", [r"palpitat.*(faint|dizz|syncope|chakkar)"], 2),
    ("high_fever_rigors", "High fever with rigors", [r"high fever", r"rigor", r"tez bukhar"], 3),
    ("dehydration", "Dehydration (dry tongue, no urine, sunken eyes)", [r"no urine", r"sunken eyes", r"dry tongue.*vomit"], 2),
    ("abdominal_guarding", "Severe abdominal pain + guarding/vomiting blood", [r"abdominal.*severe", r"pet dard.*tez"], 2),
]

def detect_red_flags(normalised_text: str, conditions: str = "") -> list[dict]:
    blob = f"{normalised_text or ''} {str(conditions or '').lower()}"
    hits: list[dict] = []
    for fid, label, pats, hint in FLAGS:
        for p in pats:
            if re.search(p, blob):
                hits.append({"id": fid, "label": label, "suggests_min_level": hint, "pattern": p})
                break
    # de-dup
    seen, out = set(), []
    for h in hits:
        if h["id"] not in seen:
            seen.add(h["id"]); out.append(h)
    return out
