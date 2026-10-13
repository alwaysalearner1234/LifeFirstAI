"""Multilingual complaint normalisation (English + Hinglish transliteration).

Patients write as they speak: 'seene me dard', 'saans phool raha', 'bukhar'.
We lowercase, strip punctuation, and map common transliterated phrases to
canonical English tokens APPENDED to the text (original kept) so TF-IDF sees both.
"""
from __future__ import annotations
import re

TRANSLIT_MAP = {
    # chest pain
    "seene me dard": "chest pain", "seene mein dard": "chest pain", "chhati me dard": "chest pain",
    "chest me dard": "chest pain", "dil me dard": "chest pain",
    # breathlessness
    "saans phool": "breathlessness", "sans phool": "breathlessness", "saans nahi aa": "breathlessness",
    "dam ghut": "breathlessness", "huff": "breathlessness",
    # faint / unconscious
    "behosh": "unconscious", "chakkar": "dizziness", "gash": "faint",
    # blood
    "khoon": "blood", "khoon aa": "bleeding", "bleeding ho": "bleeding",
    # fever
    "bukhar": "fever", "bukhaar": "fever", "tez bukhar": "high fever",
    # pain
    "dard": "pain", "pet dard": "abdominal pain", "sar dard": "headache", "sir dard": "headache",
    # cough/cold
    "khansi": "cough", "sardi": "cold", "zukam": "cold", "naak beh": "runny nose",
    # vomiting/diarrhoea
    "ulti": "vomiting", "dast": "diarrhoea", "dast lag": "diarrhoea", "potty": "stool",
    # weakness/paralysis
    "kamzori": "weakness", "lakwa": "paralysis", "sunna": "numbness",
    # pregnancy/labour
    "prasav": "labour", "delivery dard": "labour pain",
    # misc
    "ghabrahat": "palpitation anxiety", "pasina": "sweating", "thand lag": "chills",
}

def normalize_complaint(text: str | None) -> str:
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"\s+", " ", t)
    out = t
    for k, v in TRANSLIT_MAP.items():
        if k in t:
            out += f" {v}"
    out = re.sub(r"[^a-z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
