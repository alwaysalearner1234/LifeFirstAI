"""Abstention gate: know when to say 'cannot assess — send to nurse now'.

Abstain when:
- model max probability < threshold (default 0.45), OR
- >=3 vitals missing AND no red flags AND complaint very short/empty, OR
- vitals quarantined as implausible with conflicting text.

Abstention is SAFER than a confident wrong answer: scorer charges 4 on an
emergency abstain vs 20+ for under-triaging it, and 1.5 on non-emergency.
"""
from __future__ import annotations
import math

ABSTAIN_MESSAGE = "cannot assess — send to nurse now"

def should_abstain(max_proba: float, n_missing_vitals: int, n_red_flags: int,
                   complaint_len: int, quality_flags: list[str],
                   threshold: float = 0.45) -> tuple[bool, str]:
    if max_proba < threshold:
        return True, f"low model confidence ({max_proba:.2f} < {threshold})"
    if n_missing_vitals >= 3 and n_red_flags == 0 and complaint_len < 12:
        return True, f"too little signal ({n_missing_vitals}/5 vitals missing, {complaint_len} chars)"
    if any("quarantined" in q for q in quality_flags) and n_red_flags == 0 and max_proba < 0.6:
        return True, "implausible vitals quarantined + uncertain text"
    return False, ""
