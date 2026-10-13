import math

def test_spo2_r1_forces_l1():
    from src.escalation_rules import apply_escalation
    lvl, trig = apply_escalation({"spo2": 85.0, "sbp": 120.0, "dbp": 80.0, "pulse": 90.0, "temp_c": 37.0}, [], 40)
    assert lvl == 1 and trig[0]["id"] == "R1"

def test_chest_pain_combo_forces_l2():
    from src.escalation_rules import apply_escalation
    flags = [{"id": "chest_pain", "label": "x", "suggests_min_level": 2, "pattern": "x"},
             {"id": "chest_pain_accompanied", "label": "y", "suggests_min_level": 2, "pattern": "y"}]
    lvl, trig = apply_escalation({"spo2": 98.0, "sbp": 130.0, "dbp": 85.0, "pulse": 95.0, "temp_c": 37.0}, flags, 50)
    assert lvl == 2 and any(t["id"] == "R5" for t in trig)

def test_temp_96F_not_96C():
    from src.vitals import clean_vitals
    clean, flags = clean_vitals({"sbp": 120, "dbp": 80, "pulse": 80, "temp_c": 96, "spo2": 98})
    assert abs(clean["temp_c"] - 35.6) < 0.3, clean
    assert any("F_to_C" in f for f in flags)

def test_zero_bp_is_missing_not_zero():
    from src.vitals import clean_vitals
    clean, flags = clean_vitals({"sbp": 0, "dbp": 0, "pulse": 80, "temp_c": 37, "spo2": 98})
    assert math.isnan(clean["sbp"]) and "sbp_missing" in flags
