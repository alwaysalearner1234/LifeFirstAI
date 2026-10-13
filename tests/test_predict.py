def test_system_loads_and_fast():
    import os
    if not os.path.exists("models/triage_model.joblib"):
        import pytest; pytest.skip("train first")
    from src.predict import TriageSystem
    s = TriageSystem("models")
    o = s.predict_one(55, "M", 150, 95, 110, 37.2, 93, "seene me dard with pasina", "hypertension", "")
    assert o["triage_level"] <= 2
    assert o["latency_ms"] < 500
    assert any(f["id"] == "chest_pain" for f in o["red_flags"])

def test_empty_abstains():
    import os
    if not os.path.exists("models/triage_model.joblib"):
        import pytest; pytest.skip("train first")
    from src.predict import TriageSystem
    s = TriageSystem("models")
    o = s.predict_one(40, "M", 0, 0, 0, 0, 0, "", "", "")
    assert o["abstain"] is True and "nurse" in o["abstain_message"]
