def test_asymmetric_20x():
    from evaluate import case_cost
    under_emerg = case_cost(1, 2, False, True)
    over = case_cost(2, 1, False, False)
    assert under_emerg == 20.0 and over == 1.0

def test_abstain_cheaper_than_miss():
    from evaluate import case_cost
    assert case_cost(1, 5, False, True) == 80.0
    assert case_cost(1, 5, True, True) == 4.0

def test_recall_counts_abstain_as_safe():
    import pandas as pd
    from evaluate import score
    df = pd.DataFrame({"triage_level": [1, 5], "is_emergency": [1, 0]})
    r = score(df, [5, 5], [True, False])
    assert r["emergency_recall_sensitivity"] == 1.0
