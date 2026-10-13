"""Quick demo: 6 archetypal patients through the full safety stack."""
from src.predict import TriageSystem
import sys
sys.path.insert(0, ".")

CASES = [
    dict(age=55, sex="M", sbp=150, dbp=95, pulse=110, temp_c=37.2, spo2=93,
         complaint="seene me dard with pasina and saans phool raha", conditions="hypertension"),
    dict(age=68, sex="F", sbp=230, dbp=120, pulse=98, temp_c=37.0, spo2=96,
         complaint="face drooping one side slurred speech", conditions=""),
    dict(age=30, sex="F", sbp=0, dbp=0, pulse=0, temp_c=0, spo2=0, complaint="halka bukhar",
         conditions="", medications=""),
    dict(age=25, sex="M", sbp=118, dbp=76, pulse=82, temp_c=99.5, spo2=98,
         complaint="mild fever cold cough since 2 days", conditions=""),
    dict(age=2, sex="M", sbp=95, dbp=60, pulse=130, temp_c=39.2, spo2=96,
         complaint="baby fever poor feeding", conditions=""),
    dict(age=40, sex="M", sbp=120, dbp=80, pulse=80, temp_c=36.8, spo2=99, complaint="",
         conditions="", medications=""),
]

if __name__ == "__main__":
    s = TriageSystem("models")
    for c in CASES:
        o = s.predict_one(**c)
        print(f"L{o['triage_level']}{' [ABSTAIN]' if o['abstain'] else ''} {o['department']:16s} "
              f"| esc={[e['id'] for e in o['escalation']]} flags={[f['id'] for f in o['red_flags']]} "
              f"| {c['complaint'][:50]} ({o['latency_ms']}ms)")
