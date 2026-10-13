# Sample intake (1k rows, same schema as the 30k set). Full set:
#   python scripts/make_data.py --n 30000 --out data/intake_30k.csv
# Columns: age, sex, sbp, dbp, pulse, temp_c, spo2, complaint, conditions,
#          medications, arrival_time | labels: triage_level, department, is_emergency
