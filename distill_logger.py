import os
import json
from datetime import datetime

LOG_PATH = "logs/distillation_data.jsonl"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_for_distillation(user_query: str, raw_data: list, teacher_output: str):
    sample = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "data": raw_data,
        "teacher_output": teacher_output
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")
