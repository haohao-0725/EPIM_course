"""
generate_example_data.py

產生示範 CSV 資料，存放於 data/example/。
此資料由 mock_signal 產生，用於測試分析腳本。
執行：python generate_example_data.py（從專案根目錄）
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import csv
from mydaq_vibration.mock_signal import generate_mock_signal

os.makedirs("data/example", exist_ok=True)

for motor_on, label in [(False, "motor_off"), (True, "motor_on")]:
    t, v = generate_mock_signal(sampling_rate=10000, duration=2.0, motor_on=motor_on, seed=42)
    filepath = f"data/example/mock_{label}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "voltage_v"])
        for ti, vi in zip(t, v):
            writer.writerow([f"{ti:.6f}", f"{vi:.6f}"])
    print(f"Generated: {filepath} ({len(t)} samples)")
