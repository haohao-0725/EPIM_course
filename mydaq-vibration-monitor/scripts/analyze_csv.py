"""
scripts/analyze_csv.py

讀取 CSV 訊號檔案，輸出時域統計與 FFT 圖。

使用方式：
    python scripts/analyze_csv.py <csv_path>
    python scripts/analyze_csv.py data/raw/motor_on/signal.csv
    python scripts/analyze_csv.py data/raw/motor_on/signal.csv --fs 10000 --max-freq 500 --save

參數：
    csv_path         CSV 檔案路徑（必填）
    --fs             取樣率 Hz（若 CSV 無此資訊，預設 10000）
    --max-freq       FFT 圖最大顯示頻率（預設 500 Hz）
    --save           是否儲存圖片到 figures/ 資料夾
    --no-show        不開啟圖形視窗（通常與 --save 搭配使用）
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd

# 確保能找到 src 目錄
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mydaq_vibration.analysis import (
    compute_time_features,
    compute_fft,
    find_dominant_frequency,
)
from mydaq_vibration.plotting import plot_combined


def parse_args():
    parser = argparse.ArgumentParser(
        description="讀取 CSV 振動資料，輸出時域統計與 FFT 頻譜圖。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python scripts/analyze_csv.py data/raw/motor_on/signal.csv
  python scripts/analyze_csv.py data/example/mock_motor_on.csv --save --no-show
        """,
    )
    parser.add_argument("csv_path", help="CSV 檔案路徑")
    parser.add_argument("--fs",       type=int,   default=10000, help="取樣率 Hz（預設：10000）")
    parser.add_argument("--max-freq", type=float, default=500.0, help="FFT 顯示最大頻率（預設：500 Hz）")
    parser.add_argument("--save",     action="store_true",        help="儲存圖片到 figures/ 資料夾")
    parser.add_argument("--no-show",  action="store_true",        help="不開啟圖形視窗")
    return parser.parse_args()


def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    讀取 CSV 並回傳 time_array, voltage_array。
    支援兩種格式：
      - 有 time_s, voltage_v 欄位
      - 只有 voltage_v 欄位（time 自動計算）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到 CSV 檔案：{filepath}")

    df = pd.read_csv(filepath)

    if "voltage_v" not in df.columns:
        raise ValueError(
            f"CSV 缺少 'voltage_v' 欄位。\n"
            f"實際欄位：{list(df.columns)}"
        )

    voltage = df["voltage_v"].to_numpy(dtype=float)

    if "time_s" in df.columns:
        time = df["time_s"].to_numpy(dtype=float)
    else:
        # 根據 fs 自動計算時間軸
        time = np.arange(len(voltage))  # 暫時用索引，fs 之後再換算

    return time, voltage


def print_stats(features: dict, csv_path: str):
    """在終端機印出分析結果。"""
    filename = os.path.basename(csv_path)
    print()
    print("=" * 55)
    print(f"  分析結果：{filename}")
    print("=" * 55)
    print(f"  Mean          : {features['mean']:+.6f} V")
    print(f"  RMS           : {features['rms']:.6f} V")
    print(f"  Peak-to-Peak  : {features['peak_to_peak']:.6f} V")
    print(f"  Peak          : {features['peak']:.6f} V")
    print(f"  Crest Factor  : {features['crest_factor']:.3f}")
    print(f"  Dominant Freq : {features.get('dominant_freq', '?'):.2f} Hz")
    print("=" * 55)
    print()


def main():
    args = parse_args()

    print(f"[analyze_csv] 讀取：{args.csv_path}")

    try:
        time_arr, voltage_arr = load_csv(args.csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 若 time_s 是索引，換算成實際時間
    if time_arr[-1] < 1e-6 or time_arr[0] == 0 and np.all(np.diff(time_arr) == 1):
        time_arr = time_arr / args.fs

    n = len(voltage_arr)
    print(f"   取樣點數：{n}，時間長度：{n / args.fs:.2f} s，取樣率：{args.fs} Hz")

    # 計算特徵
    features = compute_time_features(voltage_arr)
    freqs, amps = compute_fft(voltage_arr, args.fs)
    dominant_freq = find_dominant_frequency(freqs, amps)
    features["dominant_freq"] = dominant_freq

    print_stats(features, args.csv_path)

    # 決定儲存路徑
    save_path = None
    if args.save:
        stem = os.path.splitext(os.path.basename(args.csv_path))[0]
        figures_dir = os.path.join(_PROJECT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, f"{stem}_analysis.png")

    # 繪圖
    show = not args.no_show
    plot_combined(
        time_arr, voltage_arr,
        freqs, amps,
        title=f"Analysis: {os.path.basename(args.csv_path)}",
        max_freq=args.max_freq,
        dominant_freq=dominant_freq,
        save_path=save_path,
        show=show,
    )

    if save_path:
        print(f"[OK] 圖片已儲存：{save_path}")


if __name__ == "__main__":
    main()
