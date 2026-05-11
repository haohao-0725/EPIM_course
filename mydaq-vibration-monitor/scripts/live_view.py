"""
scripts/live_view.py

快速啟動即時波形顯示 GUI。

使用方式：
    python scripts/live_view.py
    python scripts/live_view.py --mode mock
    python scripts/live_view.py --mode nidaq
"""

import sys
import os
import argparse

# 確保能找到 src 目錄
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="啟動即時波形顯示 GUI。")
    parser.add_argument(
        "--mode",
        choices=["mock", "nidaq"],
        default="mock",
        help="模式選擇：mock（模擬，預設）或 nidaq（實機）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[live_view] 啟動 GUI，模式：{args.mode}")

    from mydaq_vibration.gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
