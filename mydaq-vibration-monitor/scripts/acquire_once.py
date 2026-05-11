"""
scripts/acquire_once.py

從 NI myDAQ 擷取一次資料並儲存為 CSV。

使用方式：
    python scripts/acquire_once.py
    python scripts/acquire_once.py --channel Dev1/ai0 --fs 10000 --duration 5 --condition motor_off

參數：
    --channel    NI myDAQ 通道，例如 Dev1/ai0（預設：Dev1/ai0）
    --fs         取樣率 Hz（預設：10000）
    --duration   擷取時間 s（預設：5）
    --condition  量測條件標籤，例如 motor_on / motor_off（預設：motor_off）
    --output     輸出目錄（預設：data/raw/<condition>/）
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
    parser = argparse.ArgumentParser(
        description="從 NI myDAQ 擷取一次資料並儲存 CSV。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--channel",   default="Dev1/ai0", help="NI myDAQ 通道 (預設: Dev1/ai0)")
    parser.add_argument("--fs",        type=int,   default=10000, help="取樣率 Hz (預設: 10000)")
    parser.add_argument("--duration",  type=float, default=5.0,   help="擷取時間 s (預設: 5.0)")
    parser.add_argument("--condition", default="motor_off",        help="量測條件標籤 (預設: motor_off)")
    parser.add_argument("--output",    default=None,               help="輸出目錄 (預設: data/raw/<condition>/)")
    parser.add_argument("--vmin",      type=float, default=-10.0,  help="電壓最小值 V (預設: -10.0)")
    parser.add_argument("--vmax",      type=float, default=10.0,   help="電壓最大值 V (預設: 10.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  NI myDAQ 一次性資料擷取")
    print("=" * 55)
    print(f"  通道    : {args.channel}")
    print(f"  取樣率  : {args.fs} Hz")
    print(f"  時間    : {args.duration} s")
    print(f"  條件    : {args.condition}")
    print(f"  總取樣點: {int(args.fs * args.duration)}")
    print("=" * 55)
    print()

    # 確認 nidaqmx 可用
    try:
        import nidaqmx  # noqa: F401
    except ImportError:
        print("❌ 找不到 nidaqmx 套件，無法進行實機擷取。")
        print()
        print("此電腦可能沒有安裝 NI-DAQmx driver 或 nidaqmx Python 套件。")
        print("  安裝 Python 套件：pip install nidaqmx")
        print()
        print("若要在沒有硬體的電腦產生 mock 資料，請使用：")
        print("  python scripts/live_view.py  （含 --mode mock 選項）")
        sys.exit(1)

    # 執行擷取
    from mydaq_vibration.acquisition import acquire_signal, save_csv, generate_filename

    print("正在擷取資料，請稍候...")
    try:
        time_arr, voltage_arr = acquire_signal(
            channel=args.channel,
            sampling_rate=args.fs,
            duration=args.duration,
            voltage_min=args.vmin,
            voltage_max=args.vmax,
        )
    except RuntimeError as e:
        print(f"❌ 擷取失敗：{e}")
        sys.exit(1)

    print(f"✅ 擷取完成，共 {len(voltage_arr)} 筆資料。")
    print(f"   電壓範圍：{voltage_arr.min():.4f} ~ {voltage_arr.max():.4f} V")

    # 決定輸出路徑
    filename = generate_filename(
        condition=args.condition,
        channel=args.channel,
        sampling_rate=args.fs,
    )
    output_dir = args.output or os.path.join(_PROJECT_ROOT, "data", "raw", args.condition)
    filepath = os.path.join(output_dir, filename)

    save_csv(time_arr, voltage_arr, filepath)

    print()
    print(f"✅ 完成！檔案路徑：{filepath}")
    print()
    print("下一步可以執行分析：")
    print(f"  python scripts/analyze_csv.py {filepath}")


if __name__ == "__main__":
    main()
