"""
scripts/check_device.py

列出目前連接的 NI 裝置。

使用方式：
    python scripts/check_device.py

若沒有安裝 nidaqmx 套件或沒有連接裝置，
會顯示清楚的錯誤訊息，不會 crash。
"""

import sys
import os

# 確保能找到 src 目錄（若從專案根目錄執行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def check_nidaqmx_available() -> bool:
    """檢查 nidaqmx 套件是否已安裝。"""
    try:
        import nidaqmx  # noqa: F401
        return True
    except ImportError:
        return False


def list_ni_devices():
    """列出所有 NI 裝置並顯示詳細資訊。"""
    try:
        from nidaqmx.system import System
        system = System.local()
        devices = list(system.devices)

        if not devices:
            print("⚠️  NI-DAQmx driver 已安裝，但未偵測到任何 NI 裝置。")
            print()
            print("請確認：")
            print("  1. NI myDAQ 已連接 USB。")
            print("  2. 打開 NI MAX，確認裝置出現在 'Devices and Interfaces' 下。")
            print("  3. 重新插拔 USB 後再試。")
            return

        print(f"✅ 找到 {len(devices)} 個 NI 裝置：")
        print()
        for i, device in enumerate(devices, 1):
            print(f"  [{i}] 裝置名稱  : {device.name}")
            try:
                print(f"       產品類型  : {device.product_type}")
            except Exception:
                pass
            try:
                serial = device.serial_num
                print(f"       序號      : {serial}")
            except Exception:
                pass
            try:
                ai_channels = [ch.name for ch in device.ai_physical_chans]
                if ai_channels:
                    print(f"       AI 通道   : {', '.join(ai_channels[:6])}", end="")
                    if len(ai_channels) > 6:
                        print(f" ... 共 {len(ai_channels)} 個")
                    else:
                        print()
            except Exception:
                pass
            print()

        print("提示：在 acquire_once.py 或 GUI 中，使用裝置名稱加上通道，例如：")
        first_name = devices[0].name
        print(f"  --channel {first_name}/ai0")

    except Exception as e:
        print(f"❌ 讀取 NI 裝置時發生錯誤：{e}")
        print()
        print("可能原因：")
        print("  - NI-DAQmx driver 尚未安裝（請安裝 NI-DAQmx）。")
        print("  - 驅動程式版本與 Python nidaqmx 套件不相容。")


def main():
    print("=" * 55)
    print("  NI myDAQ 裝置檢查工具")
    print("  EPIM Course — NI DAQ Vibration Monitor")
    print("=" * 55)
    print()

    # 步驟 1：確認 Python 套件
    if not check_nidaqmx_available():
        print("❌ 找不到 nidaqmx Python 套件。")
        print()
        print("這台電腦尚未安裝 nidaqmx，有兩種處理方式：")
        print()
        print("  選項 A：安裝 nidaqmx（需要先安裝 NI-DAQmx driver）")
        print("    pip install nidaqmx")
        print()
        print("  選項 B：使用 mock mode 開發（不需要硬體）")
        print("    python scripts/analyze_csv.py  # 分析 CSV")
        print("    python src/mydaq_vibration/gui.py  # 開啟 GUI（選 mock mode）")
        print()
        print("NI-DAQmx driver 下載：")
        print("  https://www.ni.com/zh-tw/support/downloads/drivers/download.ni-daq-mx.html")
        sys.exit(1)

    print("✅ nidaqmx Python 套件：已安裝")
    print()

    # 步驟 2：列出裝置
    list_ni_devices()


if __name__ == "__main__":
    main()
