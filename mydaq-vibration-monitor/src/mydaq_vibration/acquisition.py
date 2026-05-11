"""
acquisition.py

NI myDAQ 實機資料擷取模組。
此模組僅在有 NI-DAQmx driver 且連接 myDAQ 時才可使用。
沒有 NI myDAQ 的電腦請使用 mock_signal.py。
"""

import numpy as np
import csv
import os
from datetime import datetime


def acquire_signal(
    channel: str = "Dev1/ai0",
    sampling_rate: int = 10000,
    duration: float = 5.0,
    voltage_min: float = -10.0,
    voltage_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    從 NI myDAQ 擷取一段有限長度的電壓訊號。

    Parameters
    ----------
    channel : str
        NI myDAQ 通道名稱，例如 "Dev1/ai0"。
    sampling_rate : int
        取樣率 (Hz)，例如 10000。
    duration : float
        擷取持續時間 (秒)。
    voltage_min : float
        輸入電壓最小值 (V)。
    voltage_max : float
        輸入電壓最大值 (V)。

    Returns
    -------
    time_array : np.ndarray
        時間軸陣列 (秒)。
    voltage_array : np.ndarray
        電壓訊號陣列 (V)。

    Raises
    ------
    ImportError
        若未安裝 nidaqmx 套件。
    RuntimeError
        若 NI-DAQmx 無法找到指定裝置或通道。
    """
    try:
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration
    except ImportError:
        raise ImportError(
            "找不到 nidaqmx 套件。\n"
            "請在有安裝 NI-DAQmx driver 的電腦上執行：\n"
            "  pip install nidaqmx\n"
            "或在此電腦使用 mock mode：\n"
            "  from mydaq_vibration.mock_signal import generate_mock_signal"
        )

    n_samples = int(sampling_rate * duration)

    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                channel,
                min_val=voltage_min,
                max_val=voltage_max,
            )
            task.timing.cfg_samp_clk_timing(
                rate=sampling_rate,
                samps_per_chan=n_samples,
            )
            data = task.read(number_of_samples_per_channel=n_samples)
    except Exception as e:
        raise RuntimeError(
            f"NI myDAQ 擷取失敗：{e}\n"
            "請確認：\n"
            "  1. myDAQ 已連接且 NI MAX 中可以看到裝置。\n"
            "  2. device_channel 設定正確 (例如 Dev1/ai0)。\n"
            "  3. NI MAX Test Panel 已關閉，避免占用裝置。\n"
            "  4. 電壓範圍設定合理。"
        )

    voltage_array = np.array(data)
    time_array = np.linspace(0, duration, n_samples, endpoint=False)
    return time_array, voltage_array


def save_csv(
    time_array: np.ndarray,
    voltage_array: np.ndarray,
    filepath: str,
) -> str:
    """
    將時域訊號儲存為 CSV 檔案。

    Parameters
    ----------
    time_array : np.ndarray
        時間軸陣列 (秒)。
    voltage_array : np.ndarray
        電壓訊號陣列 (V)。
    filepath : str
        儲存路徑（含檔名），例如 "data/raw/motor_on/signal.csv"。

    Returns
    -------
    str
        實際儲存路徑。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "voltage_v"])
        for t, v in zip(time_array, voltage_array):
            writer.writerow([f"{t:.6f}", f"{v:.6f}"])

    print(f"[acquisition] 資料已儲存：{filepath}")
    return filepath


def generate_filename(
    condition: str = "motor_off",
    channel: str = "Dev1/ai0",
    sampling_rate: int = 10000,
    mounting: str = "unknown",
) -> str:
    """
    依照命名規則產生 CSV 檔名。

    格式：YYYYMMDD_HHMMSS_condition_mounting_channel_fsHz.csv
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 把 channel 中的 / 轉換成更安全的格式
    ch_safe = channel.replace("/", "_").replace("\\", "_")
    filename = f"{now}_{condition}_{mounting}_{ch_safe}_{sampling_rate}Hz.csv"
    return filename
