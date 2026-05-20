# -*- coding: utf-8 -*-
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
from collections import OrderedDict
from collections.abc import Sequence


def normalize_channels(channels: str | Sequence[str]) -> list[str]:
    """
    將通道設定整理成乾淨的 channel list。

    channels 可為單一字串、逗號分隔字串，或字串序列。
    """
    if isinstance(channels, str):
        raw_channels = channels.replace("\n", ",").split(",")
    else:
        raw_channels = list(channels)

    normalized = [str(ch).strip() for ch in raw_channels if str(ch).strip()]
    if not normalized:
        raise ValueError("至少需要指定一個 NI myDAQ 通道，例如 Dev1/ai0。")
    return normalized


def sanitize_channel_label(channel: str) -> str:
    """將 NI channel 名稱轉成適合用於 CSV 欄位與圖例的標籤。"""
    return (
        channel.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "")
        .replace(",", "_")
    )


def _group_channels_by_device(channels: Sequence[str]) -> OrderedDict[str, list[str]]:
    """
    依裝置名稱分組，例如 Dev1/ai0 與 Dev1/ai1 會放入同一組。

    同一裝置內的多通道會放在同一個 NI-DAQmx task，以取得較好的同步性。
    跨 USB 裝置時則會分成多個 task，屬於軟體近似同步。
    """
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for channel in channels:
        device_name = channel.split("/", 1)[0]
        groups.setdefault(device_name, []).append(channel)
    return groups


def _task_data_to_channel_dict(data, channels: Sequence[str]) -> OrderedDict[str, np.ndarray]:
    """將 DAQmx read 結果轉成 {channel_label: voltage_array}。"""
    labels = [sanitize_channel_label(ch) for ch in channels]
    array = np.asarray(data, dtype=float)

    if len(labels) == 1:
        return OrderedDict([(labels[0], np.asarray(array, dtype=float))])

    if array.ndim != 2 or array.shape[0] != len(labels):
        raise RuntimeError(
            "NI-DAQmx 回傳的多通道資料形狀不符合預期，"
            f"channels={channels}, shape={array.shape}"
        )

    return OrderedDict(
        (label, np.asarray(array[idx], dtype=float))
        for idx, label in enumerate(labels)
    )


def _import_nidaqmx():
    """集中處理 nidaqmx 匯入錯誤訊息。"""
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TaskMode
    except ImportError:
        raise ImportError(
            "找不到 nidaqmx 套件。\n"
            "請在有安裝 NI-DAQmx driver 的電腦上執行：\n"
            "  pip install nidaqmx\n"
            "或在此電腦使用 mock mode。"
        )
    return nidaqmx, AcquisitionType, TaskMode


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
    time_array, voltage_by_channel = acquire_multi_signal(
        channels=[channel],
        sampling_rate=sampling_rate,
        duration=duration,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    voltage_array = next(iter(voltage_by_channel.values()))
    return time_array, voltage_array


def acquire_multi_signal(
    channels: str | Sequence[str],
    sampling_rate: int = 10000,
    duration: float = 5.0,
    voltage_min: float = -10.0,
    voltage_max: float = 10.0,
) -> tuple[np.ndarray, OrderedDict[str, np.ndarray]]:
    """
    從一個或多個 NI myDAQ channel 擷取有限長度的電壓訊號。

    同一台裝置的多個 channel 會放在同一個 DAQmx task；跨裝置時會建立
    多個 task 並在同一個函式內盡量連續啟動，屬於軟體近似同步。
    """
    nidaqmx, AcquisitionType, TaskMode = _import_nidaqmx()

    n_samples = int(sampling_rate * duration)
    channel_list = normalize_channels(channels)
    groups = _group_channels_by_device(channel_list)
    voltage_by_channel: OrderedDict[str, np.ndarray] = OrderedDict()
    tasks = []

    try:
        for task_channels in groups.values():
            task = nidaqmx.Task()
            tasks.append((task, task_channels))
            for task_channel in task_channels:
                task.ai_channels.add_ai_voltage_chan(
                    task_channel,
                    min_val=voltage_min,
                    max_val=voltage_max,
                )
            task.timing.cfg_samp_clk_timing(
                rate=sampling_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples,
            )
            task.control(TaskMode.TASK_COMMIT)

        for task, _ in tasks:
            task.start()

        for task, task_channels in tasks:
            data = task.read(
                number_of_samples_per_channel=n_samples,
                timeout=duration + 10.0,
            )
            voltage_by_channel.update(_task_data_to_channel_dict(data, task_channels))
    except Exception as e:
        raise RuntimeError(
            f"NI myDAQ 擷取失敗：{e}\n"
            "請確認：\n"
            "  1. myDAQ 已連接且 NI MAX 中可以看到裝置。\n"
            "  2. channel 設定正確 (例如 Dev1/ai0 或 Dev1/ai0, Dev1/ai1)。\n"
            "  3. NI MAX Test Panel 已關閉，避免占用裝置。\n"
            "  4. 電壓範圍設定合理。"
        )
    finally:
        for task, _ in tasks:
            try:
                task.close()
            except Exception:
                pass

    time_array = np.linspace(0, duration, n_samples, endpoint=False)
    return time_array, voltage_by_channel


class MultiChannelSignalReader:
    """
    持續讀取多個 NI myDAQ channel 的背景 reader。

    GUI 即時擷取時使用此類別，避免每個 chunk 都重新建立 DAQmx task。
    """

    def __init__(
        self,
        channels: str | Sequence[str],
        sampling_rate: int = 10000,
        chunk_duration: float = 1.0,
        voltage_min: float = -10.0,
        voltage_max: float = 10.0,
    ):
        self.channels = normalize_channels(channels)
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.chunk_samples = int(sampling_rate * chunk_duration)
        self._tasks = []
        self._nidaqmx = None
        self._acquisition_type = None
        self._task_mode = None

    def __enter__(self):
        nidaqmx, AcquisitionType, TaskMode = _import_nidaqmx()
        self._nidaqmx = nidaqmx
        self._acquisition_type = AcquisitionType
        self._task_mode = TaskMode

        for task_channels in _group_channels_by_device(self.channels).values():
            task = nidaqmx.Task()
            self._tasks.append((task, task_channels))
            for task_channel in task_channels:
                task.ai_channels.add_ai_voltage_chan(
                    task_channel,
                    min_val=self.voltage_min,
                    max_val=self.voltage_max,
                )
            task.timing.cfg_samp_clk_timing(
                rate=self.sampling_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(self.chunk_samples * 10, self.chunk_samples),
            )
            task.control(TaskMode.TASK_COMMIT)

        for task, _ in self._tasks:
            task.start()

        return self

    def read_chunk(self) -> OrderedDict[str, np.ndarray]:
        """讀取下一個 chunk，回傳 {channel_label: voltage_array}。"""
        voltage_by_channel: OrderedDict[str, np.ndarray] = OrderedDict()
        for task, task_channels in self._tasks:
            data = task.read(
                number_of_samples_per_channel=self.chunk_samples,
                timeout=self.chunk_duration + 5.0,
            )
            voltage_by_channel.update(_task_data_to_channel_dict(data, task_channels))
        return voltage_by_channel

    def close(self):
        """停止並關閉所有 DAQmx task。"""
        for task, _ in self._tasks:
            try:
                task.stop()
            except Exception:
                pass
            try:
                task.close()
            except Exception:
                pass
        self._tasks.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


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


def save_multi_csv(
    time_array: np.ndarray,
    voltage_by_channel: dict[str, np.ndarray],
    filepath: str,
) -> str:
    """
    將多通道時域訊號儲存為 CSV 檔案。

    CSV 欄位格式：
        time_s,<channel_label>_voltage_v,...
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    channel_labels = list(voltage_by_channel.keys())
    if not channel_labels:
        raise ValueError("沒有可儲存的通道資料。")

    min_len = min(len(time_array), *(len(voltage_by_channel[label]) for label in channel_labels))
    trimmed_time = time_array[:min_len]
    trimmed_voltage = {
        label: np.asarray(voltage_by_channel[label])[:min_len]
        for label in channel_labels
    }

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", *[f"{label}_voltage_v" for label in channel_labels]])
        for idx, t in enumerate(trimmed_time):
            row = [f"{t:.6f}"]
            row.extend(f"{trimmed_voltage[label][idx]:.6f}" for label in channel_labels)
            writer.writerow(row)

    print(f"[acquisition] 多通道資料已儲存：{filepath}")
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
