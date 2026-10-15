# -*- coding: utf-8 -*-
"""
批次分析 raw data，輸出加速度 FFT、瀑布圖與摘要表。

此腳本預設分析 data/rawdata_20260527/faulty_3th 內的 CSV 檔案。
檔名主體會被視為 PMSM 馬達設定頻率，例如 10.csv 代表 10 Hz。
加速度換算沿用 GUI 即時監控頁面的公式：
acceleration (g) = (voltage - 2.5) / 0.1
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mydaq_vibration.analysis import compute_fft


@dataclass(frozen=True)
class SpectrumResult:
    """儲存單一 CSV、單一通道的分析結果。"""

    setting_frequency_hz: float
    csv_path: Path
    channel_label: str
    sample_rate_hz: float
    sample_count: int
    duration_s: float
    acceleration_rms_g: float
    acceleration_peak_to_peak_g: float
    dominant_frequency_hz: float
    dominant_amplitude_g: float
    freqs: np.ndarray
    amplitudes: np.ndarray


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="批次輸出加速度 FFT 與瀑布圖。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "data" / "rawdata_20260527" / "faulty_3th"),
        help="raw data CSV 目錄。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "results"),
        help="分析結果輸出目錄。",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="輸出子資料夾與 summary 標題名稱；未指定時使用 input-dir 的資料夾名稱。",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="若 CSV 沒有 time_s 欄位時使用的取樣率。",
    )
    parser.add_argument("--min-freq", type=float, default=0.0, help="FFT 顯示最小頻率。")
    parser.add_argument("--max-freq", type=float, default=100.0, help="FFT 顯示最大頻率。")
    parser.add_argument("--sensor-bias-v", type=float, default=2.5, help="震動規零 g 偏壓。")
    parser.add_argument("--sensor-sensitivity-v-per-g", type=float, default=0.1, help="震動規靈敏度。")
    return parser.parse_args()


def parse_setting_frequency(csv_path: Path) -> float:
    """從檔名解析馬達設定頻率。"""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", csv_path.stem)
    if not match:
        raise ValueError(f"無法從檔名解析設定頻率：{csv_path.name}")
    return float(match.group(0))


def safe_label(label: str) -> str:
    """將通道名稱轉成可作為檔名的字串。"""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", label).strip("_")


def load_csv(csv_path: Path, fallback_fs: float | None) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """讀取 CSV，回傳時間軸、取樣率與各通道電壓。"""
    df = pd.read_csv(csv_path)
    if "voltage_v" in df.columns:
        voltage_by_channel = {"voltage_v": df["voltage_v"].to_numpy(dtype=float)}
    else:
        voltage_columns = [column for column in df.columns if column.endswith("_voltage_v")]
        if not voltage_columns:
            raise ValueError(f"{csv_path.name} 缺少 voltage_v 或 *_voltage_v 欄位。")
        voltage_by_channel = {
            column.replace("_voltage_v", ""): df[column].to_numpy(dtype=float)
            for column in voltage_columns
        }

    first_voltage = next(iter(voltage_by_channel.values()))
    if "time_s" in df.columns:
        time_s = df["time_s"].to_numpy(dtype=float)
        diffs = np.diff(time_s)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            raise ValueError(f"{csv_path.name} 的 time_s 欄位無法推算取樣率。")
        sample_rate_hz = float(1.0 / np.median(diffs))
    elif fallback_fs is not None:
        sample_rate_hz = float(fallback_fs)
        time_s = np.arange(len(first_voltage), dtype=float) / sample_rate_hz
    else:
        raise ValueError(f"{csv_path.name} 沒有 time_s 欄位，請用 --fs 指定取樣率。")

    return time_s, sample_rate_hz, voltage_by_channel


def find_dominant_in_range(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    min_freq: float,
    max_freq: float,
) -> tuple[float, float]:
    """在指定頻率範圍內找出最大加速度振幅。"""
    search_min = max(min_freq, 1.0)
    mask = (freqs >= search_min) & (freqs <= max_freq)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_freqs = freqs[mask]
    local_amps = amplitudes[mask]
    idx = int(np.argmax(local_amps))
    return float(local_freqs[idx]), float(local_amps[idx])


def voltage_to_acceleration_g(voltage: np.ndarray, bias_v: float, sensitivity_v_per_g: float) -> np.ndarray:
    """將震動規電壓轉換為加速度 g。"""
    return (voltage - bias_v) / sensitivity_v_per_g


def plot_fft(
    result: SpectrumResult,
    output_path: Path,
    min_freq: float,
    max_freq: float,
) -> None:
    """輸出單一 FFT 圖。"""
    mask = (result.freqs >= min_freq) & (result.freqs <= max_freq)
    plot_freqs = result.freqs[mask]
    plot_amps = result.amplitudes[mask]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(plot_freqs, plot_amps, color="#2f6fbb", linewidth=0.9)
    ax.fill_between(plot_freqs, plot_amps, color="#2f6fbb", alpha=0.16)

    if min_freq <= result.setting_frequency_hz <= max_freq:
        ax.axvline(result.setting_frequency_hz, color="#6a994e", linestyle="--", linewidth=1.1, label="Setting frequency")

    third_order_hz = result.setting_frequency_hz * 3.0
    if min_freq <= third_order_hz <= max_freq:
        ax.axvline(third_order_hz, color="#bc4749", linestyle=":", linewidth=1.3, label="3rd order")

    if np.isfinite(result.dominant_frequency_hz):
        ax.axvline(result.dominant_frequency_hz, color="#f2a900", linestyle="-.", linewidth=1.1, label="Dominant frequency")

    ax.set_title(f"FFT - Setting {result.setting_frequency_hz:g} Hz - {result.channel_label}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration Amplitude (g)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(min_freq, max_freq)
    scale_mask = mask & (result.freqs >= max(min_freq, 1.0))
    if np.any(scale_mask):
        y_max = float(np.max(result.amplitudes[scale_mask]))
        if y_max > 0:
            ax.set_ylim(bottom=-0.05 * y_max, top=1.2 * y_max)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_waterfall(
    channel_label: str,
    results: list[SpectrumResult],
    output_path: Path,
    min_freq: float,
    max_freq: float,
) -> None:
    """輸出單一通道的 2D 瀑布圖。"""
    sorted_results = sorted(results, key=lambda item: item.setting_frequency_hz)
    target_freqs = sorted_results[0].freqs
    mask = (target_freqs >= min_freq) & (target_freqs <= max_freq)
    plot_freqs = target_freqs[mask]

    amplitude_rows = []
    setting_frequencies = []
    for result in sorted_results:
        if len(result.freqs) == len(target_freqs) and np.allclose(result.freqs, target_freqs):
            amplitudes = result.amplitudes
        else:
            amplitudes = np.interp(target_freqs, result.freqs, result.amplitudes)
        amplitude_rows.append(amplitudes[mask])
        setting_frequencies.append(result.setting_frequency_hz)

    amplitude_matrix = np.vstack(amplitude_rows)
    y_values = np.asarray(setting_frequencies, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 7))
    scale_columns = plot_freqs >= max(min_freq, 1.0)
    color_max = float(np.max(amplitude_matrix[:, scale_columns])) if np.any(scale_columns) else float(np.max(amplitude_matrix))
    image = ax.imshow(
        amplitude_matrix,
        aspect="auto",
        origin="lower",
        extent=[plot_freqs[0], plot_freqs[-1], y_values[0], y_values[-1]],
        cmap="viridis",
        interpolation="nearest",
        vmax=color_max if color_max > 0 else None,
    )

    ax.plot(y_values, y_values, color="white", linestyle="--", linewidth=1.0, alpha=0.8, label="Setting frequency")
    third_order = y_values * 3.0
    in_range = (third_order >= min_freq) & (third_order <= max_freq)
    if np.any(in_range):
        ax.plot(third_order[in_range], y_values[in_range], color="#ffcc00", linestyle=":", linewidth=1.5, label="3rd order")

    ax.set_title(f"Waterfall - {channel_label}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Motor Setting Frequency (Hz)")
    ax.set_xlim(min_freq, max_freq)
    ax.set_yticks(y_values)
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Acceleration Amplitude (g)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_waterfall_3d(
    channel_label: str,
    results: list[SpectrumResult],
    output_path: Path,
    min_freq: float,
    max_freq: float,
) -> None:
    """輸出單一通道的 3D 瀑布圖。"""
    sorted_results = sorted(results, key=lambda item: item.setting_frequency_hz)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for result in sorted_results:
        mask = (result.freqs >= min_freq) & (result.freqs <= max_freq)
        plot_freqs = result.freqs[mask]
        plot_amps = result.amplitudes[mask]
        y_values = np.full_like(plot_freqs, result.setting_frequency_hz, dtype=float)
        ax.plot(plot_freqs, y_values, plot_amps, linewidth=0.85)

    setting_values = np.asarray([result.setting_frequency_hz for result in sorted_results], dtype=float)
    ax.plot(setting_values, setting_values, np.zeros_like(setting_values), color="#6a994e", linestyle="--", linewidth=1.2, label="Setting frequency")

    third_order = setting_values * 3.0
    in_range = (third_order >= min_freq) & (third_order <= max_freq)
    if np.any(in_range):
        ax.plot(third_order[in_range], setting_values[in_range], np.zeros_like(setting_values[in_range]), color="#bc4749", linestyle=":", linewidth=1.5, label="3rd order")

    ax.set_title(f"3D Waterfall - {channel_label}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Motor Setting Frequency (Hz)")
    ax.set_zlabel("Acceleration Amplitude (g)")
    ax.set_xlim(min_freq, max_freq)
    ax.set_ylim(float(np.min(setting_values)), float(np.max(setting_values)))
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_csv(results: list[SpectrumResult], output_path: Path) -> None:
    """輸出分析摘要 CSV。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "setting_frequency_hz",
                "file",
                "channel",
                "sample_rate_hz",
                "sample_count",
                "duration_s",
                "acceleration_rms_g",
                "acceleration_peak_to_peak_g",
                "dominant_frequency_hz",
                "dominant_amplitude_g",
            ]
        )
        for result in sorted(results, key=lambda item: (item.channel_label, item.setting_frequency_hz)):
            writer.writerow(
                [
                    f"{result.setting_frequency_hz:.6g}",
                    result.csv_path.name,
                    result.channel_label,
                    f"{result.sample_rate_hz:.6f}",
                    result.sample_count,
                    f"{result.duration_s:.6f}",
                    f"{result.acceleration_rms_g:.8f}",
                    f"{result.acceleration_peak_to_peak_g:.8f}",
                    f"{result.dominant_frequency_hz:.8f}",
                    f"{result.dominant_amplitude_g:.8f}",
                ]
            )


def write_summary_markdown(
    results: list[SpectrumResult],
    output_path: Path,
    input_dir: Path,
    dataset_name: str,
    min_freq: float,
    max_freq: float,
) -> None:
    """輸出繁中分析說明 Markdown。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = sorted({result.channel_label for result in results})
    frequencies = sorted({result.setting_frequency_hz for result in results})
    lines = [
        f"# {dataset_name} 批次振動分析摘要",
        "",
        f"- 資料集：`{dataset_name}`",
        f"- 輸入資料夾：`{input_dir}`",
        f"- 分析檔案數：{len(frequencies)}",
        f"- 通道：{', '.join(channels)}",
        "- 加速度換算：`acceleration (g) = (voltage - 2.5) / 0.1`",
        "- FFT 方法：沿用專案 `compute_fft`，使用 Hanning window 與單邊振幅正規化。",
        f"- 頻率顯示範圍：`{min_freq:g}~{max_freq:g} Hz`。",
        "",
        "## 計算方式",
        "",
        "- 取樣率：若 CSV 有 `time_s`，使用相鄰時間差中位數推估，`fs = 1 / median(diff(time_s))`。",
        "- 加速度：`a_g[n] = (v[n] - V_bias) / S`，本次使用 `V_bias = 2.5 V`、`S = 0.1 V/g`。",
        "- Hanning window：`x_w[n] = a_g[n] * w[n]`。",
        "- 單邊 FFT 頻率：`f[k] = k * fs / N`。",
        "- 單邊 FFT 振幅：`A[k] = 2 * abs(rfft(x_w)[k]) / sum(w)`，其中 DC 分量 `A[0]` 再除以 2。",
        "- 加速度 RMS：`sqrt(mean(a_g^2))`。",
        "- 加速度 Peak-to-Peak：`max(a_g) - min(a_g)`。",
        "- Dominant frequency：在顯示範圍內排除低於 `1 Hz` 的 DC/低頻漂移後，取最大 FFT 振幅的頻率。",
        "",
        "## 摘要表",
        "",
        "| Setting (Hz) | Channel | Dominant (Hz) | Dominant Amp (g) | Acc RMS (g) | Acc P-P (g) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (item.setting_frequency_hz, item.channel_label)):
        lines.append(
            "| "
            f"{result.setting_frequency_hz:g} | "
            f"{result.channel_label} | "
            f"{result.dominant_frequency_hz:.2f} | "
            f"{result.dominant_amplitude_g:.6f} | "
            f"{result.acceleration_rms_g:.6f} | "
            f"{result.acceleration_peak_to_peak_g:.6f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """執行批次分析流程。"""
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_name = args.dataset_name or input_dir.name
    dataset_output_dir = output_dir / dataset_name
    if not input_dir.exists():
        print(f"找不到輸入資料夾：{input_dir}")
        return 1

    csv_paths = sorted(input_dir.glob("*.csv"), key=parse_setting_frequency)
    if not csv_paths:
        print(f"輸入資料夾沒有 CSV：{input_dir}")
        return 1

    all_results: list[SpectrumResult] = []
    results_by_channel: dict[str, list[SpectrumResult]] = {}

    for csv_path in csv_paths:
        setting_frequency_hz = parse_setting_frequency(csv_path)
        time_s, sample_rate_hz, voltage_by_channel = load_csv(csv_path, args.fs)
        duration_s = float(len(time_s) / sample_rate_hz)

        for channel_label, voltage in voltage_by_channel.items():
            acceleration_g = voltage_to_acceleration_g(
                voltage,
                bias_v=args.sensor_bias_v,
                sensitivity_v_per_g=args.sensor_sensitivity_v_per_g,
            )
            freqs, amplitudes = compute_fft(acceleration_g, int(round(sample_rate_hz)))
            dominant_frequency_hz, dominant_amplitude_g = find_dominant_in_range(
                freqs,
                amplitudes,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
            )
            result = SpectrumResult(
                setting_frequency_hz=setting_frequency_hz,
                csv_path=csv_path,
                channel_label=channel_label,
                sample_rate_hz=sample_rate_hz,
                sample_count=len(acceleration_g),
                duration_s=duration_s,
                acceleration_rms_g=float(np.sqrt(np.mean(acceleration_g**2))),
                acceleration_peak_to_peak_g=float(np.max(acceleration_g) - np.min(acceleration_g)),
                dominant_frequency_hz=dominant_frequency_hz,
                dominant_amplitude_g=dominant_amplitude_g,
                freqs=freqs,
                amplitudes=amplitudes,
            )
            all_results.append(result)
            results_by_channel.setdefault(channel_label, []).append(result)

            fft_name = f"setting_{setting_frequency_hz:g}Hz_{safe_label(channel_label)}_fft.png"
            plot_fft(
                result,
                dataset_output_dir / "fft" / fft_name,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
            )

    for channel_label, channel_results in results_by_channel.items():
        waterfall_name = f"{safe_label(channel_label)}_waterfall.png"
        plot_waterfall(
            channel_label,
            channel_results,
            dataset_output_dir / "waterfall" / waterfall_name,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
        )
        waterfall_3d_name = f"{safe_label(channel_label)}_waterfall_3d.png"
        plot_waterfall_3d(
            channel_label,
            channel_results,
            dataset_output_dir / "waterfall_3d" / waterfall_3d_name,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
        )

    write_summary_csv(all_results, dataset_output_dir / "summary.csv")
    write_summary_markdown(
        all_results,
        dataset_output_dir / "summary.md",
        input_dir,
        dataset_name=dataset_name,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )

    print(f"完成分析：{len(csv_paths)} 個 CSV，{len(all_results)} 組通道結果。")
    print(f"輸出位置：{dataset_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
