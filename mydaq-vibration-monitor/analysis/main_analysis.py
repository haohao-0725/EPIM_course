# -*- coding: utf-8 -*-
"""
rawdata_main 重新分析主程式。

此腳本會讀取 data/rawdata_main 內的振動、風速與功率 CSV，輸出
FFT、瀑布圖、BPF 倍頻標示、功率回歸圖、摘要表與 Markdown 報告。
"""

from __future__ import annotations

import argparse
import csv
import math
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
class DatasetConfig:
    """儲存單一振動資料集的設定。"""

    dataset: str
    condition: str
    input_dir: Path
    has_bpf: bool


@dataclass(frozen=True)
class SpectrumResult:
    """儲存單一 CSV、單一通道的頻譜分析結果。"""

    dataset: str
    condition: str
    setting_frequency_hz: float
    channel: str
    csv_path: Path
    sample_rate_hz: float
    sample_count: int
    duration_s: float
    acceleration_rms_g: float
    acceleration_peak_to_peak_g: float
    dominant_frequency_hz: float
    dominant_amplitude_g: float
    order_1x_freq_hz: float
    order_1x_amp_g: float
    order_2x_freq_hz: float
    order_2x_amp_g: float
    order_3x_freq_hz: float
    order_3x_amp_g: float
    bpf_hz: float
    freqs: np.ndarray
    amplitudes: np.ndarray


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="Analyze rawdata_main vibration, BPF, wind speed, and power data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "data" / "rawdata_main"), help="rawdata_main 資料夾。")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "results" / "main_analysis"), help="分析輸出資料夾。")
    parser.add_argument("--min-freq", type=float, default=0.0, help="FFT 與瀑布圖顯示最小頻率。")
    parser.add_argument("--max-freq", type=float, default=100.0, help="FFT 與瀑布圖顯示最大頻率。")
    parser.add_argument("--fft-y-max", type=float, default=None, help="單張 FFT 圖的加速度上限；未指定則自動縮放。")
    parser.add_argument("--waterfall-min-freq", type=float, default=1.0, help="瀑布圖顯示最小頻率；預設排除 DC 與低頻漂移。")
    parser.add_argument("--waterfall-vmax", type=float, default=0.07, help="2D 瀑布圖色階上限；設定成 0.07 便於放大觀察主要頻譜差異。")
    parser.add_argument("--waterfall-z-max", type=float, default=0.07, help="3D 瀑布圖 Z 軸加速度上限；設定成 0.07 便於放大觀察主要頻譜差異。")
    parser.add_argument("--surface-max-frequency-points", type=int, default=1400, help="3D 連續曲面最多使用的頻率取樣點數。")
    parser.add_argument("--sensor-bias-v", type=float, default=2.5, help="震動規 0 g 偏壓。")
    parser.add_argument("--sensor-sensitivity-v-per-g", type=float, default=0.1, help="震動規靈敏度。")
    parser.add_argument("--bpf-diameter-m", type=float, default=0.05, help="BPF 暫定傳動輪內徑 D。")
    parser.add_argument("--belt-length-m", type=float, default=0.50, help="BPF 暫定皮帶總長度 L。")
    parser.add_argument("--bpf-harmonics", type=int, default=6, help="BPF 圖上標示的倍頻數。")
    parser.add_argument("--order-band-half-width-hz", type=float, default=0.5, help="1X/2X/3X 與 BPF 倍頻取峰值的半頻寬。")
    return parser.parse_args()


def parse_setting_frequency(csv_path: Path) -> float:
    """從檔名解析馬達設定頻率。"""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", csv_path.stem)
    if not match:
        raise ValueError(f"無法從檔名解析設定頻率：{csv_path.name}")
    return float(match.group(0))


def safe_label(label: str) -> str:
    """將任意標籤轉為安全檔名片段。"""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", label).strip("_")


def voltage_to_acceleration_g(voltage: np.ndarray, bias_v: float, sensitivity_v_per_g: float) -> np.ndarray:
    """將震動規輸出電壓轉換為加速度 g。"""
    return (voltage - bias_v) / sensitivity_v_per_g


def load_vibration_csv(csv_path: Path) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """讀取振動 CSV，回傳時間軸、取樣率與各通道電壓。"""
    df = pd.read_csv(csv_path)
    if "time_s" not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少 time_s 欄位，無法推算取樣率。")

    voltage_columns = [column for column in df.columns if column.endswith("_voltage_v")]
    if "voltage_v" in df.columns:
        voltage_by_channel = {"voltage_v": df["voltage_v"].to_numpy(dtype=float)}
    elif voltage_columns:
        voltage_by_channel = {
            column.replace("_voltage_v", ""): df[column].to_numpy(dtype=float)
            for column in voltage_columns
        }
    else:
        raise ValueError(f"{csv_path.name} 缺少 voltage_v 或 *_voltage_v 欄位。")

    time_s = df["time_s"].to_numpy(dtype=float)
    diffs = np.diff(time_s)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError(f"{csv_path.name} 的 time_s 無法推算取樣率。")
    sample_rate_hz = float(1.0 / np.median(diffs))
    return time_s, sample_rate_hz, voltage_by_channel


def dominant_in_range(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    min_freq: float,
    max_freq: float,
) -> tuple[float, float]:
    """排除低頻漂移後，在指定頻率範圍內找最大振幅。"""
    mask = (freqs >= max(min_freq, 1.0)) & (freqs <= max_freq)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_freqs = freqs[mask]
    local_amps = amplitudes[mask]
    idx = int(np.argmax(local_amps))
    return float(local_freqs[idx]), float(local_amps[idx])


def band_peak(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    target_hz: float,
    half_width_hz: float,
    min_freq: float,
    max_freq: float,
) -> tuple[float, float]:
    """取指定目標頻率附近頻帶中的最大振幅。"""
    if not np.isfinite(target_hz):
        return float("nan"), float("nan")
    lower = max(min_freq, target_hz - half_width_hz)
    upper = min(max_freq, target_hz + half_width_hz)
    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_freqs = freqs[mask]
    local_amps = amplitudes[mask]
    idx = int(np.argmax(local_amps))
    return float(local_freqs[idx]), float(local_amps[idx])


def compute_bpf_hz(setting_frequency_hz: float, diameter_m: float, belt_length_m: float) -> float:
    """使用暫定幾何參數計算 Belt Pass Frequency。"""
    rpm = setting_frequency_hz * 60.0
    return (rpm * math.pi * diameter_m / belt_length_m) / 60.0


def bpf_harmonic_targets(base_bpf_hz: float, harmonics: int, max_freq: float) -> list[tuple[int, float]]:
    """產生在顯示範圍內的 BPF 倍頻清單。"""
    targets = []
    for harmonic in range(1, harmonics + 1):
        target_hz = base_bpf_hz * harmonic
        if target_hz <= max_freq:
            targets.append((harmonic, target_hz))
    return targets


def plot_fft(
    result: SpectrumResult,
    output_path: Path,
    min_freq: float,
    max_freq: float,
    fft_y_max: float | None,
    bpf_harmonics: int,
) -> None:
    """輸出單一 FFT 圖，並標示 order 與 BPF 倍頻。"""
    mask = (result.freqs >= min_freq) & (result.freqs <= max_freq)
    plot_freqs = result.freqs[mask]
    plot_amps = result.amplitudes[mask]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(plot_freqs, plot_amps, color="#1f77b4", linewidth=0.9)
    ax.fill_between(plot_freqs, plot_amps, color="#1f77b4", alpha=0.14)

    order_styles = [
        ("1X", result.setting_frequency_hz, "#2ca02c", "--"),
        ("2X", result.setting_frequency_hz * 2.0, "#ff7f0e", ":"),
        ("3X", result.setting_frequency_hz * 3.0, "#d62728", "-."),
    ]
    for label, target_hz, color, linestyle in order_styles:
        if min_freq <= target_hz <= max_freq:
            ax.axvline(target_hz, color=color, linestyle=linestyle, linewidth=1.05, label=label)

    if np.isfinite(result.bpf_hz):
        for harmonic, target_hz in bpf_harmonic_targets(result.bpf_hz, bpf_harmonics, max_freq):
            if target_hz >= min_freq:
                ax.axvline(target_hz, color="#7f3c8d", linestyle=(0, (2, 2)), linewidth=0.9, alpha=0.78)
                ax.text(
                    target_hz,
                    0.98,
                    f"BPF {harmonic}x",
                    rotation=90,
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="#7f3c8d",
                )

    if np.isfinite(result.dominant_frequency_hz):
        ax.axvline(result.dominant_frequency_hz, color="#8c564b", linestyle="-", linewidth=0.9, label="Dominant")

    ax.set_title(f"FFT - {result.condition} - {result.setting_frequency_hz:g} Hz - {result.channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration Amplitude (g)")
    ax.set_xlim(min_freq, max_freq)
    if fft_y_max is not None:
        ax.set_ylim(0.0, fft_y_max)
    elif np.any(mask):
        y_max = float(np.max(plot_amps[plot_freqs >= max(min_freq, 1.0)])) if np.any(plot_freqs >= max(min_freq, 1.0)) else float(np.max(plot_amps))
        if y_max > 0:
            ax.set_ylim(0.0, y_max * 1.18)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_waterfall_2d(
    dataset: str,
    condition: str,
    channel: str,
    results: list[SpectrumResult],
    output_path: Path,
    min_freq: float,
    max_freq: float,
    waterfall_vmax: float | None,
) -> None:
    """輸出單一資料集與通道的 2D 瀑布圖。"""
    sorted_results = sorted(results, key=lambda item: item.setting_frequency_hz)
    target_freqs = sorted_results[0].freqs
    mask = (target_freqs >= min_freq) & (target_freqs <= max_freq)
    plot_freqs = target_freqs[mask]
    setting_values = np.asarray([item.setting_frequency_hz for item in sorted_results], dtype=float)

    rows = []
    for result in sorted_results:
        if len(result.freqs) == len(target_freqs) and np.allclose(result.freqs, target_freqs):
            amplitudes = result.amplitudes
        else:
            amplitudes = np.interp(target_freqs, result.freqs, result.amplitudes)
        rows.append(amplitudes[mask])
    amplitude_matrix = np.vstack(rows)

    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = waterfall_vmax
    if vmax is None:
        scale_mask = plot_freqs >= max(min_freq, 1.0)
        vmax = float(np.max(amplitude_matrix[:, scale_mask])) if np.any(scale_mask) else float(np.max(amplitude_matrix))
    image = ax.imshow(
        amplitude_matrix,
        aspect="auto",
        origin="lower",
        extent=[plot_freqs[0], plot_freqs[-1], setting_values[0], setting_values[-1]],
        cmap="viridis",
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax if vmax and vmax > 0 else None,
    )
    ax.plot(setting_values, setting_values, color="white", linestyle="--", linewidth=1.0, alpha=0.85, label="1X")
    ax.plot(setting_values * 2.0, setting_values, color="#ffcc00", linestyle=":", linewidth=1.2, alpha=0.9, label="2X")
    ax.plot(setting_values * 3.0, setting_values, color="#ff9f1c", linestyle="-.", linewidth=1.2, alpha=0.9, label="3X")
    bpf_values = np.asarray([item.bpf_hz for item in sorted_results], dtype=float)
    if np.any(np.isfinite(bpf_values)):
        ax.plot(bpf_values, setting_values, color="#e0aaff", linestyle=(0, (2, 2)), linewidth=1.1, label="BPF")

    ax.set_title(f"Waterfall - {condition} - {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Motor Setting Frequency (Hz)")
    ax.set_xlim(min_freq, max_freq)
    ax.set_yticks(setting_values)
    ax.legend(loc="upper right")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Acceleration Amplitude (g)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_waterfall_3d(
    condition: str,
    channel: str,
    results: list[SpectrumResult],
    output_path: Path,
    min_freq: float,
    max_freq: float,
    waterfall_z_max: float | None,
    max_frequency_points: int,
) -> None:
    """輸出單一資料集與通道的 3D 連續曲面瀑布圖。"""
    sorted_results = sorted(results, key=lambda item: item.setting_frequency_hz)
    target_freqs = sorted_results[0].freqs
    mask = (target_freqs >= min_freq) & (target_freqs <= max_freq)
    plot_freqs = target_freqs[mask]
    if len(plot_freqs) > max_frequency_points:
        indices = np.unique(np.linspace(0, len(plot_freqs) - 1, max_frequency_points, dtype=int))
        plot_freqs = plot_freqs[indices]
    else:
        indices = None

    rows = []
    settings = []
    for result in sorted_results:
        if len(result.freqs) == len(target_freqs) and np.allclose(result.freqs, target_freqs):
            amplitudes = result.amplitudes[mask]
        else:
            amplitudes = np.interp(target_freqs, result.freqs, result.amplitudes)[mask]
        if indices is not None:
            amplitudes = amplitudes[indices]
        if waterfall_z_max is not None:
            amplitudes = np.clip(amplitudes, 0.0, waterfall_z_max)
        rows.append(amplitudes)
        settings.append(result.setting_frequency_hz)

    amplitude_matrix = np.vstack(rows)
    settings = np.asarray(settings, dtype=float)
    freq_grid, setting_grid = np.meshgrid(plot_freqs, settings)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        freq_grid,
        setting_grid,
        amplitude_matrix,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.92,
        vmin=0.0,
        vmax=waterfall_z_max,
        rcount=len(settings),
        ccount=min(len(plot_freqs), max_frequency_points),
    )
    colorbar = fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08)
    colorbar.set_label("Acceleration Amplitude (g)")

    ax.plot(settings, settings, np.zeros_like(settings), color="#2ca02c", linestyle="--", linewidth=1.2, label="1X")
    ax.plot(settings * 2.0, settings, np.zeros_like(settings), color="#ff7f0e", linestyle=":", linewidth=1.2, label="2X")
    ax.plot(settings * 3.0, settings, np.zeros_like(settings), color="#d62728", linestyle="-.", linewidth=1.2, label="3X")
    bpf_values = np.asarray([item.bpf_hz for item in sorted_results], dtype=float)
    if np.any(np.isfinite(bpf_values)):
        ax.plot(bpf_values, settings, np.zeros_like(settings), color="#7f3c8d", linestyle=(0, (2, 2)), linewidth=1.1, label="BPF")

    ax.set_title(f"Continuous Waterfall Surface - {condition} - {channel}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Motor Setting Frequency (Hz)")
    ax.set_zlabel("Acceleration Amplitude (g)")
    ax.set_xlim(min_freq, max_freq)
    ax.set_ylim(float(np.min(settings)), float(np.max(settings)))
    if waterfall_z_max is not None:
        ax.set_zlim(0.0, waterfall_z_max)
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_vibration(args: argparse.Namespace, datasets: list[DatasetConfig], output_dir: Path) -> list[SpectrumResult]:
    """分析全部振動 CSV 並輸出 FFT 與瀑布圖。"""
    all_results: list[SpectrumResult] = []
    grouped: dict[tuple[str, str], list[SpectrumResult]] = {}

    for dataset_config in datasets:
        csv_paths = sorted(dataset_config.input_dir.glob("*.csv"), key=parse_setting_frequency)
        if not csv_paths:
            raise FileNotFoundError(f"找不到振動 CSV：{dataset_config.input_dir}")

        for csv_path in csv_paths:
            setting_hz = parse_setting_frequency(csv_path)
            time_s, sample_rate_hz, voltage_by_channel = load_vibration_csv(csv_path)
            duration_s = float(len(time_s) / sample_rate_hz)
            bpf_hz = compute_bpf_hz(setting_hz, args.bpf_diameter_m, args.belt_length_m) if dataset_config.has_bpf else float("nan")

            for channel, voltage in voltage_by_channel.items():
                acceleration_g = voltage_to_acceleration_g(voltage, args.sensor_bias_v, args.sensor_sensitivity_v_per_g)
                freqs, amplitudes = compute_fft(acceleration_g, int(round(sample_rate_hz)))
                dominant_freq, dominant_amp = dominant_in_range(freqs, amplitudes, args.min_freq, args.max_freq)
                order_values = {}
                for order in [1, 2, 3]:
                    order_freq, order_amp = band_peak(
                        freqs,
                        amplitudes,
                        setting_hz * order,
                        args.order_band_half_width_hz,
                        args.min_freq,
                        args.max_freq,
                    )
                    order_values[f"{order}x_freq"] = order_freq
                    order_values[f"{order}x_amp"] = order_amp

                result = SpectrumResult(
                    dataset=dataset_config.dataset,
                    condition=dataset_config.condition,
                    setting_frequency_hz=setting_hz,
                    channel=channel,
                    csv_path=csv_path,
                    sample_rate_hz=sample_rate_hz,
                    sample_count=len(acceleration_g),
                    duration_s=duration_s,
                    acceleration_rms_g=float(np.sqrt(np.mean(acceleration_g**2))),
                    acceleration_peak_to_peak_g=float(np.max(acceleration_g) - np.min(acceleration_g)),
                    dominant_frequency_hz=dominant_freq,
                    dominant_amplitude_g=dominant_amp,
                    order_1x_freq_hz=order_values["1x_freq"],
                    order_1x_amp_g=order_values["1x_amp"],
                    order_2x_freq_hz=order_values["2x_freq"],
                    order_2x_amp_g=order_values["2x_amp"],
                    order_3x_freq_hz=order_values["3x_freq"],
                    order_3x_amp_g=order_values["3x_amp"],
                    bpf_hz=bpf_hz,
                    freqs=freqs,
                    amplitudes=amplitudes,
                )
                all_results.append(result)
                grouped.setdefault((dataset_config.dataset, channel), []).append(result)

                fft_name = f"{dataset_config.dataset}_{setting_hz:g}Hz_{safe_label(channel)}_fft.png"
                plot_fft(
                    result,
                    output_dir / "figures" / "fft" / fft_name,
                    args.min_freq,
                    args.max_freq,
                    args.fft_y_max,
                    args.bpf_harmonics,
                )

    dataset_lookup = {item.dataset: item.condition for item in datasets}
    for (dataset, channel), results in grouped.items():
        safe_channel = safe_label(channel)
        waterfall_min_freq = max(args.min_freq, args.waterfall_min_freq)
        plot_waterfall_2d(
            dataset,
            dataset_lookup[dataset],
            channel,
            results,
            output_dir / "figures" / "waterfall" / f"{dataset}_{safe_channel}_waterfall.png",
            waterfall_min_freq,
            args.max_freq,
            args.waterfall_vmax,
        )
        plot_waterfall_3d(
            dataset_lookup[dataset],
            channel,
            results,
            output_dir / "figures" / "waterfall" / f"{dataset}_{safe_channel}_waterfall_3d.png",
            waterfall_min_freq,
            args.max_freq,
            args.waterfall_z_max,
            args.surface_max_frequency_points,
        )

    return all_results


def write_vibration_tables(results: list[SpectrumResult], output_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """輸出振動 FFT 指標與 BPF 倍頻指標。"""
    rows = []
    bpf_rows = []
    for result in sorted(results, key=lambda item: (item.dataset, item.setting_frequency_hz, item.channel)):
        rows.append(
            {
                "dataset": result.dataset,
                "condition": result.condition,
                "setting_frequency_hz": result.setting_frequency_hz,
                "file": result.csv_path.name,
                "channel": result.channel,
                "sample_rate_hz": result.sample_rate_hz,
                "sample_count": result.sample_count,
                "duration_s": result.duration_s,
                "acceleration_rms_g": result.acceleration_rms_g,
                "acceleration_peak_to_peak_g": result.acceleration_peak_to_peak_g,
                "dominant_frequency_hz": result.dominant_frequency_hz,
                "dominant_amplitude_g": result.dominant_amplitude_g,
                "order_1x_freq_hz": result.order_1x_freq_hz,
                "order_1x_amp_g": result.order_1x_amp_g,
                "order_2x_freq_hz": result.order_2x_freq_hz,
                "order_2x_amp_g": result.order_2x_amp_g,
                "order_3x_freq_hz": result.order_3x_freq_hz,
                "order_3x_amp_g": result.order_3x_amp_g,
                "bpf_hz": result.bpf_hz,
            }
        )
        if np.isfinite(result.bpf_hz):
            for harmonic, target_hz in bpf_harmonic_targets(result.bpf_hz, args.bpf_harmonics, args.max_freq):
                peak_freq, peak_amp = band_peak(
                    result.freqs,
                    result.amplitudes,
                    target_hz,
                    args.order_band_half_width_hz,
                    args.min_freq,
                    args.max_freq,
                )
                bpf_rows.append(
                    {
                        "dataset": result.dataset,
                        "condition": result.condition,
                        "setting_frequency_hz": result.setting_frequency_hz,
                        "channel": result.channel,
                        "bpf_base_hz": result.bpf_hz,
                        "harmonic": harmonic,
                        "target_frequency_hz": target_hz,
                        "peak_frequency_hz": peak_freq,
                        "peak_amplitude_g": peak_amp,
                    }
                )

    vibration_df = pd.DataFrame(rows)
    bpf_df = pd.DataFrame(bpf_rows)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    vibration_df.to_csv(tables_dir / "vibration_fft_metrics.csv", index=False, encoding="utf-8")
    bpf_df.to_csv(tables_dir / "bpf_metrics.csv", index=False, encoding="utf-8")
    return vibration_df, bpf_df


def build_order_comparison(vibration_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """建立健康與故障馬達的 1X/2X/3X 倍頻比較表與圖。"""
    order_rows = []
    for row in vibration_df.itertuples(index=False):
        for order in [1, 2, 3]:
            order_rows.append(
                {
                    "dataset": row.dataset,
                    "condition": row.condition,
                    "setting_frequency_hz": float(row.setting_frequency_hz),
                    "channel": row.channel,
                    "order": f"{order}X",
                    "order_number": order,
                    "peak_frequency_hz": float(getattr(row, f"order_{order}x_freq_hz")),
                    "peak_amplitude_g": float(getattr(row, f"order_{order}x_amp_g")),
                }
            )
    order_long_df = pd.DataFrame(order_rows)

    detail_df = pivot_order_comparison(order_long_df, ["setting_frequency_hz", "channel", "order", "order_number"])
    summary_long_df = (
        order_long_df.groupby(["dataset", "condition", "setting_frequency_hz", "order", "order_number"], as_index=False)
        .agg(
            peak_amplitude_g=("peak_amplitude_g", "mean"),
            peak_frequency_hz=("peak_frequency_hz", "mean"),
        )
    )
    summary_df = pivot_order_comparison(summary_long_df, ["setting_frequency_hz", "order", "order_number"])

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(tables_dir / "order_health_fault_comparison_by_channel.csv", index=False, encoding="utf-8")
    summary_df.to_csv(tables_dir / "order_health_fault_summary.csv", index=False, encoding="utf-8")

    figures_dir = output_dir / "figures" / "comparison"
    plot_order_amplitude_comparison(summary_df, figures_dir / "order_amplitude_comparison.png")
    plot_order_ratio_heatmap(summary_df, figures_dir / "order_ratio_heatmap.png")
    return detail_df, summary_df


def pivot_order_comparison(order_df: pd.DataFrame, index_columns: list[str]) -> pd.DataFrame:
    """將長格式倍頻資料轉成健康/故障比較欄位。"""
    amp_pivot = (
        order_df.pivot_table(index=index_columns, columns="dataset", values="peak_amplitude_g", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    freq_pivot = (
        order_df.pivot_table(index=index_columns, columns="dataset", values="peak_frequency_hz", aggfunc="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    comparison_df = amp_pivot.rename(
        columns={
            "healthy_unit2_motor": "healthy_amp_g",
            "faulty_unit3_motor": "faulty_amp_g",
        }
    )
    freq_pivot = freq_pivot.rename(
        columns={
            "healthy_unit2_motor": "healthy_peak_frequency_hz",
            "faulty_unit3_motor": "faulty_peak_frequency_hz",
        }
    )
    comparison_df = comparison_df.merge(freq_pivot, on=index_columns, how="left")
    comparison_df["faulty_minus_healthy_amp_g"] = comparison_df["faulty_amp_g"] - comparison_df["healthy_amp_g"]
    comparison_df["faulty_to_healthy_amp_ratio"] = comparison_df["faulty_amp_g"] / comparison_df["healthy_amp_g"]
    comparison_df["faulty_amp_increase_percent"] = (comparison_df["faulty_to_healthy_amp_ratio"] - 1.0) * 100.0
    return comparison_df.sort_values(index_columns)


def plot_order_amplitude_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    """輸出 1X/2X/3X 健康與故障振幅折線比較圖。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    colors = {"healthy": "#1f77b4", "faulty": "#d62728"}
    for ax, order in zip(axes, ["1X", "2X", "3X"]):
        order_df = summary_df[summary_df["order"] == order].sort_values("setting_frequency_hz")
        ax.plot(
            order_df["setting_frequency_hz"],
            order_df["healthy_amp_g"],
            marker="o",
            linewidth=1.4,
            color=colors["healthy"],
            label="Healthy unit2",
        )
        ax.plot(
            order_df["setting_frequency_hz"],
            order_df["faulty_amp_g"],
            marker="s",
            linewidth=1.4,
            color=colors["faulty"],
            label="Faulty unit3",
        )
        ax.fill_between(
            order_df["setting_frequency_hz"],
            order_df["healthy_amp_g"],
            order_df["faulty_amp_g"],
            color="#d62728",
            alpha=0.10,
        )
        ax.set_title(f"{order} Order Amplitude")
        ax.set_ylabel("Amplitude (g)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Motor Setting Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_order_ratio_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """輸出故障/健康倍頻振幅比值熱圖。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = summary_df.pivot(index="order", columns="setting_frequency_hz", values="faulty_to_healthy_amp_ratio")
    pivot = pivot.reindex(["1X", "2X", "3X"])
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    image = ax.imshow(values, aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=max(3.0, float(np.nanmax(values))))
    ax.set_title("Faulty / Healthy Order Amplitude Ratio")
    ax.set_xlabel("Motor Setting Frequency (Hz)")
    ax.set_ylabel("Order")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.1f}x", ha="center", va="center", fontsize=9, color="black")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Amplitude Ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_wind_speed(wind_dir: Path) -> pd.DataFrame:
    """讀取風速 CSV。"""
    mapping = {
        "healthy_unit2_motor": wind_dir / "wind_speed_data_2th.csv",
        "faulty_unit3_motor": wind_dir / "wind_speed_data_3th.csv",
    }
    rows = []
    for dataset, path in mapping.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到風速 CSV：{path}")
        df = pd.read_csv(path, header=None, names=["setting_frequency_hz", "wind_speed_mps"])
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "setting_frequency_hz": float(row.setting_frequency_hz),
                    "wind_speed_mps": float(row.wind_speed_mps),
                }
            )
    return pd.DataFrame(rows)


def parse_power_filename(path: Path) -> tuple[str, float]:
    """從功率檔名解析設備與設定頻率。"""
    match = re.match(r"(unit[23])_([-+]?\d+(?:\.\d+)?)Hz", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"無法從功率檔名解析 unit 與 Hz：{path.name}")
    unit = match.group(1).lower()
    dataset = "healthy_unit2_motor" if unit == "unit2" else "faulty_unit3_motor"
    return dataset, float(match.group(2))


def load_power(power_dir: Path) -> pd.DataFrame:
    """讀取功率 CSV，使用三相公式計算平均功率。"""
    rows = []
    for path in sorted(power_dir.glob("*.csv")):
        dataset, setting_hz = parse_power_filename(path)
        df = pd.read_csv(path)
        current_col = "輸出電流(A)"
        voltage_col = "輸出電壓(V)"
        pf_col = "輸出PF()"
        required = {current_col, voltage_col, pf_col}
        if not required.issubset(df.columns):
            raise ValueError(f"{path.name} 缺少功率欄位：{required - set(df.columns)}")
        current_a = df[current_col].to_numpy(dtype=float)
        voltage_v = df[voltage_col].to_numpy(dtype=float)
        power_factor = df[pf_col].to_numpy(dtype=float)
        power_w = math.sqrt(3.0) * voltage_v * current_a * power_factor
        rows.append(
            {
                "dataset": dataset,
                "setting_frequency_hz": setting_hz,
                "file": path.name,
                "sample_count": len(df),
                "mean_current_a": float(np.mean(current_a)),
                "mean_voltage_v": float(np.mean(voltage_v)),
                "mean_power_factor": float(np.mean(power_factor)),
                "mean_power_w": float(np.mean(power_w)),
                "std_power_w": float(np.std(power_w, ddof=1)) if len(power_w) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def analyze_power_and_wind(raw_root: Path, output_dir: Path) -> pd.DataFrame:
    """建立健康功率回歸線並計算故障耗能率。"""
    wind_df = load_wind_speed(raw_root / "wind_speed")
    power_df = load_power(raw_root / "about_power")
    combined = power_df.merge(wind_df, on=["dataset", "setting_frequency_hz"], how="left")
    if combined["wind_speed_mps"].isna().any():
        missing = combined[combined["wind_speed_mps"].isna()][["dataset", "setting_frequency_hz"]]
        raise ValueError(f"功率資料缺少對應風速：{missing.to_dict(orient='records')}")

    healthy = combined[combined["dataset"] == "healthy_unit2_motor"].sort_values("wind_speed_mps")
    if len(healthy) < 2:
        raise ValueError("健康資料不足，無法建立風速對功率回歸線。")
    slope, intercept = np.polyfit(healthy["wind_speed_mps"], healthy["mean_power_w"], deg=1)

    combined["expected_healthy_power_w"] = slope * combined["wind_speed_mps"] + intercept
    combined["energy_overuse_percent"] = np.nan
    faulty_mask = combined["dataset"] == "faulty_unit3_motor"
    combined.loc[faulty_mask, "energy_overuse_percent"] = (
        combined.loc[faulty_mask, "mean_power_w"] / combined.loc[faulty_mask, "expected_healthy_power_w"] - 1.0
    ) * 100.0
    combined["regression_slope_w_per_mps"] = slope
    combined["regression_intercept_w"] = intercept

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tables_dir / "power_wind_regression.csv", index=False, encoding="utf-8")

    plot_power_wind_regression(healthy, combined[faulty_mask], slope, intercept, output_dir / "figures" / "power_wind_regression.png")
    return combined


def plot_power_wind_regression(
    healthy_df: pd.DataFrame,
    faulty_df: pd.DataFrame,
    slope: float,
    intercept: float,
    output_path: Path,
) -> None:
    """輸出風速對平均功率回歸圖。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_min = min(float(healthy_df["wind_speed_mps"].min()), float(faulty_df["wind_speed_mps"].min()))
    x_max = max(float(healthy_df["wind_speed_mps"].max()), float(faulty_df["wind_speed_mps"].max()))
    x_line = np.linspace(max(0.0, x_min * 0.85), x_max * 1.08, 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, color="#1f77b4", linewidth=1.4, label="Healthy regression")
    ax.scatter(healthy_df["wind_speed_mps"], healthy_df["mean_power_w"], color="#1f77b4", s=65, label="Healthy unit2")
    for row in healthy_df.itertuples(index=False):
        ax.annotate(f"{row.setting_frequency_hz:g} Hz", (row.wind_speed_mps, row.mean_power_w), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.scatter(faulty_df["wind_speed_mps"], faulty_df["mean_power_w"], color="#d62728", marker="x", s=90, linewidths=2.0, label="Faulty unit3")
    for row in faulty_df.itertuples(index=False):
        ax.annotate(f"{row.setting_frequency_hz:g} Hz", (row.wind_speed_mps, row.mean_power_w), textcoords="offset points", xytext=(6, -13), fontsize=9)

    ax.set_title("Wind Speed vs Mean Power")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Mean Three-Phase Power (W)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    vibration_df: pd.DataFrame,
    bpf_df: pd.DataFrame,
    order_summary_df: pd.DataFrame,
    power_df: pd.DataFrame,
) -> None:
    """輸出繁體中文 Markdown 分析報告。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = (
        vibration_df.groupby(["dataset", "condition", "setting_frequency_hz"], as_index=False)
        .agg(
            mean_rms_g=("acceleration_rms_g", "mean"),
            max_rms_g=("acceleration_rms_g", "max"),
            mean_1x_amp_g=("order_1x_amp_g", "mean"),
            mean_2x_amp_g=("order_2x_amp_g", "mean"),
            mean_3x_amp_g=("order_3x_amp_g", "mean"),
        )
        .sort_values(["setting_frequency_hz", "dataset"])
    )
    pivot_rms = aggregate.pivot(index="setting_frequency_hz", columns="dataset", values="mean_rms_g").reset_index()
    if {"healthy_unit2_motor", "faulty_unit3_motor"}.issubset(pivot_rms.columns):
        pivot_rms["faulty_to_healthy_rms_ratio"] = pivot_rms["faulty_unit3_motor"] / pivot_rms["healthy_unit2_motor"]
    else:
        pivot_rms["faulty_to_healthy_rms_ratio"] = np.nan

    faulty_power = power_df[power_df["dataset"] == "faulty_unit3_motor"].copy()
    energy_text = "無故障功率資料可計算。"
    if not faulty_power.empty:
        row = faulty_power.sort_values("setting_frequency_hz").iloc[0]
        energy_text = (
            f"故障 unit3 目前只有 `{row['setting_frequency_hz']:.0f} Hz` 功率資料；"
            f"實測平均功率為 `{row['mean_power_w']:.2f} W`，"
            f"同風速健康回歸預期功率為 `{row['expected_healthy_power_w']:.2f} W`，"
            f"耗能率為 `{row['energy_overuse_percent']:.2f}%`。"
        )

    lines = [
        "# rawdata_main 重新資料分析報告",
        "",
        "## 分析設定",
        "",
        f"- 輸入資料根目錄：`{Path(args.raw_root).resolve()}`",
        f"- 輸出資料夾：`{Path(args.output_dir).resolve()}`",
        f"- 頻率顯示範圍：`{args.min_freq:g}~{args.max_freq:g} Hz`",
        f"- 瀑布圖顯示範圍：`{args.waterfall_min_freq:g}~{args.max_freq:g} Hz`、`0~{args.waterfall_z_max:g} g`",
        f"- 加速度換算：`a_g = (voltage - {args.sensor_bias_v:g}) / {args.sensor_sensitivity_v_per_g:g}`",
        f"- BPF 暫定參數：`D = {args.bpf_diameter_m:g} m`、`L = {args.belt_length_m:g} m`、`RPM = input_hz * 60`",
        "",
        "## 振動摘要",
        "",
        "| Setting (Hz) | Healthy Mean RMS (g) | Faulty Mean RMS (g) | Faulty / Healthy |",
        "|---:|---:|---:|---:|",
    ]
    for row in pivot_rms.itertuples(index=False):
        lines.append(
            f"| {row.setting_frequency_hz:g} | "
            f"{getattr(row, 'healthy_unit2_motor', float('nan')):.5f} | "
            f"{getattr(row, 'faulty_unit3_motor', float('nan')):.5f} | "
            f"{row.faulty_to_healthy_rms_ratio:.2f} |"
        )

    finite_order_df = order_summary_df[np.isfinite(order_summary_df["faulty_to_healthy_amp_ratio"])].copy()
    finite_order_df = finite_order_df.sort_values("faulty_to_healthy_amp_ratio", ascending=False)
    lines.extend(
        [
            "",
            "## 倍頻健康/故障比較",
            "",
            "下表為雙通道平均後，故障 unit3 相對健康 unit2 差異最大的倍頻項目。",
            "",
            "| Setting (Hz) | Order | Healthy Amp (g) | Faulty Amp (g) | Faulty / Healthy | Increase (%) |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in finite_order_df.head(10).itertuples(index=False):
        lines.append(
            f"| {row.setting_frequency_hz:g} | {row.order} | {row.healthy_amp_g:.6f} | "
            f"{row.faulty_amp_g:.6f} | {row.faulty_to_healthy_amp_ratio:.2f} | "
            f"{row.faulty_amp_increase_percent:.1f} |"
        )

    lines.extend(
        [
            "",
            "## BPF 摘要",
            "",
            "BPF 僅套用在故障 unit3 motor；目前 D、L 為暫定值，之後可以用 CLI 參數快速重跑。",
            "",
            "| Setting (Hz) | Channel | BPF Base (Hz) | Strongest BPF Harmonic | Peak Amp (g) |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    if bpf_df.empty:
        lines.append("| NA | NA | NA | NA | NA |")
    else:
        strongest_bpf = (
            bpf_df.sort_values("peak_amplitude_g", ascending=False)
            .groupby(["setting_frequency_hz", "channel"], as_index=False)
            .first()
            .sort_values(["setting_frequency_hz", "channel"])
        )
        for row in strongest_bpf.itertuples(index=False):
            lines.append(
                f"| {row.setting_frequency_hz:g} | {row.channel} | {row.bpf_base_hz:.3f} | "
                f"{int(row.harmonic)} | {row.peak_amplitude_g:.6f} |"
            )

    lines.extend(
        [
            "",
            "## 風速與功率",
            "",
            energy_text,
            "",
            "| Dataset | Setting (Hz) | Wind Speed (m/s) | Mean Power (W) | Expected Healthy Power (W) | Energy Overuse (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in power_df.sort_values(["dataset", "setting_frequency_hz"]).itertuples(index=False):
        overuse = "" if pd.isna(row.energy_overuse_percent) else f"{row.energy_overuse_percent:.2f}"
        lines.append(
            f"| {row.dataset} | {row.setting_frequency_hz:g} | {row.wind_speed_mps:.3f} | "
            f"{row.mean_power_w:.2f} | {row.expected_healthy_power_w:.2f} | {overuse} |"
        )

    lines.extend(
        [
            "",
            "## 輸出檔案",
            "",
            "- `tables/vibration_fft_metrics.csv`",
            "- `tables/bpf_metrics.csv`",
            "- `tables/order_health_fault_comparison_by_channel.csv`",
            "- `tables/order_health_fault_summary.csv`",
            "- `tables/power_wind_regression.csv`",
            "- `figures/fft/*.png`",
            "- `figures/waterfall/*.png`",
            "- `figures/comparison/order_amplitude_comparison.png`",
            "- `figures/comparison/order_ratio_heatmap.png`",
            "- `figures/power_wind_regression.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """執行 rawdata_main 完整分析。"""
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not raw_root.exists():
        print(f"找不到 rawdata_main：{raw_root}")
        return 1

    datasets = [
        DatasetConfig("healthy_unit2_motor", "Healthy unit2 motor", raw_root / "healthy_unit2_motor", False),
        DatasetConfig("faulty_unit3_motor", "Faulty unit3 motor", raw_root / "faulty_unit3_motor", True),
    ]

    results = analyze_vibration(args, datasets, output_dir)
    vibration_df, bpf_df = write_vibration_tables(results, output_dir, args)
    order_detail_df, order_summary_df = build_order_comparison(vibration_df, output_dir)
    power_df = analyze_power_and_wind(raw_root, output_dir)
    write_report(output_dir / "analysis_report.md", args, vibration_df, bpf_df, order_summary_df, power_df)

    print(f"完成 rawdata_main 分析：{output_dir}")
    print(f"振動結果：{len(vibration_df)} rows")
    print(f"BPF 結果：{len(bpf_df)} rows")
    print(f"倍頻比較：{len(order_detail_df)} channel rows，{len(order_summary_df)} summary rows")
    print(f"功率回歸：{len(power_df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
