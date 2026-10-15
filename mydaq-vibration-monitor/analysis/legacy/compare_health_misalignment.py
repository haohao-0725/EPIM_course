# -*- coding: utf-8 -*-
"""
比較健康風機與軸不對中風機的振動頻譜與風速表現。

此腳本使用 2026-05-27 的 raw data：
  - healthy_2th：健康風機
  - faulty_3th：馬達搭配皮帶輪且存在軸不對中案例
  - wind_speed：兩台風機於 10、15、20 Hz 的風速資料

輸出內容包含比較圖、CSV 數值表與 Markdown 分析報告。
"""

from __future__ import annotations

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


RAW_ROOT = PROJECT_ROOT / "data" / "rawdata_20260527"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "comparison_healthy_vs_misalignment"
MIN_FREQ_HZ = 0.0
MAX_FREQ_HZ = 100.0
SENSOR_BIAS_V = 2.5
SENSOR_SENSITIVITY_V_PER_G = 0.1
ORDER_BAND_HALF_WIDTH_HZ = 0.5


@dataclass(frozen=True)
class DatasetConfig:
    """單一資料集的路徑與顯示名稱。"""

    dataset_name: str
    condition_label: str
    input_dir: Path


DATASETS = [
    DatasetConfig(
        dataset_name="healthy_2th",
        condition_label="Healthy fan",
        input_dir=RAW_ROOT / "healthy_2th",
    ),
    DatasetConfig(
        dataset_name="faulty_3th",
        condition_label="Misalignment fan",
        input_dir=RAW_ROOT / "faulty_3th",
    ),
]


def parse_setting_frequency(csv_path: Path) -> float:
    """從檔名解析馬達設定頻率。"""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", csv_path.stem)
    if not match:
        raise ValueError(f"無法從檔名解析設定頻率：{csv_path.name}")
    return float(match.group(0))


def voltage_to_acceleration_g(voltage: np.ndarray) -> np.ndarray:
    """將震動規輸出電壓轉成加速度 g。"""
    return (voltage - SENSOR_BIAS_V) / SENSOR_SENSITIVITY_V_PER_G


def load_vibration_csv(csv_path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """讀取振動 CSV，回傳取樣率與各通道電壓。"""
    df = pd.read_csv(csv_path)
    voltage_columns = [column for column in df.columns if column.endswith("_voltage_v")]
    if "voltage_v" in df.columns:
        voltage_by_channel = {"voltage_v": df["voltage_v"].to_numpy(dtype=float)}
    elif voltage_columns:
        voltage_by_channel = {
            column.replace("_voltage_v", ""): df[column].to_numpy(dtype=float)
            for column in voltage_columns
        }
    else:
        raise ValueError(f"{csv_path.name} 缺少可分析的電壓欄位。")

    if "time_s" not in df.columns:
        raise ValueError(f"{csv_path.name} 缺少 time_s，無法推估取樣率。")
    time_s = df["time_s"].to_numpy(dtype=float)
    diffs = np.diff(time_s)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError(f"{csv_path.name} 的 time_s 無法推估取樣率。")
    sample_rate_hz = float(1.0 / np.median(diffs))
    return sample_rate_hz, voltage_by_channel


def band_peak(freqs: np.ndarray, amplitudes: np.ndarray, target_hz: float) -> tuple[float, float]:
    """取指定頻率附近頻帶的最大振幅。"""
    if not np.isfinite(target_hz) or target_hz > MAX_FREQ_HZ:
        return float("nan"), float("nan")
    lower = max(MIN_FREQ_HZ, target_hz - ORDER_BAND_HALF_WIDTH_HZ)
    upper = min(MAX_FREQ_HZ, target_hz + ORDER_BAND_HALF_WIDTH_HZ)
    mask = (freqs >= lower) & (freqs <= upper)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_freqs = freqs[mask]
    local_amps = amplitudes[mask]
    idx = int(np.argmax(local_amps))
    return float(local_freqs[idx]), float(local_amps[idx])


def dominant_frequency(freqs: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float]:
    """排除 1 Hz 以下低頻漂移後，取 0~100 Hz 內最大振幅頻率。"""
    mask = (freqs >= 1.0) & (freqs <= MAX_FREQ_HZ)
    local_freqs = freqs[mask]
    local_amps = amplitudes[mask]
    idx = int(np.argmax(local_amps))
    return float(local_freqs[idx]), float(local_amps[idx])


def analyze_vibration() -> pd.DataFrame:
    """計算兩組振動資料的 RMS、P-P、dominant 與 1X/2X/3X。"""
    rows: list[dict[str, float | str]] = []
    for config in DATASETS:
        csv_paths = sorted(config.input_dir.glob("*.csv"), key=parse_setting_frequency)
        for csv_path in csv_paths:
            setting_hz = parse_setting_frequency(csv_path)
            sample_rate_hz, voltage_by_channel = load_vibration_csv(csv_path)
            for channel, voltage in voltage_by_channel.items():
                acceleration_g = voltage_to_acceleration_g(voltage)
                freqs, amplitudes = compute_fft(acceleration_g, int(round(sample_rate_hz)))
                dom_freq_hz, dom_amp_g = dominant_frequency(freqs, amplitudes)
                order_values: dict[str, float] = {}
                for order in [1, 2, 3]:
                    order_freq_hz, order_amp_g = band_peak(freqs, amplitudes, setting_hz * order)
                    order_values[f"order_{order}x_freq_hz"] = order_freq_hz
                    order_values[f"order_{order}x_amp_g"] = order_amp_g
                order_1x_amp = order_values["order_1x_amp_g"]
                order_3x_amp = order_values["order_3x_amp_g"]
                rows.append(
                    {
                        "dataset": config.dataset_name,
                        "condition": config.condition_label,
                        "setting_frequency_hz": setting_hz,
                        "channel": channel,
                        "sample_rate_hz": sample_rate_hz,
                        "acceleration_rms_g": float(np.sqrt(np.mean(acceleration_g**2))),
                        "acceleration_peak_to_peak_g": float(np.max(acceleration_g) - np.min(acceleration_g)),
                        "dominant_frequency_hz": dom_freq_hz,
                        "dominant_amplitude_g": dom_amp_g,
                        "order_1x_freq_hz": order_values["order_1x_freq_hz"],
                        "order_1x_amp_g": order_1x_amp,
                        "order_2x_freq_hz": order_values["order_2x_freq_hz"],
                        "order_2x_amp_g": order_values["order_2x_amp_g"],
                        "order_3x_freq_hz": order_values["order_3x_freq_hz"],
                        "order_3x_amp_g": order_3x_amp,
                        "order_3x_to_1x_amp_ratio": float(order_3x_amp / order_1x_amp) if order_1x_amp > 0 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def load_wind_speed() -> pd.DataFrame:
    """讀取健康與異常風機風速資料。"""
    mapping = {
        "healthy_2th": RAW_ROOT / "wind_speed" / "wind_speed_data_2th.csv",
        "faulty_3th": RAW_ROOT / "wind_speed" / "wind_speed_data_3th.csv",
    }
    condition_lookup = {config.dataset_name: config.condition_label for config in DATASETS}
    rows = []
    for dataset, path in mapping.items():
        df = pd.read_csv(path, header=None, names=["setting_frequency_hz", "wind_speed_mps"])
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition_lookup[dataset],
                    "setting_frequency_hz": float(row.setting_frequency_hz),
                    "wind_speed_mps": float(row.wind_speed_mps),
                }
            )
    return pd.DataFrame(rows)


def build_wind_comparison(wind_df: pd.DataFrame) -> pd.DataFrame:
    """建立健康基線與異常風機風速比較。"""
    pivot = wind_df.pivot(index="setting_frequency_hz", columns="dataset", values="wind_speed_mps").reset_index()
    pivot["wind_speed_ratio_faulty_to_healthy"] = pivot["faulty_3th"] / pivot["healthy_2th"]
    pivot["wind_speed_loss_percent"] = (1.0 - pivot["wind_speed_ratio_faulty_to_healthy"]) * 100.0
    pivot["air_speed_cubed_proxy_ratio"] = pivot["wind_speed_ratio_faulty_to_healthy"] ** 3
    pivot["air_speed_cubed_proxy_loss_percent"] = (1.0 - pivot["air_speed_cubed_proxy_ratio"]) * 100.0
    return pivot


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 輸出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def plot_rms_comparison(vibration_df: pd.DataFrame, output_path: Path) -> None:
    """輸出 RMS 比較圖。"""
    channels = sorted(vibration_df["channel"].unique())
    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 7), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    colors = {"healthy_2th": "#2f6fbb", "faulty_3th": "#bc4749"}
    for ax, channel in zip(axes, channels):
        channel_df = vibration_df[vibration_df["channel"] == channel]
        for dataset, group in channel_df.groupby("dataset"):
            group = group.sort_values("setting_frequency_hz")
            ax.plot(
                group["setting_frequency_hz"],
                group["acceleration_rms_g"],
                marker="o",
                label=dataset,
                color=colors[dataset],
            )
        ax.set_title(f"Acceleration RMS - {channel}")
        ax.set_ylabel("RMS (g)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    axes[-1].set_xlabel("Motor Setting Frequency (Hz)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_order_comparison(vibration_df: pd.DataFrame, output_path: Path) -> None:
    """輸出 1X/2X/3X 平均 order 振幅比較圖。"""
    aggregate = (
        vibration_df.groupby(["dataset", "setting_frequency_hz"], as_index=False)[
            ["order_1x_amp_g", "order_2x_amp_g", "order_3x_amp_g"]
        ]
        .mean()
        .sort_values(["dataset", "setting_frequency_hz"])
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"healthy_2th": "#2f6fbb", "faulty_3th": "#bc4749"}
    for ax, order_column, title in zip(
        axes,
        ["order_1x_amp_g", "order_2x_amp_g", "order_3x_amp_g"],
        ["1X Order", "2X Order", "3X Order"],
    ):
        for dataset, group in aggregate.groupby("dataset"):
            ax.plot(
                group["setting_frequency_hz"],
                group[order_column],
                marker="o",
                label=dataset,
                color=colors[dataset],
            )
        ax.set_title(f"{title} Amplitude")
        ax.set_ylabel("Amplitude (g)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
    axes[-1].set_xlabel("Motor Setting Frequency (Hz)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_wind_speed(wind_comparison_df: pd.DataFrame, output_path: Path) -> None:
    """輸出風速與健康基線比較圖。"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(wind_comparison_df["setting_frequency_hz"], wind_comparison_df["healthy_2th"], marker="o", label="Healthy fan", color="#2f6fbb")
    ax.plot(wind_comparison_df["setting_frequency_hz"], wind_comparison_df["faulty_3th"], marker="o", label="Misalignment fan", color="#bc4749")
    ax.set_title("Wind Speed Comparison")
    ax.set_xlabel("Motor Setting Frequency (Hz)")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_wind_loss(wind_comparison_df: pd.DataFrame, output_path: Path) -> None:
    """輸出風速損失與風速三次方代理指標損失。"""
    x = np.arange(len(wind_comparison_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, wind_comparison_df["wind_speed_loss_percent"], width, label="Wind speed loss", color="#bc4749")
    ax.bar(x + width / 2, wind_comparison_df["air_speed_cubed_proxy_loss_percent"], width, label="Speed cubed proxy loss", color="#f2a900")
    ax.set_title("Performance Loss vs Healthy Baseline")
    ax.set_xlabel("Motor Setting Frequency (Hz)")
    ax.set_ylabel("Loss (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in wind_comparison_df["setting_frequency_hz"]])
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def format_percent(value: float) -> str:
    """格式化百分比。"""
    return f"{value:.1f}%"


def write_report(
    vibration_df: pd.DataFrame,
    wind_comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """輸出課程專題分析報告草稿。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = (
        vibration_df.groupby(["dataset", "setting_frequency_hz"], as_index=False)
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
    pivot_rms["rms_ratio_faulty_to_healthy"] = pivot_rms["faulty_3th"] / pivot_rms["healthy_2th"]
    dominant_order_df = vibration_df.copy()
    dominant_order_df["dominant_order"] = (dominant_order_df["dominant_frequency_hz"] / dominant_order_df["setting_frequency_hz"]).round(1)
    dominant_counts = (
        dominant_order_df.groupby(["dataset", "dominant_order"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "dominant_order"])
    )
    order_ratio_at_key_speed = (
        vibration_df[vibration_df["setting_frequency_hz"].isin([10.0, 20.0])]
        .groupby(["dataset", "setting_frequency_hz"], as_index=False)
        .agg(
            mean_1x_amp_g=("order_1x_amp_g", "mean"),
            mean_3x_amp_g=("order_3x_amp_g", "mean"),
            mean_3x_to_1x=("order_3x_to_1x_amp_ratio", "mean"),
        )
        .sort_values(["setting_frequency_hz", "dataset"])
    )

    overlap_wind = wind_comparison_df.copy()
    worst_wind = overlap_wind.sort_values("wind_speed_loss_percent", ascending=False).iloc[0]
    worst_proxy = overlap_wind.sort_values("air_speed_cubed_proxy_loss_percent", ascending=False).iloc[0]
    rms_ratio_mean = float(pivot_rms["rms_ratio_faulty_to_healthy"].mean())
    rms_ratio_at_10 = float(pivot_rms.loc[pivot_rms["setting_frequency_hz"] == 10.0, "rms_ratio_faulty_to_healthy"].iloc[0])
    rms_ratio_at_20 = float(pivot_rms.loc[pivot_rms["setting_frequency_hz"] == 20.0, "rms_ratio_faulty_to_healthy"].iloc[0])

    lines = [
        "# 健康風機與軸不對中風機比較分析",
        "",
        "## 分析目的",
        "",
        "本分析對照健康風機 `healthy_2th` 與馬達搭配皮帶輪且存在軸不對中的風機 `faulty_3th`。分析重點包含振動訊號式異常診斷，以及以風速資料建立健康基線後比較異常機台的輸出性能。",
        "",
        "## 目前資料與限制",
        "",
        "- 振動資料：兩台風機皆有 2、4、6、8、10、12、14、16、18、20 Hz 設定頻率，各 10 秒、取樣率約 10000 Hz、兩個 myDAQ 通道。",
        "- 風速資料：健康與異常風機皆有 10、15、20 Hz 三個量測點。",
        "- 能效限制：目前資料未包含電壓、電流、功率因數或實功率，因此不能計算真正的設備效率或 kW/風量。以下能效討論先以健康風機為 CMVP 基線，使用風速作為輸出性能代理指標。",
        "",
        "## 計算方式",
        "",
        f"- 加速度換算：`a_g[n] = (v[n] - {SENSOR_BIAS_V}) / {SENSOR_SENSITIVITY_V_PER_G}`。",
        "- FFT：使用專案 `compute_fft`，Hanning window 後做單邊 FFT 振幅正規化。",
        f"- 顯示與診斷頻率範圍：`{MIN_FREQ_HZ:g}~{MAX_FREQ_HZ:g} Hz`。",
        f"- Order 分析：在目標 order 頻率 `setting_frequency * order` 的 `±{ORDER_BAND_HALF_WIDTH_HZ:g} Hz` 頻帶中取最大振幅，計算 1X、2X、3X。",
        "- 風速三次方代理指標：同一量測截面與空氣密度假設下，氣流動能尺度可用 `wind_speed^3` 作為粗略輸出性能 proxy；此值不是實際能效。",
        "",
        "## 主要結果",
        "",
        f"- 異常風機的平均振動 RMS 約為健康風機的 `{rms_ratio_mean:.2f}` 倍；在 10 Hz 約 `{rms_ratio_at_10:.2f}` 倍，在 20 Hz 約 `{rms_ratio_at_20:.2f}` 倍。",
        f"- 風速損失最大出現在 `{worst_wind['setting_frequency_hz']:.0f} Hz`：健康 `{worst_wind['healthy_2th']:.2f} m/s`，異常 `{worst_wind['faulty_3th']:.2f} m/s`，下降 `{format_percent(worst_wind['wind_speed_loss_percent'])}`。",
        f"- 以風速三次方作為輸出性能 proxy，損失最大出現在 `{worst_proxy['setting_frequency_hz']:.0f} Hz`，proxy 下降 `{format_percent(worst_proxy['air_speed_cubed_proxy_loss_percent'])}`。",
        "- 健康風機的 dominant frequency 幾乎都落在 1X；軸不對中風機則有較多結果落在 3X 或其他倍頻附近，頻譜型態較不單純。",
        "",
        "## 風速基線比較",
        "",
        "| Setting (Hz) | Healthy (m/s) | Misalignment (m/s) | Wind Speed Loss | Speed Cubed Proxy Loss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in wind_comparison_df.itertuples(index=False):
        lines.append(
            f"| {row.setting_frequency_hz:g} | {row.healthy_2th:.2f} | {row.faulty_3th:.2f} | "
            f"{row.wind_speed_loss_percent:.1f}% | {row.air_speed_cubed_proxy_loss_percent:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 振動 RMS 比較",
            "",
            "| Setting (Hz) | Healthy Mean RMS (g) | Misalignment Mean RMS (g) | Faulty / Healthy |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in pivot_rms.itertuples(index=False):
        lines.append(
            f"| {row.setting_frequency_hz:g} | {row.healthy_2th:.4f} | {row.faulty_3th:.4f} | {row.rms_ratio_faulty_to_healthy:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 倍頻特徵觀察",
            "",
            "| Dataset | Dominant Order | Count |",
            "|---|---:|---:|",
        ]
    )
    for row in dominant_counts.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.dominant_order:.1f}X | {row.count} |")

    lines.extend(
        [
            "",
            "| Dataset | Setting (Hz) | Mean 1X Amp (g) | Mean 3X Amp (g) | 3X / 1X |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in order_ratio_at_key_speed.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.setting_frequency_hz:g} | {row.mean_1x_amp_g:.5f} | {row.mean_3x_amp_g:.5f} | {row.mean_3x_to_1x:.2f} |"
        )

    lines.extend(
        [
            "",
            "## CMVP 角度的下一步",
            "",
            "- 邊界：建議定義為單一風機含馬達、皮帶輪傳動與葉輪，輸出端為固定位置/固定截面的風速或風量。",
            "- 量測：除了風速，後續應同步量測輸入功率，例如功率計或三相電壓、電流、功率因數。",
            "- 基線：以健康風機在相同頻率、相同量測位置、相同環境條件下的 `功率 -> 風速/風量` 關係作為 baseline。",
            "- 比較：異常狀態應比較同頻率下的振動特徵、輸入功率、風速/風量，以及 `輸出 proxy / 輸入功率`。目前資料只能完成振動診斷與風速輸出基線比較。",
            "",
            "## 輸出圖檔",
            "",
            "- `figures/vibration_rms_comparison.png`",
            "- `figures/order_amplitude_comparison.png`",
            "- `figures/wind_speed_comparison.png`",
            "- `figures/wind_loss_comparison.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """執行比較分析。"""
    vibration_df = analyze_vibration()
    wind_df = load_wind_speed()
    wind_comparison_df = build_wind_comparison(wind_df)

    write_csv(vibration_df, OUTPUT_DIR / "tables" / "vibration_order_metrics.csv")
    write_csv(wind_df, OUTPUT_DIR / "tables" / "wind_speed_raw.csv")
    write_csv(wind_comparison_df, OUTPUT_DIR / "tables" / "wind_speed_comparison.csv")

    plot_rms_comparison(vibration_df, OUTPUT_DIR / "figures" / "vibration_rms_comparison.png")
    plot_order_comparison(vibration_df, OUTPUT_DIR / "figures" / "order_amplitude_comparison.png")
    plot_wind_speed(wind_comparison_df, OUTPUT_DIR / "figures" / "wind_speed_comparison.png")
    plot_wind_loss(wind_comparison_df, OUTPUT_DIR / "figures" / "wind_loss_comparison.png")

    write_report(vibration_df, wind_comparison_df, OUTPUT_DIR / "analysis_report.md")
    print(f"完成健康與軸不對中比較分析：{OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
