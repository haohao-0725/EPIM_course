"""
analysis.py

訊號分析模組。

提供時域與頻域特徵計算：
  - mean, RMS, peak-to-peak
  - FFT 與 dominant frequency
"""

import numpy as np
from scipy import signal as scipy_signal


# ─────────────────────────────────────────────
# 時域特徵
# ─────────────────────────────────────────────

def compute_mean(voltage: np.ndarray) -> float:
    """計算訊號平均值 (V)。"""
    return float(np.mean(voltage))


def compute_rms(voltage: np.ndarray) -> float:
    """
    計算均方根值 (RMS)。

    RMS = sqrt(mean(x^2))
    """
    return float(np.sqrt(np.mean(voltage ** 2)))


def compute_peak_to_peak(voltage: np.ndarray) -> float:
    """
    計算峰值到峰值 (Peak-to-Peak)。

    Peak-to-peak = max(x) - min(x)
    """
    return float(np.max(voltage) - np.min(voltage))


def compute_peak(voltage: np.ndarray) -> float:
    """計算訊號最大絕對值 (Peak)。"""
    return float(np.max(np.abs(voltage)))


def compute_crest_factor(voltage: np.ndarray) -> float:
    """
    計算波峰因數 (Crest Factor)。

    Crest Factor = Peak / RMS
    波峰因數可以用來判斷衝擊性，通常 >3 需注意。
    """
    rms = compute_rms(voltage)
    if rms == 0:
        return float("inf")
    return compute_peak(voltage) / rms


def compute_time_features(voltage: np.ndarray) -> dict:
    """
    一次計算所有時域特徵，回傳字典。

    Returns
    -------
    dict with keys:
        mean, rms, peak_to_peak, peak, crest_factor
    """
    return {
        "mean":           compute_mean(voltage),
        "rms":            compute_rms(voltage),
        "peak_to_peak":   compute_peak_to_peak(voltage),
        "peak":           compute_peak(voltage),
        "crest_factor":   compute_crest_factor(voltage),
    }


# ─────────────────────────────────────────────
# 頻域特徵
# ─────────────────────────────────────────────

def compute_fft(
    voltage: np.ndarray,
    sampling_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算單邊 FFT 頻譜。

    Parameters
    ----------
    voltage : np.ndarray
        時域訊號。
    sampling_rate : int
        取樣率 (Hz)。

    Returns
    -------
    freqs : np.ndarray
        頻率陣列 (Hz)，僅包含正頻率部分（0 到 fs/2）。
    amplitudes : np.ndarray
        對應振幅陣列 (V)，已做正規化。
    """
    n = len(voltage)
    # 使用 Hanning window 減少頻譜洩漏
    window = np.hanning(n)
    windowed = voltage * window
    # FFT
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    # 振幅正規化：乘以 2/n 取單邊，再除以 window sum 修正
    window_correction = np.sum(window)
    amplitudes = (2.0 / window_correction) * np.abs(fft_vals)
    # DC 分量不需乘以 2
    amplitudes[0] /= 2.0
    return freqs, amplitudes


def find_dominant_frequency(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    min_freq: float = 1.0,
) -> float:
    """
    找出振幅最大的頻率（主頻）。

    Parameters
    ----------
    freqs : np.ndarray
        頻率陣列 (Hz)。
    amplitudes : np.ndarray
        對應振幅陣列。
    min_freq : float
        最小搜尋頻率 (Hz)，用於過濾 DC 分量。

    Returns
    -------
    float
        主頻 (Hz)。
    """
    mask = freqs >= min_freq
    filtered_freqs = freqs[mask]
    filtered_amps = amplitudes[mask]
    idx = np.argmax(filtered_amps)
    return float(filtered_freqs[idx])


def compute_frequency_features(
    voltage: np.ndarray,
    sampling_rate: int,
) -> dict:
    """
    計算所有頻域特徵，回傳字典。

    Returns
    -------
    dict with keys:
        dominant_frequency_hz, freqs, amplitudes
    """
    freqs, amplitudes = compute_fft(voltage, sampling_rate)
    dominant_freq = find_dominant_frequency(freqs, amplitudes)

    return {
        "dominant_frequency_hz": dominant_freq,
        "freqs":                 freqs,
        "amplitudes":            amplitudes,
    }


def compute_all_features(
    voltage: np.ndarray,
    sampling_rate: int,
) -> dict:
    """
    計算全部時域 + 頻域特徵，回傳合併字典。

    Returns
    -------
    dict 包含 mean, rms, peak_to_peak, peak,
                crest_factor, dominant_frequency_hz,
                freqs, amplitudes
    """
    time_feats = compute_time_features(voltage)
    freq_feats = compute_frequency_features(voltage, sampling_rate)
    return {**time_feats, **freq_feats}
