"""
plotting.py

圖表繪製模組。

使用 matplotlib 繪製：
  - 時域波形圖
  - FFT 頻譜圖
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


# ─────────────────────────────────────────────
# 預設樣式
# ─────────────────────────────────────────────

PLOT_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#4a4a8a",
    "axes.labelcolor":  "#e0e0ff",
    "xtick.color":      "#aaaacc",
    "ytick.color":      "#aaaacc",
    "text.color":       "#e0e0ff",
    "grid.color":       "#2a2a5a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  1.0,
}


def plot_time_domain(
    time_array: np.ndarray,
    voltage_array: np.ndarray,
    title: str = "Time Domain Signal",
    label: str = "Voltage (V)",
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    繪製時域波形圖。

    Parameters
    ----------
    time_array : np.ndarray
        時間軸 (秒)。
    voltage_array : np.ndarray
        電壓訊號 (V)。
    title : str
        圖表標題。
    label : str
        圖例標籤。
    save_path : str or None
        若指定，儲存圖片到此路徑。
    show : bool
        是否立即顯示圖表。
    ax : plt.Axes or None
        若提供，繪製到此 Axes；否則新建 Figure。

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.plot(time_array, voltage_array, color="#00d4ff", linewidth=0.8, label=label, alpha=0.9)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("Voltage (V)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#ffffff")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[plotting] 時域圖已儲存：{save_path}")

        if show:
            plt.show()

        return fig


def plot_fft(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    title: str = "FFT Spectrum",
    max_freq: float | None = None,
    dominant_freq: float | None = None,
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    繪製 FFT 頻譜圖。

    Parameters
    ----------
    freqs : np.ndarray
        頻率陣列 (Hz)。
    amplitudes : np.ndarray
        對應振幅陣列 (V)。
    title : str
        圖表標題。
    max_freq : float or None
        顯示的最大頻率 (Hz)，None 表示顯示全部。
    dominant_freq : float or None
        若提供，在圖上標記主頻。
    save_path : str or None
        若指定，儲存圖片到此路徑。
    show : bool
        是否立即顯示圖表。
    ax : plt.Axes or None
        若提供，繪製到此 Axes；否則新建 Figure。

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(PLOT_STYLE):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        if max_freq is not None:
            mask = freqs <= max_freq
            plot_freqs = freqs[mask]
            plot_amps = amplitudes[mask]
        else:
            plot_freqs = freqs
            plot_amps = amplitudes

        ax.plot(plot_freqs, plot_amps, color="#ff6b6b", linewidth=0.9, alpha=0.9)
        ax.fill_between(plot_freqs, plot_amps, alpha=0.2, color="#ff6b6b")

        if dominant_freq is not None:
            # 找主頻對應振幅
            idx = np.argmin(np.abs(plot_freqs - dominant_freq))
            dom_amp = plot_amps[idx]
            ax.axvline(x=dominant_freq, color="#ffd700", linewidth=1.2, linestyle="--", alpha=0.8)
            ax.annotate(
                f"  {dominant_freq:.1f} Hz",
                xy=(dominant_freq, dom_amp),
                color="#ffd700",
                fontsize=9,
            )

        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Amplitude (V)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#ffffff")
        ax.grid(True)

        fig.tight_layout()

        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[plotting] FFT 圖已儲存：{save_path}")

        if show:
            plt.show()

        return fig


def plot_combined(
    time_array: np.ndarray,
    voltage_array: np.ndarray,
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    title: str = "Vibration Analysis",
    max_freq: float = 500.0,
    dominant_freq: float | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    同時繪製時域圖與 FFT 頻譜圖（上下排列）。

    Parameters
    ----------
    time_array, voltage_array : 時域資料
    freqs, amplitudes : FFT 資料
    title : 整體圖表標題
    max_freq : FFT 顯示最大頻率
    dominant_freq : 主頻標記
    save_path : 儲存路徑
    show : 是否顯示

    Returns
    -------
    plt.Figure
    """
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
        fig.suptitle(title, fontsize=15, fontweight="bold", color="#ffffff", y=1.01)

        plot_time_domain(
            time_array, voltage_array,
            title="Time Domain",
            show=False, ax=ax1,
        )
        plot_fft(
            freqs, amplitudes,
            title="FFT Spectrum",
            max_freq=max_freq,
            dominant_freq=dominant_freq,
            show=False, ax=ax2,
        )

        fig.tight_layout()

        if save_path:
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"[plotting] 合併圖已儲存：{save_path}")

        if show:
            plt.show()

        return fig
