"""
mock_signal.py

產生模擬振動訊號，供沒有 NI myDAQ 的電腦開發與測試使用。

模擬訊號組成：
  - 基頻 (預設 60 Hz) 正弦波
  - 二次諧波 (120 Hz)
  - 白噪聲
  - motor_on 時振幅較大
"""

import numpy as np


def generate_mock_signal(
    sampling_rate: int = 10000,
    duration: float = 5.0,
    motor_on: bool = False,
    fundamental_hz: float = 60.0,
    noise_amplitude: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    產生模擬振動訊號。

    Parameters
    ----------
    sampling_rate : int
        取樣率 (Hz)。
    duration : float
        訊號持續時間 (秒)。
    motor_on : bool
        True 表示馬達啟動狀態（振幅較大）。
        False 表示馬達關閉狀態（低振幅）。
    fundamental_hz : float
        基頻 (Hz)，預設為 60 Hz（模擬電源頻率相關振動）。
    noise_amplitude : float
        白噪聲振幅 (V)。
    seed : int or None
        隨機種子，設定後可重現相同訊號；None 表示每次隨機。

    Returns
    -------
    time_array : np.ndarray
        時間軸陣列 (秒)。
    voltage_array : np.ndarray
        模擬電壓訊號 (V)。

    Examples
    --------
    >>> t, v = generate_mock_signal(sampling_rate=10000, duration=2.0, motor_on=True)
    >>> print(t.shape, v.shape)
    (20000,) (20000,)
    """
    rng = np.random.default_rng(seed)

    n_samples = int(sampling_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    if motor_on:
        # 馬達啟動：基頻振幅較大，並有明顯諧波
        amp_fundamental = 1.0    # V，基頻 60 Hz
        amp_harmonic2 = 0.4      # V，二次諧波 120 Hz
        amp_harmonic3 = 0.15     # V，三次諧波 180 Hz
        dc_offset = 0.02         # 輕微 DC 偏移
    else:
        # 馬達關閉：低振幅，以噪聲為主
        amp_fundamental = 0.05   # V，殘留 60 Hz（環境電磁干擾）
        amp_harmonic2 = 0.02     # V
        amp_harmonic3 = 0.005    # V
        dc_offset = 0.0

    # 基頻分量
    signal = amp_fundamental * np.sin(2 * np.pi * fundamental_hz * t)
    # 二次諧波
    signal += amp_harmonic2 * np.sin(2 * np.pi * 2 * fundamental_hz * t + np.pi / 6)
    # 三次諧波
    signal += amp_harmonic3 * np.sin(2 * np.pi * 3 * fundamental_hz * t + np.pi / 3)
    # DC 偏移
    signal += dc_offset
    # 白噪聲
    signal += rng.normal(0, noise_amplitude, n_samples)

    return t, signal


def generate_transient_signal(
    sampling_rate: int = 10000,
    duration: float = 5.0,
    transient_time: float = 2.0,
    transient_amplitude: float = 2.0,
    transient_duration: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    產生包含暫態衝擊（transient）的模擬訊號。

    Parameters
    ----------
    sampling_rate : int
        取樣率 (Hz)。
    duration : float
        訊號持續時間 (秒)。
    transient_time : float
        暫態衝擊發生的時間點 (秒)。
    transient_amplitude : float
        衝擊振幅 (V)。
    transient_duration : float
        衝擊持續時間 (秒)。
    seed : int or None
        隨機種子。

    Returns
    -------
    time_array : np.ndarray
        時間軸陣列 (秒)。
    voltage_array : np.ndarray
        含暫態衝擊的模擬電壓訊號 (V)。
    """
    t, signal = generate_mock_signal(
        sampling_rate=sampling_rate,
        duration=duration,
        motor_on=True,
        seed=seed,
    )

    # 加入指數衰減衝擊
    n_samples = len(t)
    transient = np.zeros(n_samples)
    decay_rate = 50.0  # 衰減速率

    for i, ti in enumerate(t):
        dt = ti - transient_time
        if 0 <= dt <= transient_duration:
            transient[i] = transient_amplitude * np.exp(-decay_rate * dt)

    signal += transient
    return t, signal
