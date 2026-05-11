"""
gui.py

NI myDAQ 振動監控 GUI 主程式。

功能：
  - 選擇 mock / nidaq mode
  - 設定 channel、取樣率、duration
  - 顯示即時波形（使用 pyqtgraph）
  - 顯示 RMS 與 Peak-to-Peak 數值
  - 可儲存 CSV

使用方式：
    python -m mydaq_vibration.gui
  或：
    python src/mydaq_vibration/gui.py
"""

import sys
import os
import csv
import time
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QGroupBox,
    QFileDialog, QStatusBar, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QFont, QPalette, QColor

import pyqtgraph as pg

# ─────────────────────────────────────────────
# 確保 src 在 import path 上（開發用）
# ─────────────────────────────────────────────
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mydaq_vibration.mock_signal import generate_mock_signal
from mydaq_vibration.analysis import compute_rms, compute_peak_to_peak, compute_fft, find_dominant_frequency


# ─────────────────────────────────────────────
# 樣式設定
# ─────────────────────────────────────────────

DARK_BG      = "#0f0f1a"
PANEL_BG     = "#1a1a2e"
ACCENT_BLUE  = "#00d4ff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED   = "#ff4d6d"
ACCENT_GOLD  = "#ffd700"
TEXT_PRIMARY = "#e0e0ff"
TEXT_DIM     = "#8888aa"
BORDER_COLOR = "#2a2a5a"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', Arial, sans-serif;
}}
QGroupBox {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    margin-top: 8px;
    padding: 8px;
    font-weight: bold;
    color: {ACCENT_BLUE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QLabel {{
    color: {TEXT_PRIMARY};
}}
QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT_BLUE};
}}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 4px 8px;
    color: {TEXT_PRIMARY};
}}
QComboBox:focus {{
    border: 1px solid {ACCENT_BLUE};
}}
QComboBox::drop-down {{
    border: none;
}}
QPushButton {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 16px;
    color: {TEXT_PRIMARY};
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #252545;
    border-color: {ACCENT_BLUE};
}}
QPushButton:pressed {{
    background-color: #1a1a3a;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    border-color: #222240;
}}
QPushButton#btn_start {{
    background-color: #0a3a2a;
    border-color: {ACCENT_GREEN};
    color: {ACCENT_GREEN};
}}
QPushButton#btn_start:hover {{
    background-color: #0d4a35;
}}
QPushButton#btn_stop {{
    background-color: #3a0a15;
    border-color: {ACCENT_RED};
    color: {ACCENT_RED};
}}
QPushButton#btn_stop:hover {{
    background-color: #4a0d1e;
}}
QPushButton#btn_save {{
    background-color: #1a1a00;
    border-color: {ACCENT_GOLD};
    color: {ACCENT_GOLD};
}}
QPushButton#btn_save:hover {{
    background-color: #252500;
}}
QStatusBar {{
    background-color: {PANEL_BG};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER_COLOR};
}}
"""


# ─────────────────────────────────────────────
# 資料採集執行緒（非阻塞）
# ─────────────────────────────────────────────

class AcquisitionThread(QThread):
    """在背景執行緒中持續產生訊號資料。"""

    new_data = Signal(np.ndarray, np.ndarray)  # time_array, voltage_array
    error_occurred = Signal(str)

    def __init__(
        self,
        mode: str,
        channel: str,
        sampling_rate: int,
        chunk_duration: float,
        motor_on: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.mode = mode
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.motor_on = motor_on
        self._running = False

    def run(self):
        self._running = True
        chunk_count = 0

        if self.mode == "nidaq":
            try:
                import nidaqmx
            except ImportError:
                self.error_occurred.emit(
                    "找不到 nidaqmx 套件！\n請切換到 mock mode 或在此電腦安裝 NI-DAQmx。"
                )
                return

        while self._running:
            try:
                if self.mode == "mock":
                    t, v = generate_mock_signal(
                        sampling_rate=self.sampling_rate,
                        duration=self.chunk_duration,
                        motor_on=self.motor_on,
                        seed=None,
                    )
                    self.new_data.emit(t, v)
                    time.sleep(self.chunk_duration * 0.9)

                else:  # nidaq
                    from mydaq_vibration.acquisition import acquire_signal
                    t, v = acquire_signal(
                        channel=self.channel,
                        sampling_rate=self.sampling_rate,
                        duration=self.chunk_duration,
                    )
                    self.new_data.emit(t, v)

            except Exception as e:
                self.error_occurred.emit(str(e))
                break

            chunk_count += 1

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
# 主視窗
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    """主視窗：NI myDAQ 振動監控 GUI。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NI myDAQ Vibration Monitor — EPIM Course")
        self.setMinimumSize(1100, 700)

        self._acq_thread: AcquisitionThread | None = None
        self._last_time: np.ndarray | None = None
        self._last_voltage: np.ndarray | None = None
        self._buffer_time: list[np.ndarray] = []
        self._buffer_voltage: list[np.ndarray] = []
        self._buffer_max_chunks = 10  # 保留最近 N 個 chunk

        self._build_ui()
        self._apply_pyqtgraph_style()

    # ── UI 建構 ──────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # 左側：控制面板
        control_panel = self._build_control_panel()
        root_layout.addWidget(control_panel, stretch=0)

        # 右側：圖表區
        plot_area = self._build_plot_area()
        root_layout.addWidget(plot_area, stretch=1)

        # 狀態列
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就緒。請選擇模式並按下 Start Live View。")

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── 模式選擇 ──
        mode_group = QGroupBox("模式設定 / Mode")
        mode_layout = QVBoxLayout(mode_group)

        mode_layout.addWidget(QLabel("擷取模式 (Mode):"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["mock (模擬)", "nidaq (實機)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.combo_mode)

        mode_layout.addWidget(QLabel("馬達狀態 (Motor):"))
        self.combo_motor = QComboBox()
        self.combo_motor.addItems(["motor_off (關閉)", "motor_on (啟動)"])
        mode_layout.addWidget(self.combo_motor)

        layout.addWidget(mode_group)

        # ── 參數設定 ──
        param_group = QGroupBox("採集參數 / Parameters")
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("通道 (Channel):"))
        self.edit_channel = QLineEdit("Dev1/ai0")
        self.edit_channel.setEnabled(False)  # mock 模式下禁用
        param_layout.addWidget(self.edit_channel)

        param_layout.addWidget(QLabel("取樣率 (Sampling Rate, Hz):"))
        self.edit_fs = QLineEdit("10000")
        param_layout.addWidget(self.edit_fs)

        param_layout.addWidget(QLabel("每次擷取時間 (Chunk, s):"))
        self.edit_duration = QLineEdit("1.0")
        param_layout.addWidget(self.edit_duration)

        layout.addWidget(param_group)

        # ── 實驗標籤 ──
        label_group = QGroupBox("實驗標籤 / Condition")
        label_layout = QVBoxLayout(label_group)

        label_layout.addWidget(QLabel("量測條件:"))
        self.combo_condition = QComboBox()
        self.combo_condition.addItems([
            "motor_off",
            "motor_on",
            "background_noise",
            "installation_test",
        ])
        label_layout.addWidget(self.combo_condition)

        layout.addWidget(label_group)

        # ── 按鈕 ──
        btn_group = QGroupBox("控制 / Control")
        btn_layout = QVBoxLayout(btn_group)

        self.btn_start = QPushButton("▶  Start Live View")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_stop)

        self.btn_save = QPushButton("💾  Save CSV")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save_csv)
        btn_layout.addWidget(self.btn_save)

        layout.addWidget(btn_group)

        # ── 統計數值 ──
        stat_group = QGroupBox("即時統計 / Live Stats")
        stat_layout = QVBoxLayout(stat_group)

        self.lbl_rms_val = self._make_stat_label("-- V")
        self._add_stat_row(stat_layout, "RMS:", self.lbl_rms_val, ACCENT_GREEN)

        self.lbl_pp_val = self._make_stat_label("-- V")
        self._add_stat_row(stat_layout, "Peak-to-Peak:", self.lbl_pp_val, ACCENT_BLUE)

        self.lbl_dominant_val = self._make_stat_label("-- Hz")
        self._add_stat_row(stat_layout, "Dominant Freq:", self.lbl_dominant_val, ACCENT_GOLD)

        layout.addWidget(stat_group)
        layout.addStretch()

        return panel

    def _make_stat_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Consolas", 13, QFont.Bold))
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lbl

    def _add_stat_row(self, layout: QVBoxLayout, key: str, val_label: QLabel, color: str):
        row = QHBoxLayout()
        key_lbl = QLabel(key)
        key_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        val_label.setStyleSheet(f"color: {color};")
        row.addWidget(key_lbl)
        row.addStretch()
        row.addWidget(val_label)
        layout.addLayout(row)

    def _build_plot_area(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 時域圖
        self.plot_time = pg.PlotWidget(title="Time Domain — Voltage (V)")
        self.plot_time.setLabel("left", "Voltage", units="V")
        self.plot_time.setLabel("bottom", "Time", units="s")
        self.curve_time = self.plot_time.plot(pen=pg.mkPen(color="#00d4ff", width=1))
        layout.addWidget(self.plot_time, stretch=1)

        # FFT 圖
        self.plot_fft = pg.PlotWidget(title="FFT Spectrum — Amplitude (V)")
        self.plot_fft.setLabel("left", "Amplitude", units="V")
        self.plot_fft.setLabel("bottom", "Frequency", units="Hz")
        self.plot_fft.setXRange(0, 500)
        self.curve_fft = self.plot_fft.plot(pen=pg.mkPen(color="#ff6b6b", width=1))
        layout.addWidget(self.plot_fft, stretch=1)

        return widget

    def _apply_pyqtgraph_style(self):
        pg.setConfigOption("background", PANEL_BG)
        pg.setConfigOption("foreground", TEXT_PRIMARY)

        for plot in [self.plot_time, self.plot_fft]:
            plot.getPlotItem().getAxis("left").setPen(pg.mkPen(color=BORDER_COLOR))
            plot.getPlotItem().getAxis("bottom").setPen(pg.mkPen(color=BORDER_COLOR))
            plot.showGrid(x=True, y=True, alpha=0.3)

    # ── 事件處理 ──────────────────────────────

    def _on_mode_changed(self, index: int):
        is_nidaq = (index == 1)
        self.edit_channel.setEnabled(is_nidaq)
        if is_nidaq:
            self.status_bar.showMessage("nidaq 模式：請確認 NI myDAQ 已連接並在 NI MAX 中可見。")
        else:
            self.status_bar.showMessage("mock 模式：使用模擬訊號，無需硬體。")

    def _on_start(self):
        mode = "mock" if self.combo_mode.currentIndex() == 0 else "nidaq"
        motor_on = "motor_on" in self.combo_motor.currentText()
        channel = self.edit_channel.text().strip()

        try:
            fs = int(self.edit_fs.text())
            duration = float(self.edit_duration.text())
        except ValueError:
            self.status_bar.showMessage("❌ 取樣率或 chunk 時間格式錯誤，請輸入數字。")
            return

        # 清除緩衝
        self._buffer_time.clear()
        self._buffer_voltage.clear()

        # 建立採集執行緒
        self._acq_thread = AcquisitionThread(
            mode=mode,
            channel=channel,
            sampling_rate=fs,
            chunk_duration=duration,
            motor_on=motor_on,
        )
        self._acq_thread.new_data.connect(self._on_new_data)
        self._acq_thread.error_occurred.connect(self._on_error)
        self._acq_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.status_bar.showMessage(f"✅ Live view 已啟動 | 模式: {mode} | fs: {fs} Hz | chunk: {duration} s")

    def _on_stop(self):
        if self._acq_thread:
            self._acq_thread.stop()
            self._acq_thread.wait()
            self._acq_thread = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # 有資料才允許儲存
        if self._buffer_voltage:
            self.btn_save.setEnabled(True)
            self.status_bar.showMessage("⏹ 已停止。可按 Save CSV 儲存當前緩衝資料。")
        else:
            self.status_bar.showMessage("⏹ 已停止。無資料可儲存。")

    def _on_new_data(self, time_array: np.ndarray, voltage_array: np.ndarray):
        """接收新資料，更新圖表與統計數值。"""
        self._last_time = time_array
        self._last_voltage = voltage_array

        # 存入緩衝（保留最近 N 個 chunk）
        self._buffer_time.append(time_array)
        self._buffer_voltage.append(voltage_array)
        if len(self._buffer_time) > self._buffer_max_chunks:
            self._buffer_time.pop(0)
            self._buffer_voltage.pop(0)

        # 合併最近資料顯示
        all_v = np.concatenate(self._buffer_voltage)
        n_total = len(all_v)
        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        all_t = np.linspace(0, n_total / fs, n_total, endpoint=False)

        # 更新時域圖
        self.curve_time.setData(all_t, all_v)

        # 計算 FFT
        freqs, amps = compute_fft(all_v, fs)
        self.curve_fft.setData(freqs, amps)

        # 更新統計
        rms = compute_rms(all_v)
        pp = compute_peak_to_peak(all_v)
        dominant = find_dominant_frequency(freqs, amps)

        self.lbl_rms_val.setText(f"{rms:.4f} V")
        self.lbl_pp_val.setText(f"{pp:.4f} V")
        self.lbl_dominant_val.setText(f"{dominant:.1f} Hz")

    def _on_error(self, msg: str):
        self.status_bar.showMessage(f"❌ 錯誤：{msg}")
        self._on_stop()

    def _on_save_csv(self):
        if not self._buffer_voltage:
            self.status_bar.showMessage("❌ 沒有資料可儲存。")
            return

        condition = self.combo_condition.currentText()
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{now_str}_{condition}.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 CSV",
            os.path.join("data", "raw", condition, default_name),
            "CSV Files (*.csv)",
        )
        if not filepath:
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        all_v = np.concatenate(self._buffer_voltage)
        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        all_t = np.linspace(0, len(all_v) / fs, len(all_v), endpoint=False)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "voltage_v"])
            for t, v in zip(all_t, all_v):
                writer.writerow([f"{t:.6f}", f"{v:.6f}"])

        self.status_bar.showMessage(f"✅ 資料已儲存：{filepath} ({len(all_v)} 筆)")

    def closeEvent(self, event):
        if self._acq_thread:
            self._acq_thread.stop()
            self._acq_thread.wait()
        event.accept()


# ─────────────────────────────────────────────
# 進入點
# ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setApplicationName("NI myDAQ Vibration Monitor")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
