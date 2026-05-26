# -*- coding: utf-8 -*-
"""
gui.py  — NI myDAQ Vibration Monitor
色系：信紙米白  |  側邊選單頁面切換
"""

import sys, os, csv, time, re
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QGroupBox,
    QFileDialog, QStatusBar, QStackedWidget, QSizePolicy,
    QSpacerItem, QFrame, QScrollArea,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor
import pyqtgraph as pg

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mydaq_vibration.mock_signal import generate_mock_signal
from mydaq_vibration.acquisition import (
    MultiChannelSignalReader,
    normalize_channels,
    sanitize_channel_label,
    save_multi_csv,
)
from mydaq_vibration.analysis import (
    compute_rms, compute_peak_to_peak, compute_fft, find_dominant_frequency,
    compute_time_features,
)
from mydaq_vibration.plotting import plot_combined

# ── 色彩 ──────────────────────────────────────
BG        = "#FAF6F0"   # 信紙米白背景
PANEL     = "#FFF9F2"   # 面板
SIDEBAR   = "#EDE4D8"   # 側邊欄
BORDER    = "#D4C5A9"   # 邊框
TEXT      = "#2C2010"   # 主要文字（深棕）
TEXT_DIM  = "#8A7A6A"   # 次要文字
ACCENT    = "#4A8C8C"   # 主色調（藍綠）
ACCENT_H  = "#3A7070"   # Hover
SUCCESS   = "#5A8A5A"   # 綠色
DANGER    = "#A04040"   # 紅色
GOLD      = "#8A7020"   # 金棕

SS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI', Arial; font-size: 12px; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 5px; margin-top: 8px;
    padding: 6px 6px 5px 6px; color: {ACCENT}; font-weight: bold;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; background: {PANEL}; }}
QLineEdit {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 8px; color: {TEXT};
}}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 5px 8px; color: {TEXT};
}}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {TEXT}; selection-background-color: {ACCENT}; }}
QPushButton {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 5px 12px; color: {TEXT}; font-weight: bold;
}}
QPushButton:hover {{ background: {SIDEBAR}; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#btn_start {{ border-color: {SUCCESS}; color: {SUCCESS}; background: #F0F7F0; }}
QPushButton#btn_start:hover {{ background: #E0F0E0; }}
QPushButton#btn_stop {{ border-color: {DANGER}; color: {DANGER}; background: #F7F0F0; }}
QPushButton#btn_stop:hover {{ background: #F0E0E0; }}
QPushButton#btn_save {{ border-color: {GOLD}; color: {GOLD}; background: #F7F5E8; }}
QPushButton#btn_save:hover {{ background: #F0EDD8; }}
QPushButton#nav_btn {{
    background: transparent; border: none; border-radius: 6px;
    padding: 12px 16px; color: {TEXT_DIM}; font-size: 14px; text-align: left;
}}
QPushButton#nav_btn:hover {{ background: {BORDER}; color: {TEXT}; }}
QPushButton#nav_btn_active {{
    background: {ACCENT}; border: none; border-radius: 6px;
    padding: 12px 16px; color: white; font-size: 14px; text-align: left; font-weight: bold;
}}
QStatusBar {{ background: {SIDEBAR}; color: {TEXT_DIM}; border-top: 1px solid {BORDER}; }}
QLabel#stat_val {{ font-family: Consolas; font-size: 16px; font-weight: bold; }}
"""

STATUS_NORMAL_SS = f"background:{SIDEBAR}; color:{TEXT_DIM}; border-top:1px solid {BORDER};"
STATUS_RECORDING_SS = f"background:{DANGER}; color:white; border-top:1px solid {DANGER}; font-weight:bold;"


def make_status_callback(callback):
    """建立可接受錄製狀態參數的狀態列 callback。"""
    raw_callback = callback or (lambda *args: None)

    def wrapped(message: str, recording: bool | None = None):
        try:
            raw_callback(message, recording)
        except TypeError:
            raw_callback(message)

    return wrapped


# ── 採集執行緒 ─────────────────────────────────

class AcquisitionThread(QThread):
    new_data = Signal(np.ndarray, object)
    error_occurred = Signal(str)

    def __init__(self, mode, channels, fs, chunk_dur, motor_on, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.channels = normalize_channels(channels)
        self.fs, self.chunk_dur, self.motor_on = fs, chunk_dur, motor_on
        self._running = False

    def run(self):
        self._running = True
        if self.mode == "nidaq":
            try:
                import nidaqmx  # noqa
            except ImportError:
                self.error_occurred.emit("找不到 nidaqmx！請切換 mock 模式。")
                return
        try:
            if self.mode == "mock":
                self._run_mock()
            else:
                self._run_nidaq()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_mock(self):
        channel_labels = [sanitize_channel_label(ch) for ch in self.channels]
        while self._running:
            voltage_by_channel = {}
            time_array = None
            for idx, label in enumerate(channel_labels):
                t, v = generate_mock_signal(
                    self.fs,
                    self.chunk_dur,
                    self.motor_on,
                    fundamental_hz=60.0 + idx * 8.0,
                    noise_amplitude=0.05 + idx * 0.01,
                )
                time_array = t
                voltage_by_channel[label] = v
            self.new_data.emit(time_array, voltage_by_channel)
            time.sleep(self.chunk_dur * 0.9)

    def _run_nidaq(self):
        with MultiChannelSignalReader(
            self.channels,
            sampling_rate=self.fs,
            chunk_duration=self.chunk_dur,
        ) as reader:
            n_samples = int(self.fs * self.chunk_dur)
            t = np.linspace(0, self.chunk_dur, n_samples, endpoint=False)
            while self._running:
                voltage_by_channel = reader.read_chunk()
                self.new_data.emit(t, voltage_by_channel)

    def stop(self):
        self._running = False


# ── 頁面 1：即時監控 ───────────────────────────

class LivePage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self._status = make_status_callback(status_callback)
        self._buf_t = []
        self._buf_v_by_channel = {}
        self._record_v_by_channel = {}
        self._channel_order = []
        self._total_samples = 0   # 追蹤總取樣點數
        self._MAX = 10
        self._thread = None
        self.curves_t = {}
        self.curves_f = {}
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左控制欄（放入 ScrollArea，避免內容被截斷）
        ctrl_inner = QWidget()
        ctrl_inner.setStyleSheet(f"background:{PANEL};")
        cl = QVBoxLayout(ctrl_inner)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        # 模式
        g1 = QGroupBox("模式 / Mode")
        gl1 = QVBoxLayout(g1); gl1.setSpacing(3)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["mock (模擬)", "nidaq (實機)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode)
        self.combo_motor = QComboBox()
        self.combo_motor.addItems(["motor_off", "motor_on"])
        gl1.addWidget(QLabel("擷取模式")); gl1.addWidget(self.combo_mode)
        gl1.addWidget(QLabel("馬達狀態")); gl1.addWidget(self.combo_motor)
        cl.addWidget(g1)

        # 參數
        g2 = QGroupBox("採集參數")
        gl2 = QVBoxLayout(g2); gl2.setSpacing(3)
        self.edit_ch  = QLineEdit("Dev1/ai0, Dev1/ai1"); self.edit_ch.setEnabled(False)
        self.edit_fs  = QLineEdit("10000")
        self.edit_dur = QLineEdit("1.0")
        gl2.addWidget(QLabel("通道 (逗號分隔)")); gl2.addWidget(self.edit_ch)
        gl2.addWidget(QLabel("取樣率 (Hz)"));   gl2.addWidget(self.edit_fs)
        gl2.addWidget(QLabel("Chunk 時間 (s)")); gl2.addWidget(self.edit_dur)
        cl.addWidget(g2)

        # 軸範圍
        g_yaxis = QGroupBox("圖表軸範圍")
        gyl = QVBoxLayout(g_yaxis); gyl.setSpacing(3)
        # 時域圖
        gyl.addWidget(QLabel("時域 Voltage (V):"))
        row_td = QHBoxLayout()
        self.edit_td_ymin = QLineEdit("auto"); self.edit_td_ymin.setFixedWidth(74)
        self.edit_td_ymax = QLineEdit("auto"); self.edit_td_ymax.setFixedWidth(74)
        row_td.addWidget(QLabel("Min")); row_td.addWidget(self.edit_td_ymin)
        row_td.addWidget(QLabel("Max")); row_td.addWidget(self.edit_td_ymax)
        gyl.addLayout(row_td)
        # FFT Y 軸
        gyl.addWidget(QLabel("FFT Acceleration (g):"))
        row_fft = QHBoxLayout()
        self.edit_fft_ymin = QLineEdit("auto"); self.edit_fft_ymin.setFixedWidth(74)
        self.edit_fft_ymax = QLineEdit("auto"); self.edit_fft_ymax.setFixedWidth(74)
        row_fft.addWidget(QLabel("Min")); row_fft.addWidget(self.edit_fft_ymin)
        row_fft.addWidget(QLabel("Max")); row_fft.addWidget(self.edit_fft_ymax)
        gyl.addLayout(row_fft)

        # FFT X 軸
        gyl.addWidget(QLabel("FFT Frequency (Hz):"))
        row_fft_x = QHBoxLayout()
        self.edit_fft_xmin = QLineEdit("0"); self.edit_fft_xmin.setFixedWidth(74)
        self.edit_fft_xmax = QLineEdit("500"); self.edit_fft_xmax.setFixedWidth(74)
        row_fft_x.addWidget(QLabel("Min")); row_fft_x.addWidget(self.edit_fft_xmin)
        row_fft_x.addWidget(QLabel("Max")); row_fft_x.addWidget(self.edit_fft_xmax)
        gyl.addLayout(row_fft_x)

        hint = QLabel("Min/Max 都輸入數字可固定範圍；\n輸入 auto 則自動縮放。")
        hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        hint.setWordWrap(True)
        gyl.addWidget(hint)
        cl.addWidget(g_yaxis)

        # 條件
        g3 = QGroupBox("實驗條件")
        gl3 = QVBoxLayout(g3); gl3.setSpacing(3)
        self.combo_cond = QComboBox()
        self.combo_cond.addItems(["motor_off","motor_on","background_noise","installation_test"])
        gl3.addWidget(self.combo_cond)
        cl.addWidget(g3)

        # 按鈕
        g4 = QGroupBox("控制")
        gl4 = QVBoxLayout(g4); gl4.setSpacing(4)
        self.btn_start = QPushButton("▶  Start Live View"); self.btn_start.setObjectName("btn_start")
        self.btn_stop  = QPushButton("■  Stop");            self.btn_stop.setObjectName("btn_stop")
        self.btn_save  = QPushButton("  Save CSV");         self.btn_save.setObjectName("btn_save")
        self.btn_stop.setEnabled(False); self.btn_save.setEnabled(False)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_save.clicked.connect(self._save)
        for b in [self.btn_start, self.btn_stop, self.btn_save]:
            gl4.addWidget(b)
        cl.addWidget(g4)

        # 統計
        g5 = QGroupBox("即時統計")
        gl5 = QVBoxLayout(g5); gl5.setSpacing(3)
        self.lbl_rms  = self._stat_row(gl5, "RMS (V)",        SUCCESS)
        self.lbl_pp   = self._stat_row(gl5, "Peak-to-Peak (V)", ACCENT)
        self.lbl_dom  = self._stat_row(gl5, "Dominant Freq",   GOLD)
        # 已記錄時間
        dur_row = QHBoxLayout()
        dur_k = QLabel("已記錄:"); dur_k.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        self.lbl_recorded = QLabel("0.0 s")
        self.lbl_recorded.setStyleSheet(
            f"color:{TEXT}; font-family:Consolas; font-size:12px; font-weight:bold;")
        self.lbl_recorded.setAlignment(Qt.AlignRight)
        dur_row.addWidget(dur_k); dur_row.addStretch(); dur_row.addWidget(self.lbl_recorded)
        gl5.addLayout(dur_row)
        cl.addWidget(g5)
        cl.addStretch()

        # 用 QScrollArea 包住左欄，支援捲動
        scroll = QScrollArea()
        scroll.setWidget(ctrl_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-right: 1px solid {BORDER}; background: {PANEL}; }}"
            f"QScrollBar:vertical {{ width: 6px; background: {SIDEBAR}; }}"
            f"QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}"
        )
        root.addWidget(scroll)

        # 右側圖表
        pg.setConfigOption("background", PANEL)
        pg.setConfigOption("foreground", TEXT)
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(12)

        self.plot_t = pg.PlotWidget(title="Time Domain")
        self.plot_t.setLabel("left", "Voltage", units="V")
        self.plot_t.setLabel("bottom", "Time", units="s")
        self.plot_t.showGrid(x=True, y=True, alpha=0.4)
        self.plot_t.getPlotItem().titleLabel.setText("Time Domain",
            color=TEXT, size="13pt")
        self.plot_t.addLegend(offset=(-12, 12))
        pl.addWidget(self.plot_t)

        self.plot_f = pg.PlotWidget(title="FFT Spectrum (Acceleration)")
        self.plot_f.setLabel("left", "Amplitude", units="g")
        self.plot_f.setLabel("bottom", "Frequency", units="Hz")
        self.plot_f.setXRange(0, 500)
        self.plot_f.showGrid(x=True, y=True, alpha=0.4)
        self.plot_f.addLegend(offset=(-12, 12))
        pl.addWidget(self.plot_f)
        root.addWidget(plots, stretch=1)
        self._reset_curves(["Dev1_ai0", "Dev1_ai1"])

    def _stat_row(self, layout, label, color):
        row = QHBoxLayout()
        k = QLabel(label + ":"); k.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        v = QLabel("--")
        v.setStyleSheet(f"color:{color}; font-family:Consolas; font-size:13px; font-weight:bold;")
        v.setAlignment(Qt.AlignRight)
        row.addWidget(k); row.addStretch(); row.addWidget(v)
        layout.addLayout(row)
        return v

    def _on_mode(self, idx):
        self.edit_ch.setEnabled(idx == 1 and self.btn_start.isEnabled())

    def _start(self):
        try:
            fs  = int(self.edit_fs.text())
            dur = float(self.edit_dur.text())
            channels = normalize_channels(self.edit_ch.text())
        except ValueError:
            self._status("[錯誤] 取樣率、Chunk 時間或通道格式不正確。")
            return
        self._channel_order = [sanitize_channel_label(ch) for ch in channels]
        self._buf_t.clear()
        self._buf_v_by_channel = {label: [] for label in self._channel_order}
        self._record_v_by_channel = {label: [] for label in self._channel_order}
        self._total_samples = 0
        self.lbl_recorded.setText("0.0 s")
        self._reset_curves(self._channel_order)
        mode = "mock" if self.combo_mode.currentIndex() == 0 else "nidaq"
        self._thread = AcquisitionThread(
            mode,
            channels, fs, dur,
            "motor_on" in self.combo_motor.currentText(),
        )
        self._thread.new_data.connect(self._on_data)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._status(
            f"[Live] 擷取中 | 模式: {mode} | channels: {', '.join(self._channel_order)} | "
            f"fs: {fs} Hz | chunk: {dur} s — 按 Stop 結束",
            True,
        )

    def _stop(self):
        if self._thread:
            self._thread.stop(); self._thread.wait(); self._thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        has_data = any(self._record_v_by_channel.values())
        self.btn_save.setEnabled(has_data)
        if has_data:
            self._status("[停止] 擷取已停止。緩衝區有資料，可按 [Save CSV] 儲存。", False)
        else:
            self._status("[停止] 擷取已停止，無資料可儲存。", False)

    def _on_data(self, t, voltage_by_channel):
        # 累計總取樣點數（不受 buffer 限制）
        if not voltage_by_channel:
            return

        first_voltage = next(iter(voltage_by_channel.values()))
        self._total_samples += len(first_voltage)
        self._buf_t.append(t)

        for label, voltage in voltage_by_channel.items():
            if label not in self._buf_v_by_channel:
                self._buf_v_by_channel[label] = []
                self._record_v_by_channel[label] = []
                self._channel_order.append(label)
                self._reset_curves(self._channel_order)
            self._buf_v_by_channel[label].append(voltage)
            self._record_v_by_channel[label].append(np.asarray(voltage, dtype=float).copy())

        if len(self._buf_t) > self._MAX:
            self._buf_t.pop(0)
            for chunks in self._buf_v_by_channel.values():
                if chunks:
                    chunks.pop(0)

        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        stats = {}

        for label in self._channel_order:
            chunks = self._buf_v_by_channel.get(label, [])
            if not chunks:
                continue
            all_v = np.concatenate(chunks)
            all_t = np.linspace(0, len(all_v)/fs, len(all_v), endpoint=False)

            # ── 時域圖 ──
            self.curves_t[label].setData(all_t, all_v)

            # ── 加速度轉換後的 FFT ──
            # 公式：acceleration (g) = (voltage - 2.5) / 0.1
            accel = (all_v - 2.5) / 0.1
            freqs, amps = compute_fft(accel, fs)
            self.curves_f[label].setData(freqs, amps)

            rms = compute_rms(all_v)
            pp  = compute_peak_to_peak(all_v)
            dom = find_dominant_frequency(freqs, amps)
            stats[label] = (rms, pp, dom)

        self._apply_yrange(self.plot_t, self.edit_td_ymin, self.edit_td_ymax)
        self._apply_yrange(self.plot_f, self.edit_fft_ymin, self.edit_fft_ymax)
        self._apply_xrange(self.plot_f, self.edit_fft_xmin, self.edit_fft_xmax)

        # ── 統計 ──
        self.lbl_rms.setText("\n".join(f"{k}: {v[0]:.4f} V" for k, v in stats.items()))
        self.lbl_pp.setText("\n".join(f"{k}: {v[1]:.4f} V" for k, v in stats.items()))
        self.lbl_dom.setText("\n".join(f"{k}: {v[2]:.1f} Hz" for k, v in stats.items()))

        # ── 已記錄時間 ──
        recorded_sec = self._total_samples / fs
        self.lbl_recorded.setText(f"{recorded_sec:.1f} s")

        n_chunks = len(self._buf_t)
        if stats:
            first_label = next(iter(stats))
            rms, pp, dom = stats[first_label]
            self._status(
                f"[Live] {first_label} RMS: {rms:.4f} V   Pp: {pp:.4f} V   "
                f"主頻: {dom:.1f} Hz   channels: {len(stats)}   "
                f"已記錄: {recorded_sec:.1f} s   緩衝: {n_chunks}/{self._MAX} chunks"
            )

    def _reset_curves(self, channel_labels):
        """依照目前通道建立時域與 FFT 曲線。"""
        palette = [ACCENT, DANGER, GOLD, SUCCESS, "#6E6AA8", "#C46A4A"]

        for curve in self.curves_t.values():
            self.plot_t.removeItem(curve)
        for curve in self.curves_f.values():
            self.plot_f.removeItem(curve)

        for plot_widget in (self.plot_t, self.plot_f):
            legend = plot_widget.getPlotItem().legend
            if legend is not None:
                legend.clear()

        self.curves_t = {}
        self.curves_f = {}
        for idx, label in enumerate(channel_labels):
            color = palette[idx % len(palette)]
            self.curves_t[label] = self.plot_t.plot(
                pen=pg.mkPen(color, width=1.5),
                name=label,
            )
            self.curves_f[label] = self.plot_f.plot(
                pen=pg.mkPen(color, width=1.5),
                name=label,
            )

    @staticmethod
    def _apply_yrange(plot_widget, edit_min: QLineEdit, edit_max: QLineEdit):
        """依照輸入欄位套用 Y 軸範圍；輸入 auto 則自動縮放。"""
        LivePage._apply_axis_range(plot_widget, edit_min, edit_max, axis="y")

    @staticmethod
    def _apply_xrange(plot_widget, edit_min: QLineEdit, edit_max: QLineEdit):
        """依照輸入欄位套用 X 軸範圍；輸入 auto 則自動縮放。"""
        LivePage._apply_axis_range(plot_widget, edit_min, edit_max, axis="x")

    @staticmethod
    def _parse_axis_value(edit: QLineEdit) -> float | None:
        """讀取軸範圍輸入；空字串或 auto 表示自動縮放。"""
        text = edit.text().strip().lower()
        if text == "" or text == "auto":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _apply_axis_range(plot_widget, edit_min: QLineEdit, edit_max: QLineEdit, axis: str):
        """套用指定軸向範圍；支援固定範圍、單邊限制與自動縮放。"""
        axis_min = LivePage._parse_axis_value(edit_min)
        axis_max = LivePage._parse_axis_value(edit_max)
        view_box = plot_widget.getViewBox()
        if axis == "x":
            limit_keys = {"xMin": axis_min, "xMax": axis_max}
            clear_keys = {"xMin": None, "xMax": None}
            set_range = plot_widget.setXRange
        else:
            limit_keys = {"yMin": axis_min, "yMax": axis_max}
            clear_keys = {"yMin": None, "yMax": None}
            set_range = plot_widget.setYRange

        if axis_min is None or axis_max is None:
            view_box.setLimits(**limit_keys)
            plot_widget.enableAutoRange(axis=axis)
            return

        if axis_min >= axis_max:
            return

        view_box.setLimits(**clear_keys)
        set_range(axis_min, axis_max, padding=0)

    def _on_error(self, msg: str):
        self._status(f"[錯誤] {msg}", False)
        self._stop()

    def _save(self):
        if not any(self._record_v_by_channel.values()):
            self._status("[儲存] 沒有資料可儲存。")
            return
        cond = self.combo_cond.currentText()
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{cond}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV",
            os.path.join("data", "raw", cond, name), "CSV (*.csv)")
        if not path:
            self._status("[儲存] 已取消。")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        voltage_by_channel = {
            label: np.concatenate(chunks)
            for label, chunks in self._record_v_by_channel.items()
            if chunks
        }
        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        min_len = min(len(v) for v in voltage_by_channel.values())
        all_t = np.linspace(0, min_len/fs, min_len, endpoint=False)
        save_multi_csv(all_t, voltage_by_channel, path)
        self._status(
            f"[儲存完成] {os.path.basename(path)}  "
            f"({len(voltage_by_channel)} channels, {min_len} 筆/通道)"
        )


# ── 頁面 2：掃頻助手 ───────────────────────────

class SweepPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self._status = make_status_callback(status_callback)
        self._buf_t = []
        self._buf_v_by_channel = {}
        self._record_v_by_channel = {}
        self._channel_order = []
        self._total_samples = 0
        self._target_samples = 0
        self._MAX = 10
        self._thread = None
        self._finishing = False
        self.curves_t = {}
        self.curves_f = {}
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        ctrl_inner = QWidget()
        ctrl_inner.setStyleSheet(f"background:{PANEL};")
        cl = QVBoxLayout(ctrl_inner)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        g1 = QGroupBox("模式 / Mode")
        gl1 = QVBoxLayout(g1); gl1.setSpacing(3)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["mock (模擬)", "nidaq (實機)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode)
        self.combo_motor = QComboBox()
        self.combo_motor.addItems(["motor_off", "motor_on"])
        gl1.addWidget(QLabel("擷取模式")); gl1.addWidget(self.combo_mode)
        gl1.addWidget(QLabel("馬達狀態")); gl1.addWidget(self.combo_motor)
        cl.addWidget(g1)

        g2 = QGroupBox("採集參數")
        gl2 = QVBoxLayout(g2); gl2.setSpacing(3)
        self.edit_ch = QLineEdit("Dev1/ai0, Dev1/ai1"); self.edit_ch.setEnabled(False)
        self.edit_fs = QLineEdit("10000")
        self.edit_dur = QLineEdit("0.5")
        gl2.addWidget(QLabel("通道 (逗號分隔)")); gl2.addWidget(self.edit_ch)
        gl2.addWidget(QLabel("取樣率 (Hz)")); gl2.addWidget(self.edit_fs)
        gl2.addWidget(QLabel("Chunk 時間 (s)")); gl2.addWidget(self.edit_dur)
        cl.addWidget(g2)

        g_sweep = QGroupBox("掃頻設定")
        gsl = QVBoxLayout(g_sweep); gsl.setSpacing(3)
        row_freq = QHBoxLayout()
        self.edit_freq = QLineEdit("30")
        self.edit_step = QLineEdit("5")
        row_freq.addWidget(QLabel("目前 Hz")); row_freq.addWidget(self.edit_freq)
        row_freq.addWidget(QLabel("間隔 Hz")); row_freq.addWidget(self.edit_step)
        gsl.addLayout(row_freq)
        row_rec = QHBoxLayout()
        self.edit_record_sec = QLineEdit("5.0")
        self.combo_cond = QComboBox()
        self.combo_cond.addItems(["sweep","motor_on","motor_off","installation_test"])
        row_rec.addWidget(QLabel("擷取秒數")); row_rec.addWidget(self.edit_record_sec)
        gsl.addLayout(row_rec)
        gsl.addWidget(QLabel("量測條件")); gsl.addWidget(self.combo_cond)
        self.edit_session = QLineEdit(datetime.now().strftime("sweep_%Y%m%d_%H%M%S"))
        gsl.addWidget(QLabel("Session 名稱")); gsl.addWidget(self.edit_session)
        row_out = QHBoxLayout()
        self.edit_output = QLineEdit(os.path.join("data", "raw", "sweep"))
        btn_out = QPushButton("瀏覽...")
        btn_out.setFixedWidth(78)
        btn_out.clicked.connect(self._browse_output)
        row_out.addWidget(self.edit_output); row_out.addWidget(btn_out)
        gsl.addWidget(QLabel("輸出目錄")); gsl.addLayout(row_out)
        cl.addWidget(g_sweep)

        g_axis = QGroupBox("圖表軸範圍")
        gal = QVBoxLayout(g_axis); gal.setSpacing(3)
        gal.addWidget(QLabel("時域 Voltage (V):"))
        row_td = QHBoxLayout()
        self.edit_td_ymin = QLineEdit("auto"); self.edit_td_ymin.setFixedWidth(74)
        self.edit_td_ymax = QLineEdit("auto"); self.edit_td_ymax.setFixedWidth(74)
        row_td.addWidget(QLabel("Min")); row_td.addWidget(self.edit_td_ymin)
        row_td.addWidget(QLabel("Max")); row_td.addWidget(self.edit_td_ymax)
        gal.addLayout(row_td)
        gal.addWidget(QLabel("FFT Acceleration (g):"))
        row_fft_y = QHBoxLayout()
        self.edit_fft_ymin = QLineEdit("auto"); self.edit_fft_ymin.setFixedWidth(74)
        self.edit_fft_ymax = QLineEdit("auto"); self.edit_fft_ymax.setFixedWidth(74)
        row_fft_y.addWidget(QLabel("Min")); row_fft_y.addWidget(self.edit_fft_ymin)
        row_fft_y.addWidget(QLabel("Max")); row_fft_y.addWidget(self.edit_fft_ymax)
        gal.addLayout(row_fft_y)
        gal.addWidget(QLabel("FFT Frequency (Hz):"))
        row_fft_x = QHBoxLayout()
        self.edit_fft_xmin = QLineEdit("0"); self.edit_fft_xmin.setFixedWidth(74)
        self.edit_fft_xmax = QLineEdit("500"); self.edit_fft_xmax.setFixedWidth(74)
        row_fft_x.addWidget(QLabel("Min")); row_fft_x.addWidget(self.edit_fft_xmin)
        row_fft_x.addWidget(QLabel("Max")); row_fft_x.addWidget(self.edit_fft_xmax)
        gal.addLayout(row_fft_x)
        cl.addWidget(g_axis)

        g4 = QGroupBox("控制")
        gl4 = QVBoxLayout(g4); gl4.setSpacing(4)
        self.btn_start = QPushButton("▶  Start Current Frequency"); self.btn_start.setObjectName("btn_start")
        self.btn_stop = QPushButton("■  Abort"); self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start_step)
        self.btn_stop.clicked.connect(lambda: self._abort())
        gl4.addWidget(self.btn_start); gl4.addWidget(self.btn_stop)
        cl.addWidget(g4)

        g5 = QGroupBox("目前進度")
        gl5 = QVBoxLayout(g5); gl5.setSpacing(3)
        self.lbl_freq = self._stat_row(gl5, "目前頻率", GOLD)
        self.lbl_next = self._stat_row(gl5, "下一頻率", ACCENT)
        self.lbl_time = self._stat_row(gl5, "已錄製", SUCCESS)
        self.lbl_file = QLabel("--")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        gl5.addWidget(self.lbl_file)
        cl.addWidget(g5)
        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; border-right: 1px solid {BORDER}; background: {PANEL}; }}"
            f"QScrollBar:vertical {{ width: 6px; background: {SIDEBAR}; }}"
            f"QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}"
        )
        root.addWidget(scroll)

        pg.setConfigOption("background", PANEL)
        pg.setConfigOption("foreground", TEXT)
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(12)

        self.plot_t = pg.PlotWidget(title="Sweep Time Domain")
        self.plot_t.setLabel("left", "Voltage", units="V")
        self.plot_t.setLabel("bottom", "Time", units="s")
        self.plot_t.showGrid(x=True, y=True, alpha=0.4)
        self.plot_t.addLegend(offset=(-12, 12))
        pl.addWidget(self.plot_t)

        self.plot_f = pg.PlotWidget(title="Sweep FFT Spectrum (Acceleration)")
        self.plot_f.setLabel("left", "Amplitude", units="g")
        self.plot_f.setLabel("bottom", "Frequency", units="Hz")
        self.plot_f.setXRange(0, 500)
        self.plot_f.showGrid(x=True, y=True, alpha=0.4)
        self.plot_f.addLegend(offset=(-12, 12))
        pl.addWidget(self.plot_f)
        root.addWidget(plots, stretch=1)
        self._reset_curves(["Dev1_ai0", "Dev1_ai1"])
        self._update_frequency_labels()

    def _stat_row(self, layout, label, color):
        row = QHBoxLayout()
        k = QLabel(label + ":"); k.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        v = QLabel("--")
        v.setStyleSheet(f"color:{color}; font-family:Consolas; font-size:13px; font-weight:bold;")
        v.setAlignment(Qt.AlignRight)
        row.addWidget(k); row.addStretch(); row.addWidget(v)
        layout.addLayout(row)
        return v

    def _on_mode(self, idx):
        self.edit_ch.setEnabled(idx == 1)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "選擇掃頻輸出目錄", self.edit_output.text())
        if path:
            self.edit_output.setText(path)

    def _start_step(self):
        try:
            fs = int(self.edit_fs.text())
            dur = float(self.edit_dur.text())
            record_sec = float(self.edit_record_sec.text())
            current_freq = float(self.edit_freq.text())
            float(self.edit_step.text())
            channels = normalize_channels(self.edit_ch.text())
            if fs <= 0 or dur <= 0 or record_sec <= 0:
                raise ValueError
        except ValueError:
            self._status("[掃頻錯誤] 頻率、間隔、秒數、取樣率或 Chunk 時間格式不正確。")
            return

        self._channel_order = [sanitize_channel_label(ch) for ch in channels]
        self._buf_t.clear()
        self._buf_v_by_channel = {label: [] for label in self._channel_order}
        self._record_v_by_channel = {label: [] for label in self._channel_order}
        self._total_samples = 0
        self._target_samples = int(round(fs * record_sec))
        self._finishing = False
        self._reset_curves(self._channel_order)
        self._update_frequency_labels()

        mode = "mock" if self.combo_mode.currentIndex() == 0 else "nidaq"
        self._thread = AcquisitionThread(
            mode,
            channels, fs, dur,
            "motor_on" in self.combo_motor.currentText(),
        )
        self._thread.new_data.connect(self._on_data)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_inputs_enabled(False)
        self._status(
            f"[掃頻] {current_freq:g} Hz 錄製中 | 目標 {record_sec:.2f} s | "
            f"fs: {fs} Hz | channels: {', '.join(self._channel_order)}",
            True,
        )

    def _abort(self, show_status: bool = True):
        if self._thread:
            self._thread.stop(); self._thread.wait(); self._thread = None
        self._finishing = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_inputs_enabled(True)
        if show_status:
            self._status("[掃頻] 已中止目前頻率點，未自動存檔。", False)

    def _on_data(self, t, voltage_by_channel):
        if self._finishing or not voltage_by_channel:
            return

        first_voltage = next(iter(voltage_by_channel.values()))
        self._total_samples += len(first_voltage)
        self._buf_t.append(t)

        for label, voltage in voltage_by_channel.items():
            if label not in self._buf_v_by_channel:
                self._buf_v_by_channel[label] = []
                self._record_v_by_channel[label] = []
                self._channel_order.append(label)
                self._reset_curves(self._channel_order)
            voltage_arr = np.asarray(voltage, dtype=float)
            self._buf_v_by_channel[label].append(voltage_arr)
            self._record_v_by_channel[label].append(voltage_arr.copy())

        if len(self._buf_t) > self._MAX:
            self._buf_t.pop(0)
            for chunks in self._buf_v_by_channel.values():
                if chunks:
                    chunks.pop(0)

        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        stats = {}
        for label in self._channel_order:
            chunks = self._buf_v_by_channel.get(label, [])
            if not chunks:
                continue
            all_v = np.concatenate(chunks)
            all_t = np.linspace(0, len(all_v) / fs, len(all_v), endpoint=False)
            self.curves_t[label].setData(all_t, all_v)

            accel = (all_v - 2.5) / 0.1
            freqs, amps = compute_fft(accel, fs)
            self.curves_f[label].setData(freqs, amps)
            stats[label] = (
                compute_rms(all_v),
                compute_peak_to_peak(all_v),
                find_dominant_frequency(freqs, amps),
            )

        LivePage._apply_yrange(self.plot_t, self.edit_td_ymin, self.edit_td_ymax)
        LivePage._apply_yrange(self.plot_f, self.edit_fft_ymin, self.edit_fft_ymax)
        LivePage._apply_xrange(self.plot_f, self.edit_fft_xmin, self.edit_fft_xmax)

        recorded_sec = min(self._total_samples / fs, float(self.edit_record_sec.text()))
        target_sec = float(self.edit_record_sec.text())
        self.lbl_time.setText(f"{recorded_sec:.2f} / {target_sec:.2f} s")

        if stats:
            first_label = next(iter(stats))
            rms, pp, dom = stats[first_label]
            self._status(
                f"[掃頻] {self.edit_freq.text()} Hz | {recorded_sec:.2f}/{target_sec:.2f} s | "
                f"{first_label} RMS: {rms:.4f} V | 主頻: {dom:.1f} Hz"
            )

        if self._total_samples >= self._target_samples:
            self._finish_step()

    def _finish_step(self):
        if self._finishing:
            return
        self._finishing = True
        if self._thread:
            self._thread.stop(); self._thread.wait(); self._thread = None

        try:
            filepath, rows = self._save_current_step()
            self.lbl_file.setText(os.path.basename(filepath))
            self._append_manifest(filepath, rows)
            self._advance_frequency()
            self._status(f"[掃頻完成] 已儲存 {os.path.basename(filepath)}，下一頻率已更新。", False)
        except Exception as e:
            self._status(f"[掃頻錯誤] 自動存檔失敗：{e}", False)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_inputs_enabled(True)
        self._update_frequency_labels()
        self._finishing = False

    def _save_current_step(self):
        if not any(self._record_v_by_channel.values()):
            raise RuntimeError("沒有資料可儲存。")

        fs = int(self.edit_fs.text())
        current_freq = float(self.edit_freq.text())
        target_len = self._target_samples
        voltage_by_channel = {
            label: np.concatenate(chunks)[:target_len]
            for label, chunks in self._record_v_by_channel.items()
            if chunks
        }
        min_len = min(len(v) for v in voltage_by_channel.values())
        all_t = np.linspace(0, min_len / fs, min_len, endpoint=False)

        condition = self.combo_cond.currentText()
        session = self._safe_name(self.edit_session.text() or "sweep")
        output_root = self.edit_output.text().strip() or os.path.join("data", "raw", "sweep")
        output_dir = os.path.join(output_root, session)
        freq_label = self._format_frequency_label(current_freq)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{condition}_{freq_label}_{fs}Hz.csv"
        filepath = os.path.join(output_dir, filename)

        save_multi_csv(all_t, voltage_by_channel, filepath)
        return filepath, min_len

    def _append_manifest(self, filepath: str, rows: int):
        session = self._safe_name(self.edit_session.text() or "sweep")
        output_root = self.edit_output.text().strip() or os.path.join("data", "raw", "sweep")
        manifest_path = os.path.join(output_root, session, "manifest.csv")
        is_new = not os.path.exists(manifest_path)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "timestamp", "frequency_hz", "record_seconds", "sampling_rate_hz",
                    "channels", "rows_per_channel", "condition", "filepath",
                ])
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                f"{float(self.edit_freq.text()):g}",
                f"{float(self.edit_record_sec.text()):g}",
                self.edit_fs.text().strip(),
                ";".join(self._channel_order),
                rows,
                self.combo_cond.currentText(),
                filepath,
            ])

    def _advance_frequency(self):
        current_freq = float(self.edit_freq.text())
        step = float(self.edit_step.text())
        self.edit_freq.setText(f"{current_freq + step:g}")

    def _update_frequency_labels(self):
        try:
            current_freq = float(self.edit_freq.text())
            step = float(self.edit_step.text())
            self.lbl_freq.setText(f"{current_freq:g} Hz")
            self.lbl_next.setText(f"{current_freq + step:g} Hz")
        except ValueError:
            self.lbl_freq.setText("--")
            self.lbl_next.setText("--")

    def _set_inputs_enabled(self, enabled: bool):
        for widget in [
            self.combo_mode, self.combo_motor, self.edit_ch, self.edit_fs, self.edit_dur,
            self.edit_freq, self.edit_step, self.edit_record_sec, self.combo_cond,
            self.edit_session, self.edit_output,
        ]:
            widget.setEnabled(enabled)
        self.edit_ch.setEnabled(enabled and self.combo_mode.currentIndex() == 1)

    def _reset_curves(self, channel_labels):
        """依照目前通道建立時域與 FFT 曲線。"""
        palette = [ACCENT, DANGER, GOLD, SUCCESS, "#6E6AA8", "#C46A4A"]
        for curve in self.curves_t.values():
            self.plot_t.removeItem(curve)
        for curve in self.curves_f.values():
            self.plot_f.removeItem(curve)
        for plot_widget in (self.plot_t, self.plot_f):
            legend = plot_widget.getPlotItem().legend
            if legend is not None:
                legend.clear()
        self.curves_t = {}
        self.curves_f = {}
        for idx, label in enumerate(channel_labels):
            color = palette[idx % len(palette)]
            self.curves_t[label] = self.plot_t.plot(pen=pg.mkPen(color, width=1.5), name=label)
            self.curves_f[label] = self.plot_f.plot(pen=pg.mkPen(color, width=1.5), name=label)

    def _on_error(self, msg: str):
        self._status(f"[掃頻錯誤] {msg}", False)
        self._abort(show_status=False)

    @staticmethod
    def _format_frequency_label(frequency_hz: float) -> str:
        text = f"{frequency_hz:g}".replace("-", "minus_").replace(".", "p")
        return f"freq_{text}Hz"

    @staticmethod
    def _safe_name(text: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
        return safe.strip("._") or "sweep"


# ── 頁面 3：訊號分析 ───────────────────────────

class AnalysisPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("訊號分析 / Signal Analysis")
        title.setStyleSheet(f"font-size:20px; font-weight:bold; color:{ACCENT}; padding-bottom:4px;")
        root.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); root.addWidget(sep)

        # 檔案選取
        file_grp = QGroupBox("選擇 CSV 檔案")
        fl = QHBoxLayout(file_grp)
        self.edit_file = QLineEdit(); self.edit_file.setPlaceholderText("點擊右側按鈕選擇 CSV...")
        self.edit_file.setReadOnly(True)
        btn_browse = QPushButton("瀏覽...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse)
        fl.addWidget(self.edit_file); fl.addWidget(btn_browse)
        root.addWidget(file_grp)

        # 參數
        param_grp = QGroupBox("分析參數")
        pl = QHBoxLayout(param_grp)
        pl.addWidget(QLabel("取樣率 (Hz):"))
        self.edit_fs = QLineEdit("10000"); self.edit_fs.setFixedWidth(100)
        pl.addWidget(self.edit_fs)
        pl.addWidget(QLabel("   FFT 最小頻率 (Hz):"))
        self.edit_minf = QLineEdit("0"); self.edit_minf.setFixedWidth(100)
        pl.addWidget(self.edit_minf)
        pl.addWidget(QLabel("   FFT 最大頻率 (Hz):"))
        self.edit_maxf = QLineEdit("500"); self.edit_maxf.setFixedWidth(100)
        pl.addWidget(self.edit_maxf)
        pl.addStretch()
        root.addWidget(param_grp)

        # 執行按鈕
        btn_run = QPushButton("  執行分析並顯示圖表")
        btn_run.setFixedHeight(40)
        btn_run.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;border:none;border-radius:6px;font-size:14px;font-weight:bold;}}"
            f"QPushButton:hover{{background:{ACCENT_H};}}"
        )
        btn_run.clicked.connect(self._run)
        root.addWidget(btn_run)

        # 統計結果
        res_grp = QGroupBox("統計結果")
        rl = QVBoxLayout(res_grp)
        self._res_labels = {}
        fields = [("Mean (V)", TEXT), ("RMS (V)", SUCCESS),
                  ("Peak-to-Peak (V)", ACCENT), ("Peak (V)", GOLD),
                  ("Crest Factor", TEXT_DIM), ("Dominant Freq (Hz)", DANGER)]
        for label, color in fields:
            row = QHBoxLayout()
            k = QLabel(label + ":"); k.setFixedWidth(200)
            k.setStyleSheet(f"color:{TEXT_DIM};")
            v = QLabel("--")
            v.setStyleSheet(f"font-family:Consolas; font-size:15px; font-weight:bold; color:{color};")
            row.addWidget(k); row.addWidget(v); row.addStretch()
            rl.addLayout(row)
            self._res_labels[label] = v
        root.addWidget(res_grp)
        root.addStretch()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 CSV", "data", "CSV (*.csv)")
        if path: self.edit_file.setText(path)

    def _run(self):
        path = self.edit_file.text()
        if not path or not os.path.exists(path):
            return
        import pandas as pd
        try:
            df = pd.read_csv(path)
            if "voltage_v" in df.columns:
                voltage_col = "voltage_v"
            else:
                voltage_cols = [col for col in df.columns if col.endswith("_voltage_v")]
                if not voltage_cols:
                    raise ValueError("CSV 缺少 voltage_v 或 *_voltage_v 欄位。")
                voltage_col = voltage_cols[0]
            v = df[voltage_col].to_numpy(dtype=float)
            fs = int(self.edit_fs.text())
            min_f = float(self.edit_minf.text())
            max_f = float(self.edit_maxf.text())
            if min_f >= max_f:
                raise ValueError("FFT 最小頻率必須小於最大頻率。")
        except Exception as e:
            print("Error:", e); return
        feats = compute_time_features(v)
        freqs, amps = compute_fft(v, fs)
        dom = find_dominant_frequency(freqs, amps)
        n = len(v)
        t = np.linspace(0, n/fs, n, endpoint=False)
        mapping = {
            "Mean (V)": f"{feats['mean']:+.5f}",
            "RMS (V)": f"{feats['rms']:.5f}",
            "Peak-to-Peak (V)": f"{feats['peak_to_peak']:.5f}",
            "Peak (V)": f"{feats['peak']:.5f}",
            "Crest Factor": f"{feats['crest_factor']:.3f}",
            "Dominant Freq (Hz)": f"{dom:.2f}",
        }
        for k, val in mapping.items():
            self._res_labels[k].setText(val)
        plot_combined(t, v, freqs, amps,
                      title=f"{os.path.basename(path)} - {voltage_col}",
                      min_freq=min_f, max_freq=max_f, dominant_freq=dom,
                      save_path=None, show=True)


# ── 頁面 4：設定 ───────────────────────────────

class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        title = QLabel("設定 / Settings")
        title.setStyleSheet(f"font-size:20px; font-weight:bold; color:{ACCENT}; padding-bottom:4px;")
        root.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); root.addWidget(sep)

        info_grp = QGroupBox("裝置資訊")
        il = QVBoxLayout(info_grp)
        self.lbl_device_info = QLabel("點擊「檢查 NI 裝置」來偵測連接的設備。")
        self.lbl_device_info.setWordWrap(True)
        self.lbl_device_info.setStyleSheet(f"color:{TEXT_DIM}; font-family:Consolas;")
        il.addWidget(self.lbl_device_info)
        btn_check = QPushButton("檢查 NI 裝置")
        btn_check.clicked.connect(self._check_devices)
        il.addWidget(btn_check)
        root.addWidget(info_grp)

        path_grp = QGroupBox("路徑設定")
        pl = QVBoxLayout(path_grp)
        pl.addWidget(QLabel("原始資料存放目錄："))
        rh = QHBoxLayout()
        self.edit_raw = QLineEdit("data/raw")
        rh.addWidget(self.edit_raw); pl.addLayout(rh)
        pl.addWidget(QLabel("圖片輸出目錄："))
        fh = QHBoxLayout()
        self.edit_fig = QLineEdit("figures")
        fh.addWidget(self.edit_fig); pl.addLayout(fh)
        root.addWidget(path_grp)

        about_grp = QGroupBox("關於")
        al = QVBoxLayout(about_grp)
        about_txt = QLabel(
            "NI myDAQ Vibration Monitor\n"
            "EPIM Course v0.1.0\n\n"
            "支援 mock / NI-DAQmx 雙模式\n"
            "使用 Python 3.11 + PySide6 + pyqtgraph"
        )
        about_txt.setStyleSheet(f"color:{TEXT_DIM}; line-height:1.6;")
        al.addWidget(about_txt)
        root.addWidget(about_grp)
        root.addStretch()

    def _check_devices(self):
        try:
            from nidaqmx.system import System
            devs = list(System.local().devices)
            if devs:
                info = "\n".join(f"  [{i+1}] {d.name}  ({d.product_type})" for i, d in enumerate(devs))
                self.lbl_device_info.setText(f"找到 {len(devs)} 個裝置：\n{info}")
            else:
                self.lbl_device_info.setText("NI-DAQmx 已安裝，但未偵測到裝置。\n請確認 myDAQ 已插入 USB。")
        except ImportError:
            self.lbl_device_info.setText("未安裝 nidaqmx 套件。\n此電腦請使用 mock 模式。")
        except Exception as e:
            self.lbl_device_info.setText(f"錯誤：{e}")


# ── 主視窗 ──────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NI myDAQ Vibration Monitor — EPIM Course")
        self.setMinimumSize(960, 600)
        self._pages = []
        self._nav_btns = []
        self._build()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 側邊導覽欄 ──
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(f"background:{SIDEBAR}; border-right:1px solid {BORDER};")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 20, 12, 20)
        sl.setSpacing(4)

        logo = QLabel("Vibration\nMonitor")
        logo.setStyleSheet(
            f"font-size:18px; font-weight:bold; color:{ACCENT};"
            f"padding-bottom:12px; border-bottom:1px solid {BORDER}; margin-bottom:8px;"
        )
        logo.setAlignment(Qt.AlignCenter)
        sl.addWidget(logo)

        nav_items = [
            ("📡  即時監控", LivePage),
            ("🧭  掃頻助手", SweepPage),
            ("📊  訊號分析", AnalysisPage),
            ("⚙️  設定",    SettingsPage),
        ]
        self.stack = QStackedWidget()

        for i, (label, PageClass) in enumerate(nav_items):
            btn = QPushButton(label)
            btn.setObjectName("nav_btn" if i != 0 else "nav_btn_active")
            btn.setCheckable(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, idx=i: self._switch(idx))
            sl.addWidget(btn)
            self._nav_btns.append(btn)

            # LivePage 需要 status callback
            if PageClass is LivePage:
                page = PageClass(status_callback=self._set_status)
            else:
                page = PageClass()
            self.stack.addWidget(page)
            self._pages.append(page)

        sl.addStretch()
        root.addWidget(sidebar)
        root.addWidget(self.stack, stretch=1)

        self._sb = QStatusBar()
        self._sb.setStyleSheet(STATUS_NORMAL_SS)
        self._sb.showMessage("就緒 — 請至 [即時監控] 頁面按下 Start Live View 開始擷取")
        self.setStatusBar(self._sb)
        self._cur = 0

    def _set_status(self, msg: str, recording: bool | None = None):
        if recording is True:
            self._sb.setStyleSheet(STATUS_RECORDING_SS)
        elif recording is False:
            self._sb.setStyleSheet(STATUS_NORMAL_SS)
        self._sb.showMessage(msg)

    def _switch(self, idx: int):
        self._nav_btns[self._cur].setObjectName("nav_btn")
        self._nav_btns[self._cur].setStyle(self._nav_btns[self._cur].style())
        self._cur = idx
        self._nav_btns[idx].setObjectName("nav_btn_active")
        self._nav_btns[idx].setStyle(self._nav_btns[idx].style())
        self.stack.setCurrentIndex(idx)

    def closeEvent(self, event):
        for page in self._pages:
            thread = getattr(page, "_thread", None)
            if thread:
                thread.stop(); thread.wait()
        event.accept()


# ── 入口 ──────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(SS)
    app.setApplicationName("NI myDAQ Vibration Monitor")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
