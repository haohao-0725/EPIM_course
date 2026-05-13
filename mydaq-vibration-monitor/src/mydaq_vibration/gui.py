"""
gui.py  — NI myDAQ Vibration Monitor
色系：信紙米白  |  側邊選單頁面切換
"""

import sys, os, csv, time
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

# ── 採集執行緒 ─────────────────────────────────

class AcquisitionThread(QThread):
    new_data = Signal(np.ndarray, np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, mode, channel, fs, chunk_dur, motor_on, parent=None):
        super().__init__(parent)
        self.mode, self.channel = mode, channel
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
        while self._running:
            try:
                if self.mode == "mock":
                    t, v = generate_mock_signal(self.fs, self.chunk_dur, self.motor_on)
                    self.new_data.emit(t, v)
                    time.sleep(self.chunk_dur * 0.9)
                else:
                    from mydaq_vibration.acquisition import acquire_signal
                    t, v = acquire_signal(self.channel, self.fs, self.chunk_dur)
                    self.new_data.emit(t, v)
            except Exception as e:
                self.error_occurred.emit(str(e))
                break

    def stop(self):
        self._running = False


# ── 頁面 1：即時監控 ───────────────────────────

class LivePage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._buf_t, self._buf_v = [], []
        self._total_samples = 0   # 追蹤總取樣點數
        self._MAX = 10
        self._thread = None
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
        self.edit_ch  = QLineEdit("Dev1/ai0"); self.edit_ch.setEnabled(False)
        self.edit_fs  = QLineEdit("10000")
        self.edit_dur = QLineEdit("1.0")
        gl2.addWidget(QLabel("通道 (Channel)")); gl2.addWidget(self.edit_ch)
        gl2.addWidget(QLabel("取樣率 (Hz)"));   gl2.addWidget(self.edit_fs)
        gl2.addWidget(QLabel("Chunk 時間 (s)")); gl2.addWidget(self.edit_dur)
        cl.addWidget(g2)

        # Y 軸範圍
        g_yaxis = QGroupBox("Y 軸範圍")
        gyl = QVBoxLayout(g_yaxis); gyl.setSpacing(3)
        # 時域圖
        gyl.addWidget(QLabel("時域 Voltage (V):"))
        row_td = QHBoxLayout()
        self.edit_td_ymin = QLineEdit("auto"); self.edit_td_ymin.setFixedWidth(70)
        self.edit_td_ymax = QLineEdit("auto"); self.edit_td_ymax.setFixedWidth(70)
        row_td.addWidget(QLabel("Min")); row_td.addWidget(self.edit_td_ymin)
        row_td.addWidget(QLabel("Max")); row_td.addWidget(self.edit_td_ymax)
        gyl.addLayout(row_td)
        # FFT 圖
        gyl.addWidget(QLabel("FFT Acceleration (g):"))
        row_fft = QHBoxLayout()
        self.edit_fft_ymin = QLineEdit("auto"); self.edit_fft_ymin.setFixedWidth(70)
        self.edit_fft_ymax = QLineEdit("auto"); self.edit_fft_ymax.setFixedWidth(70)
        row_fft.addWidget(QLabel("Min")); row_fft.addWidget(self.edit_fft_ymin)
        row_fft.addWidget(QLabel("Max")); row_fft.addWidget(self.edit_fft_ymax)
        gyl.addLayout(row_fft)
        hint = QLabel("輸入數字固定範圍，\n輸入 auto 自動縮放")
        hint.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
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
        scroll.setFixedWidth(250)
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
        self.curve_t = self.plot_t.plot(pen=pg.mkPen(ACCENT, width=1.5))
        pl.addWidget(self.plot_t)

        self.plot_f = pg.PlotWidget(title="FFT Spectrum (Acceleration)")
        self.plot_f.setLabel("left", "Amplitude", units="g")
        self.plot_f.setLabel("bottom", "Frequency", units="Hz")
        self.plot_f.setXRange(0, 500)
        self.plot_f.showGrid(x=True, y=True, alpha=0.4)
        self.curve_f = self.plot_f.plot(pen=pg.mkPen(DANGER, width=1.5),
                                        fillLevel=0, brush=pg.mkBrush(DANGER + "30"))
        pl.addWidget(self.plot_f)
        root.addWidget(plots, stretch=1)

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

    def _start(self):
        try:
            fs  = int(self.edit_fs.text())
            dur = float(self.edit_dur.text())
        except ValueError:
            self._status("[錯誤] 取樣率或 Chunk 時間格式不正確，請輸入數字。")
            return
        self._buf_t.clear(); self._buf_v.clear()
        self._total_samples = 0
        self.lbl_recorded.setText("0.0 s")
        mode = "mock" if self.combo_mode.currentIndex() == 0 else "nidaq"
        self._thread = AcquisitionThread(
            mode,
            self.edit_ch.text().strip(), fs, dur,
            "motor_on" in self.combo_motor.currentText(),
        )
        self._thread.new_data.connect(self._on_data)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._status(f"[Live] 擷取中 | 模式: {mode} | fs: {fs} Hz | chunk: {dur} s — 按 Stop 結束")

    def _stop(self):
        if self._thread:
            self._thread.stop(); self._thread.wait(); self._thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        has_data = bool(self._buf_v)
        self.btn_save.setEnabled(has_data)
        if has_data:
            self._status("[停止] 擷取已停止。緩衝區有資料，可按 [Save CSV] 儲存。")
        else:
            self._status("[停止] 擷取已停止，無資料可儲存。")

    def _on_data(self, t, v):
        # 累計總取樣點數（不受 buffer 限制）
        self._total_samples += len(v)
        self._buf_t.append(t); self._buf_v.append(v)
        if len(self._buf_t) > self._MAX:
            self._buf_t.pop(0); self._buf_v.pop(0)

        all_v = np.concatenate(self._buf_v)
        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        all_t = np.linspace(0, len(all_v)/fs, len(all_v), endpoint=False)

        # ── 時域圖 ──
        self.curve_t.setData(all_t, all_v)
        self._apply_yrange(self.plot_t, self.edit_td_ymin, self.edit_td_ymax)

        # ── 加速度轉換後的 FFT ──
        # 公式：acceleration (g) = (voltage - 2.5) / 0.1
        accel = (all_v - 2.5) / 0.1
        freqs, amps = compute_fft(accel, fs)
        self.curve_f.setData(freqs, amps)
        self._apply_yrange(self.plot_f, self.edit_fft_ymin, self.edit_fft_ymax)

        # ── 統計 ──
        rms = compute_rms(all_v)
        pp  = compute_peak_to_peak(all_v)
        dom = find_dominant_frequency(freqs, amps)
        self.lbl_rms.setText(f"{rms:.4f} V")
        self.lbl_pp.setText(f"{pp:.4f} V")
        self.lbl_dom.setText(f"{dom:.1f} Hz")

        # ── 已記錄時間 ──
        recorded_sec = self._total_samples / fs
        self.lbl_recorded.setText(f"{recorded_sec:.1f} s")

        n_chunks = len(self._buf_t)
        self._status(
            f"[Live] RMS: {rms:.4f} V   Pp: {pp:.4f} V   "
            f"主頻: {dom:.1f} Hz   已記錄: {recorded_sec:.1f} s   緩衝: {n_chunks}/{self._MAX} chunks"
        )

    @staticmethod
    def _apply_yrange(plot_widget, edit_min: QLineEdit, edit_max: QLineEdit):
        """依照輸入欄位套用 Y 軸範圍；輸入 auto 則自動縮放。"""
        txt_min = edit_min.text().strip().lower()
        txt_max = edit_max.text().strip().lower()
        try:
            ymin = float(txt_min)
        except ValueError:
            ymin = None
        try:
            ymax = float(txt_max)
        except ValueError:
            ymax = None

        if ymin is not None and ymax is not None:
            plot_widget.setYRange(ymin, ymax, padding=0)
        elif ymin is not None:
            plot_widget.enableAutoRange(axis="y")
            vb = plot_widget.getViewBox()
            vb.setLimits(yMin=ymin)
        elif ymax is not None:
            plot_widget.enableAutoRange(axis="y")
            vb = plot_widget.getViewBox()
            vb.setLimits(yMax=ymax)
        else:
            plot_widget.enableAutoRange(axis="y")

    def _on_error(self, msg: str):
        self._status(f"[錯誤] {msg}")
        self._stop()

    def _save(self):
        if not self._buf_v:
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
        all_v = np.concatenate(self._buf_v)
        fs = int(self.edit_fs.text()) if self.edit_fs.text().isdigit() else 10000
        all_t = np.linspace(0, len(all_v)/fs, len(all_v), endpoint=False)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "voltage_v"])
            for ti, vi in zip(all_t, all_v):
                w.writerow([f"{ti:.6f}", f"{vi:.6f}"])
        self._status(f"[儲存完成] {os.path.basename(path)}  ({len(all_v)} 筆資料)")


# ── 頁面 2：訊號分析 ───────────────────────────

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
            v = df["voltage_v"].to_numpy(dtype=float)
            fs = int(self.edit_fs.text())
            max_f = float(self.edit_maxf.text())
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
                      title=os.path.basename(path),
                      max_freq=max_f, dominant_freq=dom,
                      save_path=None, show=True)


# ── 頁面 3：設定 ───────────────────────────────

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
        self._sb.setStyleSheet(f"background:{SIDEBAR}; color:{TEXT_DIM}; border-top:1px solid {BORDER};")
        self._sb.showMessage("就緒 — 請至 [即時監控] 頁面按下 Start Live View 開始擷取")
        self.setStatusBar(self._sb)
        self._cur = 0

    def _set_status(self, msg: str):
        self._sb.showMessage(msg)

    def _switch(self, idx: int):
        self._nav_btns[self._cur].setObjectName("nav_btn")
        self._nav_btns[self._cur].setStyle(self._nav_btns[self._cur].style())
        self._cur = idx
        self._nav_btns[idx].setObjectName("nav_btn_active")
        self._nav_btns[idx].setStyle(self._nav_btns[idx].style())
        self.stack.setCurrentIndex(idx)

    def closeEvent(self, event):
        live: LivePage = self._pages[0]
        if live._thread:
            live._thread.stop(); live._thread.wait()
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
