# NI myDAQ 振動訊號擷取與分析

> **EPIM Course — Electromechanical Principles and Monitoring**  
> 課程專案：用 Python 擷取 NI myDAQ 振動感測器訊號，進行時域與頻域分析。

---

## 📋 專案目標

| 功能 | 說明 |
|------|------|
| Mock mode | 不需硬體，用模擬訊號開發與測試 |
| NI myDAQ mode | 從 `Dev1/ai0` 讀取真實振動訊號 |
| 即時波形 GUI | 顯示時域訊號與 FFT 頻譜 |
| 訊號分析 | RMS、Peak-to-Peak、主頻 |
| CSV 儲存 | 保留原始電壓資料 |
| 跨電腦同步 | 透過 Git 在 A/B/C 電腦同步程式碼 |

---

## 🗂 專案結構

```
mydaq-vibration-monitor/
├── README.md
├── requirements.txt
├── .gitignore
├── config.example.yaml        ← 複製成 config.yaml 使用
├── generate_example_data.py   ← 產生示範 CSV
│
├── src/mydaq_vibration/
│   ├── __init__.py
│   ├── acquisition.py    ← NI myDAQ 實機擷取
│   ├── mock_signal.py    ← 模擬振動訊號產生器
│   ├── analysis.py       ← RMS / FFT 等訊號分析
│   ├── plotting.py       ← matplotlib 時域圖 / FFT 圖
│   └── gui.py            ← PySide6 即時監控 GUI
│
├── scripts/
│   ├── check_device.py   ← 列出 NI 裝置
│   ├── acquire_once.py   ← 擷取一次並存 CSV
│   ├── live_view.py      ← 快速啟動 GUI
│   └── analyze_csv.py    ← 分析 CSV 並畫圖
│
├── data/
│   ├── raw/              ← 原始量測資料（不納入 Git）
│   └── example/          ← 示範資料（納入 Git）
│
└── figures/              ← 輸出圖片（不納入 Git）
```

---

## ⚙️ 環境建立（Hao_ASCL 使用 conda）

### 建立 conda 環境

```bash
conda create -n epim_course python=3.11 -y
conda activate epim_course
```

### 安裝套件

```bash
cd mydaq-vibration-monitor
pip install -r requirements.txt
```

### 確認安裝成功

```bash
python -c "import numpy, scipy, matplotlib, pandas, pyqtgraph, PySide6, yaml; print('All OK')"
```

---

## 🖥️ 其他電腦環境建立（venv 方式）

> 適用於沒有 conda 的 B / C 電腦。

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

之後每次使用：

```bash
.venv\Scripts\activate
```

---

## 🚀 快速開始

### 1. 產生示範資料（第一次使用先執行）

```bash
cd mydaq-vibration-monitor
python generate_example_data.py
```

會產生 `data/example/mock_motor_off.csv` 和 `data/example/mock_motor_on.csv`。

---

### 2. 分析示範 CSV

```bash
python scripts/analyze_csv.py data/example/mock_motor_on.csv
```

輸出：
- 終端機顯示 RMS、Peak-to-Peak、主頻
- 開啟時域圖 + FFT 頻譜圖

儲存圖片：

```bash
python scripts/analyze_csv.py data/example/mock_motor_on.csv --save --no-show
```

---

### 3. 啟動即時監控 GUI（mock mode）

```bash
python scripts/live_view.py
# 或
python src/mydaq_vibration/gui.py
```

GUI 說明：
- 左側 **模式設定** → 選擇 `mock (模擬)`
- 選擇馬達狀態：`motor_off` 或 `motor_on`
- 按 **▶ Start Live View**
- 右側即時顯示時域波形與 FFT 頻譜
- 左側即時顯示 RMS、Peak-to-Peak、主頻
- 按 **■ Stop** 後可按 **💾 Save CSV** 儲存

---

## 🔌 B 電腦使用 NI myDAQ

### 必須安裝

1. [NI-DAQmx driver](https://www.ni.com/zh-tw/support/downloads/drivers/download.ni-daq-mx.html)
2. NI MAX（隨 NI-DAQmx 安裝）
3. Python 3.11 + `pip install -r requirements.txt`

### 確認裝置

```bash
python scripts/check_device.py
```

正常輸出範例：
```
✅ 找到 1 個 NI 裝置：

  [1] 裝置名稱  : Dev1
       產品類型  : NI myDAQ
       AI 通道   : Dev1/ai0, Dev1/ai1
```

### 確認 device name

每台電腦的裝置名稱可能不同（不一定是 `Dev1`）。  
請在 NI MAX 的 **Devices and Interfaces** 確認名稱，再修改指令中的 `--channel`。

### 擷取一次資料

```bash
python scripts/acquire_once.py --channel Dev1/ai0 --fs 10000 --duration 5 --condition motor_on
```

### 使用 GUI 實機擷取

```bash
python scripts/live_view.py
```

GUI 中選擇 `nidaq (實機)` 模式，填入正確的 Channel（如 `Dev1/ai0`）。

---

## 🔄 Git 跨電腦同步流程

### A 電腦開發後 push

```bash
git add .
git commit -m "feat: 新增 XXX 功能"
git push
```

### B / C 電腦使用前 pull

```bash
git pull
```

### 注意事項

| 項目 | 說明 |
|------|------|
| 每台電腦各自建立環境 | 不要 commit `.venv/` 或 conda 環境 |
| `data/raw/` 不納入 Git | 量測資料太大，用雲端或外接硬碟共享 |
| `config.yaml` 不納入 Git | 每台電腦 device name 可能不同 |
| `data/example/` 納入 Git | 少量示範資料，方便測試 |

---

## ❗ 常見錯誤排除

### 錯誤：找不到 nidaqmx

```
ImportError: No module named 'nidaqmx'
```

**解決**：
1. 若此電腦有 NI-DAQmx driver：`pip install nidaqmx`
2. 若此電腦只做開發：使用 **mock mode**（不需要 `nidaqmx`）

---

### 錯誤：找不到 NI 裝置

```
❌ NI-DAQmx driver 已安裝，但未偵測到任何 NI 裝置。
```

**解決**：
1. 確認 myDAQ 已插入 USB
2. 打開 NI MAX → Devices and Interfaces
3. 若看不到裝置，重新插拔 USB
4. 確認 device name（可能不是 `Dev1`）

---

### 錯誤：NI MAX 可以看到，但 Python 讀不到

**解決**：
1. 關閉 NI MAX 的 Test Panel（它會佔用裝置）
2. 重新執行 `python scripts/check_device.py`
3. 確認 channel 名稱正確

---

### GUI crash 在沒有 myDAQ 的電腦

**這是 bug，請回報**。正確行為：GUI 應預設 mock mode，不 crash。

確認 GUI 開啟後左上角選的是 `mock (模擬)`，而非 `nidaq (實機)`。

---

## 📦 套件說明

| 套件 | 用途 |
|------|------|
| `nidaqmx` | 呼叫 NI-DAQmx，讀取 myDAQ 訊號 |
| `numpy` | 數值運算、FFT |
| `scipy` | 訊號處理、window function |
| `matplotlib` | 匯出靜態圖表 |
| `pandas` | CSV 讀取 |
| `pyqtgraph` | 即時波形顯示 |
| `PySide6` | GUI 框架 |
| `pyyaml` | 讀取 YAML 設定檔 |

---

## 📐 訊號分析說明

| 指標 | 公式 | 意義 |
|------|------|------|
| RMS | √(mean(x²)) | 訊號能量，反映整體振動強度 |
| Peak-to-Peak | max(x) - min(x) | 振幅範圍 |
| Peak | max(|x|) | 最大瞬時值 |
| Crest Factor | Peak / RMS | >3 通常表示有衝擊性振動 |
| Dominant Freq | argmax(FFT) | 最主要的振動頻率 |

---

## 📄 未來規劃

詳見 [`mydaq_vibration_project_spec.md`](../mydaq_vibration_project_spec.md)（完整版規格書）。

下一階段規劃：
- Welch PSD 頻譜分析
- 高通 / 低通 / 帶通濾波器
- Excel 報表輸出
- Jupyter Notebook 分析範本
- 感測器安裝比較實驗流程

---

*EPIM Course | NI myDAQ Vibration Monitor v0.1.0*
