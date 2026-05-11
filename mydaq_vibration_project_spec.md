# NI myDAQ 振動訊號擷取與分析專案規格書

> 目的：建立一個可以在本地開發、跨電腦同步、並最終連接 NI myDAQ 讀取振動感測器訊號的 Python 專案。  
> 使用情境：A 電腦先開發與測試介面，B 電腦連接 NI myDAQ 實際量測，必要時 C 電腦可臨時接手。  
> 核心原則：先用 mock data 完成開發流程，再切換到 NI-DAQmx 實機擷取。

---

## 1. 專案目標

本專案要完成以下功能：

1. 讀取 NI myDAQ 類比輸入通道，例如 `Dev1/ai0`。
2. 擷取振動感測器輸出的電壓訊號。
3. 支援馬達「未啟動」與「啟動」兩種狀態的資料擷取。
4. 儲存 CSV 資料，方便後續分析與報告使用。
5. 顯示簡單的即時波形介面。
6. 對擷取資料做基本訊號分析：
   - 時域波形
   - RMS
   - Peak-to-peak
   - FFT 頻譜
   - PSD 或 Welch spectrum
7. 支援沒有 myDAQ 的電腦使用 mock mode 開發 UI 與分析流程。
8. 透過 Git 在 A / B / C 電腦之間同步程式碼。

---

## 2. 建議技術選型

### 2.1 Python 版本

建議使用：

```bash
Python 3.11
```

原因：

- 與 `nidaqmx`、`numpy`、`scipy`、`pandas`、`pyqtgraph` 相容性穩定。
- 避免過新版本 Python 造成套件相容問題。

---

### 2.2 必要 Python 套件

建議建立 `requirements.txt`：

```txt
nidaqmx
numpy
scipy
matplotlib
pandas
pyqtgraph
PySide6
openpyxl
ipykernel
jupyter
pyyaml
```

套件用途：

| 套件 | 用途 |
|---|---|
| `nidaqmx` | 呼叫 NI-DAQmx，讀取 myDAQ 訊號 |
| `numpy` | 數值運算、陣列處理、FFT |
| `scipy` | 訊號處理、濾波、Welch PSD、window function |
| `matplotlib` | 匯出圖表、離線分析圖 |
| `pandas` | CSV 儲存與讀取 |
| `pyqtgraph` | 即時波形顯示 |
| `PySide6` | GUI 介面框架，供 pyqtgraph 使用 |
| `openpyxl` | 輸出 Excel 報表 |
| `ipykernel` / `jupyter` | Notebook 分析 |
| `pyyaml` | 讀取 YAML 設定檔 |

---

## 3. NI 軟體與硬體環境

### 3.1 B 電腦，也就是實際接 myDAQ 的電腦，必須安裝

1. NI-DAQmx
2. NI-ELVISmx
3. NI MAX
4. Python 3.11
5. 本專案 requirements

### 3.2 A / C 電腦可以不安裝 NI driver

如果 A 或 C 電腦只是開發 GUI、分析流程、整理資料，不一定需要安裝 NI-DAQmx。  
但程式必須支援：

```text
mock mode
```

也就是沒有 myDAQ 時，使用模擬訊號代替真實量測資料。

---

## 4. 建議專案資料夾結構

請建立以下專案結構：

```text
mydaq-vibration-monitor/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example
│
├─ config/
│  ├─ default.yaml
│  ├─ local.example.yaml
│  └─ README.md
│
├─ src/
│  └─ mydaq_vibration/
│     ├─ __init__.py
│     │
│     ├─ main.py
│     │
│     ├─ acquisition/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ nidaq_reader.py
│     │  ├─ mock_reader.py
│     │  └─ device_check.py
│     │
│     ├─ analysis/
│     │  ├─ __init__.py
│     │  ├─ time_features.py
│     │  ├─ frequency_features.py
│     │  └─ filters.py
│     │
│     ├─ ui/
│     │  ├─ __init__.py
│     │  ├─ app.py
│     │  ├─ main_window.py
│     │  └─ live_plot.py
│     │
│     ├─ io/
│     │  ├─ __init__.py
│     │  ├─ csv_writer.py
│     │  └─ file_naming.py
│     │
│     └─ utils/
│        ├─ __init__.py
│        ├─ config_loader.py
│        └─ logging_setup.py
│
├─ scripts/
│  ├─ check_device.py
│  ├─ acquire_once.py
│  ├─ live_view.py
│  ├─ analyze_csv.py
│  └─ generate_mock_data.py
│
├─ notebooks/
│  ├─ 01_quick_fft_analysis.ipynb
│  └─ README.md
│
├─ data/
│  ├─ raw/
│  │  ├─ motor_off/
│  │  └─ motor_on/
│  ├─ processed/
│  └─ example/
│
├─ figures/
│
├─ reports/
│
└─ tests/
   ├─ test_time_features.py
   └─ test_frequency_features.py
```

---

## 5. 各資料夾功能說明

### `src/mydaq_vibration/acquisition/`

負責資料擷取。

建議使用抽象介面，讓 NI 實機與 mock data 可以互換。

#### `base.py`

定義資料擷取介面，例如：

```python
class BaseReader:
    def read_samples(self, n_samples: int):
        raise NotImplementedError
```

#### `nidaq_reader.py`

實際用 `nidaqmx` 讀 myDAQ。

功能：

- 設定 device channel，例如 `Dev1/ai0`
- 設定 sampling rate
- 設定 voltage range
- 讀取 finite samples
- 回傳 numpy array

#### `mock_reader.py`

產生模擬振動訊號。

mock signal 可包含：

- 60 Hz 正弦波
- 120 Hz 諧波
- 白噪聲
- 啟動狀態下振幅變大
- 可選擇加入 transient

用途：

- A 電腦沒有 myDAQ 時也能開發 UI
- 測試 FFT 與 RMS 是否正常
- agent 可以不用硬體也能幫忙開發

#### `device_check.py`

列出目前可用 NI 裝置。

功能類似：

```python
from nidaqmx.system import System

system = System.local()
for device in system.devices:
    print(device.name, device.product_type)
```

---

### `src/mydaq_vibration/analysis/`

負責訊號分析。

#### `time_features.py`

計算時域特徵：

- mean
- standard deviation
- RMS
- peak
- peak-to-peak
- crest factor

#### `frequency_features.py`

計算頻域特徵：

- FFT
- single-sided amplitude spectrum
- Welch PSD
- dominant frequency
- 頻帶能量，例如 0–100 Hz、100–500 Hz、500–1000 Hz

#### `filters.py`

可先保留簡單濾波功能：

- high-pass filter 去 DC / 低頻漂移
- low-pass filter 去高頻雜訊
- band-pass filter 針對特定振動頻帶

注意：第一版不要過度濾波。先保留原始資料，再額外產生處理後資料。

---

### `src/mydaq_vibration/ui/`

負責簡單介面。

第一版 GUI 只需要做到：

1. 選擇模式：
   - `mock`
   - `nidaq`
2. 設定 device channel，例如 `Dev1/ai0`
3. 設定 sampling rate
4. 設定 duration
5. 按鈕：
   - Start live view
   - Acquire once
   - Save CSV
   - Stop
6. 即時顯示：
   - 時域波形
   - RMS 數值
   - peak-to-peak 數值
7. 可以選擇資料標籤：
   - `motor_off`
   - `motor_on`
   - `installation_test`
   - `background_noise`

第一版不需要過度美化，重點是穩定。

---

### `src/mydaq_vibration/io/`

負責資料儲存與命名。

CSV 建議欄位：

```csv
time_s,voltage_v
0.0000,0.0123
0.0001,0.0131
...
```

metadata 建議另外存一份 JSON 或 YAML，例如：

```yaml
experiment_id: 2026-05-11_001
condition: motor_on
sensor_axis: unknown
device_channel: Dev1/ai0
sampling_rate_hz: 10000
duration_s: 5
voltage_range: [-10, 10]
mounting_method: tape
motor_status: on
notes: first test on motor housing
```

---

## 6. 設定檔建議

### `config/default.yaml`

```yaml
acquisition:
  mode: mock
  device_channel: "Dev1/ai0"
  sampling_rate_hz: 10000
  duration_s: 5
  voltage_min: -10.0
  voltage_max: 10.0
  samples_per_read: 1000

sensor:
  type: "voltage_output_accelerometer"
  sensitivity_v_per_g: null
  offset_v: null
  axis: "unknown"

experiment:
  default_condition: "motor_off"
  default_mounting_method: "unknown"
  operator: "unknown"

paths:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  figures_dir: "figures"
```

### `config/local.example.yaml`

這份作為範例，實際使用者複製成 `local.yaml`。

```yaml
acquisition:
  mode: nidaq
  device_channel: "Dev1/ai0"
  sampling_rate_hz: 10000
  duration_s: 10
  voltage_min: -10.0
  voltage_max: 10.0

experiment:
  operator: "your_name"
```

### 注意

`config/local.yaml` 不要 commit 到 Git，因為每台電腦的 device name、路徑或操作者可能不同。

---

## 7. `.gitignore` 建議

請建立 `.gitignore`：

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
.venv/
venv/
env/

# IDE
.vscode/settings.json
.idea/

# OS
.DS_Store
Thumbs.db

# Local config
.env
config/local.yaml

# Data files
data/raw/
data/processed/
figures/
reports/*.pdf
reports/*.docx

# Jupyter
.ipynb_checkpoints/

# Logs
logs/
*.log

# Build
build/
dist/
*.egg-info/
```

說明：

- 原始量測資料可能很大，不建議直接放 Git。
- 若需要同步資料，使用雲端硬碟、NAS、Git LFS，或只放小型 example data。
- `data/example/` 可以保留少量示範資料並 commit。

---

## 8. Git 開發規範

### 8.1 分支建議

基本分支：

```text
main
dev
feature/*
```

用途：

| 分支 | 用途 |
|---|---|
| `main` | 穩定版本，可以在 B 電腦直接使用 |
| `dev` | 日常開發整合 |
| `feature/ui-live-view` | 開發特定功能 |
| `feature/nidaq-reader` | 開發 NI 擷取功能 |
| `feature/fft-analysis` | 開發頻譜分析 |

如果專案很小，也可以只用：

```text
main
feature/*
```

但至少不要直接在 `main` 上亂改。

---

### 8.2 Commit message 建議

使用清楚格式：

```text
feat: add mock signal reader
feat: add NI DAQ finite acquisition
fix: handle missing nidaqmx driver
docs: update setup guide
refactor: split acquisition and analysis modules
test: add RMS calculation test
```

常用類型：

| 類型 | 意義 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修正錯誤 |
| `docs` | 文件 |
| `refactor` | 重構 |
| `test` | 測試 |
| `chore` | 環境、設定、雜項 |

---

### 8.3 跨電腦同步流程

#### 在 A 電腦開發前

```bash
git pull
```

#### 開發完成後

```bash
git status
git add .
git commit -m "feat: add live waveform viewer"
git push
```

#### 到 B 電腦使用前

```bash
git pull
```

#### 如果 B 電腦只是量測，不建議在 B 電腦改大量程式

B 電腦最好保持穩定，用來：

- 確認 myDAQ
- 擷取資料
- 存 CSV
- 做基本測試

主要開發放在 A 電腦。

---

### 8.4 每台電腦需要各自建立虛擬環境

不要把 `.venv/` 放進 Git。

每台電腦第一次設定：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

之後每次使用：

```bash
.venv\Scripts\activate
```

---

## 9. A / B / C 電腦分工建議

### A 電腦：主要開發機

可能沒有 myDAQ。

負責：

- GUI 開發
- mock mode 測試
- FFT / RMS 分析
- 報表產生
- 程式重構
- 文件撰寫

需要：

- Python 3.11
- requirements
- Git
- VS Code

不一定需要：

- NI-DAQmx
- NI-ELVISmx
- NI MAX

---

### B 電腦：實際量測機

連接 NI myDAQ。

負責：

- NI MAX 看訊號
- 實機擷取
- 儲存 raw CSV
- 簡單確認 FFT
- 不負責大規模開發

需要：

- Windows
- NI-DAQmx
- NI-ELVISmx
- NI MAX
- Python 3.11
- requirements
- Git

---

### C 電腦：臨時備援機

用途：

- 臨時開發
- 資料分析
- 或臨時接 myDAQ

注意：

- 第一次使用前先 `git pull`
- 建立自己的 `.venv`
- 複製 `config/local.example.yaml` 成 `config/local.yaml`
- 確認 device name 是否相同，不要假設一定是 `Dev1`

---

## 10. 第一階段開發任務

請 agent 優先完成以下功能。

### Phase 1：環境與 mock mode

1. 建立專案結構。
2. 建立 `requirements.txt`。
3. 建立 `config/default.yaml`。
4. 建立 `mock_reader.py`。
5. 建立 `time_features.py`。
6. 建立 `frequency_features.py`。
7. 建立 `scripts/generate_mock_data.py`。
8. 建立 `scripts/analyze_csv.py`。

Phase 1 完成條件：

- 不需要 myDAQ。
- 可以產生 mock vibration signal。
- 可以儲存 CSV。
- 可以讀 CSV 並畫時域圖與 FFT 圖。

---

### Phase 2：NI myDAQ 擷取

1. 建立 `device_check.py`。
2. 建立 `nidaq_reader.py`。
3. 建立 `scripts/check_device.py`。
4. 建立 `scripts/acquire_once.py`。

Phase 2 完成條件：

- 可以列出 NI 裝置。
- 可以從 `Dev1/ai0` 擷取固定秒數資料。
- 可以儲存 CSV。
- 如果沒有 `nidaqmx` 或沒有裝置，錯誤訊息要清楚，不要直接 crash。

---

### Phase 3：簡單 GUI

1. 建立 `ui/app.py`。
2. 建立 `ui/main_window.py`。
3. 建立 `ui/live_plot.py`。
4. GUI 支援 mock / nidaq 模式切換。
5. GUI 支援 Start / Stop / Acquire / Save。
6. 顯示即時時域波形。
7. 顯示 RMS 與 peak-to-peak。

Phase 3 完成條件：

- A 電腦可用 mock mode 開 GUI。
- B 電腦可切換 nidaq mode 擷取。
- GUI 不需要很漂亮，但必須穩定。

---

## 11. 基本命令設計

建議提供以下 script：

### 確認 NI 裝置

```bash
python scripts/check_device.py
```

### 產生模擬資料

```bash
python scripts/generate_mock_data.py --condition motor_on --duration 5 --fs 10000
```

### 擷取一次 myDAQ

```bash
python scripts/acquire_once.py --channel Dev1/ai0 --fs 10000 --duration 5 --condition motor_off
```

### 分析 CSV

```bash
python scripts/analyze_csv.py data/raw/motor_on/example.csv
```

### 開啟 GUI

```bash
python -m mydaq_vibration.main
```

或：

```bash
python src/mydaq_vibration/main.py
```

---

## 12. 資料命名規則

建議 raw data 檔名：

```text
YYYYMMDD_HHMMSS_condition_mounting_channel_fs.csv
```

範例：

```text
20260511_153012_motor_off_tape_ai0_10000Hz.csv
20260511_153820_motor_on_tape_ai0_10000Hz.csv
20260511_154245_motor_on_magnet_ai0_10000Hz.csv
```

metadata 檔案可同名：

```text
20260511_153012_motor_off_tape_ai0_10000Hz.yaml
```

---

## 13. 實驗資料分類建議

```text
data/raw/
├─ background_noise/
├─ motor_off/
├─ motor_on/
├─ mounting_test/
└─ invalid/
```

分類說明：

| 資料夾 | 用途 |
|---|---|
| `background_noise` | 感測器未安裝於馬達時的環境訊號 |
| `motor_off` | 感測器安裝於馬達，但馬達未啟動 |
| `motor_on` | 馬達啟動後的量測資料 |
| `mounting_test` | 比較不同安裝方式 |
| `invalid` | 明顯錯誤或待確認資料 |

---

## 14. 程式設計注意事項

### 14.1 不要把硬體讀取與分析綁死

錯誤做法：

```python
data = read_from_nidaq()
fft = do_fft(data)
plot(fft)
```

建議做法：

```python
reader = create_reader(mode="mock" or "nidaq")
data = reader.read_samples(n_samples)
features = analyze(data, fs)
```

這樣 A 電腦可以 mock，B 電腦可以 nidaq。

---

### 14.2 不要只存處理後資料

必須保留 raw voltage data。

原因：

- 後續可以重新分析
- 濾波參數可以重調
- 報告需要可追溯性
- 避免前處理錯誤後無法復原

---

### 14.3 不要假設 device name 永遠是 `Dev1`

程式要允許 config 或 CLI 指定：

```bash
--channel Dev2/ai0
```

每台電腦在 NI MAX 裡看到的裝置名稱可能不同。

---

### 14.4 沒有 NI driver 時要優雅處理

A 電腦可能沒有 `nidaqmx` 或沒有 NI driver。

程式應顯示：

```text
NI-DAQmx is not available. Please use mock mode or install NI-DAQmx on the measurement computer.
```

不要讓整個 GUI crash。

---

### 14.5 取樣率與 duration 不要寫死

應該放在 config 或 GUI 裡。

預設：

```yaml
sampling_rate_hz: 10000
duration_s: 5
```

後續可改：

```yaml
sampling_rate_hz: 20000
duration_s: 10
```

---

## 15. 感測器安裝與量測記錄欄位

這次課程任務需要討論「如何將感測器安裝至馬達，量取正確振動訊號」。

程式或 metadata 建議記錄：

```yaml
mounting_method: "tape"
mounting_location: "motor housing side"
sensor_axis: "radial"
cable_fixed: true
motor_status: "on"
motor_speed_rpm: null
notes: "sensor fixed by tape; cable not fully fixed"
```

建議安裝方式比較：

| 安裝方式 | 優點 | 缺點 |
|---|---|---|
| 螺絲固定 | 剛性最好，訊號最可靠 | 需要馬達外殼有孔位 |
| 磁吸座 | 快速、可重複安裝 | 需要鐵磁表面，可能有額外共振 |
| 瞬間膠 / 強力膠 | 接觸較穩 | 不易拆卸 |
| 雙面膠 | 方便 | 可能太軟，影響高頻訊號 |
| 膠帶 | 最簡單 | 容易量到膠帶或線材晃動 |

第一版實驗至少比較：

1. 感測器未接觸馬達的背景訊號。
2. 感測器固定於馬達，馬達未啟動。
3. 感測器固定於馬達，馬達啟動。
4. 若時間允許，比較不同固定方式。

---

## 16. 建議第一版 README 內容

`README.md` 應包含：

1. 專案目的
2. 硬體需求
3. 軟體需求
4. 安裝方式
5. 如何使用 mock mode
6. 如何使用 NI myDAQ mode
7. 如何在 NI MAX 確認 device name
8. 如何擷取資料
9. 如何分析 CSV
10. Git 同步注意事項
11. 常見錯誤排除

---

## 17. 常見錯誤排除

### 找不到 `nidaqmx`

可能原因：

- 沒有安裝 Python package

處理：

```bash
python -m pip install nidaqmx
```

---

### Python package 有裝，但找不到 NI device

可能原因：

- 沒有安裝 NI-DAQmx driver
- myDAQ 沒插好
- NI MAX 裡沒有看到裝置
- device name 不是 `Dev1`

處理：

1. 打開 NI MAX。
2. 確認 Devices and Interfaces 裡有 myDAQ。
3. 記下 device name。
4. 修改 `config/local.yaml` 的 `device_channel`。

---

### NI MAX 看得到訊號，但 Python 讀不到

可能原因：

- channel name 錯誤
- NI MAX 的 task 正在占用裝置
- Python 腳本沒有關閉上一個 task
- voltage range 設錯

處理：

1. 關閉 NI MAX test panel。
2. 重插 myDAQ。
3. 確認 `Dev1/ai0` 是否正確。
4. 重新執行 `scripts/check_device.py`。

---

### GUI 在沒有 myDAQ 的電腦 crash

這是程式設計錯誤。

修正原則：

- import `nidaqmx` 應該只放在 `nidaq_reader.py` 內。
- GUI 啟動時不要強制初始化 NI device。
- 預設模式應為 `mock`。

---

## 18. Agent 開發要求

請 agent 依照以下要求建立專案：

1. 先建立完整資料夾架構。
2. 第一版先完成 mock mode。
3. 所有硬體相關程式獨立在 `nidaq_reader.py`。
4. 沒有 myDAQ 時仍可執行 GUI。
5. 所有參數從 config 或 CLI 讀取，不要硬寫死。
6. 保留 raw data，不要只存處理後資料。
7. data/raw 預設不納入 Git。
8. 需要建立清楚 README。
9. 需要建立最少兩個測試：
   - RMS 計算測試
   - FFT dominant frequency 測試
10. 程式碼要能在 Windows 上執行。

---

## 19. 建議最小可行版本 MVP

MVP 不需要很複雜，只要完成：

1. `mock_reader.py` 可以產生 60 Hz + 120 Hz + noise 的資料。
2. `analyze_csv.py` 可以畫時域圖與 FFT。
3. `nidaq_reader.py` 可以從 `Dev1/ai0` 讀 5 秒。
4. `acquire_once.py` 可以存 CSV。
5. GUI 可以顯示 mock live waveform。
6. README 寫清楚如何切換到 NI mode。

完成以上就可以支援目前課程任務。

---

## 20. 最後開發策略

建議順序：

```text
先 mock → 再分析 → 再 GUI → 再 NI 實機 → 最後整理實驗流程
```

不要一開始就直接寫 NI 實機 GUI。  
原因是硬體、driver、sensor 接線、NI MAX 都可能出問題。  
先用 mock mode 把程式架構做好，等到 B 電腦確認 NI MAX 看得到訊號，再接上 `nidaq_reader.py` 即可。
