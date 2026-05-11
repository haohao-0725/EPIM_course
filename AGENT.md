# NI myDAQ 振動訊號擷取與分析 — Agent 說明

## 用途

此 repository 用於 EPIM 課程的 NI myDAQ 振動訊號擷取、分析、即時監控 GUI 開發，以及相關說明文件撰寫。

在此 repo 中工作時，請優先遵守以下原則：

- 撰寫可重現的 Python 實作
- 提供清楚且具教學性的說明
- 維持一致的專案結構
- 產出可直接用於繳交或展示的結果

---

## 語言使用說明

在聊天室以及程式碼開頭說明註解段落，都必須使用繁體中文。只有在輸出的圖表中以及使用者特別說明這兩種情況才使用英文。

舉例來說：
- 製作一份報告？說明文字都使用繁體中文，僅圖表中的文字是英文。
- 製作程式碼？僅開頭說明註解是繁體中文，其餘皆可用英文。

---

## 電腦環境識別

此專案橫跨三台不同電腦，開始工作前**必須先確認目前位於哪一台電腦**，並依照對應規則操作：

| 代號 | 作業系統 | 主要用途 | 備註 |
|------|----------|----------|------|
| **A 電腦** | Windows（實驗室桌機） | 大型功能開發、架構調整、較耗時的實驗 | 主要開發機 |
| **B 電腦** | Windows | git pull 後連接 NI myDAQ 硬體擷取資料 | 不做大幅修改，以硬體操作為主 |
| **C 電腦** | macOS | 臨時溝通、快速修改程式碼 | 無硬體連接，僅做 mock mode 開發或文件編輯 |

確認電腦的方式（於對話開始時請使用者說明，或依系統提示判斷）：
- Windows：使用 PowerShell 原生指令操作檔案
- macOS：使用標準 Unix shell 指令（`ls`、`find`、`grep` 等皆可）

**不得假設電腦環境。** 若不確定目前在哪台電腦，請明確詢問使用者。

---

## 執行環境

### A / B 電腦（Windows）— Conda 環境

所有 Python 相關工作必須在 `epim_course` 環境下執行。**每次開啟新的 terminal session 時，必須先執行以下指令啟動環境：**

```powershell
conda activate epim_course
```

啟動後確認環境正確：

```powershell
conda info --envs
# 確認 epim_course 前方有星號 (*) 標記
```

**重要：不得假設環境已啟動。** 每次執行腳本前，必須在同一個 shell session 中先執行 `conda activate epim_course`。

### C 電腦（macOS）— venv 環境

```bash
source .venv/bin/activate
```

確認環境：

```bash
which python
# 應指向 .venv/bin/python
```

### 禁止行為（所有電腦通用）

- **不得**在未取得使用者明確指示的情況下切換至其他環境
- **不得**自行使用 `pip install` 或 `conda install` 安裝新套件；若缺少套件，應告知使用者並等待確認
- **不得**使用硬編碼的絕對路徑；所有路徑應以腳本所在位置為基準的相對路徑表示

---

## 中文字元編碼

由於專案中包含繁體中文註解與說明，請依照電腦環境處理編碼問題。

### Windows（A / B 電腦）

在 PowerShell 執行 Python 腳本前，先確保終端機輸出編碼正確：

```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 所有腳本開頭（通用）

所有 Python 腳本開頭應加入：

```python
# -*- coding: utf-8 -*-
```

若腳本需要讀寫含中文內容的文字檔，請明確指定編碼：

```python
with open("file.txt", "r", encoding="utf-8") as f:
    ...
```

---

## 檔案操作指令

### Windows（A / B 電腦）

**不得使用 `rg`（ripgrep）搜尋檔案**；此工具在 Windows 環境中不可靠，一律使用 PowerShell 原生指令：

| 用途 | 指令範例 |
|------|---------|
| 列出目錄內容（含子目錄） | `Get-ChildItem src/ -Recurse` |
| 搜尋特定副檔名的檔案 | `Get-ChildItem src/ -Recurse -Filter "*.py"` |
| 在檔案內容中搜尋字串 | `Select-String -Path src/*.py -Pattern "acquisition"` |
| 確認檔案是否存在 | `Test-Path data/example/mock_motor_on.csv` |
| 查看檔案大小 | `Get-Item figures/fft.png \| Select-Object Length` |

### macOS（C 電腦）

可使用標準 Unix 工具：`ls`、`find`、`grep`、`cat` 等。

---

## 圖表與視覺化規則

- 所有出現在圖表中的文字都必須使用英文，包含：標題、副標題、座標軸標籤、圖例、註記等。
- 圖檔檔名也應使用英文。
- **所有圖表均必須輸出成檔案**，不得依賴互動式視窗顯示。
- 使用 `matplotlib` 時，**一律在腳本開頭加入**（必須在其他 matplotlib import 之前）：

```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
```

- 圖表儲存至 `figures/` 目錄，可依類型建立子資料夾分類，例如：
  - `figures/time_domain/` — 時域波形圖
  - `figures/fft/` — 頻譜圖
  - `figures/comparison/` — 多條件比較圖

- 圖表輸出格式預設使用 `.png`，解析度設定為 `dpi=150` 以上。
- 儲存圖表後必須呼叫 `plt.close()` 以釋放記憶體。

---

## 報告與文件輸出

- 需要整理資料或輸出報告時，一律使用 **Markdown 格式**（`.md` 檔案）。
- 報告中說明文字使用繁體中文；圖表內文字使用英文。
- 報告內容應與程式碼及產生出的圖表保持一致，不得虛構數值結果或實驗發現。
- 說明應保持精簡、技術正確，讓課程助教或老師能快速理解。

---

## 程式撰寫規則

- 所有原始碼都必須包含足夠的註解：每個函式、類別與非顯而易見的邏輯區塊都必須有說明。
- 所有註解、docstring 與程式中的說明文字，一律使用繁體中文。
- 除非使用者明確要求，程式識別字、檔名、函式名稱、類別名稱、CLI 參數與輸出檔名一律使用英文。
- 優先選擇可讀性高的實作方式：變數命名應具語義（如 `sample_rate` 而非 `sr`）。
- 修改既有程式時，除非任務明確要求重新設計，否則應保留原本的演算法意圖。

---

## 執行與驗證流程

修改程式後，應執行最小且相關的腳本確認無 runtime error。

### Windows（A / B 電腦）

```powershell
conda activate epim_course
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python scripts/check_device.py   # 或對應腳本
```

確認輸出檔案存在：

```powershell
Get-ChildItem figures/ -Recurse
Test-Path figures/fft/motor_on_fft.png
Get-Item figures/fft/motor_on_fft.png | Select-Object Name, Length
```

### macOS（C 電腦）

```bash
source .venv/bin/activate
python scripts/live_view.py   # 或對應腳本
ls -lh figures/
```

在最終回覆中，應說明：
- 執行了哪些命令
- 修改或新增了哪些檔案（含相對路徑）
- 驗證結果（是否通過、是否有警告）

若因缺少套件、缺少硬體或需求不明而無法執行，請明確指出阻塞原因，不要自行猜測或跳過。

---

## 跨電腦 Git 同步注意事項

| 項目 | 說明 |
|------|------|
| 每台電腦各自建立環境 | 不要 commit `.venv/` 或 conda 環境資料夾 |
| `data/raw/` 不納入 Git | 量測資料太大，用雲端或外接硬碟共享 |
| `config.yaml` 不納入 Git | 每台電腦 device name 可能不同 |
| `data/example/` 納入 Git | 少量示範資料，方便測試 |
| `figures/` 不納入 Git | 產出圖片，執行後本地重新生成 |

---

## 目前專案脈絡

- 本 repo 為 `mydaq-vibration-monitor`，主要結構在 `src/mydaq_vibration/` 與 `scripts/`。
- 支援兩種執行模式：`mock`（無硬體，模擬訊號）與 `nidaq`（實機，B 電腦專用）。
- C 電腦僅能使用 mock mode 開發與測試。
- 現有腳本已呈現出將圖表輸出到 `figures/` 的慣例，新增內容應與此保持一致。