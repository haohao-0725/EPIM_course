# data/example/ 目錄

此目錄存放示範資料，**會被 Git 追蹤**，方便在 A/B/C 電腦同步。

示範資料由 mock_signal 產生，格式與實機資料完全相同，
可用於測試 `analyze_csv.py` 是否正常運作。

## 使用方式

```bash
python scripts/analyze_csv.py data/example/mock_motor_off.csv
python scripts/analyze_csv.py data/example/mock_motor_on.csv
```
