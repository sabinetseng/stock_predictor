# X_Y_VARIABLE_DEFINITION.md — 預測目標（Y）與特徵（X）定義

> 本檔為系統中「模型要預測什麼（Y）」與「模型用什麼來預測（X）」的唯一權威定義。
> 引用位置：`apps/stocks/model_training.py`（FEATURE_COLUMNS 註解與 MODEL_INFO 模型說明）、Dashboard 首頁「本次訓練將用到的資料與參數」資訊卡（`GET /stocks/model-info/` API，隨「預測模型」下拉動態切換）。

---

## 一、預測目標（Y）

### Return20

```
Return20 = (P(t+20) − P(t)) / P(t)
```

* `P(t)`：起算日 t 的**還原收盤價**（除權息調整後，避免配息造成假下跌）
* `t+20`：往後數第 **20 個交易日**（非自然日），直接取 `StockPrice` 第 i+20 筆
* 設定：`LABEL_LOOKAHEAD_TRADING_DAYS = 20`

### 標籤版本（二選一，寫入 `StockLabel.label_version`）

| 版本 | 規則 | 設定 |
|---|---|---|
| **MVP 單門檻**（`mvp`） | `Return20 > 3%` → 標籤 1，否則 0 | `LABEL_RETURN_THRESHOLD = 0.03` |
| **進階雙關卡**（`advanced`，**預設**） | 以下兩個條件**同時成立**才標籤 1：<br>① **超額報酬關卡**：個股 Return20 − 大盤(^TWII) Return20 > 門檻<br>② **停損關卡**：未來 20 個交易日內，任一日最低價不曾跌破 t 日收盤 × 95% | `USE_ADVANCED_LABEL_GATE = True`<br>`STOP_LOSS_RATIO = 0.95` |

> 用「雙關卡」的原因：單看漲幅會把「先暴漲又崩跌」的走勢標成好買點；加上停損關卡後，只有「贏過大盤且過程不會讓你停損出場」的樣本才是 1。切換方式：首頁下拉選單或 `build_labels --mvp`；**自選日期一鍵訓練**（`POST /stocks/full-pipeline-train/`，需提供自選 `dates` 四個日期）與**一鍵重訓練**（`POST /stocks/walkforward-retrain/`，最新 60/20/20）都會依所選標籤版本，在「建立標籤」步驟自動重算對應版本的標籤。

---

## 二、特徵（X）：共 32 個數值特徵

定義於 `apps/stocks/model_training.py` 的 `FEATURE_COLUMNS`（日頻 30 個存 `StockFeatureDaily`、月頻 2 個存 `StockFeatureMonthly`）。各特徵的計算式詳見 `README.md`〈特徵清單〉一節，此處列出清單與分類：

| 分類 | 特徵 | 數量 |
|---|---|---|
| 技術面 | `ma20_slope`, `ma60_slope`, `rsi`, `macd` | 4 |
| 籌碼面 | `institutional_net_buy_ratio`, `margin_usage_ratio` | 2 |
| 相對大盤 | `excess_return_vs_benchmark`, `beta_vs_benchmark` | 2 |
| 跨市場總經 | `usd_twd_change_rate`, `sox_change_rate`, `vix_level_percentile`, `vix_change_rate` | 4 |
| 籌碼面進階 | `inst_net_buy_20d`, `inst_accel`, `margin_mom` | 3 |
| 技術面進階 | `bias_ma20`, `bias_ma60`, `vol_ratio_5_20`, `volatility_20d`, `rsi_diff_3d`, `rsi_monthly`, `macd_monthly` | 7 |
| 跨市場進階 | `sox_mom`, `rs_vs_sox`, `usd_twd`, `vix` | 4 |
| ETF 折溢價（僅 ETF 有值） | `etf_premium_discount`, `etf_premium_ma5`, `etf_premium_z20`, `etf_premium_chg_5d` | 4 |
| 月頻營收 | `revenue_mom`, `revenue_yoy`（存 `StockFeatureMonthly`） | 2 |

### 缺值處理規則

* 「整欄皆空」的特徵視為該標的**結構性不可得**（如 ETF 無月營收、個股無折溢價），訓練時自動剔除並記錄在模型檔 `feature_columns`
* 混合個股＋ETF 訓練時取「所有股票都有的特徵交集」
* 其餘零星缺值整列捨棄（保守策略）
* 樹模型（LightGBM/XGBoost）推論端對結構性缺欄以 NaN 傳入、原生支援缺值；TFT 推論端要求最新 `seq_len` 個「特徵完整」交易日才能組窗
* **資料源 NaN 於清洗層即剔除**：價格／營收／時序（匯率、指數）任一關鍵數值為 NaN 時整列不寫入（`apps/sync/data_cleaning.py` 的 `clean_*_rows`）；籌碼 NaN 則補 0 並記錄 `fixed_nan`
* **寫入邊界雙保險**：所有 `update_or_create` 的 defaults 都先過 `clean_defaults()`，NaN/±Inf 統一轉 NULL——PostgreSQL 可存 NaN 但 MariaDB/MySQL 會踢例外，此層確保兩種資料庫都能穩定寫入（詳見 README〈6.2 資料清洗與 MariaDB 相容性〉）

---

## 三、時間安全（look-ahead bias 防護）

* 所有跨表對齊一律用 `merge_asof` 對齊「當天或之前最近一筆」資料
* 月營收以**保守生效日**（不是所屬月份）生效——生效日＝`max(次月 10 日, 資料源日期)`。依公開資訊觀測站（MOPS）營業額資訊申報規定「每月 10 日前公告」，真實公告日必不晚於次月 10 日，故以此日為準，保證模型任何一天拿到的營收都是真實世界已公布的資訊（2026-09 實測：FinMind 的 date 欄位為「次月 1 日」月級粒度，直接採用會比真實世界提早約 4~9 天看到營收；而 MOPS 各查詢介面均不提供歷史每家公司真實公告日，故取法定期限最為嚴謹。可用 `backfill_revenue_effective_dates --verify-mops N` 以 MOPS 官方彙總檔交叉驗證營收數值）
* 大盤/匯率指數缺日最多前值填補 5 天（VIX 百分位另有 expand 最小歷史窗）

## 四、誤導特徵診斷（什麼情形下會「剔除特徵」）

* 誤導特徵診斷**已納入訓練流程**：`run_training` 訓練完成後自動呼叫 `diagnostics.attach_auto_diagnosis`——診斷端用與訓練端同一套 `_load_dataset`（含上述防作弊對齊）組「驗證區間」資料，餵給 TreeSHAP／置換重要性，報告存回 `ModelTrainingRun.diagnostics`（超參數 `auto_shap=false` 可關閉；TFT 置換重要性時間預算 60 秒）。自動診斷只診斷、不重訓；手動 `python manage.py diagnose_model --auto --retrain` 會把 `exclude_features`／`monotone_constraints` 注入 `ModelTrainingRun.hyperparams`，`run_training` 再從 `used_cols` 剔除後重訓（保留給歷史舊紀錄補診斷與帶修正重訓比對）。
* 剔除後剩餘特徵不得為空，且剔除只影響「被打上診斷旗標的那次訓練」——不改變 `FEATURE_COLUMNS` 的權威定義。
* 詳細判準、情況 A／B 與指令：`MODEL_TUNING_GUIDE.md` 第八節；白話版：`小白名詞手冊.md` 2-5／5-8。
