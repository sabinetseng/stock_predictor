# 台股預測系統 - Django 專案

完整的資料管線已全部實作完成：

1. **爬蟲（原始資料入庫）**：美金匯率（yfinance）、跨市場指數（^TWII/^SOX/^VIX）、股票行情/籌碼/月營收（FinMind）
2. **資料清洗**：寫入資料庫前先經過 `apps/sync/data_cleaning.py` 清洗（移除無效值、重複資料、邏輯不一致、離群值警告；NaN 於清洗層剔除、寫入邊界再轉 NULL，PostgreSQL／MariaDB 皆相容，見〈6.2 資料清洗與 MariaDB 相容性〉）
3. **特徵工程**：`python manage.py build_features` 計算日頻/月頻特徵（技術面、籌碼面、相對大盤、跨市場總經、ETF 折溢價，共 32 個數值特徵）
4. **標籤生成**：`python manage.py build_labels` 計算 Return20 標籤（MVP 單一門檻 / Advanced 雙關卡）
5. **模型訓練**：LightGBM / XGBoost / TFT（Temporal Fusion Transformer 序列模型）訓練、驗證（AUC/PR-AUC）、n_repeats 重複訓練統計（平均值 ± 標準差）、walk-forward 自動化多輪回測
6. **預測與前端**：AI 預測 API + 量化決策看板（K 線 + 均線 + AI 買賣點、動態突破機率、特徵監控）+ 相關性分析（Pearson/Spearman/Lagged/Rolling）。模型選擇與買進門檻等預測參數集中於決策看板右欄「策略設定」面板（舊獨立設定頁已轉址併入）。

## 目錄結構

```
stock_predictor/
├── manage.py
├── smoke_check.py                # 一鍵健檢：資料層／模型層／API 層／模板 45+ 項自動驗證
├── requirements.txt
├── .env                        # 已內建，含資料庫連線設定，需依你本機環境調整密碼
├── venv/                       # 虛擬環境（**不隨 zip 分享**；收到專案請自行 `python -m venv venv` 並 `pip install -r requirements.txt` 重建）
├── stock_predictor/            # 專案設定
│   ├── settings.py             # 含資料庫、標籤門檻、跨市場代碼等設定
│   ├── urls.py                 # 主路由，彙整各 App
│   └── wsgi.py
└── apps/
    ├── dashboard/               # 首頁：標準作業流程按鈕列、訓練設定卡（模型/超參數/60-20-20 切分/自選日期模式/一鍵訓練管線）、本次訓練資訊卡（model-info 動態切換）、訓練紀錄表格、最新訓練結果摘要卡
    ├── stocks/                   # 股票資料、特徵、標籤、預測（核心，FinMind 爬蟲已完成）
    ├── fx_data/                 # 美金匯率資料與更新（yfinance 爬蟲已完成）
    ├── market_data/             # ^TWII / ^SOX / ^VIX 與交易日行事曆（已完成）
    ├── analysis/                 # 相關性分析（後端 API + 前端圖表皆已完成）
    └── sync/                     # 共用的 DataSyncLog 與 data_cleaning.py（NaN/離群值清洗、sanitize_val/clean_defaults 寫入閘門），任何「檢查新資料」動作都會寫入這裡
```

## 各 App 現況

| App | 狀態 |
|---|---|
| `stocks` | 爬蟲、特徵工程（含 ETF 折溢價）、標籤生成、LightGBM / XGBoost / TFT 模型訓練與預測、ETF 淨值同步、walk-forward 自動化皆已完成 |
| `fx_data` | 美金匯率爬蟲已完成 |
| `market_data` | 大盤/費半/VIX 爬蟲、交易日行事曆已完成 |
| `sync` | `DataSyncLog` 已可用，掛在 Django Admin |
| `dashboard` | 莫蘭迪綠卡片式 UI（純文字按鈕、無 emoji icon）；標準作業流程五步按鈕全接 API；「訓練/驗證區間設定」支援自動 60/20/20 切分、滾動視窗年數、**自選日期模式**（四個日期完全手動）與**兩顆一鍵訓練按鈕**（一鍵重訓練：同步→特徵→標籤→最新 60/20/20 訓練；自選日期一鍵訓練：同步→特徵→標籤→訓練）；「本次訓練將用到的資料與參數」資訊卡隨模型下拉動態切換（model-info API）；訓練紀錄表格（AUC/PR-AUC 平均±標準差、彩色狀態徽章、分頁）自動載入更新，**一鍵重訓練／自選日期一鍵訓練的紀錄全部顯示**；重訓練完成訊息長期保留於「最新訓練結果摘要」卡；首頁**預測模型預設 TFT**（可切換 LightGBM/XGBoost）；相關性分析位於 `/analysis/`，「策略設定」面板內建於決策看板右欄 |
| `analysis` | 後端 API 已完成並測試（Pearson/Spearman/Lagged/Rolling），前端已繪製成互動圖表 |

`stocks` 個股詳細頁為「決策看板」單頁版：左側 K 線 + 均線 + AI 買賣點主圖、動態突破機率 vs 門檻、特徵監控（可下拉選特徵）畫在中欄，左欄「即時因子監控」，右欄內建「策略設定」面板，可即時調整模型/進場門檻並一鍵進行 AI 預測；門檻一改，指標卡與圖表立即重算。舊網址 `/stocks/<代碼>/strategy-settings/` 會自動轉址回決策看板。

## 快速開始（venv 已經建好，套件也已安裝）

### 1. 啟用虛擬環境

> 執行 `python3 -m venv venv`，再 `pip install -r requirements.txt` 即可。

**Mac / Linux**
```bash
source venv/bin/activate
```

**Windows（PowerShell）**
```powershell
.\venv\Scripts\Activate.ps1
```

> 再 `pip install -r requirements.txt` 即可。

### 2. 確認 `.env` 內的資料庫密碼

打開 `.env`，把 `DATABASE_URL` 改成你本機 PostgreSQL 的實際帳密：
```

# postgresql資料庫設定
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'stock_predictor',         # 必須是在 pgAdmin 裡已經建立好的資料庫
        'USER': 'postgres',              # 這是使用者名稱
        'PASSWORD': 'your_password',
        'HOST': 'localhost',             # 這裡一定要是 localhost 或 127.0.0.1，不能是 postgres
        'PORT': '5432',
    }
}

# MariaDB資料庫設定
#  pip install mysqlclient

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',  # MariaDB 共用 MySQL 的後端
        'NAME': 'stock_predictor',           # 在 MariaDB 中建立好的資料庫名稱
        'USER': 'root',                      # 你的 MariaDB 使用者名稱 (例如 root)
        'PASSWORD': 'your_password',         # 你的 MariaDB 密碼
        'HOST': 'localhost',                 # 資料庫主機位置
        'PORT': '3306',                      # MariaDB 預設連接埠
    }
}


```

先在 PostgreSQL 裡建立好 `stock_predictor` 這個資料庫（`CREATE DATABASE stock_predictor;`）。

### 3. 建立資料表

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

### 4. 測試爬蟲、特徵工程與模型訓練

```bash
python manage.py sync_fx_data
python manage.py sync_market_data
python manage.py sync_stock_data 2330

python manage.py build_features 2330    # 計算日頻/月頻特徵
python manage.py build_labels 2330      # 計算 Return20 標籤（進階雙關卡版本，預設）
python manage.py build_labels 2330 --mvp   # MVP 單門檻版本
python manage.py import_etf_nav 0050 --csv nav_0050.csv   # 匯入 ETF 歷史淨值（折溢價回補）

# 訓練模型（訓練/驗證區間需先確定該區間已跑過 build_features + build_labels）
python manage.py train_model \
    --train-start 2018-01-01 --train-end 2024-12-31 \
    --val-start 2025-01-01 --val-end 2025-12-31

# 可選參數：
#   --model-type {lightgbm,xgboost,tft}     （CLI／後端 API 預設 lightgbm；網頁端下拉預設 TFT。tft 需 PyTorch，CPU 版即可）
#   --hyperparams '{"n_estimators": 300}'   （JSON，只傳想覆蓋的欄位，其餘用系統預設值）
#   --n-repeats 5                           （重複訓練 N 次，回報 AUC 平均值 ± 標準差）
```

訓練完成的模型會存在 `trained_models/`（已加入 `.gitignore`，不會被提交）。
跑完可以到 `/admin/` 後台檢查資料、`Stocks > Model training runs` 檢查訓練結果（AUC/PR-AUC）。

首頁的「進行 AI 預測」按鈕或決策看板右欄「策略設定」面板的「進行 AI 預測」會自動載入**該架構最近一次訓練完成**的模型；首頁「預測模型」下拉與決策看板右欄「AI 模型架構」下拉**皆以 TFT 為預設值**，切換架構後訓練與圖表即改用該架構的最新完成模型（三種架構各有獨立的模型版本）。AI 買進機率門檻直接在決策看板（`/stocks/<代碼>/`）右欄調整，改門檻會即時重算指標卡與機率圖。
「建立訓練紀錄」按鈕送出日期區間後，會**同步**觸發訓練（資料量大時要等一下，可設定 `TRAINING_ASYNC=True` 改用 Celery 非同步，見下方說明）。

### 5. Walk-forward 自動化多輪訓練（驗證模型能否套用到未來5-10年）

```bash
python manage.py walk_forward_train --start-year 2018 --end-year 2025 --initial-train-years 3
```

會自動跑好幾輪訓練（訓練窗格逐年擴大、驗證窗格逐年往後滑動），跑完在終端機會印出每一輪的 AUC/PR-AUC 趨勢表，
也可以到 Admin 的 `Stocks > Model training runs` 依 `months_from_train_end` 排序查看。

### 6. 相關性分析

前端頁面：`GET /analysis/?stock_code=2330`（預設 `BENCHMARK_STOCK_CODE`）

```bash
# JSON API（供前端圖表呼叫）
GET /analysis/correlation/2330/?compare_target=USD_TWD
```
`compare_target` 可用值：`USD_TWD` / `TWII` / `SOX` / `VIX`。回傳 Pearson、Spearman、Lagged、Rolling 四種分析結果，前端已繪製成互動圖表（相關係數卡片、Lag 相關係數長條圖、Rolling Correlation 折線圖）。

### 6.1 決策看板與策略設定

首頁輸入股票代碼後：
- **查看圖表 / 詳情（含策略設定＋相關性分析）** → `/stocks/<代碼>/`：單頁決策看板——左欄「即時因子監控」＋診斷/買賣紀律，中欄 K 線 + 均線 + AI 買賣點、動態突破機率 vs 門檻、特徵監控，底部「相關性分析」面板（USD_TWD／TWII／SOX／VIX，Pearson／Spearman／落後期／滾動四圖）；右欄策略設定面板（模型架構、買進機率門檻即時重算、「進行 AI 預測」按鈕與結果卡）
- **AI 推論結果卡會建議賣出／停損**：卡片狀態機有六態——空手時「機率 ≥ 門檻 → 新觸發建倉 (BUY)、否則維持觀望 (WAIT)」；持倉中（最近一次訊號是買點且尚未出場）以「最近買點收盤 → 最新收盤」報酬比對停利/停損門檻：達停利（預設 +6%）→「建議停利賣出 (TAKE PROFIT)」、觸停損（預設 −4.5%）→「建議停損賣出 (STOP LOSS)」、其餘「持倉觀察中 (HOLD)」（顯示進場價、浮動損益與門檻）；最新機率日剛觸發賣點 →「賣出訊號觸發 (SELL)」並標註原因（停利／停損／機率跌破門檻）。門檻由後端 API 帶入（`take_profit_pct`/`stop_loss_pct`，與歷史買賣點模擬 S-2 同一組常數），K 線上的 S 旗標即這套紀律的歷史軌跡（歷史模擬買點／賣點次數也顯示在卡片上）
- 決策看板右欄「**進行 AI 預測**」按鈕＝**純預測**：直接以「AI 模型架構」下拉選定架構「最近一次訓練完成」的模型對最新特徵做推論，並重繪全部圖表＋重算相關性分析。**它不是完整管線**——不同步資料、不建特徵、不建標籤；要一口氣「同步→特徵→標籤→訓練→預測」請在首頁用「自選日期一鍵訓練」，或先走標準作業流程 1→4 步
- 首頁「訓練/驗證區間設定」有「自動切分（60/20/20）」按鈕：依資料最早/最新日期自動填入 Train 前 60%、Val 次 20%，最後 20% 留作盲測不參與訓練；另有「**自選日期模式**」checkbox——勾選後四個日期欄位（訓練起訖、驗證起訖）完全手動、頁面載入不自動覆寫（checkbox 下方說明行即時顯示目前模式），取消勾選則自動以最新 60/20/20 重新填入
- **ETF／指標型標的（如 0050、0056）沒有月營收**：系統自動把「整欄皆空」的特徵（revenue_mom/yoy）從該次訓練剔除並記錄在模型檔（feature_columns），推論端同樣自動放行、以 NaN 餵給 LightGBM/XGBoost 原生缺值處理；「建立特徵」對無營收標的不再誤報失敗。**反向亦然**：個股沒有 ETF 折溢價特徵（etf_premium_*），推論端同樣以 NaN 結構性放行——即使最近完成的模型是用 ETF 資料訓練的，個股預測也能正常執行（反之：ETF 模型缺折溢價欄位不影響 ETF 預測；個股模型缺營收欄位不影響 ETF 預測）。特徵剔除的判定單位是「每一檔參與訓練的股票」：混合個股+ETF 訓練時，系統會自動取「所有股票都有的特徵交集」（ETF 缺營收特徵、個股缺折溢價特徵，兩組都會被剔除），因此**兩種標的的樣本都能保留**，混合訓練可正常運作；單一標的訓練行為不變
- **滾動式重訓練（避免訓練邊界停留在過去）**：首頁日期欄位載入時自動填成「目前資料的最新 60/20/20」，另有「**一鍵重訓練（最新 60/20/20，含同步／特徵／標籤）**」按鈕——完整管線：**同步資料（個股行情/籌碼/月營收＋美金匯率＋TWII/SOX/VIX 指數＋ETF 淨值）→ 建立特徵 → 建立標籤 → 訓練**，且 60/20/20 切分在**同步完資料後**才重算（資料每次增長，切分自動右移，模型永遠學到最新資料）——`predict()` 永遠使用最近一次完成的模型，因此重訓後決策看板即套用新模型。可設定「滾動視窗年數」（預設 3 年、0＝全部歷史）只訓練近期資料。排程化：`python manage.py walkforward_retrain --stock-codes 2330 --dry-run` 先預覽切割日，去掉 `--dry-run` 真正訓練，掛 Windows 排程任務／cron 即可每日或每週自動滾動更新
- **訓練納入最新資料（勾選「訓練納入最新資料」／指令加 `--include-latest`）**：驗證區間右端延伸至最新交易日，訓練完成後再以「訓練＋驗證」全部資料重新擬合最終模型——模型學到今天為止的市場；AUC 等評估指標仍以原切分誠實計算。未勾選時維持三段切分，最後 20% 保留為盲測區。此設定（`refit_on_all` 旗標）會保存在訓練紀錄的 `hyperparams` 欄位（非模型超參數，僅供事後查驗）
- **一鍵完整訓練管線（同步資料→建立特徵→建立標籤→訓練）**：首頁兩顆一鍵按鈕都是這條完整管線——「**一鍵重訓練**」（`POST /stocks/walkforward-retrain/`）在**同步完資料後**以最新資料自動切 60/20/20；「**自選日期一鍵訓練**」（`POST /stocks/full-pipeline-train/`，必須帶 `dates`）使用上方四個自選日期（需先填滿、訓練結束日必須早於驗證起始日，前端與後端雙重驗證）。管線包含：個股行情/籌碼/月營收＋美金匯率＋TWII/SOX/VIX 指數＋ETF 淨值同步 → 日頻/月頻特徵 → 依標籤版本建立標籤 → 建立並執行訓練。單一股票某步失敗只記錄不中斷（其他股票繼續），某一步全部失敗才中止（`dry_run: true` 只預覽切分與步驟、完全不寫入；`include_latest` 同「訓練納入最新資料」）。**已移除「全自動一鍵訓練」**：它與「一鍵重訓練」（最新資料 60/20/20）行為重疊易混淆，最新 60/20/20 一鍵訓練統一由「一鍵重訓練」按鈕提供

### 6.1.1 三顆按鈕功能對照（一鍵重訓練／自選日期一鍵訓練／進行 AI 預測）

| 功能 | **一鍵重訓練**（最新 60/20/20） | **自選日期一鍵訓練** | **進行 AI 預測**（決策看板） |
|---|---|---|---|
| 同步資料（行情/籌碼/月營收/匯率/指數/ETF 淨值） | **有**——管線第①步全量同步（同步後才重算 60/20/20） | 有——管線第①步全量同步 | 無 |
| 建立特徵（日頻＋月頻 32 項） | **有**——管線第②步 | 有——管線第②步 | 無 |
| 建立標籤（進階雙關卡／MVP，依所選版本） | **有**——管線第③步 | 有——管線第③步 | 無 |
| 訓練/驗證切分方式 | **有，最新資料 60/20/20 自動切分**（60% 訓練、20% 驗證、20% 盲測） | **自選四個日期**（`train_end` 必須早於 `validation_start`，前後端雙重驗證；不做 60/20/20） | 不訓練——直接用「AI 模型架構」下拉選定架構**最近一次訓練完成**的模型推論 |
| 時間滾動式（滾動視窗年數） | **有**——「滾動視窗（年）」可設（預設 3 年、0＝全部歷史），切分隨資料成長自動右移 | 無——日期完全由你指定（`lookback_years` 已不適用） | 無——由該次訓練決定 |
| 訓練納入最新資料（include_latest） | 有（可勾選） | 有（可勾選） | 不適用 |
| 股票範圍 | 可填（逗號分隔，留空＝全部股票） | 可填（同左） | 由所選架構最後訓練使用的股票範圍決定 |
| 寫入紀錄 | 有——`ModelTrainingRun`（顯示在首頁訓練紀錄表格） | 有——`ModelTrainingRun`（顯示在首頁訓練紀錄表格） | 有——`StockPrediction`（AI 預測結果）＋重繪看板圖表 |
| 對應 API | `POST /stocks/walkforward-retrain/` | `POST /stocks/full-pipeline-train/`（必須帶 `dates`） | `GET /stocks/<代碼>/prediction-settings/?model_type=...` |

> 決策看板的機率曲線、買賣訊號（`chart-data` / `signals`）與「進行 AI 預測」共用「該架構最近一次訓練完成」的模型——因此**先按「一鍵重訓練」或「自選日期一鍵訓練」完成後，決策看板會自動套用新模型與新時間區間；「進行 AI 預測」就是讓該模型對目前最新資料實際跑一次推論**。

- **本次訓練將用到的資料與參數（資訊卡）**：即時顯示訓練/驗證期間、股票範圍、標籤版本、重複次數、切分模式（自動/自選），以及所選模型的完整說明、超參數（含中文說明、與特徵清單一致的列式排版）與 32 個特徵分類清單；切換「預測模型」下拉即動態更新，不是單一靜態說明。資料來源：`GET /stocks/model-info/`（可 `?model_type=lightgbm|xgboost|tft` 過濾）
- **特徵重要性 Top 20 與 Walk-Forward 穩定性評估（新增）**：樹模型（LightGBM／XGBoost）訓練完成後把 `feature_importance`（LightGBM＝split、XGBoost＝gain）存進訓練紀錄，首頁卡片以長條圖顯示 Top 20，單一特徵佔比 ≥35% 顯示「模型可能背下股票代碼」警告；「Walk-Forward 穩定性評估」卡片依模型類型彙整每輪 AUC／PR-AUC，輸出「穩定性：足／不足」與「參考價值百分比」。另加入樹模型 early stopping（50 輪）與 1800 秒訓練超時防護（詳見 MODEL_TUNING_GUIDE.md 五／六節）
- **ETF 折溢價特徵（新增）**：折溢價率＝(收盤價−淨值)/淨值×100（正＝溢價、負＝折價），是 ETF 進出場的重要考量。系統新增 `EtfNavData` 淨值資料表，並為 ETF 標的計算 4 個專用特徵：`etf_premium_discount`（當日折溢價）、`etf_premium_ma5`（5日均折溢價）、`etf_premium_z20`（20日 z-score，偏離近期常態的程度）、`etf_premium_chg_5d`（5日變化）。非 ETF 標的這組特徵整組留空、訓練自動剔除；淨值過期超過 7 天不會與最新股價硬湊（防止失真）。**歷史淨值回補**：免費歷史淨值 API 目前不可得，請自投信官網／證交所下載歷史淨值另存 CSV（欄位 `date,nav[,close]`，中文表頭亦可），執行 `python manage.py import_etf_nav 0050 --csv nav.csv` 匯入後重跑「建立特徵」；上線後每日「檢查是否有新資料」也會對 ETF 自動累積當日淨值
- **跨市場資料同步已納入一鍵管線**：「檢查是否有新資料」會連動同步美金匯率與 TWII/SOX/VIX 指數（日頻特徵的必要輸入）。另外 yfinance 在含中文的專案路徑下會因 curl 無法讀取 CA 憑證而讓所有下載回傳空值（被誤判為「沒有新資料」），`settings.py` 已內建自動修復（將憑證複製到純 ASCII 暫存路徑並設定 `SSL_CERT_FILE`）；TWII 對齊改用 merge_asof＋有限度前值填補，避免指數缺日讓 beta 的 rolling 視窗連環變 NaN
- **預測基準日語義**：基準日＝「最後一個特徵完整的交易日」，不是明天——模型用 t 日收盤可得的特徵預測 t 起 20 個交易日；融券等籌碼資料為 T+1 公布，因此當日收盤後跑管線，隔日早上基準日才會推進到當天
- 舊網址相容：`/stocks/<代碼>/strategy-settings/` 與 `/analysis/?stock_code=<代碼>` 皆自動轉址到決策看板對應位置

### 6.2 資料清洗與 MariaDB 相容性（NaN 防護）

專案同時支援 PostgreSQL 與 MariaDB/MySQL，但兩者對 NaN 的行為不同：**PostgreSQL 可以把 NaN 存進數值欄位，MariaDB/MySQL 會直接丟例外**（`Out of range` / `Incorrect value`）。資料源（FinMind/yfinance）偶爾會對特定日期/標的回傳 NaN，若未攔截，逐筆寫入會「前面幾筆成功、碰到 NaN 那筆整個同步中斷」——畫面上就是「有寫入資料但出現 NaN 錯誤」。現行防護有兩層：

1. **清洗層（`apps/sync/data_cleaning.py`）**
   - `clean_stock_price_rows` / `clean_revenue_rows` / `clean_time_series_rows`：關鍵數值（OHLC、還原收盤、營收金額、匯率、指數收盤）為 **NaN 時整列剔除**（NaN 是廢列，不能存 NULL 污染特徵；注意 pandas 中 `NaN <= 0` 為 False，只用 `<= 0` 過濾攔不到 NaN）
   - `clean_chip_data_rows`：買賣超/資券欄 NaN **補 0** 並在 report 記錄 `fixed_nan`
2. **寫入邊界閘門（第二道保險）**
   - `sanitize_val()` / `clean_defaults()`：NaN / ±Inf / `pd.NA` 一律轉 `None`（寫成 NULL）
   - 已套用於所有寫入點：`stocks`（行情/籌碼/營收）、`fx_data`（匯率）、`market_data`（指數）的 `update_or_create(defaults=...)`

> 曾有同學的 MariaDB 環境在「同步個股資料」出現 NaN 錯誤、資料寫了一半：正是舊版清洗只用 `<= 0` 過濾且逐筆寫入沒有容錯。現行版本已從源頭擋掉；若資料庫仍殘留舊的髒資料，建議重建資料庫（`DROP` 後重新 `migrate`）再完整同步一次。

### 7.（選用）啟用 Celery 非同步訓練

預設訓練是同步執行（API 會等到訓練完成才回應）。要改成非同步：

1. 啟動 Redis（`brew install redis && brew services start redis`，或用 Docker：`docker run -d -p 6379:6379 redis`）
2. 另開一個終端機視窗，啟用虛擬環境後執行：`celery -A stock_predictor worker --loglevel=info`
3. 把 `.env` 裡的 `TRAINING_ASYNC` 改成 `True`

沒有做這三步的話，請保持 `TRAINING_ASYNC=False`（預設值），系統會用原本同步的方式訓練，不影響使用。

### 8. 啟動伺服器

```bash
python manage.py runserver
```

瀏覽器打開 `http://127.0.0.1:8000/`。

## 特徵清單（32 個數值特徵）

### 原始資料層（爬蟲存入資料庫）

| 資料 | 資料庫表 | 欄位 |
|---|---|---|
| 三大法人買賣超（外資/投信） | `StockChipData` | `foreign_net_buy` / `trust_net_buy` |
| 融資餘額張數 | `StockChipData` | `margin_balance` |
| 單月營收金額 | `StockMonthlyRevenue` | `revenue_amount` |
| 還原收盤價 / 成交量 | `StockPrice` | `adjusted_close` / `volume` |
| SOX 指數收盤價 | `MarketIndexRate` | `close_value` (index_code="SOX") |
| 美元兌台幣匯率 | `UsdTwdRate` | `close_rate` |
| VIX 指數收盤價 | `MarketIndexRate` | `close_value` (index_code="VIX") |

### 特徵工程層（`StockFeatureDaily` + `StockFeatureMonthly`）

**技術面（原有 4 個）**
- `ma20_slope`：20 日移動平均線 (MA20) 相較於 5 天前的斜率/變化率
- `ma60_slope`：60 日移動平均線 (MA60) 相較於 5 天前的斜率/變化率
- `rsi`：日線級別 14 日 RSI 指數
- `macd`：日線級別 MACD 柱狀圖（EMA12-EMA26）

**籌碼面（原有 2 個）**
- `institutional_net_buy_ratio`：法人買賣超佔近 20 日成交量比例 (%)
- `margin_usage_ratio`：資券使用率 (%)

**相對大盤（原有 2 個）**
- `excess_return_vs_benchmark`：相對大盤 (TWII) 超額報酬
- `beta_vs_benchmark`：相對大盤 Beta 值

**跨市場（原有 4 個）**
- `usd_twd_change_rate`：美元兌台幣匯率月變動率
- `sox_change_rate`：費城半導體指數月變動率
- `vix_level_percentile`：VIX 相對水位（歷史百分位）
- `vix_change_rate`：VIX 月變動率

**新增：籌碼面進階（3 個）**
- `inst_net_buy_20d`：外資與投信近 20 個交易日的累計買賣超總和
- `inst_accel`：籌碼加速度（近 5 日日均買超 / 近 20 日日均買超）
- `margin_mom`：融資餘額相較於 20 個交易日前的月變化率 (MoM)

**新增：技術面進階（7 個）**
- `bias_ma20`：股價相對於 MA20 的乖離率
- `bias_ma60`：股價相對於 MA60 的乖離率
- `vol_ratio_5_20`：量能比（5 日平均成交量 / 20 日平均成交量）
- `volatility_20d`：近 20 個交易日每日對數報酬率的年化波動率
- `rsi_diff_3d`：日線級別 14 日 RSI 在近 3 天的增減變化量
- `rsi_monthly`：月線層級的 14 個月 RSI 指數
- `macd_monthly`：月線層級的 MACD 指數（EMA12-EMA26）

**新增：跨市場進階（4 個）**
- `sox_mom`：費城半導體指數的 20 日累積漲跌幅
- `rs_vs_sox`：個股對費半的相對強弱度（個股 20 日漲跌幅 - 費半 20 日漲跌幅）
- `usd_twd`：對齊交易日後的美元兌台幣匯率
- `vix`：對齊交易日後的 VIX 恐慌指數

**新增：ETF 折溢價（4 個，僅 ETF 標的有值；來源：`EtfNavData` 淨值表）**
- `etf_premium_discount`：當日折溢價率 (%)＝(收盤價−淨值)/淨值×100，正為溢價、負為折價
- `etf_premium_ma5`：折溢價率近 5 日平均 (%)
- `etf_premium_z20`：折溢價率近 20 日 z-score（目前偏離近期常態幾個標準差；大幅正 z＝異常溢價、均值回歸壓力大）
- `etf_premium_chg_5d`：折溢價率相較 5 日前的變化 (百分點)

> 折溢價歷史資料回補：`python manage.py import_etf_nav <代碼> --csv <檔案>`（欄位 `date,nav[,close]`）；每日「檢查是否有新資料」會對 ETF 自動累積當日淨值。淨值過期超過 7 天時特徵留空，不與最新股價硬湊。

**月頻（原有 2 個，存於 `StockFeatureMonthly`）**
- `revenue_mom`：單月營收相較於上個月的成長率 (MoM)
- `revenue_yoy`：單月營收相較於去年同月的成長率 (YoY)

## 提醒

- `TA-Lib` 需要系統底層 C 函式庫才能安裝，這個 venv 裡**還沒裝**，先不影響已完成的爬蟲功能；等要做技術指標特徵時再另外處理（Mac 用 `brew install ta-lib`，Windows 需要下載對應的 wheel 檔）
- **TFT 需要 PyTorch**（CPU 版即可，`requirements.txt` 已含 `torch>=2.0`）；未安裝時選 TFT 訓練會收到清楚的安裝指引而非難懂的錯誤。TFT 為序列模型：每個樣本是「最近 seq_len 個交易日」的特徵視窗（標籤＝視窗最後一天），推論端會自動收集最新 seq_len 個特徵完整交易日組成視窗預測；CPU 訓練時間明顯比樹模型長，建議先用小 `epochs` / 小 `d_model` 試跑。調參指引見 `MODEL_TUNING_GUIDE.md` 第四節
- **一鍵健檢**：執行 `python smoke_check.py` 可一次驗證資料層（特徵定義、ETF 淨值）、模型層（三種模型 bundle 可載入、序列旗標正確、雙標的端到端預測）、API 層（20+ 端點含 POST 幂等重算與一鍵管線 dry-run）、模板層（TFT 預設、摘要卡、UI 元素、無 emoji 檢查）；任何 FAIL 以非 0 結束碼離開，適合掛 Windows 排程／cron 每日自檢
- **盲測區策略回測**：`python manage.py backtest_strategy 2330`——把模型機率變成一筆筆交易（訊號日次一交易日收盤進場、持有 20 個交易日、進場價 95% 觸價停損、成本 0.585% 來回），輸出勝率／年化夏普／最大回撤／對比個股與大盤買進持有；`--json report.json` 另存完整報告（含每筆交易）。最近訓練若沒留下盲測區（驗證區延伸至資料最新日）會自動改在驗證區回測並標註「非盲測、僅供參考」；檔案已遺失的舊模型紀錄會自動跳過
- **SHAP 誤導特徵診斷（黃金流程，已納入訓練流程）**：訓練完成後**自動**對驗證區間跑 SHAP 誤導特徵診斷並把報告存回 `ModelTrainingRun.diagnostics`（超參數 `auto_shap=false` 可關閉；TFT 置換重要性時間預算 60 秒，`shap_time_budget` 可調），首頁「SHAP 誤導特徵診斷」卡片與 Walk-Forward 穩定性表都會顯示。另保留手動指令 `python manage.py diagnose_model 2330 --model-type lightgbm --auto --retrain`——用於歷史舊紀錄補診斷、或帶修正重訓比對新舊 AUC：兩棵樹用 TreeSHAP（獨大/作弊特徵、SHAP 方向違反領域直覺、依賴曲線鋸齒三類證據），TFT 用逐股置換重要性；`--auto` 自動剔除獨大特徵＋套用領域直覺單調約束後重訓，`--retrain` 產出「新舊 AUC 對比」（情況 A：略降但更安全／情況 B：提升去噪成功）。需 `pip install shap`（`requirements.txt` 已含 `shap>=0.52`）
- FinMind（免費版）與 yfinance / Yahoo Finance 都不需要 API Token
- `.env` 不可提交到 Git
- 標籤版本預設為進階雙關卡（`USE_ADVANCED_LABEL_GATE = True`），如需改用 MVP 單門檻請在首頁下拉選單或加 `--label-version mvp` / `--mvp` 參數
- 訓練/驗證日期不需手寫：首頁載入時自動填成「目前資料的最新 60/20/20」（隨資料成長自動右移），可手動覆寫或按「自動切分」重算，或勾選「自選日期模式」後完全手動；日期區間也可透過 API 手動指定（見 `ModelTrainingRun`）
- 新增特徵後：需 `python manage.py makemigrations` + `python manage.py migrate` 建立新欄位，並重新執行 `python manage.py build_features <股票代碼>` 重新計算特徵
- **MariaDB 部署提醒（團隊協作）**：同學本機若使用 MariaDB，`settings.py` 的 `DATABASES` 改用 MySQL 後端（`django.db.backends.mysql`）並 `pip install mysqlclient`；本專案已內建 NaN 清洗與寫入邊界閘門（見 6.2），PostgreSQL／MariaDB 都能穩定同步。專案以 zip 分享時**排除 `venv/`**，收到的人請重建虛擬環境後 `pip install -r requirements.txt`，並先跑 `python smoke_check.py` 健檢
