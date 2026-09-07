# 股票預測模型調參與防過擬合指南

> 調參的核心目標：**在提升預測能力的同時，嚴格防止模型「過擬合」（Overfitting，即模型死記歷史數據，導致線上實測大虧）。**
> 判斷模型「訓練得好」**不能只看訓練集分數**，必須看**驗證集（Validation）或回測（Backtest）**的指標（如 MSE 降低、IC 提升、或策略夏普比率提高）。當**驗證集錯誤率最低，且訓練集與驗證集的誤差最接近**時，就是最佳參數。

---

## 一、如何判斷「訓練得好不好」？（評估標準）

**不能只看訓練集分數**，必須看「驗證集 / 回測」表現：

| 評估指標 | 良好訊號 | 危險訊號（過擬合） |
|---|---|---|
| 損失函數（MSE / LogLoss） | 訓練集與驗證集**同步下降**並收斂 | 訓練集持續下降，但驗證集**反彈上升** |
| 準確率 / AUC | 驗證集 AUC 穩定（±0.03 內） | 訓練集 95%，驗證集只有 55% |
| 特徵重要度 | 分散在多個技術指標 | 過度依賴某一個股票代碼或單一特徵 |
| Walk-Forward 回測 | 多輪區間皆穩定 | 第一輪好，第二輪以後大幅衰退 |

**一句話總結：驗證集錯誤率最低，且與訓練集差距最小，才是最佳參數。**

---

## 二、LightGBM 調參指南（Leaf-wise 葉子生長，重點管「樹的複雜度」）

### 核心可調參數

| 參數 | 作用 | 股票預測建議值 |
|---|---|---|
| `num_leaves` | 控制樹的複雜度（最重要） | **15 - 63**，不要超過 127 |
| `max_depth` | 限制樹的高度 | **3 - 8** |
| `learning_rate` | 學習率（每步更新幅度） | **0.01 - 0.05**（調小要配 `n_estimators` 加大） |
| `min_data_in_leaf` | 葉子節點最小樣本數 | **500 - 2000**（防資料雜訊） |
| `feature_fraction` | 每次建樹取多少特徵比例 | **0.6 - 0.8**（增加隨機性） |
| `bagging_fraction` | 每次取多少股票/交易日樣本 | **0.7 - 0.9** |
| `lambda_l2` | L2 正則化 | **1.0 - 10.0** |

### 調整邏輯（防過擬合順序）

1. **先降 `num_leaves` / `max_depth`**（讓樹變簡單）
2. **調高 `min_data_in_leaf`**（確保每條規則都有足夠股票支持）
3. **調高 `bagging_fraction` / `feature_fraction`**（讓模型不要每次都看同樣數據）
4. **調高 `lambda_l2`**（壓低不重要特徵的權重）
5. **調低 `learning_rate` + 調高 `n_estimators`**（精細學習，但用 early stopping 防止過度）

---

## 三、XGBoost 調參（Level-wise 對參數敏感）

### 參數

| 參數 | 作用 | 建議範圍 |
|---|---|---|
| `max_depth` | 樹深（複雜度） | **3 - 6**（比 LightGBM 更嚴格） |
| `learning_rate` / `eta` | 學習率 | **0.01 - 0.1** |
| `subsample` | 行採樣比例 | **0.7 - 0.9** |
| `colsample_bytree` | 特徵採樣 | **0.6 - 0.8** |
| `gamma` | 最小損失下降（後剪枝） | **0.1 - 1.0**（越大越保守） |
| `reg_alpha` | L1 正則化 | **0.1 - 1.0** |
| `reg_lambda` | L2 正則化 | **1.0 - 10.0**（可有效稀疏化特徵） |

### 調整邏輯（跟 LightGBM 不同）

* **嚴格限制 max_depth**（股票設 3-6，結構愈簡單、泛化能力愈強）
* **提高 gamma**（大於 0.5）：讓節點在建立更複雜規則時，需「實質損失下降」才允許分裂
* **調高 subsample / colsample_bytree**：增加隨機性，防止模型背下少數極端行情
* **調整 reg_alpha / reg_lambda**：直接對特徵權重做稀疏化或懲罰，剔除雜訊特徵

---

## 四、TFT（Temporal Fusion Transformer）調參（序列模型，本專案已支援）

> 本專案已內建 TFT（`apps/stocks/tft_model.py`，純 PyTorch、CPU 可訓練）。每個樣本是「最近 seq_len 個交易日」的特徵視窗，標籤＝視窗最後一天；內部由可變選擇網路（VSN）自動學習每天該重視哪些特徵、LSTM 擷取時序動態、多頭自注意力讓視窗內任意兩天直接互動。
>
> **預設值注意**：首頁「預測模型」下拉與決策看板「AI 模型架構」下拉目前都**預設選 TFT**；但 CLI `train_model` 與後端 API 的預設仍是 lightgbm（需明確傳 `--model-type tft` 或 `model_type: "tft"`）。三種架構各自維護獨立的模型版本與訓練紀錄。

### 參數（`DEFAULT_HYPERPARAMS["tft"]`，範圍見 `HYPERPARAM_BOUNDS`）

| 參數 | 預設 | 允許範圍 | 作用／防過擬合建議 |
|---|---|---|---|
| `seq_len` | 30 | 5 ~ 120 | 視窗長度：越長越能看見趨勢，但需要的歷史越多、可用樣本越少 |
| `d_model` | 32 | 8 ~ 128 | 內部隱藏維度：越大表達力越強**也最容易 overfit**，先從 8~32 起步 |
| `n_heads` | 4 | 1 ~ 8 | 自注意力的頭數 |
| `num_layers` | 2 | 1 ~ 6 | LSTM 層數：加深易 overfit、CPU 也明顯變慢 |
| `dropout` | 0.1 | 0 ~ 0.5 | **最重要的正則化旋鈕**；訓練/驗證 AUC 分岔時優先調大 |
| `learning_rate` | 0.003 | 0.001 ~ 1.0 | AdamW 學習率，太大不收斂 |
| `epochs` | 30 | 3 ~ 200 | 只是上限：驗證 AUC 連續數輪無進步會**自動早停**，設大也不會無限訓練 |
| `batch_size` | 128 | 16 ~ 512 | 批次大小 |

### 與樹模型不同的三個重點

1. **內建早停**：以驗證集 AUC 為準提前結束，不需要手動調 `n_estimators` 對應物。
2. **防過擬合順序**：先調大 `dropout` → 再縮小 `d_model` / `num_layers` → 最後才縮短 `seq_len`。
3. **n_repeats 標準差天然有意義**：初始化、dropout、批次順序都含隨機性，重複訓練的變異是真實的；AUC 標準差偏大時，處理方式是加大正則化（dropout），而不是像樹模型那樣檢查子抽樣比例。
4. **CPU 訓練慢是常態**：先用小 `epochs`＋小 `d_model` 驗證管線通不通，再逐步放大找最佳點。

---

## 五、前端網頁設計建議（防止用戶把模型調壞）

> 現況對照：**① 參數範圍限制——✅ 已實作**（前端 `<input type="number">` 帶 min/max，後端 `HYPERPARAM_BOUNDS` 二次驗證並拒絕超出範圍的值）；**② 「訓練中」按鈕鎖定——✅ 已實作**（所有非同步動作共用 `withButtonLoading()` disable＋顯示處理中文字，「一鍵重訓練」另有 confirm() 二次確認）；**③ Walk-Forward 表格——✅ 已落地**為首頁「訓練紀錄表格」（每輪 AUC/PR-AUC 平均±標準差、彩色狀態徽章、分頁瀏覽完整歷史）；**④ TFT 早停——✅ 後端內建**；**⑤ 一鍵訓練管線——✅ 已實作**（首頁「自選日期一鍵訓練」按鈕：同步資料→建立特徵→建立標籤→訓練一次完成，`POST /stocks/full-pipeline-train/` 需帶 `dates`（自選日期）並支援 dry-run 唯讀預覽與後端日期洩漏驗證；「全自動」模式已移除——最新 60/20/20 一鍵訓練統一由「一鍵重訓練」`POST /stocks/walkforward-retrain/` 提供（**同樣含同步／特徵／標籤**，並在同步後重算 60/20/20），避免按鈕功能混淆）；**⑥ 本次訓練資訊卡——✅ 已實作**（`GET /stocks/model-info/` 提供模型說明、32 特徵分類描述與超參數中文說明，隨「預測模型」下拉即時切換）。其中「4. 特徵重要性排序圖」與「5. Walk-Forward 每輪的驗證／回測結果表格」已實作（見下方 ✅ 標記），1~3 仍屬建議方向。

1. **「一鍵優化」按鈕（預設）**
   * 自動幫用戶填入穩健參數：`max_depth=4`, `learning_rate=0.05`, `num_leaves=31`, `min_data_in_leaf=500`
   * 讓新用戶「不會一出手就調出過擬合模型」

2. **「手動進階」切換 + Slider 範圍限制**
   * 手動模式一定要把滑桿限制在「參數合理範圍」內，例如 `num_leaves` 只能 2~127、`max_depth` 只能 2~8
   * 超出範圍前端直接阻擋（後端也要驗證）

3. **訓練完畫「損失函數曲線圖」**
   * 前端顯示兩條線：`訓練集 Loss vs 驗證集 Loss`
   * 如果兩條線**越走越開**，網頁直接跳出提示：
     >  **「模型可能過擬合了，請調低 max_depth 或調高 subsample / bagging_fraction」**

4. **特徵重要性排序圖**
   * 前端畫出 Top 20 的 `feature_importance`（LightGBM 用 split，XGBoost 用 gain/weight）
   * 如果某個特徵一枝獨秀，代表模型可能「背下股票代碼」而不是學到通用規律
   * ✅ **已實作**：訓練完成後把重要性存進 `ModelTrainingRun.feature_importance`（LightGBM＝split 使用次數、XGBoost＝gain 增益），首頁「特徵重要性 Top 20」卡片以 CSS 長條圖呈現（`GET /stocks/feature-importance/`）；單一特徵佔比 ≥35% 時顯示過擬合警告；TFT 無原生重要性不提供

5. **Walk-Forward 每輪的驗證 / 回測結果表格**
   * 用表格呈現每一輪的 AUC / PR-AUC，若第二輪以後明顯衰退，直接顯示「模型穩定性不足」
   * 提供輪迴測試後建議的參數，顯示「模型穩定性不足/足 與參考價值百分比」
   * ✅ **已實作**：`GET /stocks/model-stability/` 依模型類型彙整每輪 AUC／PR-AUC（依驗證期距訓練期月數排序），輸出「穩定性：足／不足」（≥2 輪、平均 AUC ≥0.55、最遠輪相對最初輪衰退 ≤0.05）與「參考價值百分比」（AUC ≥0.55 的輪數佔比），首頁「Walk-Forward 穩定性評估」卡片呈現每輪表格

---

## 六、防止無窮迴圈 / 計算當機的設計

調參或訓練時，如果沒有限制，很容易因為參數設太大導致伺服器卡死或算到當機。以下是這個 Django 專案裡的防護措施（1~3 已於本輪全部補上）：

### 後端（`model_training.py`）已內建的防護

目前 `HYPERPARAM_BOUNDS` 已經限制了參數範圍（例如 `n_estimators` 只允許 10~2000、`num_leaves` 2~512；TFT 專屬參數如 `seq_len` 5~120、`d_model` 8~128、`epochs` 3~200 也都有範圍保護），前端傳入超出範圍的值會被拒絕。建議再加：

1. ~~**`early_stopping_rounds`（樹模型尚未實作）**~~ ✅ **已實作**：`model_training.py` 常數 `EARLY_STOPPING_ROUNDS = 50`；LightGBM 4.x 以 `lgb.early_stopping` callback（`eval_metric="auc"`）、XGBoost 2.x 於建構子帶 `early_stopping_rounds`，皆需驗證集兩類都存在才啟用（單一類別時退回普通訓練）；`best_iteration` 一併記錄於訓練 notes。**TFT 本就內建早停**（以驗證 AUC 早停，`epochs` 只是上限）。

2. ~~**訓練時間上限**~~ ✅ **已實作**：`model_training.py` 常數 `TRAIN_TIMEOUT_SECONDS = 1800`——`run_training` 在每次重複訓練前檢查、TFT 在 `fit_frame` 的 epoch 迴圈內檢查（`max_seconds`），超時提前結束並寫入訓練 notes（已完成輪次照常計入平均值／標準差統計）。

3. **驗證集類別檢查** ✅ **維持並強化**：單一類別時 AUC／PR-AUC 記為 None、不啟用樹模型早停，並寫入訓練 notes（「驗證集僅單一類別…」）供事後查驗。

### 前端（`dashboard/templates/dashboard/index.html`）建議加

1. **Slider 滑桿綁定範圍**：每個參數都用 `<input type="range">` 搭配 min/max/step 限制，使用者無法輸入 100 這種危險值。

2. **提交前二次確認**：當 `n_estimators > 1000` 或 `max_depth > 8` 時，跳出確認視窗提醒「參數偏大，訓練時間可能很長，確定繼續？」。

3. ~~「訓練中」狀態鎖定~~ ✅ **已實作**：所有非同步動作按鈕共用 `withButtonLoading()`（disable＋顯示「處理中…」），避免連點造成多個同時執行的訓練程序；「一鍵重訓練」另加 confirm() 二次確認。

---

## 七、最後的提醒（很重要）

> **調參的核心不是「特徵愈多愈好」，而是：**  
> **「讓驗證集的表現與訓練集接近，且在未來多輪回測都穩定。」**  
> 只要掌握「**讓模型簡單、增加隨機性、加入正則化**」這三個原則，LightGBM 和 XGBoost 都可以調出穩定可用的模型；TFT 則掌握「**加大 dropout、縮小 d_model、依賴內建早停**」。

### 本專案對照

* 模型訓練程式碼：`apps/stocks/model_training.py`
* 參數預設值與範圍：`DEFAULT_HYPERPARAMS` / `HYPERPARAM_BOUNDS`
* TFT 模型實作（VSN + LSTM + 多頭自注意力）：`apps/stocks/tft_model.py`
* TFT 序列組窗與驗證邏輯：`apps/stocks/model_training.py`（`run_training` 的 tft 分支）
* 盲測區策略回測（勝率／夏普／最大回撤／對比買進持有，含手續費 0.1425%×2＋證交稅 0.3%）：`python manage.py backtest_strategy 2330`（`apps/stocks/backtest.py`；最近訓練沒留盲測區時自動退回驗證區並誠實標註「非盲測」）
* **SHAP 誤導特徵診斷（黃金流程：訓練→AUC→SHAP 抓誤導→修正→重訓→比對新舊 AUC）**：`python manage.py diagnose_model 2330 --model-type lightgbm --auto --retrain`（`apps/stocks/diagnostics.py`；兩棵樹用 TreeSHAP、TFT 用逐股置換重要性；自動修正＝剔除獨大/作弊特徵＋領域直覺單調約束，鋸齒狀特徵只能給人工分箱/平滑建議；新舊 AUC 對比分「情況 A：略降但更安全／情況 B：提升去噪成功」；完整報告存回 `ModelTrainingRun.diagnostics`）
* X/Y 變數定義：`X_Y_VARIABLE_DEFINITION.md`
* 一鍵健檢腳本（資料／模型／API／模板四層驗證）：`python smoke_check.py`
* 前端訓練設定：`apps/dashboard/templates/dashboard/index.html`
* 模型預測 API：`/stocks/<股票代碼>/predict/`
* 一鍵訓練管線 API：`POST /stocks/full-pipeline-train/`（自選日期，需帶 `dates`：同步→特徵→標籤→訓練；`dry_run: true` 只預覽）
* 滾動重訓練 API：`POST /stocks/walkforward-retrain/`（**同一條完整管線**：同步→特徵→標籤→訓練，並在同步後以最新 60/20/20 自動切分；支援 `lookback_years` 滾動視窗與 `include_latest` 納入最新資料；`dry_run: true` 只預覽）
* 模型／特徵／超參數說明 API：`GET /stocks/model-info/`（供首頁訓練資訊卡動態顯示）

---

## 八、SHAP 誤導特徵診斷——訓練循環（黃金流程完整版）

> 特徵重要性只回答「模型用了誰」，本節回答更關鍵的問題：「模型是不是被某些特徵誤導了？」
> 對應小白名詞手冊 2-5／5-8，簡報版為「包浩斯極簡版」第 18（訓練循環圖）／19（數字判讀表）頁。

### 8-1 完整迴圈

```
訓練 → 算驗證 AUC／PR-AUC（± 標準差）→ 跑 SHAP／置換重要性診斷
   │
   ├─ 沒有誤導特徵 ──────────► 收工：存模型、留報告（下次的「情況 A／B」基準）
   └─ 有誤導特徵 ─► 三選一修正：剔除獨大／加單調約束／分箱平滑（人工）
                          │
                          └────────► 用相同切分「重訓」──► 回第二格再比一次
```

### 8-2 三類誤導證據與對應修正（`apps/stocks/diagnostics.py` 的判準）

| 類型 | 判準 | 修正動作 |
|---|---|---|
| 獨大／作弊特徵（leak） | 單一特徵 SHAP 佔比 ≥ 35%（與首頁過擬合警語同門檻） | `exclude_features` 剔除後重訓 |
| 方向悖離（direction） | SHAP 與特徵值 Spearman 顯著（p < 0.05）且方向與領域直覺相反 | `monotone_constraints` 單調約束（僅樹模型） |
| 依賴曲線鋸齒（jagged） | 分箱平均 SHAP 的震盪指數 ≥ 0.5 | 人工：分箱／平滑／對數轉換（無法自動修） |

- 領域直覺方向表 `MONO_HINTS`：只對「方向近乎公理」的特徵做方向健檢（均線斜率、法人買超、營收成長＝+1；資券使用率、波動率、VIX＝−1）。RSI／乖離率／匯率／Beta／ETF 折溢價屬非單調，不檢、也不建議加約束（硬約束反而傷模型）。
- TFT 是 PyTorch 序列模型，吃不了 TreeSHAP，改用**逐股置換重要性**（打亂某特徵→看驗證 AUC 掉多少），有 `--time-budget` 秒數預算保護（預設 900 秒），超時誠實標記未測完的特徵。

### 8-3 重訓後怎麼比對（情況 A／B）

| 結果 | 解讀 | 行動 |
|---|---|---|
| **情況 A**：AUC 顯著下降 | 移除誤導特徵後線下略降——舊高分是過擬合／洩漏的虛胖 | 接受新模型：線下低一點，線上更穩更安全 |
| **情況 B**：AUC 顯著提升 | 修正特徵／加約束後去噪成功 | 接受新模型：分類能力真實變強 |
| 變化在 ±（舊標準差＋新標準差）內 | 不顯著 | 需更多證據（加大 n_repeats 或換修正組合）再下結論 |

顯著與否由 `compare_runs` 用「兩次訓練標準差總合」當門檻；報告落在 `ModelTrainingRun.diagnostics`（Admin 可查，兩顆 run 都存同一份）。

### 8-4 執行方式

**自動診斷（預設開啟，已納入訓練流程）**：`run_training` 在訓練完成後自動呼叫
`diagnostics.attach_auto_diagnosis`，對驗證區間跑 SHAP（樹模型 TreeSHAP／TFT 置換
重要性），報告存進 `run.diagnostics`——首頁「SHAP 誤導特徵診斷」卡片、Walk-Forward
穩定性表的 SHAP 欄、訓練紀錄表的 SHAP 欄都會顯示，訓練失敗不影響（診斷失敗只記在 notes）。
自動診斷**不重訓、不做新舊對比**（comparison 為 diagnosed_only）；TFT 時間預算 60 秒。

- 關閉方式：建立訓練時在超參數帶 `"auto_shap": false`（會寫回 run 紀錄供查驗）。
- TFT 時間預算：超參數 `"shap_time_budget": 秒數` 覆蓋（預設 `AUTO_TFT_TIME_BUDGET = 60`）。

**手動指令**（歷史舊紀錄補診斷、或帶修正重訓比對）：

```bash
# 只診斷（不重訓）
python manage.py diagnose_model 2330 --model-type lightgbm
# 診斷＋自動套用修正＋重訓＋比對（情況 A／B）
python manage.py diagnose_model 2330 --model-type lightgbm --auto --retrain
# 手動指定修正（TFT 只有剔除可用）
python manage.py diagnose_model --model-type tft --exclude "rsi" --retrain --time-budget 600
```

- `--auto`：自動套用所有 leak（剔除）＋ direction（單調約束，最多 `--mono-cap` 個，依 |rho| 取最顯著）；jagged 只列人工建議。
- 診斷重訓沿用**同一套訓練/驗證切分、股票範圍、標籤版本**，只注入修正旗標——比對才公平。
- 子已實作：`run_training` 從 `run.hyperparams` 讀 `exclude_features`／`monotone_constraints`（沿用 `refit_on_all` 的控制旗標慣例）；修正措施會寫回這顆 run 的 `hyperparams`，事後可查驗。
- 評估資料一律是「驗證區間」（非盲測）：診斷目的是看模型學到了什麼，不是未來表現；要真實上線證據請用盲測（`backtest_strategy`）。