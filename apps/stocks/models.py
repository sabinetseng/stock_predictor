"""
stocks App 的資料模型骨架。
對應 PROJECT_PLAN.md 第 5 節資料庫設計。
先只定義欄位結構，計算邏輯之後在 services.py 補上。
"""
from django.db import models


class StockBasicInfo(models.Model):
    """個股基本資料：股票代碼、名稱、產業分類、股本等。"""

    stock_code = models.CharField(max_length=10, unique=True, verbose_name="股票代碼")
    stock_name = models.CharField(max_length=50, verbose_name="股票名稱")
    industry = models.CharField(max_length=50, blank=True, verbose_name="產業分類")
    shares_outstanding = models.BigIntegerField(null=True, blank=True, verbose_name="流通在外股數")
    is_active = models.BooleanField(default=True, verbose_name="是否仍在交易")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "個股基本資料"

    def __str__(self):
        return f"{self.stock_code} {self.stock_name}"


class StockPrice(models.Model):
    """個股日線行情，含原始收盤價與還原收盤價。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="prices")
    trade_date = models.DateField(verbose_name="交易日期")
    open_price = models.DecimalField(max_digits=12, decimal_places=4)
    high_price = models.DecimalField(max_digits=12, decimal_places=4)
    low_price = models.DecimalField(max_digits=12, decimal_places=4)
    close_price = models.DecimalField(max_digits=12, decimal_places=4, verbose_name="原始收盤價")
    adjusted_close = models.DecimalField(
        max_digits=12, decimal_places=4, verbose_name="還原收盤價（計算 Return20 一律用這欄）"
    )
    volume = models.BigIntegerField(default=0, verbose_name="成交量")
    source_name = models.CharField(max_length=30, verbose_name="資料來源，例如 FinMind / yfinance")

    class Meta:
        verbose_name = "個股日線行情"
        unique_together = ("stock", "trade_date")
        indexes = [models.Index(fields=["stock", "trade_date"])]


class StockChipData(models.Model):
    """個股籌碼面原始資料：外資/投信買賣超、融資融券張數。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="chip_data")
    trade_date = models.DateField()
    foreign_net_buy = models.BigIntegerField(default=0, verbose_name="外資買賣超張數")
    trust_net_buy = models.BigIntegerField(default=0, verbose_name="投信買賣超張數")
    margin_balance = models.BigIntegerField(default=0, verbose_name="融資餘額張數")
    short_balance = models.BigIntegerField(default=0, verbose_name="融券餘額張數")
    source_name = models.CharField(max_length=30)

    class Meta:
        verbose_name = "個股籌碼面原始資料"
        unique_together = ("stock", "trade_date")


class StockMonthlyRevenue(models.Model):
    """個股月營收，務必保留「實際公告日期」避免 look-ahead bias。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="monthly_revenue")
    revenue_month = models.DateField(verbose_name="營收所屬月份（該月第一天）")
    announcement_date = models.DateField(verbose_name="實際公告日期，特徵對齊必須用這欄")
    revenue_amount = models.BigIntegerField()
    mom_growth = models.FloatField(null=True, blank=True, verbose_name="月增率 MoM")
    yoy_growth = models.FloatField(null=True, blank=True, verbose_name="年增率 YoY")
    source_name = models.CharField(max_length=30)

    class Meta:
        verbose_name = "個股月營收"
        unique_together = ("stock", "revenue_month")


class StockFeatureDaily(models.Model):
    """
    日頻特徵表：技術面 + 標準化後籌碼面（法人買賣超佔比、資券使用率）。
    月頻資料（營收）以 forward-fill 方式對齊到這裡，需搭配 announcement_date 判斷是否可用。
    """

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="daily_features")
    trade_date = models.DateField()

    # 技術面
    ma20_slope = models.FloatField(null=True, blank=True)
    ma60_slope = models.FloatField(null=True, blank=True)
    rsi = models.FloatField(null=True, blank=True)
    macd = models.FloatField(null=True, blank=True)

    # 籌碼面（已轉換為比例，不可用原始張數）
    institutional_net_buy_ratio = models.FloatField(
        null=True, blank=True, verbose_name="法人買賣超佔近20日成交量比例 (%)"
    )
    margin_usage_ratio = models.FloatField(
        null=True, blank=True, verbose_name="資券使用率 (%)"
    )

    # 相對大盤強弱勢
    excess_return_vs_benchmark = models.FloatField(null=True, blank=True, verbose_name="相對大盤超額報酬")
    beta_vs_benchmark = models.FloatField(null=True, blank=True)

    # 跨市場與總經（來源：fx_data.UsdTwdRate / market_data.MarketIndexRate）
    usd_twd_change_rate = models.FloatField(
        null=True, blank=True, verbose_name="美元兌台幣匯率月變動率"
    )
    sox_change_rate = models.FloatField(
        null=True, blank=True, verbose_name="費城半導體指數（^SOX）月變動率"
    )
    vix_level_percentile = models.FloatField(
        null=True, blank=True, verbose_name="VIX 相對水位（歷史百分位）"
    )
    vix_change_rate = models.FloatField(
        null=True, blank=True, verbose_name="VIX 月變動率"
    )

    # ===== 新增特徵（對應 X_Y_VARIABLE_DEFINITION.md 的完整特徵清單）=====

    # 籌碼面進階特徵
    inst_net_buy_20d = models.FloatField(
        null=True, blank=True, verbose_name="外資與投信近20個交易日累計買賣超總和"
    )
    inst_accel = models.FloatField(
        null=True, blank=True, verbose_name="籌碼加速度（近5日日均買超 / 近20日日均買超）"
    )
    margin_mom = models.FloatField(
        null=True, blank=True, verbose_name="融資餘額相較於20個交易日前的月變化率 (MoM)"
    )

    # 技術面進階特徵
    bias_ma20 = models.FloatField(
        null=True, blank=True, verbose_name="股價相對於 MA20 的乖離率"
    )
    bias_ma60 = models.FloatField(
        null=True, blank=True, verbose_name="股價相對於 MA60 的乖離率"
    )
    vol_ratio_5_20 = models.FloatField(
        null=True, blank=True, verbose_name="量能比（5日平均成交量 / 20日平均成交量）"
    )
    volatility_20d = models.FloatField(
        null=True, blank=True, verbose_name="近20個交易日每日對數報酬率的年化波動率"
    )
    rsi_diff_3d = models.FloatField(
        null=True, blank=True, verbose_name="日線級別14日RSI在近3天的增減變化量"
    )
    rsi_monthly = models.FloatField(
        null=True, blank=True, verbose_name="月線層級的14個月RSI指數"
    )
    macd_monthly = models.FloatField(
        null=True, blank=True, verbose_name="月線層級的MACD指數（EMA12-EMA26）"
    )

    # 跨市場進階特徵
    sox_mom = models.FloatField(
        null=True, blank=True, verbose_name="費城半導體指數的20日累積漲跌幅"
    )
    rs_vs_sox = models.FloatField(
        null=True, blank=True, verbose_name="個股對費半的相對強弱度（個股20日漲跌幅 - 費半20日漲跌幅）"
    )
    usd_twd = models.FloatField(
        null=True, blank=True, verbose_name="對齊交易日後的美元兌台幣匯率"
    )
    vix = models.FloatField(
        null=True, blank=True, verbose_name="對齊交易日後的VIX恐慌指數"
    )

    # ETF 折溢價特徵（僅 ETF 標的有值；來源：EtfNavData）
    etf_premium_discount = models.FloatField(
        null=True, blank=True, verbose_name="ETF 折溢價率 (%)＝(收盤價－淨值)/淨值×100，正為溢價、負為折價"
    )
    etf_premium_ma5 = models.FloatField(
        null=True, blank=True, verbose_name="ETF 折溢價率近 5 日平均 (%)"
    )
    etf_premium_z20 = models.FloatField(
        null=True, blank=True, verbose_name="ETF 折溢價率近 20 日 z-score（偏離近期常態的程度）"
    )
    etf_premium_chg_5d = models.FloatField(
        null=True, blank=True, verbose_name="ETF 折溢價率相較 5 日前的變化 (百分點)"
    )

    class Meta:
        verbose_name = "個股日頻特徵"
        unique_together = ("stock", "trade_date")


class StockFeatureMonthly(models.Model):
    """月頻特徵表：營收 MoM/YoY 等，對齊後才寫入。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="monthly_features")
    effective_date = models.DateField(verbose_name="此特徵開始生效的日期（= 公告日期）")
    revenue_mom = models.FloatField(null=True, blank=True)
    revenue_yoy = models.FloatField(null=True, blank=True)

    class Meta:
        verbose_name = "個股月頻特徵"
        unique_together = ("stock", "effective_date")


class EtfNavData(models.Model):
    """
    ETF 每日「淨值（NAV）」與折溢價原始資料。

    折溢價率定義：premium_discount_pct ＝ (收盤價 − 淨值) / 淨值 × 100
      - 正值＝溢價（市價高於淨值，買貴了）
      - 負值＝折價（市價低於淨值，相對便宜）
    折溢價具有均值回歸特性，是 ETF 進出場的重要考量因素。

    資料來源（source_name）：
      - "csv_manual"：使用者自投信官網／證交所下載的歷史淨值 CSV 匯入
        （python manage.py import_etf_nav <代碼> --csv <檔案>）
      - "twse_mis"：證交所即時報價 API，每個交易日收盤後同步當日一筆、逐日累積
    免費歷史淨值 API 目前不可得（FinMind 免費版無此資料集），因此歷史需以 CSV 回補，
    或從上線日起由每日同步自動累積。
    """

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="nav_data")
    date = models.DateField(verbose_name="資料日期")
    nav = models.DecimalField(max_digits=12, decimal_places=4, verbose_name="每單位淨值 (NAV)")
    close_price = models.DecimalField(
        max_digits=12, decimal_places=4, null=True, blank=True,
        verbose_name="當日收盤價快照（可空；若空則以 StockPrice 補算折溢價）",
    )
    premium_discount_pct = models.FloatField(
        null=True, blank=True, verbose_name="折溢價率 (%)，正為溢價、負為折價"
    )
    source_name = models.CharField(max_length=30, default="csv_manual", verbose_name="資料來源")

    class Meta:
        verbose_name = "ETF 淨值與折溢價"
        unique_together = ("stock", "date")
        indexes = [models.Index(fields=["stock", "date"])]

    def __str__(self):
        return f"{self.stock.stock_code} {self.date} NAV={self.nav}"


class StockLabel(models.Model):
    """標籤表：Return20 與二元分類結果。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="labels")
    trade_date = models.DateField(verbose_name="標籤起算日 t")
    return_20 = models.FloatField(verbose_name="Return20 = (P(t+20)-P(t))/P(t)")
    label = models.SmallIntegerField(verbose_name="0 或 1")
    label_version = models.CharField(
        max_length=20,
        default="advanced",
        verbose_name="標籤版本：mvp（單一門檻）或 advanced（雙關卡，預設）",
    )

    class Meta:
        verbose_name = "股票標籤"
        unique_together = ("stock", "trade_date", "label_version")


class StockPrediction(models.Model):
    """模型預測結果，供前端圖表與表格顯示、標記買點。"""

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="predictions")
    trade_date = models.DateField()
    predicted_probability = models.FloatField(verbose_name="模型預測為1的機率")
    predicted_label = models.SmallIntegerField()
    model_version = models.CharField(max_length=100, verbose_name="模型版本，供之後做版本管理")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "股票預測結果"
        unique_together = ("stock", "trade_date", "model_version")


class ModelTrainingRun(models.Model):
    """
    記錄每一次模型訓練所使用的訓練/驗證區間與結果。
    settings.py 裡的 TRAIN_DATE_START 等 4 個常數只是「預設值」，
    實際訓練時的區間以這張表記錄的為準，前端可手動輸入調整。

    用途：
    1. 讓訓練/驗證區間可以手動調整，不寫死。
    2. 支援 walk-forward（滾動式）回測：同一個模型可以訓練多輪、
       每輪用不同區間，用來檢驗模型是否能套用到未來更長的時間（5年/10年），
       而不是只看單一次切分的結果。
    """

    STATUS_CHOICES = [
        ("pending", "排隊中"),
        ("running", "訓練中"),
        ("completed", "已完成"),
        ("failed", "失敗"),
    ]

    # 訓練與驗證區間，皆可手動輸入，不強制等於 settings.py 的預設值
    train_start = models.DateField(verbose_name="訓練資料起始日")
    train_end = models.DateField(verbose_name="訓練資料結束日")
    validation_start = models.DateField(verbose_name="驗證資料起始日")
    validation_end = models.DateField(verbose_name="驗證資料結束日")

    model_type = models.CharField(max_length=20, default="lightgbm", verbose_name="lightgbm 或 xgboost")
    model_version = models.CharField(max_length=100, verbose_name="模型版本標籤，例如 v1_2018_2024")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # 這次訓練用哪些股票、哪個標籤版本，方便之後回頭查驗
    stock_codes = models.CharField(
        max_length=500, blank=True,
        verbose_name="訓練用的股票代碼，逗號分隔；留空代表使用資料庫內全部股票",
    )
    label_version = models.CharField(
        max_length=20, default="advanced", verbose_name="mvp 或 advanced（預設），需對應 StockLabel.label_version"
    )
    model_file_path = models.CharField(max_length=500, blank=True, verbose_name="訓練完成後的模型檔案路徑")

    # 訓練用的超參數（依 model_type 不同，接受的欄位也不同，見 model_training.py 的 DEFAULT_HYPERPARAMS）
    hyperparams = models.JSONField(
        default=dict, blank=True,
        verbose_name="訓練用的超參數，例如 n_estimators/learning_rate/num_leaves/max_depth",
    )

    # 驗證結果指標（訓練完成後才會有值）
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    auc = models.FloatField(null=True, blank=True)
    pr_auc = models.FloatField(null=True, blank=True, verbose_name="PR-AUC，不平衡標籤建議優先看這個")

    # 統計上單次訓練的結果有抽樣變異，重複訓練 N 次（不同亂數種子）才能看出結果穩不穩定
    n_repeats = models.IntegerField(default=1, verbose_name="重複訓練次數（不同亂數種子），統計上建議 >1")
    auc_std = models.FloatField(null=True, blank=True, verbose_name="N次訓練的AUC標準差，越小代表結果越穩定")
    pr_auc_std = models.FloatField(null=True, blank=True, verbose_name="N次訓練的PR-AUC標準差")

    # 距離訓練期多遠（月數），用於畫「表現 vs 距離訓練期多遠」的衰退趨勢圖
    months_from_train_end = models.IntegerField(
        null=True, blank=True, verbose_name="驗證期起始月 與 訓練期結束月 相差幾個月"
    )

    # 特徵重要性（MODEL_TUNING_GUIDE.md 五-4）：樹模型才有——LightGBM 預設 split（使用次數）、
    # XGBoost 預設 gain（增益）；內容為 [{"feature": 欄位名, "importance": 值}] 依大到小排序。
    # TFT 無原生重要性，訓練完成後記錄空清單。供首頁「特徵重要性 Top 20」卡片顯示。
    feature_importance = models.JSONField(
        default=list, blank=True,
        verbose_name="特徵重要性（樹模型訓練完成後記錄；TFT 為空清單）",
    )

    # 診斷黃金流程（MODEL_TUNING_GUIDE.md「SHAP 誤導特徵診斷」）：diagnose_model 指令
    # 對「已完成模型」跑 SHAP／置換重要性 → 誤導特徵與修正建議 →（可選）帶修正重訓後，
    # 把完整報告（含新舊 AUC 對比）存進這裡。內容結構見 apps/stocks/diagnostics.py。
    diagnostics = models.JSONField(
        default=dict, blank=True,
        verbose_name="誤導特徵診斷報告（diagnose_model 產出；未診斷過為空 dict）",
    )

    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "模型訓練紀錄"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.model_version} [{self.train_start}~{self.train_end}] -> [{self.validation_start}~{self.validation_end}]"


class ModelArtifact(models.Model):
    """訓練完成模型檔的入庫備份。

    Render 免費方案磁碟為暫態，trained_models/ 會在重新部署／重啟時被清空；
    把模型檔 bytes 存進資料庫，載入端（model_artifacts.get_model_file）在
    磁碟檔遺失時自動還原，讓「已完成」的訓練持續可用（預測／診斷／回測）。
    """

    run = models.OneToOneField(
        ModelTrainingRun, on_delete=models.CASCADE, related_name="artifact",
        verbose_name="對應的訓練紀錄",
    )
    file_name = models.CharField(max_length=500, verbose_name="模型檔名")
    content = models.BinaryField(verbose_name="模型檔內容（joblib bytes）")
    size_bytes = models.IntegerField(default=0, verbose_name="檔案大小（bytes）")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="入庫時間")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新時間")

    class Meta:
        verbose_name = "模型檔入庫備份"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"{self.file_name}（{self.size_bytes} bytes）"
