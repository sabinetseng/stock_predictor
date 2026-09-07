from django.urls import path
from . import views

app_name = "stocks"

urlpatterns = [
    # 靜態路徑要放在動態的 <str:stock_code>/ 之前，否則 Django 會依序比對，
    # 讓 <str:stock_code>/ 把 "training-runs" 這種固定字串誤判成股票代碼吃掉。
    # 手動輸入訓練/驗證區間，建立一筆訓練紀錄
    path("training-runs/create/", views.create_training_run, name="create_training_run"),
    # 查詢歷史訓練紀錄，用於畫「表現 vs 距離訓練期多遠」的趨勢圖
    path("training-runs/", views.list_training_runs, name="list_training_runs"),
    # 查詢行情資料的最早/最新日期（供「自動切分 60/20/20」按鈕使用）
    path("date-range/", views.date_range, name="date_range"),
    # 滾動式重訓練：以最新資料 60/20/20 自動切分並啟動訓練（解決訓練邊界靜止問題）
    path("walkforward-retrain/", views.walkforward_retrain, name="walkforward_retrain"),
    # 一鍵完整訓練管線（自選日期）：同步資料→建立特徵→建立標籤→訓練
    # 最新 60/20/20 一鍵訓練請用 walkforward-retrain（「一鍵重訓練」）
    path("full-pipeline-train/", views.full_pipeline_train, name="full_pipeline_train"),
    # 列出支援模型類型的描述、特徵與超參數說明（供「本次訓練資訊卡」顯示）
    path("model-info/", views.model_info, name="model_info"),
    # 特徵重要性 Top N（最近完成的樹模型訓練；LightGBM=split、XGBoost=gain，供長條圖）
    path("feature-importance/", views.feature_importance, name="feature_importance"),
    # Walk-Forward 穩定性評估（每輪 AUC/PR-AUC 與「足/不足」結論、參考價值百分比）
    path("model-stability/", views.model_stability, name="model_stability"),
    # 最近一次 SHAP 誤導特徵診斷報告（diagnose_model 產出；SHAP 重要性／誤導證據／修正建議）
    path("shap-diagnostics/", views.shap_diagnostics, name="shap_diagnostics"),

    # 以下才是動態股票代碼相關路由
    # 「進行 AI 預測」按鈕對應的 API
    path("<str:stock_code>/predict/", views.predict_stock, name="predict"),
    # 「檢查是否有新資料」按鈕對應的 API（同步行情/籌碼/月營收）
    path("<str:stock_code>/check-new-data/", views.check_new_stock_data, name="check_new_data"),
    # 「建立特徵」按鈕對應的 API（計算日頻特徵 + 月頻特徵）
    path("<str:stock_code>/build-features/", views.build_features, name="build_features"),
    # 「建立標籤」按鈕對應的 API（計算 Return20 與二元分類標籤）
    path("<str:stock_code>/build-labels/", views.build_labels, name="build_labels"),
    # 圖表資料 API（股價/大盤對照/匯率/預測標記，一次回傳）
    path("<str:stock_code>/chart-data/", views.chart_data, name="chart_data"),
    # 策略設定路由（相容性保留）：已整併進決策看板，自動轉址
    path("<str:stock_code>/strategy-settings/", views.strategy_settings, name="strategy_settings"),
    # 策略設定頁對應的 API：取得當前預測資訊
    path("<str:stock_code>/prediction-settings/", views.get_prediction_settings, name="get_prediction_settings"),
    # 查詢單一股票的行情與特徵（放最後，因為它是最寬鬆的 pattern，什麼都能匹配）
    path("<str:stock_code>/signals/", views.stock_signals, name="signals"),
    path("<str:stock_code>/", views.stock_detail, name="detail"),
]