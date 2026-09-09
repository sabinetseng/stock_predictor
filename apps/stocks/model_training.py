"""
LightGBM 模型訓練邏輯，獨立成一個檔案，避免 services.py 過於肥大。

設計原則：
- 訓練資料是「跨股票彙整」的（cross-sectional），不是只練單一股票，
  這樣模型才能學到「什麼樣的特徵組合容易上漲」的通用規律，而不是背下單一股票的走勢。
  可以透過 ModelTrainingRun.stock_codes 限定只用某幾支股票，留空則用資料庫內全部股票。
- 標籤不平衡（Return20>3% 的正樣本通常較少）用 LightGBM 的 is_unbalance=True 處理，
  不需要額外做 SMOTE。
- 驗證指標優先看 AUC / PR-AUC，而不是 accuracy（PROJECT_PLAN.md 已說明原因）。
"""
import os
import time
import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from django.conf import settings
from django.utils import timezone

from .models import StockFeatureDaily, StockFeatureMonthly, StockLabel, ModelTrainingRun

# LightGBM 4.x 的 sklearn 接口對「搭配 early stopping 的 eval_set」會持續噴 deprecation 提醒，
# 屬已知噪音（功能正常且專案仍用回調式早停），全模組層級抑制，讓訓練輸出保持乾淨。
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")

# 對應 X_Y_VARIABLE_DEFINITION.md 的特徵欄位（日頻30個 + 月頻2個 = 32個數值特徵，含 ETF 折溢價4個）
FEATURE_COLUMNS = [
    # 技術面（原有）
    "ma20_slope", "ma60_slope", "rsi", "macd",
    # 籌碼面（原有）
    "institutional_net_buy_ratio", "margin_usage_ratio",
    # 相對大盤（原有）
    "excess_return_vs_benchmark", "beta_vs_benchmark",
    # 跨市場（原有）
    "usd_twd_change_rate", "sox_change_rate", "vix_level_percentile", "vix_change_rate",
    # 新增：籌碼面進階
    "inst_net_buy_20d", "inst_accel", "margin_mom",
    # 新增：技術面進階
    "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
    "rsi_diff_3d", "rsi_monthly", "macd_monthly",
        # 新增：跨市場進階
    "sox_mom", "rs_vs_sox", "usd_twd", "vix",
    # 新增：ETF 折溢價（僅 ETF 標的有值；非 ETF 訓練時整欄皆空會自動剔除）
    "etf_premium_discount", "etf_premium_ma5", "etf_premium_z20", "etf_premium_chg_5d",
    # 月頻（原有）
    "revenue_mom", "revenue_yoy",
]

MODELS_DIR = os.path.join(settings.BASE_DIR, "trained_models")

# 訓練資料至少要有這麼多筆才有意義，太少會訓練出不可靠的模型
MIN_TRAINING_ROWS = 30

# 防護上限（MODEL_TUNING_GUIDE.md 六）：
# - EARLY_STOPPING_ROUNDS：樹模型（LightGBM/XGBoost）以驗證集指標提前停止，
#   n_estimators 只是上限，避免無限訓練到滿（驗證集需兩類都存在才啟用）
# - TRAIN_TIMEOUT_SECONDS：訓練時間上限，超過就停止後續重複訓練／TFT epoch，
#   防止使用者輸入超大參數把伺服器卡死（已完成輪次照常計入統計）
EARLY_STOPPING_ROUNDS = 50
TRAIN_TIMEOUT_SECONDS = 1800

# 每種模型可調整的超參數與預設值。前端只會送出這裡列出的欄位，其他欄位（例如 objective、
# random_state）是系統固定值，不開放調整，避免使用者不小心把訓練目標改掉。
# 每種模型可調整的超參數與預設值。前端只會送出這裡列出的欄位，其他欄位（例如 objective、
# random_state）是系統固定值，不開放調整，避免使用者不小心把訓練目標改掉。
#
# 重要：bagging_fraction/feature_fraction（LightGBM）與 subsample/colsample_bytree（XGBoost）
# 特別設成 <1.0（不是預設的1.0），是因為梯度提升樹在「沒有子抽樣」時，不同 random_state
# 建出來的樹幾乎一模一樣（因為每一步都是貪婪最佳分割，seed 幾乎不影響結果）。
# 如果保持1.0，重複訓練N次（n_repeats>1）算出來的標準差會是假的0，
# 讓使用者誤以為模型「非常穩定」，但其實只是根本沒有真的引入隨機性。
# 開啟子抽樣後，seed 才會真正影響「每棵樹看到哪些資料/特徵」，n_repeats 算出來的標準差才有意義。
DEFAULT_HYPERPARAMS = {
    "lightgbm": {
        "n_estimators": 200,        # 樹的數量，越多學得越細但越容易 overfit、訓練也越慢
        "learning_rate": 0.05,      # 學習率，越小越穩但需要更多樹才能收斂
        "num_leaves": 31,           # 單棵樹的複雜度上限，越大越容易 overfit
        "max_depth": -1,            # 樹的最大深度，-1 代表不限制（由 num_leaves 控制複雜度）
        "bagging_fraction": 0.8,    # 每棵樹隨機取80%的資料列，同時也是 n_repeats 標準差有意義的關鍵
        "feature_fraction": 0.8,    # 每棵樹隨機取80%的特徵欄位
    },
    "xgboost": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,             # XGBoost 慣用 max_depth 控制複雜度，沒有 num_leaves 概念
        "subsample": 0.8,           # 等同 LightGBM 的 bagging_fraction
        "colsample_bytree": 0.8,    # 等同 LightGBM 的 feature_fraction
    },
    # TFT（Temporal Fusion Transformer）：序列模型，每個樣本是「最近 seq_len 個交易日」的
    # 特徵視窗，視窗標籤＝最後一天的 label。CPU 訓練即可（PyTorch）。
    "tft": {
        "seq_len": 30,          # 每個樣本回看幾個交易日（視窗長度）
        "d_model": 32,          # 內部隱藏維度，越大表達力越強但越容易 overfit、越慢
        "n_heads": 4,           # 自注意力的頭數
        "num_layers": 2,        # LSTM 層數
        "dropout": 0.1,         # dropout 比例，正則化用
        "learning_rate": 0.003, # AdamW 學習率
        "epochs": 30,           # 最多訓練輪數（驗證 AUC 早停會提前結束）
        "batch_size": 128,      # 批次大小
    },
}

# 每個超參數的合理範圍，用來擋掉前端傳來的異常值（例如負數的樹的數量）
HYPERPARAM_BOUNDS = {
    "n_estimators": (10, 2000),
    "learning_rate": (0.001, 1.0),
    "num_leaves": (2, 512),
    "max_depth": (-1, 30),
    "bagging_fraction": (0.1, 1.0),
    "feature_fraction": (0.1, 1.0),
    "subsample": (0.1, 1.0),
    "colsample_bytree": (0.1, 1.0),
    # TFT 專屬
    "seq_len": (5, 120),
    "d_model": (8, 128),
    "n_heads": (1, 8),
    "num_layers": (1, 6),
    "dropout": (0.0, 0.5),
    "epochs": (3, 200),
        "batch_size": (16, 512),
}


# 每個特徵的中文描述，用於前端「本次訓練資訊卡」顯示，對應 FEATURE_COLUMNS 的順序
FEATURE_DESCRIPTIONS = {
    "ma20_slope": "20日移動平均線斜率，判斷短期趨勢方向",
    "ma60_slope": "60日移動平均線斜率，判斷中期趨勢方向",
    "rsi": "RSI 相對強弱指數（14日），超過70為超買，低於30為超賣",
    "macd": "MACD 柴斯曼趨吉擇時指標（EMA12-EMA26），動能與趨勢轉換信號",
    "institutional_net_buy_ratio": "法人買賣超佔近20日成交量比例，機構觀察家信號",
    "margin_usage_ratio": "資券使用率，槓杆交易風險指標",
    "excess_return_vs_benchmark": "相對大盤超額報酬，判斷強弱於市場",
    "beta_vs_benchmark": "Beta係數，相對大盤波動性",
    "usd_twd_change_rate": "美元兌台幣匯率月變動率，外匯因子影響",
    "sox_change_rate": "費城半導體指數月變動率，科技股相關性",
    "vix_level_percentile": "VIX 相對水位（歷史百分位），市場恐慌情緒指標",
    "vix_change_rate": "VIX 月變動率，波動性變化",
    "inst_net_buy_20d": "外資與投信近20日累計買賣超，籌碼面動能",
    "inst_accel": "籌碼加速度，近5日 vs 近20日買超變化率",
    "margin_mom": "融資餘額月變化率(MoM)，槓杆情緒指標",
    "bias_ma20": "股價相對MA20乖離率，短期超升/超跌信號",
    "bias_ma60": "股價相對MA60乖離率，中期超升/超跌信號",
    "vol_ratio_5_20": "量能比（5日均量/20日均量），成交量放縮/放大",
    "volatility_20d": "近20日年化波動率，股價波動程度",
    "rsi_diff_3d": "RSI 3日變化量，動能轉強/轉弱訊號",
    "rsi_monthly": "月線層級14月RSI，中期過熱/冷清",
    "macd_monthly": "月線層級MACD，中期趨勢與動能",
    "sox_mom": "費半20日累積漲跌幅，半導體族群動能",
    "rs_vs_sox": "個股對費半相對強弱度，族群輪動指標",
    "usd_twd": "對齊交易日的美元兌台幣匯率，外匯因子",
    "vix": "對齊交易日的VIX恐慌指數，市場情緒",
    "etf_premium_discount": "ETF折溢價率，市價 vs 淨值（僅ETF有值）",
    "etf_premium_ma5": "ETF折溢價5日均線，近期溢價趨勢（僅ETF有值）",
    "etf_premium_z20": "ETF折溢價20日z-score，偏離常態程度（僅ETF有值）",
    "etf_premium_chg_5d": "ETF折溢價5日變化，溢價擴張/收斂（僅ETF有值）",
    "revenue_mom": "月營收月增率(MoM)，基本面成長動能",
    "revenue_yoy": "月營收年增率(YoY)，基本面年成長",
}


# 特徵分類，用於前端分組顯示（順序對應 X_Y_VARIABLE_DEFINITION.md）
FEATURE_CATEGORIES = [
    ("技術面", ["ma20_slope", "ma60_slope", "rsi", "macd"]),
    ("籌碼面", ["institutional_net_buy_ratio", "margin_usage_ratio"]),
    ("相對大盤", ["excess_return_vs_benchmark", "beta_vs_benchmark"]),
    ("跨市場總經", ["usd_twd_change_rate", "sox_change_rate", "vix_level_percentile", "vix_change_rate"]),
    ("籌碼面進階", ["inst_net_buy_20d", "inst_accel", "margin_mom"]),
    ("技術面進階", ["bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
                    "rsi_diff_3d", "rsi_monthly", "macd_monthly"]),
    ("跨市場進階", ["sox_mom", "rs_vs_sox", "usd_twd", "vix"]),
    ("ETF折溢價（僅ETF）", ["etf_premium_discount", "etf_premium_ma5",
                            "etf_premium_z20", "etf_premium_chg_5d"]),
    ("月頻營收", ["revenue_mom", "revenue_yoy"]),
]


# 每種模型的完整資訊，供前端「本次訓練資訊卡」動態顯示。
# 卡片會依 model-type-select 的值即時切換顯示對應的模型說明、特徵處理方式與超參數。
MODEL_INFO = {
    "lightgbm": {
        "name": "LightGBM",
        "type": "梯度提升樹（Leaf-wise 生長）",
        "description": (
            "以 Leaf-wise（以葉子生長）策略建樹：每次只生長到「損失下降最多」的葉子，"
            "因此同等複雜度下比 Level-wise（層生長）更精準，但需透過 num_leaves/max_depth 防過擬合。"
            "對類別不平衡標籤（正樣本稀少）使用 is_unbalance=True 自動加權，無需 SMOTE。"
            "訓練速度快，CPU 即可處理上萬支股票 × 32 個特徵的資料。"
        ),
        "features_note": (
            "將特徵壓平成一行：每筆 = 一支股票在某交易日的 32 個特徵值 "
            "（cross-sectional，不考慮時間序列）。"
        ),
        "is_sequence_model": False,
        "training_time": "秒級（CPU）",
        "hyperparam_descriptions": {
            "n_estimators": "樹的數量；越多學得越細，但越容易 overfit，訓練也越慢",
            "learning_rate": "學習率；越小越穩但需更多樹才能收斂",
            "num_leaves": "單棵樹複雜度上限；越大越容易 overfit",
            "max_depth": "樹最大深度；-1 代表不限制（由 num_leaves 控制複雜度）",
            "bagging_fraction": "每棵樹隨機取部分資料列；開啟子抽樣才讓不同 random_state 產生不同模型",
            "feature_fraction": "每棵樹隨機取部分特徵欄位；增加隨機性防過擬合",
        },
    },
    "xgboost": {
        "name": "XGBoost",
        "type": "梯度提升樹（Level-wise 生長）",
        "description": (
            "以 Level-wise（以層生長）策略建樹：每次擴展同一層的所有葉子，"
            "在深度受限時更穩定，但深樹場景下效率稪低於 LightGBM。"
            "使用 subsample + colsample_bytree 做子抽樣處理類別不平衡，提升泛化能力。"
            "訓練速度快，CPU 即可處理上萬支股票 × 32 個特徵的資料。"
        ),
        "features_note": (
            "將特徵壓平成一行：每筆 = 一支股票在某交易日的 32 個特徵值 "
            "（cross-sectional，不考慮時間序列）。"
        ),
        "is_sequence_model": False,
        "training_time": "秒級（CPU）",
        "hyperparam_descriptions": {
            "n_estimators": "樹的數量；越多學得越細，但越容易 overfit，訓練也越慢",
            "learning_rate": "學習率；越小越穩但需更多樹才能收斂",
            "max_depth": "樹最大深度；XGBoost 慣用 max_depth 控制複雜度",
            "subsample": "每棵樹隨機取部分資料列，等同 LightGBM 的 bagging_fraction",
            "colsample_bytree": "每棵樹隨機取部分特徵欄位，等同 LightGBM 的 feature_fraction",
        },
    },
    "tft": {
        "name": "TFT（Temporal Fusion Transformer）",
        "type": "序列模型 / Temporal Fusion Transformer（純 PyTorch，CPU 即可）",
        "description": (
            "時間序列 Transformer：每個訓練樣本是「最近 seq_len 個交易日」的特徵視窗，"
            "標籤＝視窗最後一天的 Label。架構：Variable Selection Network（VSN）學習哪些特徵重要 "
            "→ 多層 LSTM 擷取時序動態 → 多頭自注意力讓視窗內任意兩天互相關注 "
            "→ Gated Residual Network + GLU 輸出 → 二元分類 logit。"
            "不平衡標籤用 BCEWithLogitsLoss(pos_weight=neg/pos) 處理。"
            "驗證集 AUC 早停；需要 PyTorch（CPU 版即可），訓練時間明顯比樹模型長。"
        ),
        "features_note": (
            "特徵來源與樹模型相同 32 個，但必須組成「seq_len 天的時間窗」才是一個訓練樣本；"
            "缺值處理：訓練期用每特徵平均值補 NaN，推論時亦然。"
            "推論端要求最新 seq_len 個「特徵完整」交易日才能組窗。"
        ),
        "is_sequence_model": True,
        "training_time": "分鐘級（CPU）",
        "hyperparam_descriptions": {
            "seq_len": "視窗長度：每支股票往前看這麼多交易日組成一個訓練樣本；愈長愈能看見趨勢，但需歷史越多",
            "d_model": "內部隱藏維度；越大表達力越強但越容易 overfit、訓練越慢",
            "n_heads": "自注意力頭數；讓視窗內天數從多個角度互相關注",
            "num_layers": "LSTM 編碼器層數；擷取時序動態的深度",
            "dropout": "dropout 比例；隨機丟棄神經元防 overfit，0 表示不丟棄",
            "learning_rate": "AdamW 學習率；太大訓練不穩定，太小收斂慢",
            "epochs": "最多訓練輪數；驗證集 AUC 連續無進步會提前早停",
            "batch_size": "每次更新權重用多少個視窗樣本",
        },
    },
}


def resolve_hyperparams(model_type, user_hyperparams):
    """
    把使用者傳來的超參數與該模型的預設值合併，並做型別與範圍檢查。
    使用者只需要傳想覆蓋的欄位，沒傳的就用預設值；不屬於該模型的欄位會被忽略
    （例如傳了 num_leaves 給 xgboost，因為 xgboost 沒有這個參數，會被丟掉）。

    型別防護（2026-09 新增）：
      - 整數語義欄位（n_estimators/num_leaves/seq_len/epochs/batch_size 等）只接受 int，
        拒絕浮點與字串——先前 TFT 收到 n_heads=2.7 會一路闖進建模才爆。
      - 所有數值欄位拒絕字串（前端 JSON 誤送 "200" 時，範圍比對會拋看不懂的 TypeError）。
      - XGBoost 不接受 max_depth=-1（那是 LightGBM「不限深度」的慣例）：專案慣例
        直接把它轉為 XGBoost 的 0（同樣代表不限制），避免使用者沿用 LightGBM 習慣而訓練失敗。
    """
    if model_type not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"不支援的模型類型：{model_type}，可用值：{list(DEFAULT_HYPERPARAMS.keys())}")

    resolved = dict(DEFAULT_HYPERPARAMS[model_type])
    user_hyperparams = user_hyperparams or {}

    # 具「整數語義」的超參數：允許範圍適用 int；浮點值（如 12.5）或字串一律拒絕
    INT_HYPERPARAM_KEYS = {
        "n_estimators", "num_leaves", "max_depth", "seq_len",
        "d_model", "n_heads", "num_layers", "epochs", "batch_size",
    }

    for key, value in user_hyperparams.items():
        if key not in resolved:
            continue  # 不屬於這個模型的參數，直接忽略
        if key in HYPERPARAM_BOUNDS:
            low, high = HYPERPARAM_BOUNDS[key]
            if value is None or isinstance(value, bool):
                raise ValueError(f"參數 {key} 必須是數值，收到 {value!r}（{type(value).__name__}）")
            if isinstance(value, str):
                raise ValueError(
                    f"參數 {key} 必須是數值，收到字串 {value!r}（請改成數字，例如 200 而非 \"200\"）"
                )
            if key in INT_HYPERPARAM_KEYS and not isinstance(value, int):
                raise ValueError(f"參數 {key} 必須是整數，收到 {value!r}（{type(value).__name__}）")
            if not (low <= value <= high):
                raise ValueError(f"參數 {key}={value} 超出合理範圍（{low} ~ {high}）")
            # XGBoost 不接受 -1（LightGBM 的「不限深度」慣例）；統一轉成 XGBoost 的 0＝不限
            if model_type == "xgboost" and key == "max_depth" and value == -1:
                value = 0
        resolved[key] = value

    return resolved


def _load_dataset(start_date, end_date, stock_codes=None, label_version="advanced"):
    """
    組合訓練/驗證用的資料集：StockLabel（Y）+ StockFeatureDaily（X的日頻部分）
    + StockFeatureMonthly（X的月頻部分，用 merge_asof 對齊到「最近一筆已公告」的營收特徵）。

    回傳：(merged_df, used_cols, dropped_cols) 三元組：
      - merged_df：包含 stock_code / trade_date / 特徵欄位 / label 的 DataFrame
      - used_cols：實際可用的特徵欄位（= FEATURE_COLUMNS 剔除「整欄皆空」者）
      - dropped_cols：被剔除的欄位（例如 ETF 沒有月營收 → revenue_mom/yoy 整欄全空）
    「整欄皆 NaN」的特徵代表該標的結構性不可得（如 ETF 無營收），自動剔除而不是讓
    dropna 把這個標的的樣本整批丟掉；其餘零星缺值（滾動窗格天數不足的早期資料）
    仍維持原本保守策略——整列捨棄。
    """
    label_qs = StockLabel.objects.filter(
        trade_date__gte=start_date, trade_date__lte=end_date, label_version=label_version,
    )
    if stock_codes:
        label_qs = label_qs.filter(stock__stock_code__in=stock_codes)
    label_df = pd.DataFrame(list(
        label_qs.values("stock__stock_code", "trade_date", "label")
    ))
    if label_df.empty:
        return pd.DataFrame(), [], []
    label_df = label_df.rename(columns={"stock__stock_code": "stock_code"})

    daily_qs = StockFeatureDaily.objects.filter(trade_date__gte=start_date, trade_date__lte=end_date)
    if stock_codes:
        daily_qs = daily_qs.filter(stock__stock_code__in=stock_codes)
    daily_field_names = [
        "ma20_slope", "ma60_slope", "rsi", "macd",
        "institutional_net_buy_ratio", "margin_usage_ratio",
        "excess_return_vs_benchmark", "beta_vs_benchmark",
        "usd_twd_change_rate", "sox_change_rate", "vix_level_percentile", "vix_change_rate",
        "inst_net_buy_20d", "inst_accel", "margin_mom",
        "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
        "rsi_diff_3d", "rsi_monthly", "macd_monthly",
        "sox_mom", "rs_vs_sox", "usd_twd", "vix",
        "etf_premium_discount", "etf_premium_ma5", "etf_premium_z20", "etf_premium_chg_5d",
    ]
    daily_df = pd.DataFrame(list(
        daily_qs.values("stock__stock_code", "trade_date", *daily_field_names)
    ))
    if daily_df.empty:
        return pd.DataFrame(), [], []
    daily_df = daily_df.rename(columns={"stock__stock_code": "stock_code"})

    merged = label_df.merge(daily_df, on=["stock_code", "trade_date"], how="inner")
    if merged.empty:
        return pd.DataFrame(), [], []

    # 月頻特徵：逐一股票做 merge_asof，因為不同股票的公告日期完全不同，不能整批對齊
    monthly_qs = StockFeatureMonthly.objects.all()
    if stock_codes:
        monthly_qs = monthly_qs.filter(stock__stock_code__in=stock_codes)
    monthly_df = pd.DataFrame(list(
        monthly_qs.values("stock__stock_code", "effective_date", "revenue_mom", "revenue_yoy")
    ))

    merged["trade_date"] = pd.to_datetime(merged["trade_date"])

    if monthly_df.empty:
        merged["revenue_mom"] = np.nan
        merged["revenue_yoy"] = np.nan
    else:
        monthly_df = monthly_df.rename(columns={
            "stock__stock_code": "stock_code", "effective_date": "trade_date",
        })
        monthly_df["trade_date"] = pd.to_datetime(monthly_df["trade_date"])

        parts = []
        for code, group in merged.groupby("stock_code"):
            group_sorted = group.sort_values("trade_date")
            stock_monthly = monthly_df[monthly_df["stock_code"] == code].sort_values("trade_date")
            if stock_monthly.empty:
                group_sorted = group_sorted.copy()
                group_sorted["revenue_mom"] = None
                group_sorted["revenue_yoy"] = None
            else:
                group_sorted = pd.merge_asof(
                    group_sorted, stock_monthly[["trade_date", "revenue_mom", "revenue_yoy"]],
                    on="trade_date",
                )
            # 統一月頻欄位為 float64（ETF 等整欄 NaN 時也是 NaN 而非 object/None）——
            # 避免 pd.concat 在未來 pandas 版本對「all-NA 欄位」的 dtype 判定改變引發 FutureWarning
            for _c in ("revenue_mom", "revenue_yoy"):
                group_sorted[_c] = pd.to_numeric(group_sorted[_c], errors="coerce").astype("float64")
            parts.append(group_sorted)
        merged = pd.concat(parts, ignore_index=True)

    # 「整欄皆 NaN」的特徵＝該標的結構性不可得（如 ETF 沒有月營收、個股沒有折溢價），
    # 自動剔除，而不是讓 dropna 把這個標的的樣本整批丟掉。
    # 判定單位是「每一檔參與訓練的股票」：任一檔股票的該特徵整欄皆空，
    # 就從這次訓練剔除（混合個股+ETF 訓練時自動取「所有股票都有」的特徵交集，
    # 讓兩種標的的樣本都能留下來；單一標的訓練行為不變）。
    dropped_cols = []
    for col in FEATURE_COLUMNS:
        if merged[col].isna().all():
            dropped_cols.append(col)
            continue
        any_stock_all_nan = merged.groupby("stock_code")[col].apply(lambda s: s.isna().all()).any()
        if any_stock_all_nan:
            dropped_cols.append(col)
    used_cols = [c for c in FEATURE_COLUMNS if c not in dropped_cols]
    if not used_cols:
        raise ValueError("所有特徵欄位皆為空值，無法組成訓練資料集")

    # 其餘欄位的零星缺值（滾動窗格天數還不夠的早期資料）仍整列捨棄，維持原本保守策略
    merged = merged.dropna(subset=used_cols)
    return merged, used_cols, dropped_cols


def _require_torch():
    """TFT 需要 PyTorch；未安裝時給出清楚的安裝指引而不是難懂的 ImportError。"""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "此環境尚未安裝 PyTorch，無法訓練 TFT。請先執行："
            "pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc


STALE_RUN_MINUTES = 30


def mark_stale_runs_failed():
    """把「pending/running 但超過 30 分鐘無心跳」的訓練標記為 failed。

    背景：背景訓練子行程會隨服務重新部署/重啟被殺，或（舊版同步訓練）被
    gunicorn worker timeout 終止，紀錄會永遠停在 pending/running 成為殭屍。
    以 heartbeat_at（無心跳的舊紀錄用 created_at）判斷，逾時自動標記 failed，
    讓「歷史訓練紀錄」反映真實狀態、並解除「同時只能一個訓練」的佔用。
    回傳標記筆數。
    """
    from datetime import timedelta

    from django.db.models import Q

    cutoff = timezone.now() - timedelta(minutes=STALE_RUN_MINUTES)
    stale = ModelTrainingRun.objects.filter(status__in=["pending", "running"]).filter(
        Q(heartbeat_at__lt=cutoff)
        | Q(heartbeat_at__isnull=True, created_at__lt=cutoff)
    )
    marked = 0
    for run in stale:
        run.status = "failed"
        note = (f"背景訓練中斷：超過 {STALE_RUN_MINUTES} 分鐘無心跳"
                f"（可能為服務重新部署/重啟，或訓練程序被終止）")
        run.notes = (run.notes + "；" if run.notes else "") + note
        run.save(update_fields=["status", "notes"])
        marked += 1
    return marked


def run_training(training_run_id):
    """
    執行一次模型訓練，對應一筆 ModelTrainingRun：
    1. 依 train_start~train_end 組訓練集，validation_start~validation_end 組驗證集
    2. 訓練 LightGBM 二元分類模型
    3. 用驗證集算 accuracy / precision / recall / auc / pr_auc
    4. 存模型檔案到 trained_models/，更新 ModelTrainingRun 狀態與結果

    回傳：dict，{"success": bool, "message": str, "data": {...}}
    """
    run = ModelTrainingRun.objects.get(id=training_run_id)
    run.status = "running"
    run.heartbeat_at = timezone.now()
    run.save(update_fields=["status", "heartbeat_at"])
    t_start = time.time()  # 訓練時間上限（TRAIN_TIMEOUT_SECONDS）的起算點

    # ---------------------------------------------------------
    # 心跳執行緒：訓練進行中每 60 秒更新一次 heartbeat_at，讓「歷史訓練紀錄」
    # 能分辨「還在算」與「已死掉」（超過 30 分鐘無心跳會被 mark_stale_runs_failed
    # 自動標記 failed）。訓練結束（成功或失敗）時以 stop_heartbeat 停止。
    # ---------------------------------------------------------
    import threading

    stop_heartbeat = threading.Event()

    def _heartbeat_loop():
        while not stop_heartbeat.wait(60):
            try:
                ModelTrainingRun.objects.filter(id=run.id).update(heartbeat_at=timezone.now())
            except Exception:  # noqa: BLE001
                pass  # 心跳更新失敗不影響訓練

    threading.Thread(target=_heartbeat_loop, daemon=True).start()

    # （可選）refit_on_all：由啟動端在「納入最新資料」模式下注入；
    # 先從 hyperparams 抽出此控制旗標，避免它被當成模型超參數傳入。
    # 診斷黃金流程（diagnose_model）沿用同一慣例注入兩個控制旗標：
    #   exclude_features＝強制剔除的特徵清單（剔除作弊/誤導特徵後重訓）
    #   monotone_constraints＝{特徵: +1/-1} 領域知識單調約束（僅樹模型生效）
    _raw_hp = run.hyperparams if isinstance(run.hyperparams, dict) else {}
    refit_on_all = bool(_raw_hp.pop("refit_on_all", False))
    exclude_features = [str(c) for c in (_raw_hp.pop("exclude_features", None) or [])]
    mono_map = {}
    for _k, _v in (_raw_hp.pop("monotone_constraints", None) or {}).items():
        try:
            _iv = int(_v)
        except (TypeError, ValueError):
            continue
        if _iv in (1, -1):
            mono_map[str(_k)] = _iv
    if refit_on_all or exclude_features or mono_map:
        run.hyperparams = _raw_hp
        run.save(update_fields=["hyperparams"])
    stock_code_list = (
        [c.strip() for c in run.stock_codes.split(",") if c.strip()] if run.stock_codes else None
    )

    try:
        hyperparams = resolve_hyperparams(run.model_type, run.hyperparams)
        # 把「實際使用」的完整參數（含預設值）寫回去，方便之後查驗；
        # resolve_hyperparams 的白名單會濾掉控制旗標，因此這裡把 refit_on_all
        # 補回紀錄（非模型超參數，僅供查驗該次訓練是否以「訓練＋驗證」全量重擬合）
        hyperparams["refit_on_all"] = refit_on_all
        # SHAP 自動診斷開關（非模型超參數，僅供查驗該次訓練是否自動跑診斷）；
        # 從原始 _raw_hp 讀取——resolve_hyperparams 的白名單會濾掉非模型參數。
        # 預設開啟；建立訓練時在超參數帶 auto_shap=false 可關閉。
        auto_shap = _raw_hp.get("auto_shap", True)
        hyperparams["auto_shap"] = bool(auto_shap)
        # TFT 置換重要性的自動診斷時間預算（秒）；未提供時用 diagnostics.AUTO_TFT_TIME_BUDGET
        _shap_budget = _raw_hp.get("shap_time_budget")
        if isinstance(_shap_budget, (int, float)) and _shap_budget > 0:
            hyperparams["shap_time_budget"] = int(_shap_budget)
        # 修正措施寫回紀錄（非模型超參數，供之後查驗這次訓練做過什麼修正）
        if exclude_features:
            hyperparams["exclude_features"] = exclude_features
        if mono_map:
            hyperparams["monotone_constraints"] = mono_map
        run.hyperparams = hyperparams
        run.save(update_fields=["hyperparams"])

        train_df, used_cols, dropped_cols = _load_dataset(run.train_start, run.train_end, stock_code_list, run.label_version)
        val_df, _, _ = _load_dataset(run.validation_start, run.validation_end, stock_code_list, run.label_version)

        # 診斷黃金流程「剔除作弊/誤導特徵」：只剔除實際存在的欄位，剔除後不得為空
        if exclude_features:
            used_cols = [c for c in used_cols if c not in set(exclude_features)]
            if not used_cols:
                raise ValueError("exclude_features 剔除後沒有剩餘特徵，無法訓練")

        # 單調約束（黃金流程「加入領域知識約束」）：LightGBM／XGBoost 的 sklearn 接口
        # 都接受「依 used_cols 順序的 ±1 逗號字串」（0＝不加約束），例如 "1,0,-1,..."。
        # 用字串形式最相容（LightGBM 4.x 的 dict 形式對 sklearn wrapper 不穩）。
        def _mono_kwargs():
            if not mono_map:
                return {}
            return {"monotone_constraints": ",".join(
                str(int(mono_map.get(c, 0))) for c in used_cols)}

        if len(train_df) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"訓練資料筆數不足（{len(train_df)} 筆，至少需要 {MIN_TRAINING_ROWS} 筆），"
                f"請確認該區間已執行過 build_features 與 build_labels"
            )
        if val_df.empty:
            raise ValueError("驗證資料是空的，請確認該區間已執行過 build_features 與 build_labels")

        # 驗證集必須使用與訓練集「完全相同」的特徵欄位組合；
        # 對驗證集套同樣的 dropna，防禦性排除驗證區間才出現的缺值列。
        val_df = val_df.dropna(subset=[c for c in used_cols])

        X_train, y_train = train_df[used_cols], train_df["label"]
        X_val, y_val = val_df[used_cols], val_df["label"]

        # ---------------------------------------------------------
        # 重複訓練 N 次（不同亂數種子），統計上單次訓練的結果有抽樣變異，
        # 只看一次的 AUC 沒辦法判斷這個分數是穩定的實力還是運氣。
        # 固定同一組訓練/驗證區間（不重新切分資料，避免時間序列 look-ahead bias），
        # 只改變模型訓練時的隨機性（LightGBM/XGBoost 內部的隨機取樣、初始化），
        # 重複跑 N 次後回報平均值 ± 標準差。
        # ---------------------------------------------------------
        n_repeats = max(1, run.n_repeats)
        has_both_classes = y_val.nunique() > 1
        repeat_metrics = []
        trained_models = []
        timeout_hit = False

        for i in range(n_repeats):
            # 訓練時間上限（MODEL_TUNING_GUIDE.md 六-2）：超過就停止後續重複訓練，
            # 已完成的輪次照常納入平均值／標準差統計
            if time.time() - t_start > TRAIN_TIMEOUT_SECONDS:
                timeout_hit = True
                break
            seed = 42 + i

            if run.model_type == "tft":
                # ---------------------------------------------------------
                # TFT（Temporal Fusion Transformer）：序列模型。
                # 訓練樣本＝「最近 seq_len 個交易日」的特徵視窗（標籤＝最後一天），
                # 由 fit_frame 內部逐股重建；驗證指標同樣在「視窗」級別計算。
                # 為了讓驗證窗能引用跨訓練邊界的歷史，批次驗證時把
                # 「訓練期末歷史 + 驗證區間」合成每股連續序列再推論，
                # 之後只取驗證段的機率（輸入特徵全部 ≤ 當日，無 look-ahead bias）。
                # ---------------------------------------------------------
                _require_torch()
                from .tft_model import TFTClassifier
                model = TFTClassifier(
                    seq_len=hyperparams["seq_len"], d_model=hyperparams["d_model"],
                    n_heads=hyperparams["n_heads"], num_layers=hyperparams["num_layers"],
                    dropout=hyperparams["dropout"],
                    learning_rate=hyperparams["learning_rate"],
                    epochs=hyperparams["epochs"], batch_size=hyperparams["batch_size"],
                    random_state=seed,
                    max_seconds=max(0, TRAIN_TIMEOUT_SECONDS - int(time.time() - t_start)),
                )
                model.fit_frame(train_df, val_df, list(used_cols))

                _meta_cols = ["stock_code", "trade_date"]
                _tr = train_df[_meta_cols + list(used_cols)].copy()
                _va = val_df[_meta_cols + list(used_cols)].copy()
                _tr["_pos"] = -1 - np.arange(len(_tr))   # 負數＝訓練段（只當歷史用）
                _va["_pos"] = np.arange(len(_va))        # 正數索引＝驗證段原順序
                _src = pd.concat([_tr, _va], ignore_index=True).sort_values(
                    ["stock_code", "trade_date"])
                proba_src = model.predict_proba(_src)[:, 1]
                _src = _src.assign(_proba=proba_src)
                # 依 _pos 還原順序後取驗證段（_pos>=0），順序與原 val_df / y_val 完全一致
                _sorted = _src.sort_values("_pos")
                _val_part = _sorted[_sorted["_pos"] >= 0]
                y_proba = _val_part["_proba"].to_numpy()
                y_pred = (y_proba >= 0.5).astype(int)
            else:
                # 樹模型建構抽成小函式：早停不被支援時需要用「完全相同的超參數」重建一次
                def _make_tree(with_early_stopping=False):
                    if run.model_type == "xgboost":
                        # scale_pos_weight 是 XGBoost 處理不平衡標籤的方式，等同 LightGBM 的 is_unbalance
                        pos_count = int((y_train == 1).sum())
                        neg_count = int((y_train == 0).sum())
                        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
                        kwargs = dict(
                            objective="binary:logistic",
                            scale_pos_weight=scale_pos_weight,
                            n_estimators=hyperparams["n_estimators"],
                            learning_rate=hyperparams["learning_rate"],
                            max_depth=hyperparams["max_depth"],
                            subsample=hyperparams["subsample"],
                            colsample_bytree=hyperparams["colsample_bytree"],
                            random_state=seed,
                            eval_metric="logloss",
                        )
                        if with_early_stopping:
                            # XGBoost >= 1.6：early_stopping_rounds 放建構子（2.0 已移除 fit 參數）
                            kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
                        kwargs.update(_mono_kwargs())
                        return xgb.XGBClassifier(**kwargs)
                    return lgb.LGBMClassifier(
                        objective="binary",
                        is_unbalance=True,   # 處理 Return20>3% 這種不平衡標籤，不用額外做 SMOTE
                        n_estimators=hyperparams["n_estimators"],
                        learning_rate=hyperparams["learning_rate"],
                        num_leaves=hyperparams["num_leaves"],
                        max_depth=hyperparams["max_depth"],
                        bagging_fraction=hyperparams["bagging_fraction"],
                        feature_fraction=hyperparams["feature_fraction"],
                        bagging_freq=1,  # 每1棵樹就重新抽樣一次，bagging_fraction才會真的生效
                        random_state=seed,
                        verbose=-1,
                        **_mono_kwargs(),
                    )

                es_note = ""
                if has_both_classes:
                    # early_stopping_rounds（MODEL_TUNING_GUIDE.md 六-1）：n_estimators 只是上限，
                    # 以驗證集指標提前停止；驗證集需兩類都存在才啟用（單一類別無法評估 AUC）
                    try:
                        model = _make_tree(with_early_stopping=True)
                        if run.model_type == "lightgbm":
                            try:
                                # lightgbm>=4.x 新 API：eval_X / eval_y（eval_set 已 deprecated）
                                model.fit(
                                    X_train, y_train,
                                    eval_X=X_val, eval_y=y_val, eval_metric="auc",
                                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                                )
                            except TypeError:
                                # lightgbm<4.x：退回舊 API
                                model.fit(
                                    X_train, y_train,
                                    eval_set=[(X_val, y_val)], eval_metric="auc",
                                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                                )
                        else:
                            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                        es_note = (f"，early_stopping={EARLY_STOPPING_ROUNDS}輪"
                                   f"（best_iteration={getattr(model, 'best_iteration_', None)}）")
                    except TypeError:
                        # 舊版套件不支援 callbacks／建構子參數 → 用相同超參數重建後照原方式訓練
                        model = _make_tree(with_early_stopping=False)
                        model.fit(X_train, y_train)
                else:
                    # 驗證集僅單一類別：無法評估 AUC／早停指標 → 不啟用早停，照原方式訓練
                    model = _make_tree(with_early_stopping=False)
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

            repeat_metrics.append({
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "auc": roc_auc_score(y_val, y_proba) if has_both_classes else None,
                "pr_auc": average_precision_score(y_val, y_proba) if has_both_classes else None,
            })
            trained_models.append(model)


        def _mean(key):
            values = [m[key] for m in repeat_metrics if m[key] is not None]
            return sum(values) / len(values) if values else None

        def _std(key):
            values = [m[key] for m in repeat_metrics if m[key] is not None]
            if len(values) < 2:
                return 0.0
            mean_v = sum(values) / len(values)
            variance = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
            return variance ** 0.5

        accuracy = _mean("accuracy")
        precision = _mean("precision")
        recall = _mean("recall")
        auc = _mean("auc")
        pr_auc = _mean("pr_auc")
        auc_std = _std("auc")
        pr_auc_std = _std("pr_auc")

        # 存檔用的模型：選「AUC 最接近這 N 次平均值」的那一次，不用最後一次或隨便一次，
        # 避免不小心存到運氣特別好或特別差的那次訓練結果
        if auc is not None and repeat_metrics:
            # 只以「已完成輪次」找最接近 AUC 平均的那次。先前用 range(n_repeats) 索引，
            # 當 TRAIN_TIMEOUT 中斷後續重複訓練時，repeat_metrics 不足 n_repeats 筆 →
            # IndexError 讓整次訓練被誤判為 failed（與「已完成輪次照常計入統計」矛盾）。
            closest_idx = min(
                range(len(repeat_metrics)),
                key=lambda i: abs((repeat_metrics[i]["auc"] or 0) - auc),
            )
        else:
            closest_idx = 0
        if not trained_models:
            raise RuntimeError("所有重複訓練均未完成（可能因訓練超時），沒有任何可保存的模型")
        final_model = trained_models[closest_idx]

        # ---------------------------------------------------------
        # （可選）最終重擬合 refit_on_all：
        # 上面的評估指標已用「訓練/驗證切分」誠實算完；若啟動端要求納入最新資料，
        # 這裡改用「訓練＋驗證全部資料」以相同超參數重新擬合一次再存檔——
        # 上線模型因此學到最新市場，而指標數字維持原切分結果、不灌水。
        # ---------------------------------------------------------
        if refit_on_all:
            try:
                all_df, _, _ = _load_dataset(
                    run.train_start, run.validation_end, stock_code_list, run.label_version,
                )
                fit_cols = [c for c in used_cols]
                all_df = all_df.dropna(subset=fit_cols)
                if not all_df.empty:
                    X_refit, y_refit = all_df[fit_cols], all_df["label"]
                    if run.model_type == "tft":
                        # 重擬合語意＝「訓練＋驗證全部資料以相同超參數重新擬合」：
                        # val_df 傳 None → 不做早停、跑滿 epochs，與樹模型直接 fit 全量等價
                        _require_torch()
                        from .tft_model import TFTClassifier
                        final_model = TFTClassifier(
                            seq_len=hyperparams["seq_len"], d_model=hyperparams["d_model"],
                            n_heads=hyperparams["n_heads"], num_layers=hyperparams["num_layers"],
                            dropout=hyperparams["dropout"],
                            learning_rate=hyperparams["learning_rate"],
                            epochs=hyperparams["epochs"], batch_size=hyperparams["batch_size"],
                            random_state=42,
                        )
                        final_model.fit_frame(all_df, None, fit_cols)
                    else:
                        if run.model_type == "xgboost":
                            pos_c = int((y_refit == 1).sum())
                            neg_c = int((y_refit == 0).sum())
                            spw = (neg_c / pos_c) if pos_c > 0 else 1.0
                            final_model = xgb.XGBClassifier(
                                objective="binary:logistic",
                                scale_pos_weight=spw,
                                n_estimators=hyperparams["n_estimators"],
                                learning_rate=hyperparams["learning_rate"],
                                max_depth=hyperparams["max_depth"],
                                subsample=hyperparams["subsample"],
                                colsample_bytree=hyperparams["colsample_bytree"],
                                random_state=42,
                                eval_metric="logloss",
                                **_mono_kwargs(),
                            )
                        else:
                            final_model = lgb.LGBMClassifier(
                                objective="binary",
                                is_unbalance=True,
                                n_estimators=hyperparams["n_estimators"],
                                learning_rate=hyperparams["learning_rate"],
                                num_leaves=hyperparams["num_leaves"],
                                max_depth=hyperparams["max_depth"],
                                bagging_fraction=hyperparams["bagging_fraction"],
                                feature_fraction=hyperparams["feature_fraction"],
                                bagging_freq=1,
                                random_state=42,
                                verbose=-1,
                                **_mono_kwargs(),
                            )
                        final_model.fit(X_refit, y_refit)
            except Exception:
                pass  # 重擬合失敗時退回原切分模型，確保流程不中斷

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_file_path = os.path.join(MODELS_DIR, f"{run.model_version}.joblib")
        joblib.dump(
            {"model": final_model, "feature_columns": used_cols, "hyperparams": hyperparams},
            model_file_path,
        )

        # ---------------------------------------------------------
        # 模型檔入庫備份（ModelArtifact）：Render 免費方案磁碟為暫態，重新部署／
        # 重啟會清空 trained_models/，導致「已完成」訓練無法載入。訓練完成時把
        # 檔案 bytes 存進資料庫，載入端（model_artifacts.get_model_file）在磁碟
        # 檔遺失時自動從 DB 還原。入庫失敗不影響訓練結果（磁碟上仍有檔案）。
        # ---------------------------------------------------------
        try:
            from .model_artifacts import store_artifact
            store_artifact(run, model_file_path)
        except Exception as _artifact_exc:  # noqa: BLE001
            print(f"[model_artifacts] 模型檔入庫失敗（不影響訓練）：{_artifact_exc}")

        # ---------------------------------------------------------
        # 特徵重要性（MODEL_TUNING_GUIDE.md 五-4）：僅樹模型有原生重要性——
        # LightGBM 預設 split（特徵被使用次數）、XGBoost 預設 gain（帶來的增益）；
        # TFT 無原生重要性，記錄空清單。存進 ModelTrainingRun.feature_importance
        # 供首頁「特徵重要性 Top 20」卡片顯示；單一特徵獨大時前端會提示過擬合警語。
        # ---------------------------------------------------------
        feature_importance = []
        if run.model_type in ("lightgbm", "xgboost"):
            try:
                _imp = getattr(final_model, "feature_importances_", None)
                if _imp is not None:
                    _pairs = [(col, float(v)) for col, v in zip(used_cols, _imp)]
                    _pairs.sort(key=lambda x: x[1], reverse=True)
                    feature_importance = [
                        {"feature": col, "importance": round(v, 6)} for col, v in _pairs
                    ]
            except Exception:
                feature_importance = []  # 取不到重要性不影響訓練結果

        # 記錄驗證期距離訓練期結束多久（月數），供 walk-forward 分析用（PROJECT_PLAN.md 10.6節）
        months_gap = (
            (run.validation_start.year - run.train_end.year) * 12
            + (run.validation_start.month - run.train_end.month)
        )

        run.status = "completed"
        run.accuracy = accuracy
        run.precision = precision
        run.recall = recall
        run.auc = auc
        run.pr_auc = pr_auc
        run.auc_std = auc_std
        run.pr_auc_std = pr_auc_std
        run.months_from_train_end = months_gap
        run.feature_importance = feature_importance
        run.model_file_path = model_file_path
        run.completed_at = timezone.now()
        stock_scope = run.stock_codes if run.stock_codes else "資料庫內全部股票"
        extra_notes = []
        if timeout_hit:
            extra_notes.append(
                f"訓練超時：已達 {TRAIN_TIMEOUT_SECONDS} 秒上限，後續重複訓練停止"
                f"（已完成輪次照常計入統計）"
            )
        if not has_both_classes:
            extra_notes.append("驗證集僅單一類別：AUC／PR-AUC 記為 None，且不啟用早停")
        if feature_importance:
            extra_notes.append(f"特徵重要性: 已記錄 {len(feature_importance)} 個特徵")
        # ---------------------------------------------------------
        # SHAP 誤導特徵診斷已納入訓練流程（MODEL_TUNING_GUIDE.md 五-8）：
        # 訓練完成後自動對「驗證區間」跑 TreeSHAP（樹模型）／逐股置換重要性（TFT），
        # 三類誤導證據＋SHAP 重要性報告存進 run.diagnostics——首頁「SHAP 誤導特徵診斷」
        # 卡片、Walk-Forward 表的 SHAP 欄、訓練紀錄表的 SHAP 欄都會顯示。
        # 可在建立訓練的超參數帶 auto_shap=false 關閉；TFT 可用 shap_time_budget
        # 控制置換重要性的時間預算（秒）。診斷失敗（缺 shap 套件、驗證集組不出
        # 資料等）只會記在 notes，不影響訓練結果。
        # ---------------------------------------------------------
        diag_note = None
        if hyperparams.get("auto_shap", True) and run.model_file_path:
            try:
                from . import diagnostics as _diagnostics  # 函式內匯入，避免循環 import
                _diag_err = _diagnostics.attach_auto_diagnosis(
                    run, time_budget=hyperparams.get("shap_time_budget"))
                diag_note = ("SHAP 自動診斷: 完成（誤導證據＋重要性已存入診斷報告）"
                             if not _diag_err else f"SHAP 自動診斷失敗: {_diag_err}")
            except Exception as _diag_exc:  # noqa: BLE001
                diag_note = f"SHAP 自動診斷失敗: {_diag_exc}"
        elif not hyperparams.get("auto_shap", True):
            diag_note = "SHAP 自動診斷: 已停用（auto_shap=false）"
        if diag_note:
            extra_notes.append(diag_note)
        run.notes = (
            f"訓練筆數: {len(train_df)}, 驗證筆數: {len(val_df)}, "
            f"股票範圍: {stock_scope}, 標籤版本: {run.label_version}, "
            f"重複訓練: {n_repeats}次"
            + (es_note if run.model_type in ("lightgbm", "xgboost") else "")
            + ", "
            f"使用特徵({len(used_cols)}個): {', '.join(used_cols)}, "
            f"超參數: {hyperparams}"
            + (("｜" + "；".join(extra_notes)) if extra_notes else "")
        )

        run.save()

        auc_display = f"{auc:.4f}±{auc_std:.4f}" if auc is not None else "N/A"
        pr_auc_display = f"{pr_auc:.4f}±{pr_auc_std:.4f}" if pr_auc is not None else "N/A"

        stop_heartbeat.set()

        return {
            "success": True,
            "message": (
                f"訓練完成（重複{n_repeats}次）。AUC={auc_display}, PR-AUC={pr_auc_display}"
                f"（訓練{len(train_df)}筆/驗證{len(val_df)}筆）"
            ),
            "data": {
                "accuracy": accuracy, "precision": precision, "recall": recall,
                "auc": auc, "pr_auc": pr_auc, "auc_std": auc_std, "pr_auc_std": pr_auc_std,
                "n_repeats": n_repeats, "model_file_path": model_file_path,
                "hyperparams": hyperparams, "feature_columns": FEATURE_COLUMNS,
            },
        }

    except Exception as exc:  # noqa: BLE001
        stop_heartbeat.set()
        run.status = "failed"
        run.notes = str(exc)
        run.save(update_fields=["status", "notes"])
        return {"success": False, "message": f"訓練失敗：{exc}", "data": None}
