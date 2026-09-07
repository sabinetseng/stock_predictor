"""
analysis App 商業邏輯層：股票與美金 / 大盤的關聯分析。

設計原則：
- 一律用「日報酬率」（pct_change）算相關性，不是用原始價格水準。
  原始價格水準常常因為長期趨勢（例如兩者都在漲）而算出「假相關」，
  報酬率才能反映「兩者同步變動」的真實關聯強度。
- 資料對齊：股票只有交易日，美金/VIX 每天都有報價，用 merge_asof
  對齊到「最近一筆已知資料」，避免比對到未來還沒發生的資料。
- Pearson/Spearman/Lagged 算單一數值，寫入 CorrelationResult 快取；
  Rolling 是一個時間序列，不快取（結果太多筆，快取意義不大），直接回傳給前端畫圖。
"""
import pandas as pd
from scipy import stats

from .models import CorrelationResult
from apps.stocks.models import StockBasicInfo, StockPrice
from apps.fx_data.models import UsdTwdRate
from apps.market_data.models import MarketIndexRate

# compare_target 到資料來源的對照
COMPARE_TARGETS = ["USD_TWD", "TWII", "SOX", "VIX"]


def _load_compare_series(compare_target):
    """
    依 compare_target 讀取對應的比較對象資料，統一回傳 DataFrame，欄位為 [trade_date, value]。
    """
    if compare_target == "USD_TWD":
        qs = UsdTwdRate.objects.order_by("rate_date").values("rate_date", "close_rate")
        df = pd.DataFrame(list(qs)).rename(columns={"rate_date": "trade_date", "close_rate": "value"})
    elif compare_target in ("TWII", "SOX", "VIX"):
        qs = MarketIndexRate.objects.filter(index_code=compare_target).order_by("trade_date").values(
            "trade_date", "close_value"
        )
        df = pd.DataFrame(list(qs)).rename(columns={"close_value": "value"})
    else:
        raise ValueError(f"不支援的比較對象：{compare_target}，可用值：{COMPARE_TARGETS}")

    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["value"] = df["value"].astype(float)
    return df


def _load_aligned_returns(stock_code, compare_target):
    """
    組合「股票日報酬率」與「比較對象日報酬率」的對齊資料。
    回傳：DataFrame，欄位 [trade_date, stock_return, compare_return]，已去除缺值列。
    """
    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        raise ValueError(f"找不到股票 {stock_code}，請先執行 sync_stock_data")

    price_qs = StockPrice.objects.filter(stock=stock).order_by("trade_date").values(
        "trade_date", "adjusted_close"
    )
    price_df = pd.DataFrame(list(price_qs))
    if price_df.empty:
        raise ValueError(f"{stock_code} 沒有股價資料，請先執行 sync_stock_data")

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    price_df["adjusted_close"] = price_df["adjusted_close"].astype(float)
    price_df = price_df.sort_values("trade_date")
    price_df["stock_return"] = price_df["adjusted_close"].pct_change()

    compare_df = _load_compare_series(compare_target)
    if compare_df.empty:
        raise ValueError(f"比較對象 {compare_target} 沒有資料，請先執行對應的爬蟲")

    compare_df = compare_df.sort_values("trade_date")
    compare_df["compare_return"] = compare_df["value"].pct_change()

    # merge_asof：用股票的交易日去對齊「最近一筆已知」的比較對象資料，避免用到未來資料
    merged = pd.merge_asof(
        price_df[["trade_date", "stock_return"]],
        compare_df[["trade_date", "compare_return"]],
        on="trade_date",
    )
    merged = merged.dropna(subset=["stock_return", "compare_return"])
    return merged


def calc_pearson_correlation(stock_code, compare_target):
    """
    計算股票與比較對象「日報酬率」的 Pearson 相關係數，結果寫入 CorrelationResult 快取。
    回傳：dict，{"success": bool, "message": str, "data": {"correlation": float, "p_value": float, "n": int}}
    """
    try:
        df = _load_aligned_returns(stock_code, compare_target)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "data": None}

    if len(df) < 10:
        return {"success": False, "message": f"對齊後的資料筆數太少（{len(df)}筆），無法算出可靠的相關係數", "data": None}

    corr, p_value = stats.pearsonr(df["stock_return"], df["compare_return"])

    stock = StockBasicInfo.objects.get(stock_code=stock_code)
    CorrelationResult.objects.update_or_create(
        stock=stock, compare_target=compare_target, method="pearson", lag_days=None,
        defaults={"result_value": corr},
    )

    return {
        "success": True,
        "message": f"{stock_code} 與 {compare_target} 的 Pearson 相關係數：{corr:.4f}",
        "data": {"correlation": corr, "p_value": p_value, "n": len(df)},
    }


def calc_spearman_correlation(stock_code, compare_target):
    """
    計算 Spearman 等級相關係數（不假設線性關係，對極端值較不敏感）。
    """
    try:
        df = _load_aligned_returns(stock_code, compare_target)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "data": None}

    if len(df) < 10:
        return {"success": False, "message": f"對齊後的資料筆數太少（{len(df)}筆），無法算出可靠的相關係數", "data": None}

    corr, p_value = stats.spearmanr(df["stock_return"], df["compare_return"])

    stock = StockBasicInfo.objects.get(stock_code=stock_code)
    CorrelationResult.objects.update_or_create(
        stock=stock, compare_target=compare_target, method="spearman", lag_days=None,
        defaults={"result_value": corr},
    )

    return {
        "success": True,
        "message": f"{stock_code} 與 {compare_target} 的 Spearman 相關係數：{corr:.4f}",
        "data": {"correlation": corr, "p_value": p_value, "n": len(df)},
    }


def calc_lagged_correlation(stock_code, compare_target, max_lag_days=20):
    """
    計算「比較對象領先股票 N 天」的相關係數，N 從 0 到 max_lag_days。
    例如 lag=5 代表：比較對象在 t-5 日的報酬率，與股票在 t 日的報酬率的相關性，
    藉此判斷比較對象是否對股票有領先指標的效果。

    回傳：dict，data 內是 list[{"lag": int, "correlation": float}]，每個 lag 也各自快取一筆。
    """
    try:
        df = _load_aligned_returns(stock_code, compare_target)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "data": None}

    if len(df) < max_lag_days + 10:
        return {
            "success": False,
            "message": f"對齊後的資料筆數太少（{len(df)}筆），無法計算到 lag={max_lag_days} 的相關係數",
            "data": None,
        }

    stock = StockBasicInfo.objects.get(stock_code=stock_code)
    results = []
    for lag in range(0, max_lag_days + 1):
        shifted_compare = df["compare_return"].shift(lag)
        valid = pd.concat([df["stock_return"], shifted_compare], axis=1).dropna()
        if len(valid) < 10:
            continue
        corr, _ = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        results.append({"lag": lag, "correlation": corr})

        CorrelationResult.objects.update_or_create(
            stock=stock, compare_target=compare_target, method="lagged", lag_days=lag,
            defaults={"result_value": corr},
        )

    best = max(results, key=lambda r: abs(r["correlation"])) if results else None
    message = (
        f"{stock_code} 與 {compare_target} 在 lag={best['lag']} 天時相關性最強（{best['correlation']:.4f}）"
        if best else "沒有足夠資料計算 lagged correlation"
    )

    return {"success": True, "message": message, "data": results}


def calc_rolling_correlation(stock_code, compare_target, window_days=20):
    """
    計算滾動相關係數時間序列：每一天用「過去 window_days 天」的報酬率算一次相關係數，
    用來觀察兩者的關聯強度是否隨時間改變（例如某段時間美金與股票關聯變強或變弱）。

    注意：這是時間序列，不寫入 CorrelationResult（該表設計是存單一數值），
    直接回傳給前端畫圖即可。
    """
    try:
        df = _load_aligned_returns(stock_code, compare_target)
    except ValueError as exc:
        return {"success": False, "message": str(exc), "data": None}

    if len(df) < window_days + 5:
        return {
            "success": False,
            "message": f"對齊後的資料筆數太少（{len(df)}筆），無法計算 {window_days} 天滾動相關係數",
            "data": None,
        }

    df = df.sort_values("trade_date").reset_index(drop=True)
    df["rolling_correlation"] = df["stock_return"].rolling(window_days).corr(df["compare_return"])
    df = df.dropna(subset=["rolling_correlation"])

    results = [
        {"date": row["trade_date"].strftime("%Y-%m-%d"), "correlation": row["rolling_correlation"]}
        for _, row in df.iterrows()
    ]

    return {
        "success": True,
        "message": f"已計算 {len(results)} 筆滾動相關係數（窗格：{window_days}天）",
        "data": results,
    }


def calc_all_correlations(stock_code, compare_target):
    """一次算完 Pearson/Spearman/Lagged/Rolling 四種分析，方便前端一次呼叫拿到所有結果。"""
    return {
        "pearson": calc_pearson_correlation(stock_code, compare_target),
        "spearman": calc_spearman_correlation(stock_code, compare_target),
        "lagged": calc_lagged_correlation(stock_code, compare_target),
        "rolling": calc_rolling_correlation(stock_code, compare_target),
    }
