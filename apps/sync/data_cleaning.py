"""
資料清洗（Data Cleaning）模組。

放在 apps.sync 底下，因為股票、美金、大盤三個 App 都需要用到同一套清洗邏輯，
這是跨 App 共用的工具，不屬於任何單一資料來源。

===========================================================================
為什麼資料清洗很重要？
===========================================================================
在統計 / 機器學習的預測流程裡，資料清洗常常被視為「前置雜務」而被忽略，
但它其實是整個 pipeline 裡影響最大的一步：

    Garbage in, garbage out（垃圾進，垃圾出）

模型再怎麼調參、特徵工程做得再細緻，只要訓練資料裡混入了錯誤或極端的髒資料，
學出來的規律就是錯的。更麻煩的是，髒資料造成的問題通常不會讓程式當掉、不會報錯，
而是安靜地讓模型「學到錯的東西」，這種錯誤比程式crash更難察覺、更難除錯。

本模組處理四類最常見的資料品質問題（詳細說明見 DATA_CLEANING.md）：
1. 缺失值（Missing Values）：資料源沒回傳，或轉型失敗變成 None
2. 異常值（Outliers）：數值在統計上明顯不合理，但不一定是「錯的」
   （例如真的漲停/跌停，或除權息隔日的合理跳空），所以只標記警告，不自動刪除
3. 邏輯不一致（Consistency）：例如最高價比最低價還低，這種在物理上不可能發生，
   一定是資料源或轉型過程出錯，直接視為無效資料移除
4. 重複資料（Duplicates）：同一天有兩筆紀錄，保留清單中最後一筆（通常是較新的抓取結果）
===========================================================================
"""
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 寫入資料庫前的「安全閘門」：把 NaN / inf 統一轉成 None。
# 為什麼需要：同一份程式碼必須同時跑在 PostgreSQL 與 MariaDB/MySQL 兩種後端。
#   - PostgreSQL(+psycopg2) 能接受把 NaN 存進 numeric/double 欄位；
#   - MariaDB/MySQL(+mysqlclient) 不認得 NaN，寫進 DECIMAL/DOUBLE/FLOAT/BigInt 會踢例外。
# 因此所有 update_or_create/create/bulk_create/save 的 defaults 在送進 ORM 前，
# 都應該先過 clean_defaults()，讓 NaN 變成 NULL（NULL 兩邊都接受）。
# ---------------------------------------------------------------------------
def sanitize_val(v):
    """單一值：NaN / NA / inf / -inf → None；其餘原樣回傳。"""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
        if isinstance(v, (float, int, np.floating, np.integer)) and (math.isinf(v) or math.isnan(v)):
            return None
    except (TypeError, ValueError):
        pass
    return v


def clean_defaults(data):
    """把 dict 內所有 NaN / Inf 值 -> None，供 update_or_create 的 defaults 使用。"""
    if not isinstance(data, dict):
        return {}
    return {k: sanitize_val(v) for k, v in data.items()}


def _iqr_outlier_mask(series, k=3.0):
    """
    IQR（四分位距）方法偵測離群值。
    k=3.0 是比教科書常用的 k=1.5 更寬鬆的門檻，因為股價報酬率、匯率變動本來就有厚尾分布，
    用 1.5 會把太多正常的大漲大跌也標記成異常，反而不實用。
    """
    if len(series) < 10:
        return pd.Series([False] * len(series), index=series.index)
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series([False] * len(series), index=series.index)
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def clean_stock_price_rows(rows):
    """
    清洗股價資料。
    輸入：list[dict]，欄位需含 trade_date/open_price/high_price/low_price/close_price/adjusted_close/volume
          （對應 apps.stocks.services.fetch_stock_price 的回傳格式）

    處理規則：
    1. 移除價格 <= 0 的資料列 —— 不可能是真實股價，通常是資料源錯誤（無效資料，直接刪除）
    2. 移除 high < low 的資料列 —— 邏輯不可能發生，一定是資料錯誤（無效資料，直接刪除）
    3. 移除重複日期，保留最後一筆
    4. 用 IQR 方法偵測「單日報酬率」離群值 —— 只記錄警告，不自動刪除
       （因為真的漲跌停、除權息隔日確實可能出現大幅波動，自動刪除可能誤刪合理資料，
       交由使用者依 report 內容自行判斷是否需要進一步處理）

    回傳：(cleaned_rows: list[dict], report: dict)
    """
    empty_report = {
        "total": 0, "remaining": 0,
        "removed_invalid_price": 0, "removed_high_low": 0, "removed_duplicate": 0,
        "outlier_warnings": [],
    }
    if not rows:
        return rows, empty_report

    df = pd.DataFrame(rows)
    total = len(df)

    invalid_price_mask = (
        df["open_price"].isna() | df["high_price"].isna()
        | df["low_price"].isna() | df["close_price"].isna() | df["adjusted_close"].isna()
        | (df["open_price"] <= 0) | (df["high_price"] <= 0)
        | (df["low_price"] <= 0) | (df["close_price"] <= 0) | (df["adjusted_close"] <= 0)
    )
    removed_invalid_price = int(invalid_price_mask.sum())
    df = df[~invalid_price_mask]

    high_low_mask = df["high_price"] < df["low_price"]
    removed_high_low = int(high_low_mask.sum())
    df = df[~high_low_mask]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["trade_date"], keep="last")
    removed_duplicate = before_dedup - len(df)

    outlier_warnings = []
    if len(df) > 1:
        df = df.sort_values("trade_date")
        daily_return = df["adjusted_close"].pct_change()
        valid_returns = daily_return.dropna()
        mask = _iqr_outlier_mask(valid_returns, k=3.0)
        if mask.any():
            outlier_dates = df.loc[valid_returns[mask].index, "trade_date"]
            outlier_warnings = [str(d) for d in outlier_dates.tolist()]

    report = {
        "total": total,
        "remaining": len(df),
        "removed_invalid_price": removed_invalid_price,
        "removed_high_low": removed_high_low,
        "removed_duplicate": removed_duplicate,
        "outlier_warnings": outlier_warnings,
    }
    return df.to_dict("records"), report


def clean_chip_data_rows(rows):
    """
    清洗籌碼面資料。
    輸入：list[dict]，欄位需含 trade_date/foreign_net_buy/trust_net_buy/margin_balance/short_balance

    處理規則：
    1. margin_balance / short_balance（融資融券餘額）理論上不應為負數，
       若為負數視為資料錯誤，修正為 0 並記錄警告（不整列刪除，因為買賣超欄位可能還是對的）
    2. 移除重複日期，保留最後一筆
    """
    empty_report = {"total": 0, "remaining": 0, "fixed_negative_balance": 0, "removed_duplicate": 0}
    if not rows:
        return rows, empty_report

    df = pd.DataFrame(rows)
    total = len(df)

    # 買賣超/資券欄位是 BigInteger（不接受 NULL/NaN），把 NaN 修正為 0 並記錄，
    # 避免 NaN 送到 MariaDB 踢例外；與「負數轉 0」的做法一致。
    chip_cols = ["foreign_net_buy", "trust_net_buy", "margin_balance", "short_balance"]
    present_chip = [c for c in chip_cols if c in df.columns]
    fixed_nan = 0
    if present_chip:
        nan_mask = df[present_chip].isna().any(axis=1)
        fixed_nan = int(nan_mask.sum())
        df[present_chip] = df[present_chip].fillna(0)

    negative_mask = (df["margin_balance"] < 0) | (df["short_balance"] < 0)
    fixed_negative_balance = int(negative_mask.sum())
    df.loc[df["margin_balance"] < 0, "margin_balance"] = 0
    df.loc[df["short_balance"] < 0, "short_balance"] = 0

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["trade_date"], keep="last")
    removed_duplicate = before_dedup - len(df)

    report = {
        "total": total, "remaining": len(df),
        "fixed_negative_balance": fixed_negative_balance,
        "fixed_nan": fixed_nan,
        "removed_duplicate": removed_duplicate,
    }
    return df.to_dict("records"), report


def clean_revenue_rows(rows):
    """
    清洗月營收資料。
    輸入：list[dict]，欄位需含 revenue_month/announcement_date/revenue_amount/mom_growth/yoy_growth

    處理規則：
    1. revenue_amount <= 0 視為異常，直接移除
       （公司正常經營不可能營收為 0 或負數，通常是資料源當月尚未更新或轉型錯誤）
    2. 移除重複月份，保留最後一筆
    """
    empty_report = {"total": 0, "remaining": 0, "removed_invalid_revenue": 0, "removed_duplicate": 0}
    if not rows:
        return rows, empty_report

    df = pd.DataFrame(rows)
    total = len(df)

    invalid_mask = df["revenue_amount"].isna() | (df["revenue_amount"] <= 0)
    removed_invalid_revenue = int(invalid_mask.sum())
    df = df[~invalid_mask]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["revenue_month"], keep="last")
    removed_duplicate = before_dedup - len(df)

    report = {
        "total": total, "remaining": len(df),
        "removed_invalid_revenue": removed_invalid_revenue,
        "removed_duplicate": removed_duplicate,
    }
    return df.to_dict("records"), report


def clean_time_series_rows(rows, date_key, value_key, k=3.0):
    """
    通用時間序列清洗，給美金匯率、跨市場指數這種「單一數值 + 日期」的資料共用。

    處理規則：
    1. 移除 value <= 0 的資料列
    2. 移除重複日期，保留最後一筆
    3. 用 IQR 偵測日變動率離群值，只記錄警告

    參數：
        date_key: 日期欄位名稱，例如 "rate_date" 或 "trade_date"
        value_key: 數值欄位名稱，例如 "close_rate" 或 "close_value"
    """
    empty_report = {
        "total": 0, "remaining": 0, "removed_invalid_value": 0,
        "removed_duplicate": 0, "outlier_warnings": [],
    }
    if not rows:
        return rows, empty_report

    df = pd.DataFrame(rows)
    total = len(df)

    invalid_mask = df[value_key].isna() | (df[value_key] <= 0)
    removed_invalid_value = int(invalid_mask.sum())
    df = df[~invalid_mask]

    before_dedup = len(df)
    df = df.drop_duplicates(subset=[date_key], keep="last")
    removed_duplicate = before_dedup - len(df)

    outlier_warnings = []
    if len(df) > 1:
        df = df.sort_values(date_key)
        daily_change = df[value_key].pct_change()
        valid_changes = daily_change.dropna()
        mask = _iqr_outlier_mask(valid_changes, k=k)
        if mask.any():
            outlier_dates = df.loc[valid_changes[mask].index, date_key]
            outlier_warnings = [str(d) for d in outlier_dates.tolist()]

    report = {
        "total": total,
        "remaining": len(df),
        "removed_invalid_value": removed_invalid_value,
        "removed_duplicate": removed_duplicate,
        "outlier_warnings": outlier_warnings,
    }
    return df.to_dict("records"), report


def summarize_report(report):
    """把清洗報告轉成一句簡短的中文摘要，方便寫進 DataSyncLog 的 message 欄位。"""
    removed_keys = [k for k in report if k.startswith("removed_") or k.startswith("fixed_")]
    total_removed = sum(report.get(k, 0) for k in removed_keys)
    warning_count = len(report.get("outlier_warnings", []))

    parts = [f"清洗前{report.get('total', 0)}筆"]
    if total_removed > 0:
        parts.append(f"移除/修正{total_removed}筆異常資料")
    if warning_count > 0:
        parts.append(f"{warning_count}筆離群值警告")
    if total_removed == 0 and warning_count == 0:
        parts.append("無異常")
    return "，".join(parts)
