"""market_data App 商業邏輯層：跨市場指數與交易日行事曆抓取。"""
import time
import datetime

import yfinance as yf

from .models import MarketIndexRate, TradingCalendar
from apps.sync.models import DataSyncLog
from apps.sync.data_cleaning import clean_defaults

# Yahoo Finance 代碼對照（與 MarketIndexRate.index_code 對應）
TICKER_MAP = {
    "TWII": "^TWII",  # 加權指數
    "SOX": "^SOX",    # 費城半導體指數
    "VIX": "^VIX",    # 波動度指數
}

MAX_RETRY = 3
RETRY_WAIT_SECONDS = 5


def fetch_index_history(index_code, start_date, end_date):
    """
    從 Yahoo Finance 抓取指數歷史資料。
    index_code 使用 MarketIndexRate.index_code 的值，例如 "TWII" / "SOX" / "VIX"（不是 Yahoo 代碼）。

    回傳：list[dict]，每筆格式：{"trade_date": date, "close_value": float}
    """
    if index_code not in TICKER_MAP:
        raise ValueError(f"未知的指數代碼：{index_code}，目前支援 {list(TICKER_MAP.keys())}")

    yahoo_ticker = TICKER_MAP[index_code]
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            import pandas as pd

            end_date_exclusive = _to_date(end_date) + datetime.timedelta(days=1)
            # 用 yf.download 而非 yf.Ticker().history()：
            # download 是另一條請求路徑，實測在部分環境（中文路徑/區網）下比 history 穩定
            df = yf.download(
                yahoo_ticker,
                start=str(start_date),
                end=str(end_date_exclusive),
                progress=False,
                auto_adjust=True,
            )

            if df is None or df.empty:
                return []

            # 單一代碼 + auto_adjust 時，新版 yfinance 會回 MultiIndex 欄位，攤平成單層
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            results = []
            for row_date, row in df.iterrows():
                results.append({
                    "trade_date": row_date.date(),
                    "close_value": round(float(row["Close"]), 4),
                })
            return results

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT_SECONDS)
            continue

    raise RuntimeError(f"抓取指數 {index_code} 失敗，已重試 {MAX_RETRY} 次：{last_error}")


def check_and_sync_index_data(index_code):
    """
    單一指數的「檢查是否有新資料」邏輯，與 fx_data.check_and_sync_fx_data 是同一套模式：
    比對資料庫最後日期 -> 有新資料才 upsert -> 寫 DataSyncLog。
    """
    today = datetime.date.today()

    last_record = (
        MarketIndexRate.objects.filter(index_code=index_code).order_by("-trade_date").first()
    )
    if last_record:
        fetch_start = last_record.trade_date + datetime.timedelta(days=1)
    else:
        from django.conf import settings
        fetch_start = datetime.date.fromisoformat(settings.TRAIN_DATE_START)

    if fetch_start > today:
        DataSyncLog.objects.create(
            source_type="market_index",
            target_code=index_code,
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message="已是最新狀態，無需抓取",
        )
        return {"success": True, "message": f"{index_code} 已是最新狀態", "data": {"new_records": 0}}

    try:
        new_rows = fetch_index_history(index_code, fetch_start, today)
    except Exception as exc:  # noqa: BLE001
        DataSyncLog.objects.create(
            source_type="market_index",
            target_code=index_code,
            status="failed",
            source_name="Yahoo Finance",
            message=str(exc),
        )
        return {"success": False, "message": f"{index_code} 抓取失敗：{exc}", "data": None}

    if not new_rows:
        DataSyncLog.objects.create(
            source_type="market_index",
            target_code=index_code,
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message="區間內沒有新資料",
        )
        return {"success": True, "message": f"{index_code} 已是最新狀態", "data": {"new_records": 0}}

    # 資料清洗：移除 <=0 的指數值、重複日期，並記錄變動率離群值警告
    from apps.sync.data_cleaning import clean_time_series_rows, summarize_report
    new_rows, clean_report = clean_time_series_rows(new_rows, date_key="trade_date", value_key="close_value")
    clean_summary = summarize_report(clean_report)

    if not new_rows:
        DataSyncLog.objects.create(
            source_type="market_index",
            target_code=index_code,
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message=f"抓到的資料清洗後全數無效（{clean_summary}）",
        )
        return {"success": True, "message": f"{index_code} 抓到的資料清洗後全數無效", "data": {"new_records": 0}}

    saved_count = 0
    for row in new_rows:
        MarketIndexRate.objects.update_or_create(
            index_code=index_code,
            trade_date=row["trade_date"],
            defaults=clean_defaults({
                "close_value": row["close_value"],
                "source_name": "Yahoo Finance",
            }),
        )
        saved_count += 1

    DataSyncLog.objects.create(
        source_type="market_index",
        target_code=index_code,
        status="success_new_data",
        source_name="Yahoo Finance",
        message=f"新增/更新 {saved_count} 筆資料（{clean_summary}）",
    )

    return {
        "success": True,
        "message": f"{index_code} 同步完成，共 {saved_count} 筆新資料",
        "data": {"new_records": saved_count},
    }


def sync_all_indexes():
    """依序同步 TWII / SOX / VIX 三個指數，回傳每個指數的同步結果。"""
    return {code: check_and_sync_index_data(code) for code in TICKER_MAP.keys()}


def sync_trading_calendar(start_date, end_date):
    """
    同步台股交易日行事曆。
    來源：FinMind taiwan_stock_trading_date（確認免費版有提供，優先使用）。
    """
    from FinMind.data import DataLoader

    api = DataLoader()
    last_error = None
    df = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            df = api.taiwan_stock_trading_date(start_date=str(start_date), end_date=str(end_date))
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT_SECONDS)
            continue

    if df is None:
        DataSyncLog.objects.create(
            source_type="trading_calendar", target_code="",
            status="failed", source_name="FinMind", message=str(last_error),
        )
        return {"success": False, "message": f"抓取失敗：{last_error}", "data": None}

    if df.empty:
        DataSyncLog.objects.create(
            source_type="trading_calendar", target_code="",
            status="success_no_new_data", source_name="FinMind", message="區間內沒有資料",
        )
        return {"success": True, "message": "區間內沒有交易日資料", "data": {"new_records": 0}}

    # FinMind 回傳的都是「交易日」，非交易日不會出現在清單裡，所以全部標記 is_trading_day=True
    saved_count = 0
    for _, row in df.iterrows():
        trade_date = datetime.date.fromisoformat(row["date"])
        TradingCalendar.objects.update_or_create(
            calendar_date=trade_date,
            defaults={"is_trading_day": True},
        )
        saved_count += 1

    DataSyncLog.objects.create(
        source_type="trading_calendar", target_code="",
        status="success_new_data", source_name="FinMind", message=f"新增/更新 {saved_count} 筆",
    )
    return {"success": True, "message": f"同步完成，共 {saved_count} 筆", "data": {"new_records": saved_count}}


def _to_date(value):
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))


