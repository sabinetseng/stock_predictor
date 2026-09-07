"""
fx_data App 商業邏輯層。
規則：美金資料一律從網路抓取（Yahoo Finance，不需要 API Token），不使用 CSV 匯入。
"""
import time
import datetime

import yfinance as yf

from .models import UsdTwdRate
from apps.sync.models import DataSyncLog
from apps.sync.data_cleaning import clean_defaults

# Yahoo Finance 的美元兌台幣代碼
USD_TWD_TICKER = "TWD=X"

# 抓取失敗時的重試設定
MAX_RETRY = 3
RETRY_WAIT_SECONDS = 5


def fetch_usd_twd_rate(start_date, end_date):
    """
    從 Yahoo Finance 抓取 TWD=X 歷史匯率。

    參數：
        start_date, end_date: 字串或 datetime.date，例如 "2024-01-01"

    回傳：
        list[dict]，每筆格式：{"rate_date": date, "close_rate": float}
        若抓取失敗（重試用盡後仍失敗），會拋出例外，交給呼叫端決定如何處理
        （原則：抓取失敗不可覆蓋既有資料，所以這裡不吞掉例外）。
    """
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            ticker = yf.Ticker(USD_TWD_TICKER)
            # end_date 要 +1 天，因為 yfinance 的 end 是不包含當天的
            end_date_exclusive = _to_date(end_date) + datetime.timedelta(days=1)
            df = ticker.history(start=str(start_date), end=str(end_date_exclusive))

            if df.empty:
                # 沒有資料不算錯誤（例如區間內剛好沒有交易），回傳空 list
                return []

            results = []
            for row_date, row in df.iterrows():
                results.append({
                    "rate_date": row_date.date(),
                    "close_rate": round(float(row["Close"]), 4),
                })
            return results

        except Exception as exc:  # noqa: BLE001 - 對外部 API 呼叫採寬鬆例外處理，記錄後重試
            last_error = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT_SECONDS)
            continue

    # 重試用盡仍失敗，往外拋出，讓 check_and_sync_fx_data 寫入失敗紀錄
    raise RuntimeError(f"抓取美金匯率失敗，已重試 {MAX_RETRY} 次：{last_error}")


def check_and_sync_fx_data():
    """
    「檢查美金是否有新資料」按鈕的核心邏輯：
    1. 查詢 UsdTwdRate 資料庫最後日期
    2. 向 Yahoo Finance 抓取「最後日期的下一天」到「今天」的資料
    3. 若有新資料 -> upsert 寫入，並寫入 DataSyncLog（success_new_data）
    4. 若沒有新資料 -> 寫入 DataSyncLog（success_no_new_data），回傳「已是最新狀態」
    5. 抓取失敗 -> 不覆蓋既有資料，寫入 DataSyncLog（failed），並把錯誤訊息記下來

    回傳：dict，格式如下，方便 views.py 直接包成 JsonResponse
        {"success": bool, "message": str, "data": {...}}
    """
    today = datetime.date.today()

    # 步驟1：找資料庫目前最後一筆日期，若整張表是空的就從一個合理的起始日開始抓
    last_record = UsdTwdRate.objects.order_by("-rate_date").first()
    if last_record:
        fetch_start = last_record.rate_date + datetime.timedelta(days=1)
    else:
        # 資料庫是空的：第一次抓取，預設從 settings 的訓練起始日開始
        from django.conf import settings
        fetch_start = datetime.date.fromisoformat(settings.TRAIN_DATE_START)

    # 已經是最新的，不需要再抓
    if fetch_start > today:
        DataSyncLog.objects.create(
            source_type="usd_twd",
            target_code="",
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message="已是最新狀態，無需抓取",
        )
        return {"success": True, "message": "已是最新狀態", "data": {"new_records": 0}}

    # 步驟2：抓取資料（失敗時 fetch_usd_twd_rate 會拋出例外）
    try:
        new_rows = fetch_usd_twd_rate(fetch_start, today)
    except Exception as exc:  # noqa: BLE001
        DataSyncLog.objects.create(
            source_type="usd_twd",
            target_code="",
            status="failed",
            source_name="Yahoo Finance",
            message=str(exc),
        )
        # 抓取失敗，不覆蓋既有資料，直接回報失敗
        return {"success": False, "message": f"抓取失敗：{exc}", "data": None}

    # 沒有新資料（例如區間內都是非交易日）
    if not new_rows:
        DataSyncLog.objects.create(
            source_type="usd_twd",
            target_code="",
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message="區間內沒有新資料",
        )
        return {"success": True, "message": "已是最新狀態", "data": {"new_records": 0}}

    # 資料清洗：移除 <=0 的匯率、重複日期，並記錄變動率離群值警告
    from apps.sync.data_cleaning import clean_time_series_rows, summarize_report
    new_rows, clean_report = clean_time_series_rows(new_rows, date_key="rate_date", value_key="close_rate")
    clean_summary = summarize_report(clean_report)

    if not new_rows:
        DataSyncLog.objects.create(
            source_type="usd_twd",
            target_code="",
            status="success_no_new_data",
            source_name="Yahoo Finance",
            message=f"抓到的資料清洗後全數無效（{clean_summary}）",
        )
        return {"success": True, "message": "抓到的資料清洗後全數無效", "data": {"new_records": 0}}

    # 步驟3：upsert 寫入資料庫（用 update_or_create 避免重複寫入）
    saved_count = 0
    for row in new_rows:
        UsdTwdRate.objects.update_or_create(
            rate_date=row["rate_date"],
            defaults=clean_defaults({
                "close_rate": row["close_rate"],
                "source_name": "Yahoo Finance",
            }),
        )
        saved_count += 1

    DataSyncLog.objects.create(
        source_type="usd_twd",
        target_code="",
        status="success_new_data",
        source_name="Yahoo Finance",
        message=f"新增/更新 {saved_count} 筆資料（{clean_summary}）",
    )

    return {
        "success": True,
        "message": f"同步完成，共 {saved_count} 筆新資料",
        "data": {"new_records": saved_count},
    }


def _to_date(value):
    """統一把字串或 date 轉成 datetime.date，方便內部運算。"""
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value))
