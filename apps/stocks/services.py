"""
stocks App 的商業邏輯層。
規則：資料抓取（fetch）與特徵/分析（feature）邏輯分開寫，方便測試與維護。

資料來源：FinMind（免費版，不需要 API Token）。
"""
import csv
import time
import datetime

import numpy as np
import pandas as pd
from FinMind.data import DataLoader

from .models import (
    StockBasicInfo,
    StockPrice,
    StockChipData,
    StockMonthlyRevenue,
    StockFeatureDaily,
    StockFeatureMonthly,
    StockLabel,
    StockPrediction,
    ModelTrainingRun,
    EtfNavData,
)
from apps.sync.models import DataSyncLog
from apps.fx_data.models import UsdTwdRate
from apps.market_data.models import MarketIndexRate

MAX_RETRY = 3
RETRY_WAIT_SECONDS = 5

# FinMind 三大法人「name」欄位分類對照。
# FinMind 舊版回傳中文名稱：「外資」「外資自營商」「投信」「自營商」「自營商(自行買賣)」「自營商(避險)」；
# finmind>=2.x 改回傳英文名稱：Foreign_Investor、Foreign_Dealer_Self、Investment_Trust、
# Dealer_self、Dealer_Hedging。兩種命名都要能辨識：
# 含「外資/Foreign」的都算外資買賣超、含「投信/Investment_Trust」的算投信買賣超，
# 一般自營商（Dealer_self / Dealer_Hedging）不屬於任何一類，維持不計入。
FOREIGN_KEYWORDS = ["外資", "Foreign"]
TRUST_KEYWORDS = ["投信", "Investment_Trust"]


def _get_client():
    """建立 FinMind DataLoader，免費版不需要登入/Token。"""
    return DataLoader()


def _retry_call(fetch_fn, *args, **kwargs):
    """
    共用重試機制：呼叫 fetch_fn(*args, **kwargs)，失敗時重試 MAX_RETRY 次。
    重試用盡仍失敗會拋出例外，讓呼叫端決定如何處理（原則：不可覆蓋既有資料）。
    """
    last_error = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 對外部 API 呼叫採寬鬆例外處理，記錄後重試
            last_error = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT_SECONDS)
            continue
    raise RuntimeError(f"呼叫 FinMind 失敗，已重試 {MAX_RETRY} 次：{last_error}")


# ---------------------------------------------------------
# 資料抓取（來源：FinMind，不需要 API Token）
# ---------------------------------------------------------

def fetch_stock_basic_info(stock_code):
    """
    抓取單一股票的基本資料（股票名稱、產業分類）。
    FinMind 的 taiwan_stock_info 是抓「全部股票」的總覽表，這裡抓回來後過濾出目標股票。
    """
    api = _get_client()
    df = _retry_call(api.taiwan_stock_info)

    if df.empty:
        return None

    row = df[df["stock_id"] == stock_code]
    if row.empty:
        return None

    row = row.iloc[0]
    return {
        "stock_code": stock_code,
        "stock_name": row["stock_name"],
        "industry": row.get("industry_category", ""),
    }


def fetch_stock_price(stock_code, start_date, end_date):
    """
    抓取股票日線行情，同時包含「原始收盤價」與「還原收盤價」。

    資料來源（自動 fallback）：
    1. 優先使用 FinMind：taiwan_stock_daily（原始股價）+ taiwan_stock_daily_adj（還原股價），
       兩者用日期對齊合併。計算 Return20 一定要用還原股價，但保留原始收盤價方便核對。
       注意：FinMind 免費版（free level）有 API 限制，無法抓取全部歷史股價，
       若呼叫失敗會自動改用 yfinance。
    2. yfinance fallback：抓取 {stock_code}.TW（例如 2330.TW），
       直接提供 Open/High/Low/Close/Adj Close/Volume，不需要 API Token。

    回傳：list[dict]，每筆包含 trade_date / open / high / low / close / adjusted_close / volume
    """
    api = _get_client()

    finmind_ok = False
    raw_df = None
    adj_df = None
    try:
        raw_df = _retry_call(
            api.taiwan_stock_daily, stock_id=stock_code, start_date=str(start_date), end_date=str(end_date)
        )
        adj_df = _retry_call(
            api.taiwan_stock_daily_adj, stock_id=stock_code, start_date=str(start_date), end_date=str(end_date)
        )
        finmind_ok = True
    except Exception as exc:  # noqa: BLE001 - FinMind 免費版權限不足時 fallback 到 yfinance
        finmind_ok = False

    if finmind_ok and raw_df is not None and not raw_df.empty:
        # 用日期當 key，把還原收盤價併進來
        adj_close_map = {}
        if adj_df is not None and not adj_df.empty:
            for _, row in adj_df.iterrows():
                adj_close_map[row["date"]] = float(row["close"])

        results = []
        for _, row in raw_df.iterrows():
            trade_date_str = row["date"]
            results.append({
                "trade_date": datetime.date.fromisoformat(trade_date_str),
                "open_price": float(row["open"]),
                "high_price": float(row["max"]),
                "low_price": float(row["min"]),
                "close_price": float(row["close"]),
                # 若當天沒有還原價（理論上不會發生），先退回用原始收盤價，避免整筆資料寫入失敗
                "adjusted_close": adj_close_map.get(trade_date_str, float(row["close"])),
                "volume": int(row["Trading_Volume"]),
            })
        return results

    # ===== yfinance fallback：FinMind 免費版權限不足時自動改用 yfinance =====
    return _fetch_stock_price_via_yfinance(stock_code, start_date, end_date)


def _fetch_stock_price_via_yfinance(stock_code, start_date, end_date):
    """
    用 yfinance 抓取台股行情（{stock_code}.TW），作為 FinMind 免費版權限不足時的替代來源。

    yfinance 直接提供 Open/High/Low/Close/Adj Close/Volume，
    Close 是原始收盤價，Adj Close 是還原收盤價（計算 Return20 用）。

    回傳：list[dict]，每筆包含 trade_date / open / high / low / close / adjusted_close / volume
    """
    import yfinance as yf

    yahoo_ticker = f"{stock_code}.TW"
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            ticker = yf.Ticker(yahoo_ticker)
            # yfinance 的 end 是不包含當天的，所以要 +1 天
            end_date_parsed = end_date if isinstance(end_date, datetime.date) else datetime.date.fromisoformat(str(end_date))
            end_date_exclusive = end_date_parsed + datetime.timedelta(days=1)
            df = ticker.history(start=str(start_date), end=str(end_date_exclusive))

            if df.empty:
                return []

            results = []
            for row_date, row in df.iterrows():
                # yfinance 舊版欄位是 "Adj Close"，新版（嚴格 Auto Adjust）可能只剩 "Close"
                adjusted_close = float(row["Adj Close"]) if "Adj Close" in df.columns else float(row["Close"])
                results.append({
                    "trade_date": row_date.date(),
                    "open_price": float(row["Open"]),
                    "high_price": float(row["High"]),
                    "low_price": float(row["Low"]),
                    "close_price": float(row["Close"]),
                    "adjusted_close": adjusted_close,
                    "volume": int(row["Volume"]),
                })
            return results

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_WAIT_SECONDS)
            continue

    raise RuntimeError(f"抓取股票 {yahoo_ticker}（yfinance）失敗，已重試 {MAX_RETRY} 次：{last_error}")


def fetch_stock_chip_data(stock_code, start_date, end_date):
    """
    抓取外資/投信買賣超（taiwan_stock_institutional_investors）
    與融資融券（taiwan_stock_margin_purchase_short_sale），依日期合併成一筆。

    回傳：list[dict]，每筆包含 trade_date / foreign_net_buy / trust_net_buy / margin_balance / short_balance
    """
    api = _get_client()

    inst_df = _retry_call(
        api.taiwan_stock_institutional_investors,
        stock_id=stock_code, start_date=str(start_date), end_date=str(end_date),
    )
    margin_df = _retry_call(
        api.taiwan_stock_margin_purchase_short_sale,
        stock_id=stock_code, start_date=str(start_date), end_date=str(end_date),
    )

    # 先把三大法人資料依日期彙整成 { date: {foreign_net_buy, trust_net_buy} }
    chip_by_date = {}
    if not inst_df.empty:
        for trade_date_str, group in inst_df.groupby("date"):
            foreign_net = 0
            trust_net = 0
            for _, row in group.iterrows():
                net_buy = int(row["buy"]) - int(row["sell"])
                name = str(row["name"])
                if any(keyword in name for keyword in FOREIGN_KEYWORDS):
                    foreign_net += net_buy
                elif any(keyword in name for keyword in TRUST_KEYWORDS):
                    trust_net += net_buy
            chip_by_date[trade_date_str] = {
                "foreign_net_buy": foreign_net,
                "trust_net_buy": trust_net,
            }

    # 再把融資融券資料合併進去
    margin_by_date = {}
    if not margin_df.empty:
        for _, row in margin_df.iterrows():
            margin_by_date[row["date"]] = {
                "margin_balance": int(row["MarginPurchaseTodayBalance"]),
                "short_balance": int(row["ShortSaleTodayBalance"]),
            }

    # 以兩份資料的日期聯集為主，缺值補 0（缺值代表當天可能沒有揭露，不是股價異常，可以先補 0）
    all_dates = set(chip_by_date.keys()) | set(margin_by_date.keys())
    results = []
    for trade_date_str in sorted(all_dates):
        chip = chip_by_date.get(trade_date_str, {"foreign_net_buy": 0, "trust_net_buy": 0})
        margin = margin_by_date.get(trade_date_str, {"margin_balance": 0, "short_balance": 0})
        results.append({
            "trade_date": datetime.date.fromisoformat(trade_date_str),
            "foreign_net_buy": chip["foreign_net_buy"],
            "trust_net_buy": chip["trust_net_buy"],
            "margin_balance": margin["margin_balance"],
            "short_balance": margin["short_balance"],
        })
    return results


# MOPS「營業額資訊」公告期限：依公開資訊觀測站申報規定，上市/上櫃公司應於
# 每月 10 日前公告上月份營收 ⇒ 任何公司的真實公告日必 <= 次月 10 日。
# （2026-09 實測探勘結論：MOPS 個別月營收查詢 t05st10_ifrs、月營收彙總靜態檔
#   /nas/t21/sii/t21sc03_民國_月.html、TWSE OpenAPI t187ap05_L 均「不提供」
#   每家公司的歷史真實公告日；真實公告日僅存在於每月 1~10 日的「當日彙整」佈告，
#   不留檔、不可回查。因此以法定期限作為特徵生效日，保證不早於真實公告。）
REVENUE_ANNOUNCE_DEADLINE_DAY = 10


def effective_announcement_date(revenue_month, source_date=None):
    """
    營收特徵的「保守生效日」＝max(次月 10 日, 資料源宣稱日期)。

    背景：FinMind taiwan_stock_month_revenue 的 date 欄位實測全部為「次月 1 日」
    （月級粒度，並非每家公司真實公告日）。若直接拿來當特徵生效日，模型會比真實
    世界提早約 4~9 天看到營收（look-ahead bias：真實世界多數公司在次月 5~10 日
    才公布）。改用 MOPS 營業額資訊的法定公告期限「次月 10 日」最為保守：
    模型在任何一天拿到的營收，都保證是當天前已對外公告的資訊。

    參數：
        revenue_month: 營收所屬月份（該月 1 號，date）
        source_date:   資料源宣稱的日期（可為 None）；僅在其「晚於」法定期限時採用
    """
    if revenue_month.month == 12:
        deadline = datetime.date(revenue_month.year + 1, 1, REVENUE_ANNOUNCE_DEADLINE_DAY)
    else:
        deadline = datetime.date(revenue_month.year, revenue_month.month + 1, REVENUE_ANNOUNCE_DEADLINE_DAY)
    if source_date is not None and source_date > deadline:
        return source_date
    return deadline


def fetch_stock_monthly_revenue(stock_code, start_date, end_date):
    """
    抓取月營收（taiwan_stock_month_revenue）。
    注意：FinMind 回傳的 date 欄位實測為「次月 1 日」（月級粒度），並非每家公司
    真實公告日，不可直接拿來當特徵生效日。此處一律經 effective_announcement_date()
    轉成保守生效日（次月 10 日）後再存入 announcement_date 欄位。

    回傳：list[dict]，每筆包含 revenue_month / announcement_date / revenue_amount / mom_growth / yoy_growth
    """
    api = _get_client()
    df = _retry_call(
        api.taiwan_stock_month_revenue,
        stock_id=stock_code, start_date=str(start_date), end_date=str(end_date),
    )

    if df.empty:
        return []

    # 依 revenue_year/revenue_month 排序，才能正確算 MoM/YoY
    df = df.sort_values(["revenue_year", "revenue_month"]).reset_index(drop=True)

    results = []
    for i, row in df.iterrows():
        revenue = float(row["revenue"])

        mom_growth = None
        if i > 0:
            prev_revenue = float(df.iloc[i - 1]["revenue"])
            if prev_revenue:
                mom_growth = (revenue - prev_revenue) / prev_revenue

        yoy_growth = None
        # 找去年同月的資料算 YoY
        same_month_last_year = df[
            (df["revenue_year"] == row["revenue_year"] - 1)
            & (df["revenue_month"] == row["revenue_month"])
        ]
        if not same_month_last_year.empty:
            last_year_revenue = float(same_month_last_year.iloc[0]["revenue"])
            if last_year_revenue:
                yoy_growth = (revenue - last_year_revenue) / last_year_revenue

        revenue_month = datetime.date(int(row["revenue_year"]), int(row["revenue_month"]), 1)
        results.append({
            "revenue_month": revenue_month,
            # 保守生效日：max(次月10日法定公告期限, 資料源日期)，避免提早看到未公告營收
            "announcement_date": effective_announcement_date(
                revenue_month, datetime.date.fromisoformat(str(row["date"]))
            ),
            "revenue_amount": int(revenue),
            "mom_growth": mom_growth,
            "yoy_growth": yoy_growth,
        })
    return results


def fetch_mops_month_revenue_summary(roc_year, month, market="sii", timeout=30):
    """
    抓 MOPS「月營收彙總」靜態報表（t21sc03），回傳 {公司代號: 當月營收(仟元)}。

    這是公開資訊觀測站官方數值，用途：交叉驗證 FinMind 營收數值是否正確（防資料源錯誤）。
    注意：此檔「沒有」每家公司的公告日期（只有單一「出表日期」），且 MOPS 各查詢介面
    均不提供歷史真實公告日 → 公告日一律採「次月 10 日」法定期限（見 effective_announcement_date）。

    參數：
        roc_year: 民國年（例如 115 = 西元 2026）
        month:    月份（1~12）
        market:   "sii"=國內上市（mopsov.twse.com.tw，實測可用）；
                  "tii"=國內上櫃（mopsov 無此路徑，TPEx 未提供同格式檔 → 回傳 None）
    回傳：
        dict {stock_code(str): revenue_thousand(int)}；抓不到/不支援時回傳 None
    """
    import re as _re
    import requests as _requests

    url = f"https://mopsov.twse.com.tw/nas/t21/{market}/t21sc03_{roc_year}_{month}.html"
    try:
        resp = _requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001（網路失敗時由呼叫端決定如何處理）
        return None
    if resp.status_code != 200 or len(resp.content) < 2000:
        return None

    # 檔案為 Big5(cp950) 編碼，內含數十張「產業別」分表
    html = resp.content.decode("cp950", errors="replace")

    def _strip(x):
        return _re.sub(r"\s+", "", _re.sub(r"<[^>]+>", " ", x or "").replace("&nbsp;", ""))

    result = {}
    for tb in _re.findall(r"<table[^>]*>.*?</table>", html, _re.S | _re.I):
        for row in _re.findall(r"<tr[^>]*>(.*?)</tr>", tb, _re.S | _re.I):
            cells = [_strip(c) for c in _re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, _re.S | _re.I)]
            # 公司資料列：[代號, 名稱, 當月營收, 上月營收, ...]；代號為 3~6 位數字
            if len(cells) >= 3 and _re.fullmatch(r"\d{3,6}", cells[0]):
                val = cells[2].replace(",", "")
                if val.lstrip("-").isdigit():
                    result[cells[0]] = int(val)
    return result if result else None


# ---------------------------------------------------------
# 資料同步（「檢查是否有新資料」按鈕會呼叫這裡）
# ---------------------------------------------------------

def check_and_sync_stock_data(stock_code):
    """
    「檢查是否有新資料」按鈕的核心邏輯，一次同步三種資料：行情、籌碼、月營收。
    每一種資料獨立比對最後日期、獨立寫 DataSyncLog，其中一種失敗不影響其他兩種。

    回傳：dict，彙整三種資料的同步結果，方便 views.py 包成 JsonResponse。
    """
    stock, _ = StockBasicInfo.objects.get_or_create(
        stock_code=stock_code,
        defaults={"stock_name": stock_code},
    )

    # 如果股票名稱還是暫時用代碼填的，補一次基本資料
    if stock.stock_name == stock_code:
        info = fetch_stock_basic_info(stock_code)
        if info:
            stock.stock_name = info["stock_name"]
            stock.industry = info["industry"]
            stock.save()

    results = {
        "price": _sync_stock_price(stock, stock_code),
        "chip": _sync_stock_chip(stock, stock_code),
        "revenue": _sync_stock_revenue(stock, stock_code),
    }

    overall_success = all(r["success"] for r in results.values())
    return {
        "success": overall_success,
        "message": "同步完成" if overall_success else "部分資料同步失敗，詳見 data 內容",
        "data": results,
    }


def _sync_stock_price(stock, stock_code):
    today = datetime.date.today()
    last_record = StockPrice.objects.filter(stock=stock).order_by("-trade_date").first()

    from django.conf import settings
    fetch_start = (
        last_record.trade_date + datetime.timedelta(days=1)
        if last_record else datetime.date.fromisoformat(settings.TRAIN_DATE_START)
    )

    if fetch_start > today:
        DataSyncLog.objects.create(
            source_type="stock_price", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="已是最新狀態",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    try:
        rows = fetch_stock_price(stock_code, fetch_start, today)
    except Exception as exc:  # noqa: BLE001
        DataSyncLog.objects.create(
            source_type="stock_price", target_code=stock_code,
            status="failed", source_name="FinMind", message=str(exc),
        )
        return {"success": False, "message": str(exc), "new_records": 0}

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_price", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="區間內沒有新資料",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    # 資料清洗：寫入資料庫前先清掉無效價格、邏輯不一致、重複日期，並記錄離群值警告
    from apps.sync.data_cleaning import clean_stock_price_rows, clean_defaults, summarize_report
    rows, clean_report = clean_stock_price_rows(rows)
    clean_summary = summarize_report(clean_report)

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_price", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind",
            message=f"抓到的資料清洗後全數無效（{clean_summary}）",
        )
        return {"success": True, "message": "抓到的資料清洗後全數無效", "new_records": 0}

    for row in rows:
        _defaults = clean_defaults({k: v for k, v in row.items() if k != "trade_date"})
        _defaults["source_name"] = "FinMind"
        StockPrice.objects.update_or_create(
            stock=stock, trade_date=row["trade_date"],
            defaults=_defaults,
        )

    DataSyncLog.objects.create(
        source_type="stock_price", target_code=stock_code,
        status="success_new_data", source_name="FinMind",
        message=f"新增/更新 {len(rows)} 筆（{clean_summary}）",
    )
    return {"success": True, "message": f"新增 {len(rows)} 筆", "new_records": len(rows)}


def _sync_stock_chip(stock, stock_code):
    today = datetime.date.today()
    last_record = StockChipData.objects.filter(stock=stock).order_by("-trade_date").first()

    from django.conf import settings
    fetch_start = (
        last_record.trade_date + datetime.timedelta(days=1)
        if last_record else datetime.date.fromisoformat(settings.TRAIN_DATE_START)
    )

    if fetch_start > today:
        DataSyncLog.objects.create(
            source_type="stock_chip", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="已是最新狀態",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    try:
        rows = fetch_stock_chip_data(stock_code, fetch_start, today)
    except Exception as exc:  # noqa: BLE001
        DataSyncLog.objects.create(
            source_type="stock_chip", target_code=stock_code,
            status="failed", source_name="FinMind", message=str(exc),
        )
        return {"success": False, "message": str(exc), "new_records": 0}

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_chip", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="區間內沒有新資料",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    # 資料清洗：融資融券餘額不可為負數，修正為0並記錄警告；移除重複日期
    from apps.sync.data_cleaning import clean_chip_data_rows, clean_defaults, summarize_report
    rows, clean_report = clean_chip_data_rows(rows)
    clean_summary = summarize_report(clean_report)

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_chip", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind",
            message=f"抓到的資料清洗後全數無效（{clean_summary}）",
        )
        return {"success": True, "message": "抓到的資料清洗後全數無效", "new_records": 0}

    for row in rows:
        _defaults = clean_defaults({k: v for k, v in row.items() if k != "trade_date"})
        _defaults["source_name"] = "FinMind"
        StockChipData.objects.update_or_create(
            stock=stock, trade_date=row["trade_date"],
            defaults=_defaults,
        )

    DataSyncLog.objects.create(
        source_type="stock_chip", target_code=stock_code,
        status="success_new_data", source_name="FinMind",
        message=f"新增/更新 {len(rows)} 筆（{clean_summary}）",
    )
    return {"success": True, "message": f"新增 {len(rows)} 筆", "new_records": len(rows)}


def _sync_stock_revenue(stock, stock_code):
    """
    月營收特殊處理：以「最後一筆營收所屬月份」為基準，往回多抓約 13 個月，
    但只覆寫「最後一筆所屬月起」的月份（參考月僅作為 MoM/YoY 的計算來源）。

    為什麼「往回多抓 13 個月」而不是從最後公告日+1 或從最後所屬月起抓：
      1. FinMind 最新月份常先以「佔位列」出現（revenue 有值但 MoM/YoY 為空或 NaN），
         若從「最後公告日+1天」起抓，該列日後被補上真值時已落在起點之前，永遠不會
         被回填（實測 2330 的 2026-07 即發生此狀況）。
      2. 若只從「最後所屬月起」抓，視窗第一筆正好是最後一個月——它往前沒有「前一個月」
         可算 MoM、往去年同月更不在視窗內（算不了 YoY），update_or_create 會用 None
         再覆寫一遍，等於沒修到（實測驗證此缺陷）。
      3. 因此往回 366 天起抓，讓最後一筆所屬月及其後，在視窗內「同時有前一個月（MoM）
         與去年同月（YoY）」可算；更早的月份只當參考，不寫入，避免破壞既有良好數值。
    """
    today = datetime.date.today()
    last_record = StockMonthlyRevenue.objects.filter(stock=stock).order_by("-revenue_month").first()

    from django.conf import settings
    if last_record:
        fetch_start = last_record.revenue_month - datetime.timedelta(days=366)
        keep_from = last_record.revenue_month
    else:
        fetch_start = datetime.date.fromisoformat(settings.TRAIN_DATE_START)
        keep_from = None

    if fetch_start > today:
        DataSyncLog.objects.create(
            source_type="stock_revenue", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="已是最新狀態",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    try:
        rows = fetch_stock_monthly_revenue(stock_code, fetch_start, today)
    except Exception as exc:  # noqa: BLE001
        DataSyncLog.objects.create(
            source_type="stock_revenue", target_code=stock_code,
            status="failed", source_name="FinMind", message=str(exc),
        )
        return {"success": False, "message": str(exc), "new_records": 0}

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_revenue", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind", message="區間內沒有新資料",
        )
        return {"success": True, "message": "已是最新狀態", "new_records": 0}

    # 資料清洗：營收 <= 0 視為異常移除、去除重複月份
    from apps.sync.data_cleaning import clean_revenue_rows, clean_defaults, summarize_report
    rows, clean_report = clean_revenue_rows(rows)
    clean_summary = summarize_report(clean_report)

    if not rows:
        DataSyncLog.objects.create(
            source_type="stock_revenue", target_code=stock_code,
            status="success_no_new_data", source_name="FinMind",
            message=f"抓到的資料清洗後全數無效（{clean_summary}）",
        )
        return {"success": True, "message": "抓到的資料清洗後全數無效", "new_records": 0}

    written = 0
    for row in rows:
        # 只寫「最後一筆所屬月」起的月份；更早的月份僅作為 MoM/YoY 參考，
        # 避免把已存在且正確的 MoM/YoY 用視窗內算不出來的 None 覆寫掉。
        if keep_from is not None and row["revenue_month"] < keep_from:
            continue
        _defaults = clean_defaults({
            "announcement_date": row["announcement_date"],
            "revenue_amount": row["revenue_amount"],
            "mom_growth": row["mom_growth"],
            "yoy_growth": row["yoy_growth"],
            "source_name": "FinMind",
        })
        StockMonthlyRevenue.objects.update_or_create(
            stock=stock, revenue_month=row["revenue_month"],
            defaults=_defaults,
        )
        written += 1

    DataSyncLog.objects.create(
        source_type="stock_revenue", target_code=stock_code,
        status="success_new_data", source_name="FinMind",
        message=f"新增/更新 {written} 筆（{clean_summary}）",
    )
    return {"success": True, "message": f"新增 {written} 筆", "new_records": written}


# ---------------------------------------------------------
# 特徵工程
# ---------------------------------------------------------

# 各項特徵計算用的滾動窗格天數（皆為「交易日」，不是日曆天）
MA_SHORT_WINDOW = 20
MA_LONG_WINDOW = 60
MA_SLOPE_LOOKBACK = 5       # 用幾天前的 MA 來算斜率方向
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
CHIP_RATIO_WINDOW = 20      # 法人買賣超佔比的累積天數
EXCESS_RETURN_WINDOW = 20   # 相對大盤超額報酬的計算天數
BETA_WINDOW = 60            # Beta 值的滾動迴歸天數
CROSS_MARKET_CHANGE_WINDOW = 20  # 匯率/SOX/VIX 月變動率的計算天數
VIX_PERCENTILE_MIN_HISTORY = 60  # VIX 百分位至少要有多少筆歷史資料才開始算，避免早期樣本太少失真

# ===== 新增特徵的常數 =====
INST_ACCEL_SHORT_WINDOW = 5    # 籌碼加速度的短期窗格（近5日）
INST_ACCEL_LONG_WINDOW = 20    # 籌碼加速度的長期窗格（近20日）
MARGIN_MOM_WINDOW = 20         # 融資餘額 MoM 的比較天數
VOL_RATIO_SHORT_WINDOW = 5     # 量能比短期窗格（5日均量）
VOL_RATIO_LONG_WINDOW = 20     # 量能比長期窗格（20日均量）
VOLATILITY_WINDOW = 20         # 波動率計算窗格（20日）
VOLATILITY_ANNUALIZE = 252     # 年化交易日數
RSI_DIFF_LOOKBACK = 3          # RSI 變化量的回看天數
MONTHLY_RSI_PERIOD = 14        # 月線 RSI 週期（14個月）
MONTHLY_MACD_FAST = 12         # 月線 MACD 快線 EMA 週期
MONTHLY_MACD_SLOW = 26         # 月線 MACD 慢線 EMA 週期
RS_VS_SOX_WINDOW = 20          # 個股對費半相對強弱度的計算天數

# ===== ETF 折溢價特徵的常數 =====
PREMIUM_MA_WINDOW = 5          # 折溢價均線窗格（近5日）
PREMIUM_Z_WINDOW = 20          # 折溢價 z-score 窗格（近20日）
PREMIUM_CHG_WINDOW = 5         # 折溢價變化的比較天數


def _is_etf(stock):
    """判斷標的是否為 ETF（FinMind taiwan_stock_info 的 industry_category == 'ETF'）。"""
    return bool(stock) and (stock.industry or "").strip().upper() == "ETF"


def sync_etf_nav(stock_code):
    """
    同步 ETF 每日淨值與折溢價到 EtfNavData（逐日累積）。

    來源：證交所 MIS 即時報價 API（收盤後呼叫可得當日淨值）。
    免費歷史淨值 API 目前不可得，歷史區段請用
    `python manage.py import_etf_nav <代碼> --csv <檔案>` 回補。

    非 ETF 標的直接略過（success=True、message 註明略過），讓一鍵管線不誤判失敗。
    連線或解析失敗回 success=False 與原因，不覆蓋既有資料。
    """
    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}", "rows_written": 0}
    if not _is_etf(stock):
        return {
            "success": True,
            "message": "非 ETF 標的，略過淨值/折溢價同步",
            "rows_written": 0,
            "skipped": True,
        }

    import datetime as _dt
    from decimal import Decimal, InvalidOperation

    import requests as _requests

    session = _requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        # 先拜訪主頁取得 cookie，再打報價 API
        session.get("https://mis.twse.com.tw/stock/index.jsp", headers=headers, timeout=10)
        resp = session.get(
            "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
            params={"ex_ch": f"t{stock_code}.tw"},
            headers=headers, timeout=10,
        )
        payload = resp.json()
        info_list = payload.get("msgArray") or []
        if not info_list:
            raise ValueError("即時報價無資料（msgArray 空）")
        info = info_list[0]
        nav_raw = info.get("navPerShare") or info.get("nav") or ""
        nav_val = Decimal(str(nav_raw))
        close_raw = info.get("z") or ""   # z＝最新成交價
        close_val = Decimal(str(close_raw)) if close_raw not in ("", "-") else None
        if nav_val <= 0:
            raise ValueError(f"淨值數值異常：{nav_val}")
    except (ValueError, InvalidOperation) as exc:
        return {"success": False, "message": f"ETF 淨值同步失敗：{exc}", "rows_written": 0}
    except Exception as exc:  # noqa: BLE001 - 對外部 API 採寬鬆例外處理
        return {
            "success": False,
            "message": f"ETF 淨值同步失敗（證交所即時報價暫時無法取得）：{type(exc).__name__}",
            "rows_written": 0,
        }

    today = _dt.date.today()
    premium = None
    if close_val and close_val > 0:
        premium = float((close_val - nav_val) / nav_val * 100)
    EtfNavData.objects.update_or_create(
        stock=stock, date=today,
        defaults={
            "nav": nav_val,
            "close_price": close_val,
            "premium_discount_pct": premium,
            "source_name": "twse_mis",
        },
    )
    msg = f"已寫入 {today} 淨值 {nav_val}" + (f"、折溢價 {premium:+.2f}%" if premium is not None else "")
    return {"success": True, "message": msg, "rows_written": 1}


def import_etf_nav_csv(stock_code, csv_path, source_name="csv_manual"):
    """
    從 CSV 匯入 ETF 歷史淨值（歷史回補的正式管道）。

    CSV 格式（有表頭，逗號分隔，支援中英文欄位名）：
      date,nav[,close]
      日期,淨值[,收盤價]
    例：
      date,nav
      2024-01-02,138.55

    close（收盤價）可省略；缺漏時自動用 DB 的還原收盤價補算折溢價率。
    回傳：dict，{"success", "message", "rows_written"}
    """
    from decimal import Decimal, InvalidOperation

    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}", "rows_written": 0}

    header_alias = {
        "date": "date", "日期": "date", "data_date": "date",
        "nav": "nav", "淨值": "nav", "每單位淨值": "nav",
        "close": "close", "收盤價": "close", "收盤": "close",
    }
    saved = bad_rows = 0
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            # 正規化表頭（中文/英文皆可）
            norm_fieldnames = {}
            for raw in reader.fieldnames or []:
                key = raw.strip()
                norm_fieldnames[key] = header_alias.get(key.lower()) or header_alias.get(key)
            if "date" not in norm_fieldnames.values() or "nav" not in norm_fieldnames.values():
                return {
                    "success": False,
                    "message": f"CSV 需包含 date(日期) 與 nav(淨值) 欄位，實際表頭：{reader.fieldnames}",
                    "rows_written": 0,
                }

            def pick(row, canonical):
                for raw, canon in norm_fieldnames.items():
                    if canon == canonical and row.get(raw) not in (None, ""):
                        return str(row[raw]).strip()
                return ""

            for row in reader:
                try:
                    d = pd.to_datetime(pick(row, "date")).date()
                    nav_v = Decimal(pick(row, "nav"))
                    if nav_v <= 0:
                        raise ValueError("nav <= 0")
                    close_raw = pick(row, "close")
                    close_v = Decimal(close_raw) if close_raw else None
                except (ValueError, TypeError, InvalidOperation):
                    bad_rows += 1
                    continue

                premium = None
                if close_v and close_v > 0:
                    premium = float((close_v - nav_v) / nav_v * 100)
                else:
                    # 收盤價缺漏時，用 DB 的還原收盤價補算折溢價
                    p = StockPrice.objects.filter(stock=stock, trade_date=d).first()
                    if p and p.adjusted_close:
                        premium = float((p.adjusted_close - nav_v) / nav_v * 100)

                EtfNavData.objects.update_or_create(
                    stock=stock, date=d,
                    defaults={
                        "nav": nav_v,
                        "close_price": close_v,
                        "premium_discount_pct": premium,
                        "source_name": source_name,
                    },
                )
                saved += 1
    except FileNotFoundError:
        return {"success": False, "message": f"找不到檔案：{csv_path}", "rows_written": 0}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"CSV 匯入失敗：{type(exc).__name__} - {exc}", "rows_written": 0}

    note = f"（略過 {bad_rows} 筆格式錯誤列）" if bad_rows else ""
    return {
        "success": True,
        "message": f"{stock_code} 已匯入 {saved} 筆淨值{note}",
        "rows_written": saved,
    }


def build_daily_features(stock_code):
    """
    計算日頻特徵，寫入 StockFeatureDaily。

    設計重點（避免 look-ahead bias）：
    - 所有滾動計算（rolling/pct_change/ewm）在 t 日只會用到「t 日以前」的資料，pandas 的
      rolling/ewm 本身就是這樣運作的，不會偷看未來。
    - 跨市場資料（匯率/SOX/VIX）用 merge_asof 對齊「最近一筆已知資料」，
      不會對到 t 日之後才出現的資料。
    - VIX 相對水位用 expanding（只看到目前為止的歷史）計算百分位，不是用全部資料的百分位，
      否則會讓早期的資料「看到未來」才知道自己在整個歷史分布的哪個位置。

    回傳：dict，{"success": bool, "message": str, "rows_written": int}
    """
    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}，請先執行爬蟲", "rows_written": 0}

    # 1. 讀取還原股價，這是所有技術面計算的基礎
    price_qs = StockPrice.objects.filter(stock=stock).order_by("trade_date").values(
        "trade_date", "adjusted_close", "volume"
    )
    price_df = pd.DataFrame(list(price_qs))
    if price_df.empty:
        return {"success": False, "message": "沒有股價資料，請先執行 sync_stock_data", "rows_written": 0}

    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    price_df["adjusted_close"] = price_df["adjusted_close"].astype(float)
    price_df["volume"] = price_df["volume"].astype(float)
    price_df = price_df.sort_values("trade_date").reset_index(drop=True)

    # 2. 技術面：MA 斜率、RSI、MACD
    price_df["ma20"] = price_df["adjusted_close"].rolling(MA_SHORT_WINDOW).mean()
    price_df["ma60"] = price_df["adjusted_close"].rolling(MA_LONG_WINDOW).mean()
    # 斜率定義：MA 相對 MA_SLOPE_LOOKBACK 天前的變化率，正值代表上升趨勢、負值代表下降趨勢
    price_df["ma20_slope"] = price_df["ma20"].pct_change(periods=MA_SLOPE_LOOKBACK)
    price_df["ma60_slope"] = price_df["ma60"].pct_change(periods=MA_SLOPE_LOOKBACK)

    delta = price_df["adjusted_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    price_df["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = price_df["adjusted_close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = price_df["adjusted_close"].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    price_df["macd"] = macd_line - signal_line  # 這裡存的是 MACD 柱狀圖（histogram），正負代表動能方向

    # ===== 新增技術面特徵 =====
    # 乖離率：股價相對 MA 的偏離程度（(股價 - MA) / MA）
    price_df["bias_ma20"] = (price_df["adjusted_close"] - price_df["ma20"]) / price_df["ma20"].replace(0, pd.NA)
    price_df["bias_ma60"] = (price_df["adjusted_close"] - price_df["ma60"]) / price_df["ma60"].replace(0, pd.NA)

    # 量能比：5日均量 / 20日均量
    price_df["vol_ma5"] = price_df["volume"].rolling(VOL_RATIO_SHORT_WINDOW).mean()
    price_df["vol_ma20"] = price_df["volume"].rolling(VOL_RATIO_LONG_WINDOW).mean()
    price_df["vol_ratio_5_20"] = price_df["vol_ma5"] / price_df["vol_ma20"].replace(0, pd.NA)

    # 波動率：近20日每日對數報酬率的年化標準差
    price_df["log_return"] = np.log(price_df["adjusted_close"] / price_df["adjusted_close"].shift(1))
    price_df["volatility_20d"] = (
        price_df["log_return"].rolling(VOLATILITY_WINDOW).std() * np.sqrt(VOLATILITY_ANNUALIZE)
    )

    # RSI 動能：14日 RSI 在近3天的變化量
    price_df["rsi_diff_3d"] = price_df["rsi"].diff(periods=RSI_DIFF_LOOKBACK)

    # 月線特徵：將日線資料重採樣為月線後計算 RSI 與 MACD
    price_df["year_month"] = price_df["trade_date"].dt.to_period("M")
    monthly_close = price_df.groupby("year_month")["adjusted_close"].last().reset_index()
    monthly_close["monthly_close"] = monthly_close["adjusted_close"]

    # 月線 RSI（14個月）
    m_delta = monthly_close["monthly_close"].diff()
    m_gain = m_delta.clip(lower=0)
    m_loss = -m_delta.clip(upper=0)
    m_avg_gain = m_gain.rolling(MONTHLY_RSI_PERIOD).mean()
    m_avg_loss = m_loss.rolling(MONTHLY_RSI_PERIOD).mean()
    m_rs = m_avg_gain / m_avg_loss.replace(0, pd.NA)
    monthly_close["rsi_monthly"] = 100 - (100 / (1 + m_rs))

    # 月線 MACD（EMA12 - EMA26）
    m_ema_fast = monthly_close["monthly_close"].ewm(span=MONTHLY_MACD_FAST, adjust=False).mean()
    m_ema_slow = monthly_close["monthly_close"].ewm(span=MONTHLY_MACD_SLOW, adjust=False).mean()
    monthly_close["macd_monthly"] = m_ema_fast - m_ema_slow

    # 把月線特徵對回每日資料（用 merge_asof 對齊「最近一筆已知月線」）
    monthly_close = monthly_close.rename(columns={"year_month": "ym"})
    monthly_close["ym_start"] = monthly_close["ym"].dt.start_time
    monthly_close = monthly_close.sort_values("ym_start")
    price_df = pd.merge_asof(
        price_df.sort_values("trade_date"),
        monthly_close[["ym_start", "rsi_monthly", "macd_monthly"]],
        left_on="trade_date", right_on="ym_start",
    )
    price_df = price_df.drop(columns=["ym_start", "year_month", "vol_ma5", "vol_ma20", "log_return"])

    merged = price_df

    # 3. 籌碼面：轉換為比例，不可用原始張數（CLAUDE.md 的核心規則）
    chip_qs = StockChipData.objects.filter(stock=stock).order_by("trade_date").values(
        "trade_date", "foreign_net_buy", "trust_net_buy", "margin_balance"
    )
    chip_df = pd.DataFrame(list(chip_qs))
    if not chip_df.empty:
        chip_df["trade_date"] = pd.to_datetime(chip_df["trade_date"])
        chip_df["institutional_net_buy"] = chip_df["foreign_net_buy"] + chip_df["trust_net_buy"]
        merged = merged.merge(
            chip_df[["trade_date", "institutional_net_buy", "margin_balance"]],
            on="trade_date", how="left",
        )
        merged["institutional_net_buy"] = merged["institutional_net_buy"].fillna(0)

        rolling_net_buy = merged["institutional_net_buy"].rolling(CHIP_RATIO_WINDOW).sum()
        rolling_volume = merged["volume"].rolling(CHIP_RATIO_WINDOW).sum()
        merged["institutional_net_buy_ratio"] = (rolling_net_buy / rolling_volume.replace(0, pd.NA)) * 100

        if stock.shares_outstanding:
            merged["margin_usage_ratio"] = (merged["margin_balance"].fillna(0) / stock.shares_outstanding) * 100
        else:
            # 沒有股本資料時，改用「融資餘額 / 近20日成交量」作為替代的資券使用率，
            # 避免整欄 NaN 導致訓練資料被 dropna 全數丟棄
            merged["margin_usage_ratio"] = (
                merged["margin_balance"].fillna(0) / merged["volume"].rolling(CHIP_RATIO_WINDOW).sum().replace(0, pd.NA)
            ) * 100

        # ===== 新增籌碼面特徵 =====
        # 外資與投信近20個交易日的累計買賣超總和
        merged["inst_net_buy_20d"] = merged["institutional_net_buy"].rolling(INST_ACCEL_LONG_WINDOW).sum()

        # 籌碼加速度：近5日日均買超 / 近20日日均買超
        # 當近20日日均買超為 0（代表沒有買超動能）時，加速度設為 0 而非 NaN，
        # 避免整欄 NaN 導致訓練資料被 dropna 全數丟棄
        short_avg = merged["institutional_net_buy"].rolling(INST_ACCEL_SHORT_WINDOW).mean()
        long_avg = merged["institutional_net_buy"].rolling(INST_ACCEL_LONG_WINDOW).mean()
        merged["inst_accel"] = short_avg / long_avg.replace(0, pd.NA)
        merged["inst_accel"] = merged["inst_accel"].fillna(0)

        # 融資餘額 MoM：相較於20個交易日前的變化率
        merged["margin_mom"] = merged["margin_balance"].pct_change(periods=MARGIN_MOM_WINDOW, fill_method=None)
    else:
        merged["institutional_net_buy_ratio"] = None
        merged["margin_usage_ratio"] = None
        merged["inst_net_buy_20d"] = None
        merged["inst_accel"] = None
        merged["margin_mom"] = None

    # 4. 相對大盤（^TWII）：超額報酬與 Beta
    #    改用 merge_asof（最近一筆已知）對齊——原本用「精準日期 left join」，
    #    指數只要缺一天，個股該列就整列 NaN；且 beta 的 rolling(60) 一旦掃到
    #    NaN 會連環污染後續約兩個月，這正是預測基準日卡住的元兇之一。
    benchmark_qs = MarketIndexRate.objects.filter(index_code="TWII").order_by("trade_date").values(
        "trade_date", "close_value"
    )
    benchmark_df = pd.DataFrame(list(benchmark_qs))
    if not benchmark_df.empty:
        benchmark_df["trade_date"] = pd.to_datetime(benchmark_df["trade_date"])
        benchmark_df["close_value"] = benchmark_df["close_value"].astype(float)
        benchmark_df = benchmark_df.sort_values("trade_date")

        merged = merged.sort_values("trade_date")
        merged = pd.merge_asof(
            merged,
            benchmark_df[["trade_date", "close_value"]],
            on="trade_date",
        )
        # 有限度前值填補：指數缺日（假日錯位、來源暫時落後）用最近已知值，
        # 但最多補 5 個交易日，超過仍為 NaN 以免長期使用過時資料
        merged["close_value"] = merged["close_value"].ffill(limit=5)
        merged["benchmark_return"] = merged["close_value"].pct_change(periods=EXCESS_RETURN_WINDOW)
        merged["benchmark_daily_return"] = merged["close_value"].pct_change()

        merged["stock_return"] = merged["adjusted_close"].pct_change(periods=EXCESS_RETURN_WINDOW)
        # 相對大盤超額報酬 = 個股 20 日報酬 - 大盤 20 日報酬
        merged["excess_return_vs_benchmark"] = (
            merged["stock_return"] - merged["benchmark_return"]
        )

        merged["stock_daily_return"] = merged["adjusted_close"].pct_change()
        rolling_cov = merged["stock_daily_return"].rolling(BETA_WINDOW).cov(merged["benchmark_daily_return"])
        rolling_var = merged["benchmark_daily_return"].rolling(BETA_WINDOW).var()
        merged["beta_vs_benchmark"] = rolling_cov / rolling_var.replace(0, pd.NA)
    else:
        merged["excess_return_vs_benchmark"] = None
        merged["beta_vs_benchmark"] = None

    merged = merged.sort_values("trade_date").reset_index(drop=True)

    # 5. 跨市場總經：美金匯率、SOX、VIX，用 merge_asof 對齊「最近一筆已知資料」
    fx_qs = UsdTwdRate.objects.order_by("rate_date").values("rate_date", "close_rate")
    fx_df = pd.DataFrame(list(fx_qs))
    if not fx_df.empty:
        fx_df["rate_date"] = pd.to_datetime(fx_df["rate_date"])
        fx_df["close_rate"] = fx_df["close_rate"].astype(float)
        fx_df["usd_twd_change_rate"] = fx_df["close_rate"].pct_change(periods=CROSS_MARKET_CHANGE_WINDOW)
        fx_df = fx_df.rename(columns={"rate_date": "trade_date"}).sort_values("trade_date")
        merged = pd.merge_asof(
            merged, fx_df[["trade_date", "usd_twd_change_rate", "close_rate"]], on="trade_date"
        )
        # 對齊後的美元兌台幣匯率原始值
        merged["usd_twd"] = merged["close_rate"]
        merged = merged.drop(columns=["close_rate"])
    else:
        merged["usd_twd_change_rate"] = None
        merged["usd_twd"] = None

    sox_qs = MarketIndexRate.objects.filter(index_code="SOX").order_by("trade_date").values(
        "trade_date", "close_value"
    )
    sox_df = pd.DataFrame(list(sox_qs))
    if not sox_df.empty:
        sox_df["trade_date"] = pd.to_datetime(sox_df["trade_date"])
        sox_df["close_value"] = sox_df["close_value"].astype(float)
        sox_df["sox_change_rate"] = sox_df["close_value"].pct_change(periods=CROSS_MARKET_CHANGE_WINDOW)
        # 費半 20 日累積漲跌幅（pct_change(20) 即為 20 日累積漲跌幅）
        sox_df["sox_mom"] = sox_df["close_value"].pct_change(periods=RS_VS_SOX_WINDOW)
        sox_df = sox_df.sort_values("trade_date")
        merged = pd.merge_asof(
            merged, sox_df[["trade_date", "sox_change_rate", "sox_mom"]], on="trade_date"
        )
    else:
        merged["sox_change_rate"] = None
        merged["sox_mom"] = None

    vix_qs = MarketIndexRate.objects.filter(index_code="VIX").order_by("trade_date").values(
        "trade_date", "close_value"
    )
    vix_df = pd.DataFrame(list(vix_qs))
    if not vix_df.empty:
        vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
        vix_df["close_value"] = vix_df["close_value"].astype(float)
        vix_df = vix_df.sort_values("trade_date")
        vix_df["vix_change_rate"] = vix_df["close_value"].pct_change(periods=CROSS_MARKET_CHANGE_WINDOW)
        # 相對水位：只看「目前為止」的歷史分布算百分位，避免用到未來資料
        vix_df["vix_level_percentile"] = (
            vix_df["close_value"].expanding(min_periods=VIX_PERCENTILE_MIN_HISTORY).rank(pct=True) * 100
        )
        # 改名為 vix_close_value，避免與大盤（TWII）merge 進來的 close_value 欄位衝突
        vix_df = vix_df.rename(columns={"close_value": "vix_close_value"})
        merged = pd.merge_asof(
            merged,
            vix_df[["trade_date", "vix_change_rate", "vix_level_percentile", "vix_close_value"]],
            on="trade_date",
        )
        # 對齊後的 VIX 恐慌指數原始值
        merged["vix"] = merged["vix_close_value"]
        merged = merged.drop(columns=["vix_close_value"])
    else:
        merged["vix_change_rate"] = None
        merged["vix_level_percentile"] = None
        merged["vix"] = None

    # 個股對費半的相對強弱度：個股20日漲跌幅 - 費半20日漲跌幅
    if "sox_mom" in merged.columns:
        merged["stock_20d_return"] = merged["adjusted_close"].pct_change(periods=RS_VS_SOX_WINDOW)
        merged["rs_vs_sox"] = merged["stock_20d_return"] - merged["sox_mom"]
        merged = merged.drop(columns=["stock_20d_return"])
    else:
        merged["rs_vs_sox"] = None

    # 5.5 ETF 折溢價特徵（僅 ETF 標的；非 ETF 或尚無淨值資料時整組留空，
    #     訓練端會把「整欄皆空」的特徵自動剔除，不影響非 ETF 樣本）
    etf_premium_cols = [
        "etf_premium_discount", "etf_premium_ma5", "etf_premium_z20", "etf_premium_chg_5d",
    ]
    if _is_etf(stock):
        nav_qs = EtfNavData.objects.filter(stock=stock).order_by("date").values("date", "nav")
        nav_df = pd.DataFrame(list(nav_qs))
        if not nav_df.empty:
            nav_df["date"] = pd.to_datetime(nav_df["date"])
            nav_df["nav"] = nav_df["nav"].astype(float)
            nav_df = nav_df.sort_values("date")
            # merge_asof：每個交易日對齊「當天或之前最近一筆」淨值，淨值缺日不產生 NaN、
            # 也不會用到未來的淨值（無 look-ahead bias）。
            # tolerance=7天：淨值資料若過期超過一週（例如只回補了早期歷史），
            # 寧可讓折溢價為 NaN 也不拿陳年淨值跟最新股價硬湊出失真的巨大折溢價。
            merged = pd.merge_asof(
                merged.sort_values("trade_date"),
                nav_df.rename(columns={"date": "trade_date"})[["trade_date", "nav"]],
                on="trade_date",
                tolerance=pd.Timedelta(days=7),
            )
            # 折溢價率 (%)＝(市價 − 淨值) / 淨值 × 100；市價統一用還原收盤價序列，
            # 避免混用不同來源的收盤快照
            premium = (merged["adjusted_close"] - merged["nav"]) / merged["nav"].replace(0, pd.NA) * 100
            merged["etf_premium_discount"] = premium
            merged["etf_premium_ma5"] = premium.rolling(PREMIUM_MA_WINDOW).mean()
            roll = premium.rolling(PREMIUM_Z_WINDOW)
            merged["etf_premium_z20"] = (premium - roll.mean()) / roll.std().replace(0, pd.NA)
            merged["etf_premium_chg_5d"] = premium.diff(periods=PREMIUM_CHG_WINDOW)
            merged = merged.drop(columns=["nav"])
        else:
            for col in etf_premium_cols:
                merged[col] = None
    else:
        for col in etf_premium_cols:
            merged[col] = None

    # 6. 寫入資料庫（upsert）
    feature_columns = [
        "ma20_slope", "ma60_slope", "rsi", "macd",
        "institutional_net_buy_ratio", "margin_usage_ratio",
        "excess_return_vs_benchmark", "beta_vs_benchmark",
        "usd_twd_change_rate", "sox_change_rate", "vix_level_percentile", "vix_change_rate",
        # 新增特徵
        "inst_net_buy_20d", "inst_accel", "margin_mom",
        "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
        "rsi_diff_3d", "rsi_monthly", "macd_monthly",
        "sox_mom", "rs_vs_sox", "usd_twd", "vix",
        # ETF 折溢價特徵（僅 ETF 標的有值）
        "etf_premium_discount", "etf_premium_ma5", "etf_premium_z20", "etf_premium_chg_5d",
    ]

    saved_count = 0
    for _, row in merged.iterrows():
        defaults = {}
        for col in feature_columns:
            value = row.get(col)
            defaults[col] = None if pd.isna(value) else float(value)

        StockFeatureDaily.objects.update_or_create(
            stock=stock,
            trade_date=row["trade_date"].date(),
            defaults=defaults,
        )
        saved_count += 1

    return {"success": True, "message": f"已計算並寫入 {saved_count} 筆日頻特徵", "rows_written": saved_count}


def build_monthly_features(stock_code):
    """
    將月營收轉換為月頻特徵，寫入 StockFeatureMonthly。
    用 announcement_date 當作 effective_date —— 該欄位現為「保守生效日」
    （max(次月 10 日法定公告期限, 資料源日期)，見 effective_announcement_date），
    對應到 StockMonthlyRevenue 抓取時就已經處理過的 MoM/YoY 計算結果。
    """
    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}，請先執行爬蟲", "rows_written": 0}

    revenue_qs = StockMonthlyRevenue.objects.filter(stock=stock).order_by("announcement_date").values(
        "announcement_date", "mom_growth", "yoy_growth"
    )
    revenue_df = pd.DataFrame(list(revenue_qs))
    if revenue_df.empty:
        return {"success": False, "message": "沒有月營收資料，請先執行 sync_stock_data", "rows_written": 0}

    saved_count = 0
    skipped_empty = 0
    for _, row in revenue_df.iterrows():
        mom_v = None if pd.isna(row["mom_growth"]) else float(row["mom_growth"])
        yoy_v = None if pd.isna(row["yoy_growth"]) else float(row["yoy_growth"])
        # FinMind 偶爾會回傳「月份已開、數值還沒公布」的佔位列（MoM/YoY 皆空），
        # 這種列寫進特徵表會打斷後續所有交易日的完整性檢查，必須略過。
        if mom_v is None and yoy_v is None:
            skipped_empty += 1
            continue
        StockFeatureMonthly.objects.update_or_create(
            stock=stock,
            effective_date=row["announcement_date"],
            defaults={
                "revenue_mom": mom_v,
                "revenue_yoy": yoy_v,
            },
        )
        saved_count += 1

    note = f"（略過 {skipped_empty} 筆數值未公布的佔位列）" if skipped_empty else ""
    return {
        "success": True,
        "message": f"已寫入 {saved_count} 筆月頻特徵{note}",
        "rows_written": saved_count,
    }


# ---------------------------------------------------------
# 標籤生成
# ---------------------------------------------------------

def build_labels(stock_code, use_advanced_gate=None):
    """
    計算 Return20 並產生標籤，寫入 StockLabel。

    use_advanced_gate=None（預設，依 settings.USE_ADVANCED_LABEL_GATE 決定）：
        settings.USE_ADVANCED_LABEL_GATE=True → 進階雙關卡版本
        settings.USE_ADVANCED_LABEL_GATE=False → MVP 版本

    use_advanced_gate=True（進階雙關卡版本）：
        超額報酬關卡：個股 Return20 − 大盤 Return20 > 門檻
        停損關卡：未來 20 個交易日內每日最低價不得跌破 t 日收盤 * settings.STOP_LOSS_RATIO
        兩者同時成立才標記為 1

    use_advanced_gate=False（MVP 單門檻版本）：
        Return20 > settings.LABEL_RETURN_THRESHOLD 即為 1

    注意：t+20 是指「第 20 個交易日後」，這裡直接用 StockPrice 裡第 i+20 筆資料，
    因為 StockPrice 本身只存交易日的資料，天然符合「20 個交易日」的定義。
    """
    from django.conf import settings

    # None（未指定）時依系統預設 USE_ADVANCED_LABEL_GATE 決定，預設啟用進階雙關卡
    if use_advanced_gate is None:
        use_advanced_gate = getattr(settings, "USE_ADVANCED_LABEL_GATE", True)

    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}，請先執行爬蟲", "rows_written": 0}

    lookahead = settings.LABEL_LOOKAHEAD_TRADING_DAYS
    threshold = settings.LABEL_RETURN_THRESHOLD

    price_qs = StockPrice.objects.filter(stock=stock).order_by("trade_date").values(
        "trade_date", "adjusted_close", "low_price"
    )
    price_df = pd.DataFrame(list(price_qs)).reset_index(drop=True)
    if len(price_df) <= lookahead:
        return {
            "success": False,
            "message": f"股價資料筆數（{len(price_df)}）不足 {lookahead} 天，無法計算 Return20",
            "rows_written": 0,
        }

    price_df["adjusted_close"] = price_df["adjusted_close"].astype(float)
    price_df["low_price"] = price_df["low_price"].astype(float)

    # 進階雙關卡才需要大盤資料，用 trade_date 當索引方便查值
    benchmark_series = None
    if use_advanced_gate:
        benchmark_qs = MarketIndexRate.objects.filter(index_code="TWII").order_by("trade_date").values(
            "trade_date", "close_value"
        )
        benchmark_df = pd.DataFrame(list(benchmark_qs))
        if not benchmark_df.empty:
            benchmark_df["close_value"] = benchmark_df["close_value"].astype(float)
            benchmark_series = benchmark_df.set_index("trade_date")["close_value"]

    label_version = "advanced" if use_advanced_gate else "mvp"
    saved_count = 0

    # 只能算到「倒數第 lookahead 筆」為止，因為再往後就沒有 t+20 可以看
    for i in range(len(price_df) - lookahead):
        row = price_df.iloc[i]
        future_row = price_df.iloc[i + lookahead]

        p_t = row["adjusted_close"]
        p_future = future_row["adjusted_close"]
        if p_t == 0:
            continue
        return_20 = (p_future - p_t) / p_t

        if use_advanced_gate:
            excess_ok = False
            if benchmark_series is not None and row["trade_date"] in benchmark_series.index \
                    and future_row["trade_date"] in benchmark_series.index:
                b_t = benchmark_series.loc[row["trade_date"]]
                b_future = benchmark_series.loc[future_row["trade_date"]]
                if b_t:
                    benchmark_return = (b_future - b_t) / b_t
                    excess_ok = (return_20 - benchmark_return) > threshold

            # 停損關卡：未來 20 天內（不含第 0 天）每日最低價不得跌破 t 日收盤 * STOP_LOSS_RATIO
            window_lows = price_df.iloc[i + 1: i + lookahead + 1]["low_price"]
            stop_loss_ok = window_lows.min() >= p_t * settings.STOP_LOSS_RATIO

            label = 1 if (excess_ok and stop_loss_ok) else 0
        else:
            label = 1 if return_20 > threshold else 0

        StockLabel.objects.update_or_create(
            stock=stock,
            trade_date=row["trade_date"],
            label_version=label_version,
            defaults={"return_20": return_20, "label": label},
        )
        saved_count += 1

    return {
        "success": True,
        "message": f"已計算並寫入 {saved_count} 筆標籤（版本：{label_version}）",
        "rows_written": saved_count,
    }


# ---------------------------------------------------------
# 歷史訊號模擬（進出場邏輯）
# ---------------------------------------------------------

TAKE_PROFIT_PCT = 0.06    # 波段停利 6%（內部固定值，用於買賣點模擬，非前端可調）
STOP_LOSS_PCT = 0.045     # 波段停損 4.5%（內部固定值，用於買賣點模擬，非前端可調）


def compute_history_signals(stock_code, lookback_days=250, threshold_override=None, model_type=None):
    """
    用「最近一次訓練完成」（可依 model_type 限定）的模型，對歷史上每個特徵完整的交易日做批次推論，產生：
      - probs：每日突破機率序列（供機率圖畫曲線）
      - buys/sells：穿越制訊號日（供 K 線標記 B/S）——
        買進 B：機率「向上穿越」門檻（前日<門檻、當日≥門檻）且非空頭排列的日子；
        賣出 S：機率「向下穿越」門檻的日子，或以最近一次 B 為基準觸發停利(+6%)／
        停損(-4.5%) 的日子。
    threshold_override：外部指定門檻（例：前端策略面板調整後的值）；未提供時用模型門檻。
    model_type：「lightgbm」或「xgboost」；有值時只找該架構最近一次訓練完成的模型，
               None 則用全部架構中最近一次完成的模型。
    任何失敗都回 success=False 與原因，呼叫端可安全降級。
    """
    from .model_training import FEATURE_COLUMNS

    result = {
        "success": False,
        "message": "",
        "probs": [],
        "buys": [],
        "sells": [],
        "threshold": None,
        "model_version": None,
        # 停利／停損門檻（決策看板「AI 推論結果」卡片的賣出建議與後端訊號模擬共用同一套規則，
        # 由後端帶給前端，避免兩邊數字不同步）
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
    }

    run_qs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
    if model_type:
        run_qs = run_qs.filter(model_type=model_type)
    latest_run = run_qs.order_by("-completed_at").first()
    if not latest_run:
        model_hint = f"（模型架構：{model_type}）" if model_type else ""
        result["message"] = f"尚未有訓練完成的模型{model_hint}，請先建立訓練紀錄並完成訓練"
        return result

    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        result["message"] = f"找不到股票 {stock_code}"
        return result

    import joblib

    from .model_artifacts import get_model_file
    try:
        model_path = get_model_file(latest_run)
    except FileNotFoundError as exc:
        result["message"] = f"模型檔案遺失，請重新訓練（{exc}）"
        return result

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feat_all = bundle.get("feature_columns") or FEATURE_COLUMNS
    # ETF／指標型標的沒有月營收：該欄位對此標的「結構性不可得」，以 NaN 傳入模型
    # （LightGBM / XGBoost 原生支援缺值），不因它而丟棄整列。
    has_monthly = StockFeatureMonthly.objects.filter(stock=stock).exists()
    unavailable = set() if has_monthly else {"revenue_mom", "revenue_yoy"}
    threshold = float(threshold_override) if threshold_override else (
        float(getattr(latest_run, "threshold", None) or 0.5)
    )

    daily_qs = list(
        StockFeatureDaily.objects.filter(stock=stock)
        .order_by("-trade_date")[:int(lookback_days)]
    )[::-1]  # Django 不支援負索引，先取最新 N 筆再轉回時間正序
    monthly_rows = list(
        StockFeatureMonthly.objects.filter(stock=stock).order_by("effective_date")
    )

    # 對齊月頻特徵（revenue_mom/yoy 用「最近一次已公告」的值，避免 look-ahead bias）
    dates, rows = [], []
    mi, cur_monthly = 0, None
    for f in daily_qs:
        d = f.trade_date
        while mi < len(monthly_rows) and monthly_rows[mi].effective_date <= d:
            cur_monthly = monthly_rows[mi]
            mi += 1
        row, ok = {}, True
        for col in feat_all:
            if col == "revenue_mom":
                v = cur_monthly.revenue_mom if cur_monthly else None
            elif col == "revenue_yoy":
                v = cur_monthly.revenue_yoy if cur_monthly else None
            else:
                v = getattr(f, col, None)
            if v is None:
                if col in unavailable:
                    # 結構性不可得（如 ETF 無營收）：填 NaN 交給模型原生缺值處理
                    row[col] = np.nan
                    continue
                ok = False
                break
            row[col] = float(v)
        if ok:
            dates.append(d)
            rows.append(row)

    if len(rows) < 30:
        result["message"] = f"特徵完整天數不足（僅 {len(rows)} 天），無法產生歷史訊號"
        return result

    X = pd.DataFrame(rows)[feat_all]
    probs_arr = model.predict_proba(X)[:, 1]

    closes = {
        p.trade_date: float(p.adjusted_close)
        for p in StockPrice.objects.filter(stock=stock, trade_date__lte=max(dates))
        .order_by("-trade_date")[: len(dates) + 120]
    }

    # 全日收盤的滾動均線
    all_days = sorted(closes)
    ma20_map, ma60_map = {}, {}
    s20 = s60 = 0.0
    for i, d in enumerate(all_days):
        c = closes[d]
        s20 += c
        s60 += c
        if i >= 20:
            s20 -= closes[all_days[i - 20]]
        if i >= 60:
            s60 -= closes[all_days[i - 60]]
        if i >= 19:
            ma20_map[d] = s20 / 20
        if i >= 59:
            ma60_map[d] = s60 / 60

    prob_map = {d: float(p) for d, p in zip(dates, probs_arr)}
    probs_out = [
        {"date": d.isoformat(), "probability": round(prob_map[d], 6)}
        for d in sorted(prob_map)
    ]

    # ===== 穿越制訊號：機率「向上/向下穿越門檻」即標記，不受持倉鎖定限制 =====
    seq = [d for d in sorted(prob_map) if d in closes]
    buys, sells = [], []
    last_buy_idx = -999
    last_sell_date = None
    prev_p = None
    for idx, d in enumerate(seq):
        price = closes[d]
        ma20 = ma20_map.get(d)
        ma60 = ma60_map.get(d)
        p = prob_map[d]
        if not ma20 or not ma60:
            prev_p = p
            continue
        is_bear = (price < ma60) and (ma20 < ma60)

        # B：向上穿越門檻（前日 < 門檻 ≤ 今日）且非空頭排列
        crossed_up = prev_p is not None and prev_p < threshold <= p
        if crossed_up and not is_bear:
            buys.append(d.isoformat())
            last_buy_idx = idx

        # S-1：向下穿越門檻（前日 ≥ 門檻 > 今日）
        crossed_down = prev_p is not None and prev_p >= threshold > p

        # S-2：以最近一次 B 為基準，觸發停利 / 停損
        exit_hit = False
        if last_buy_idx >= 0 and idx > last_buy_idx:
            entry_price = closes[seq[last_buy_idx]]
            gross = (price - entry_price) / entry_price
            exit_hit = gross >= TAKE_PROFIT_PCT or gross <= -STOP_LOSS_PCT

        sold = False
        if (crossed_down or exit_hit) and (last_sell_date is None or d > last_sell_date):
            sells.append(d.isoformat())
            last_sell_date = d
            sold = True

        # 賣出後重置參考基準：TP/SL 不再對著遠古買點連環觸發；
        # 需等待下一次「向上穿越」才會出現新的 B，形成一組組 B→S 循環
        if sold:
            last_buy_idx = -999

        prev_p = p

        prev_p = p

    result.update({
        "success": True,
        "message": "",
        "probs": probs_out,
        "buys": buys,
        "sells": sells,
        "threshold": threshold,
        # 交易紀律門檻（決策看板卡片顯示停利/停損建議用；與模擬 S-2 同一組常數）
        "take_profit_pct": TAKE_PROFIT_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "model_version": latest_run.model_version,
    })
    return result


# ---------------------------------------------------------
# 滾動式重訓練（Walk-Forward Retraining）：隨資料成長自動前移訓練邊界
# ---------------------------------------------------------

def compute_walkforward_split(stock_codes=None, train_ratio=0.6, val_ratio=0.2,
                              lookback_years=None):
    """
    依資料庫現有行情的最早/最新日期，算出 60/20/20 的四個日期字串。

    lookback_years（滾動視窗年數）：
      - 有值（例 3）：只取「最新日期往前推 N 年」的區間切分 → 視窗隨時間向前滑動，
        太舊的資料被排除，模型永遠聚焦近期市場結構。
      - None / 0：使用全部歷史（擴張式視窗）。
    回傳：(train_start, train_end, validation_start, validation_end)
    """
    from django.db.models import Min, Max

    qs = StockPrice.objects.all()
    codes = [c.strip() for c in (stock_codes or []) if c and c.strip()]
    if codes:
        qs = qs.filter(stock__stock_code__in=codes)
    agg = qs.aggregate(min_date=Min("trade_date"), max_date=Max("trade_date"))
    if not agg["min_date"]:
        raise ValueError("資料庫尚無股價資料，請先執行「檢查是否有新資料」")

    max_d = agg["max_date"]
    min_d = agg["min_date"]
    if lookback_years and float(lookback_years) > 0:
        candidate = max_d - datetime.timedelta(days=int(float(lookback_years) * 365))
        if candidate > min_d:
            min_d = candidate  # 滾動視窗生效，舊於 N 年的資料不納入
    span_days = (max_d - min_d).days
    train_end = min_d + datetime.timedelta(days=span_days * train_ratio)
    val_end = min_d + datetime.timedelta(days=span_days * (train_ratio + val_ratio))
    val_start = train_end + datetime.timedelta(days=1)

    fmt = lambda d: d.strftime("%Y-%m-%d")
    return fmt(min_d), fmt(train_end), fmt(val_start), fmt(val_end)


def launch_walkforward_retrain(stock_codes_text="", model_type="lightgbm",
                               label_version="advanced", n_repeats=1,
                               hyperparams=None, dry_run=False,
                               lookback_years=None, include_latest=False,
                               existing_run_id=None):
    """完整管線式重訓練：**同步資料 → 建立特徵 → 建立標籤 → 訓練**。

    與「自選日期一鍵訓練」相同的一口氣流程，差別只在切分方式：
    - 這裡不用自選日期，而是在**同步完資料後**以「目前資料庫的最新 60/20/20」
      重新計算切分（所以資料每次增長，切分自動右移，模型永遠學到最新資料）。
    - lookback_years：滾動視窗年數（例 3＝只用近 3 年資料切分）；None/0＝全部歷史。
    - include_latest：True 時驗證區間延伸至資料庫最新日，並自動注入 refit_on_all——
      最終模型以「訓練＋驗證」全部資料重新擬合，學到最新市場（評估指標仍用原切分）。
    - dry_run=True 時只預覽步驟與切割日期、不寫入任何紀錄（供前端預覽與測試）。
    - existing_run_id：網頁背景按鈕模式預先建立的 pending run id（見
      create_pipeline_run_record）；由子行程傳入，沿用該筆紀錄進行訓練。
    """
    code_list = [c.strip() for c in (stock_codes_text or "").split(",") if c.strip()]
    if not code_list:
        code_list = list(
            StockBasicInfo.objects.values_list("stock_code", flat=True).order_by("stock_code")
        )
    if not code_list:
        raise ValueError("資料庫尚無任何股票基本資料，請先執行「檢查是否有新資料」")

    # ---- dry-run：唯讀預覽（以目前資料庫為準計算切分）----
    if dry_run:
        train_start, train_end, val_start, val_end = compute_walkforward_split(
            code_list, lookback_years=lookback_years,
        )
        if include_latest:
            from django.db.models import Max as _Max
            _mx = StockPrice.objects.filter(stock__stock_code__in=code_list).aggregate(
                m=_Max("trade_date"))["m"]
            if _mx and _mx.isoformat() > val_end:
                val_end = _mx.isoformat()
        suffix = (
            "；驗證已納入最新資料，最終模型將以全量資料重擬合"
            if include_latest else "；最後 20% 留作盲測"
        )
        steps_preview = (
            f"1. 同步資料（個股行情/籌碼/月營收＋美金匯率＋TWII/SOX/VIX＋ETF淨值）→ "
            f"2. 建立特徵 → 3. 建立標籤（{label_version}）→ "
            f"4. 訓練（{model_type}，最新 60/20/20：Train {train_start}~{train_end}｜"
            f"Val {val_start}~{val_end}）"
        )
        return {
            "success": True,
            "message": f"一鍵重訓練預覽（dry-run，未寫入任何資料）：{steps_preview}{suffix}",
            "data": {
                "train_start": train_start, "train_end": train_end,
                "validation_start": val_start, "validation_end": val_end,
                "split_mode": "最新資料 60/20/20",
                "steps": ["同步資料", "建立特徵", "建立標籤", "訓練"],
            },
        }

    # === 實際管線 ===
    pipeline_notes = []

    # ---- 步驟① 同步資料（跨市場全域一次＋逐檔個股同步）----
    try:
        from apps.fx_data.services import check_and_sync_fx_data
        _fx = check_and_sync_fx_data()
        pipeline_notes.append(f"[美金] {_fx.get('message', '')}")
    except Exception as exc:  # noqa: BLE001
        pipeline_notes.append(f"[美金] 同步失敗：{exc}")

    try:
        from apps.market_data.services import sync_all_indexes
        for _code, _r in (sync_all_indexes() or {}).items():
            pipeline_notes.append(f"[{_code}] {_r.get('message', '')}")
    except Exception as exc:  # noqa: BLE001
        pipeline_notes.append(f"[指數] 同步失敗：{exc}")

    sync_ok, sync_fail = [], []
    for code in code_list:
        try:
            _r = check_and_sync_stock_data(code)
            sync_ok.append(code)
            pipeline_notes.append(f"[{code}] {_r.get('message', '')}")
        except Exception as exc:  # noqa: BLE001
            sync_fail.append(f"{code}（{type(exc).__name__}: {exc}）")
        try:
            sync_etf_nav(code)  # 非 ETF 自動略過；失敗不影響主流程
        except Exception:  # noqa: BLE001
            pass
    if not sync_ok:
        raise ValueError(f"同步資料全部失敗，管線中止：{'；'.join(sync_fail)}")

    # ---- 步驟② 建立特徵 ----
    feat_ok, feat_fail = _features_pipeline_step(code_list)
    if not feat_ok:
        raise ValueError(f"建立特徵全部失敗，管線中止：{'；'.join(feat_fail[:3])}")

    # ---- 步驟③ 建立標籤 ----
    label_ok, label_fail = _labels_pipeline_step(code_list, label_version)
    if not label_ok:
        raise ValueError(f"建立標籤全部失敗，管線中止：{'；'.join(label_fail[:3])}")

    # ---- 步驟④ 同步完成後再以「最新 60/20/20」重算切分（資料已增長）----
    train_start, train_end, val_start, val_end = compute_walkforward_split(
        code_list, lookback_years=lookback_years,
    )
    hp = dict(hyperparams or {})
    if include_latest:
        from django.db.models import Max as _Max
        _mx = StockPrice.objects.filter(stock__stock_code__in=code_list).aggregate(
            m=_Max("trade_date"))["m"]
        if _mx and _mx.isoformat() > val_end:
            val_end = _mx.isoformat()
        hp["refit_on_all"] = True

    dates_payload = {
        "train_start": train_start,
        "train_end": train_end,
        "validation_start": val_start,
        "validation_end": val_end,
    }
    if train_end >= val_start:
        raise ValueError(f"資料天數太少，切不出訓練/驗證區間（{dates_payload}）")

    # ---- 步驟⑤ 建立（或沿用預先建立）並執行訓練 ----
    _existing_run = None
    if existing_run_id:
        _existing_run = ModelTrainingRun.objects.filter(id=existing_run_id).first()
        if _existing_run is None:
            raise ValueError(f"找不到預先建立的訓練紀錄 id={existing_run_id}（可能已被清除），請重新啟動")
    run, launched_msg, run_status, metrics = _finalize_pipeline_run(
        stock_codes_text, model_type, label_version, n_repeats,
        hp, train_start, train_end, val_start, val_end,
        existing_run=_existing_run,
    )

    step_summary = (
        f"① 同步 {len(sync_ok)} 檔成功{('，失敗：' + '、'.join(sync_fail)) if sync_fail else ''}｜"
        f"② 特徵 {len(feat_ok)} 檔成功{('，失敗：' + '、'.join(feat_fail)) if feat_fail else ''}｜"
        f"③ 標籤 {len(label_ok)} 檔成功{('，失敗：' + '、'.join(label_fail)) if label_fail else ''}"
    )
    return {
        "success": True,
        "message": (
            f"一鍵重訓練完成（最新 60/20/20）：{step_summary}｜"
            f"④ 訓練 Train {train_start} ~ {train_end}｜Val {val_start} ~ {val_end}｜"
            f"版本 {run.model_version}｜{launched_msg}"
        ),
        "data": {
            "id": run.id,
            "model_version": run.model_version,
            "status": run_status,
            "metrics": metrics,
            "split_mode": "最新資料 60/20/20",
            "pipeline_notes": pipeline_notes,
            "sync_fail": sync_fail,
            "feature_fail": feat_fail,
            "label_fail": label_fail,
            **dates_payload,
        },
    }


# ---------------------------------------------------------
# 一鍵完整訓練管線：同步資料 → 建立特徵 → 建立標籤 → 訓練
# ---------------------------------------------------------

def _validate_custom_dates(dates):
    """
    驗證「自選日期模式」的四個日期：齊全、格式正確、且 train_end < validation_start。
    通過回傳四個日期字串 tuple；不合規丟 ValueError。
    """
    if not isinstance(dates, dict):
        raise ValueError("自選日期模式需要 dates=dict（train_start/train_end/validation_start/validation_end）")
    keys = ("train_start", "train_end", "validation_start", "validation_end")
    missing = [k for k in keys if not (dates.get(k) or "").strip()]
    if missing:
        raise ValueError(f"自選日期模式缺少欄位：{'、'.join(missing)}")
    parsed = {}
    for k in keys:
        try:
            parsed[k] = datetime.datetime.strptime(dates[k].strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"日期格式錯誤：{k}={dates[k]}（需 YYYY-MM-DD）")
    if parsed["train_end"] >= parsed["validation_start"]:
        raise ValueError(
            f"訓練結束日（{parsed['train_end']}）必須早於驗證起始日（{parsed['validation_start']}），避免資料洩漏"
        )
    if parsed["train_start"] >= parsed["train_end"]:
        raise ValueError(f"訓練起始日（{parsed['train_start']}）必須早於訓練結束日（{parsed['train_end']}）")
    return tuple(dates[k].strip() for k in keys)


def launch_full_pipeline_training(stock_codes_text="", model_type="lightgbm",
                                  label_version="advanced", n_repeats=1,
                                  hyperparams=None, include_latest=False,
                                  dates=None, dry_run=False, existing_run_id=None):
    """一鍵完整訓練管線（自選日期）：同步資料 → 建立特徵 → 建立標籤 → 建立並執行訓練。

    dates（必要，自選日期模式）：dict 含四個日期 → 完全依使用者自選切分。
      不再支援「全自動模式」——需要「最新 60/20/20」一鍵訓練請改用
      launch_walkforward_retrain（首頁「一鍵重訓練」按鈕）。
    dry_run=True：唯讀預覽——只計算/驗證切分，不做任何同步、寫入或訓練。
    include_latest=True：驗證區間延伸到資料庫最新交易日並注入 refit_on_all。
    existing_run_id：網頁背景按鈕模式預先建立的 pending run id（見
      create_pipeline_run_record），沿用該筆紀錄進行訓練。
    容錯原則：單一股票失敗只記錄不中斷，某一步全部股票都失敗才中止。
    """
    code_list = [c.strip() for c in (stock_codes_text or "").split(",") if c.strip()]
    if not code_list:
        code_list = list(
            StockBasicInfo.objects.values_list("stock_code", flat=True).order_by("stock_code")
        )
    if not code_list:
        raise ValueError("資料庫尚無任何股票基本資料，請先執行「檢查是否有新資料」")

    # ---- 切分計算（dry_run 也只做這步，完全唯讀）----
    if not dates:
        raise ValueError(
            "自選日期一鍵訓練需要提供 dates（四個日期：train_start/train_end/"
            "validation_start/validation_end）；若要使用最新資料 60/20/20，"
            "請改用「一鍵重訓練」（walkforward-retrain）。"
        )
    train_start, train_end, val_start, val_end = _validate_custom_dates(dates)
    split_mode = "自選日期"

    hp = dict(hyperparams or {})
    if include_latest:
        from django.db.models import Max as _Max
        _q = StockPrice.objects.filter(stock__stock_code__in=code_list)
        _mx = _q.aggregate(m=_Max("trade_date"))["m"]
        if _mx and _mx.isoformat() > val_end:
            val_end = _mx.isoformat()
        hp["refit_on_all"] = True

    if train_end >= val_start:
        raise ValueError(f"資料天數太少，切不出訓練/驗證區間（{train_start}~{train_end}｜{val_start}~{val_end}）")

    dates_payload = {
        "train_start": train_start, "train_end": train_end,
        "validation_start": val_start, "validation_end": val_end,
    }
    steps_preview = (
        f"1. 同步資料（個股行情/籌碼/月營收＋美金匯率＋TWII/SOX/VIX＋ETF淨值）→ "
        f"2. 建立特徵 → 3. 建立標籤（{label_version}）→ "
        f"4. 訓練（{model_type}，{split_mode}：Train {train_start}~{train_end}｜Val {val_start}~{val_end}）"
    )
    if dry_run:
        suffix = "；驗證已納入最新資料，最終模型將以全量資料重擬合" if include_latest else ""
        return {
            "success": True,
            "message": f"一鍵管線預覽（dry-run，未寫入任何資料）：{steps_preview}{suffix}",
            "data": {"stock_codes": code_list, "split_mode": split_mode, **dates_payload},
        }

    # === 管線主體 ===
    return _run_full_pipeline(
        code_list, model_type=model_type, label_version=label_version,
        n_repeats=n_repeats, hp=hp, dates_payload=dates_payload,
        split_mode=split_mode, stock_codes_text=stock_codes_text,
        existing_run_id=existing_run_id,
    )


def _features_pipeline_step(code_list):
    """管線步驟②：建立日頻＋月頻特徵。回傳 (feat_ok, feat_fail)。"""
    feat_ok, feat_fail = [], []
    for code in code_list:
        try:
            _d = build_daily_features(code)
            _m = build_monthly_features(code)
            # ETF／指標型標的沒有月營收屬「正常情況」而非錯誤
            _ok = bool(_d.get("success")) and (
                bool(_m.get("success")) or ("月營收" in _m.get("message", ""))
            )
            if _ok:
                feat_ok.append(code)
            else:
                feat_fail.append(f"{code}（{_d.get('message') or _m.get('message')}）")
        except Exception as exc:  # noqa: BLE001
            feat_fail.append(f"{code}（{type(exc).__name__}: {exc}）")
    return feat_ok, feat_fail


def _labels_pipeline_step(code_list, label_version):
    """管線步驟③：建立標籤。回傳 (label_ok, label_fail)。"""
    use_advanced_gate = (label_version == "advanced")
    label_ok, label_fail = [], []
    for code in code_list:
        try:
            _l = build_labels(code, use_advanced_gate=use_advanced_gate)
            if _l.get("success"):
                label_ok.append(code)
            else:
                label_fail.append(f"{code}（{_l.get('message')}）")
        except Exception as exc:  # noqa: BLE001
            label_fail.append(f"{code}（{type(exc).__name__}: {exc}）")
    return label_ok, label_fail


def create_pipeline_run_record(stock_codes_text, model_type, label_version, n_repeats,
                               hp, train_start, train_end, val_start, val_end):
    """僅「建立」一筆 pending 的 ModelTrainingRun，不執行訓練。

    背景：網頁按鈕要在背景子行程真正跑之前，就先建立一筆訓練紀錄——
    這樣使用者按下按鈕後「立刻」可以在歷史訓練紀錄看到這筆（隨子行程
    的進度由 pending → running → completed/failed），而不是等到整個管線
    跑完才出現。與 _finalize_pipeline_run 的建立邏輯保持一致（版本命名
    重複時加時間戳）。
    """
    base_version = f"v_{model_type}_{train_start}_{train_end}_val_{val_start}_{val_end}"
    version = base_version
    if ModelTrainingRun.objects.filter(model_version=version).exists():
        from django.utils import timezone as _tz
        version = f"{base_version}_r{_tz.now().strftime('%Y%m%d%H%M%S')}"

    return ModelTrainingRun.objects.create(
        train_start=train_start,
        train_end=train_end,
        validation_start=val_start,
        validation_end=val_end,
        model_type=model_type,
        model_version=version,
        stock_codes=stock_codes_text or "",
        label_version=label_version,
        hyperparams=hp,
        n_repeats=max(1, int(n_repeats)),
        status="pending",
    )


def _finalize_pipeline_run(stock_codes_text, model_type, label_version, n_repeats,
                           hp, train_start, train_end, val_start, val_end,
                           existing_run=None):
    """管線步驟④：建立（或沿用既有）ModelTrainingRun 並執行訓練。

    existing_run：背景按鈕模式預先建立的 pending run（見 create_pipeline_run_record）。
    - 為 None：新建（原本行為）。
    - 非 None：沿用該 run；管線最終計算出的日期欄位會覆蓋預估值（因為 walkforward
      的切分是在同步資料後才重算的），確保紀錄與實際訓練一致。
    回傳 (run, launched_msg, run_status, metrics)。
    """
    if existing_run is not None:
        run = existing_run
        run.train_start = train_start
        run.train_end = train_end
        run.validation_start = val_start
        run.validation_end = val_end
        run.model_type = model_type
        run.stock_codes = stock_codes_text or ""
        run.label_version = label_version
        run.hyperparams = hp
        run.n_repeats = max(1, int(n_repeats))
        run.status = "pending"
        run.save(update_fields=[
            "train_start", "train_end", "validation_start", "validation_end",
            "model_type", "stock_codes", "label_version", "hyperparams",
            "n_repeats", "status",
        ])
    else:
        run = create_pipeline_run_record(
            stock_codes_text, model_type, label_version, n_repeats,
            hp, train_start, train_end, val_start, val_end,
        )

    from django.conf import settings as dj_settings
    if getattr(dj_settings, "TRAINING_ASYNC", False):
        from .tasks import train_model_task
        train_model_task.delay(run.id)
        launched_msg = "已排入非同步訓練佇列"
        run_status = run.status
        metrics = None
    else:
        from .model_training import run_training
        training_result = run_training(run.id)
        run.refresh_from_db()
        launched_msg = training_result["message"]
        run_status = run.status
        metrics = training_result.get("data")
    return run, launched_msg, run_status, metrics


def _run_full_pipeline(code_list, model_type, label_version, n_repeats,
                       hp, dates_payload, split_mode, stock_codes_text,
                       existing_run_id=None):
    """launch_full_pipeline_training 的實際執行段（dry-run 不會走到這裡）。"""
    pipeline_notes = []

    # ---- 步驟① 同步資料（跨市場全域一次＋逐檔個股同步）----
    try:
        from apps.fx_data.services import check_and_sync_fx_data
        _fx = check_and_sync_fx_data()
        pipeline_notes.append(f"[美金] {_fx.get('message', '')}")
    except Exception as exc:  # noqa: BLE001
        pipeline_notes.append(f"[美金] 同步失敗：{exc}")

    try:
        from apps.market_data.services import sync_all_indexes
        for _code, _r in (sync_all_indexes() or {}).items():
            pipeline_notes.append(f"[{_code}] {_r.get('message', '')}")
    except Exception as exc:  # noqa: BLE001
        pipeline_notes.append(f"[指數] 同步失敗：{exc}")

    sync_ok, sync_fail = [], []
    for code in code_list:
        try:
            _r = check_and_sync_stock_data(code)
            sync_ok.append(code)
            pipeline_notes.append(f"[{code}] {_r.get('message', '')}")
        except Exception as exc:  # noqa: BLE001
            sync_fail.append(f"{code}（{type(exc).__name__}: {exc}）")
        try:
            sync_etf_nav(code)  # 非 ETF 自動略過；失敗不影響主流程
        except Exception:  # noqa: BLE001
            pass
    if not sync_ok:
        raise ValueError(f"同步資料全部失敗，管線中止：{'；'.join(sync_fail)}")

    # ---- 步驟② 建立特徵 ----
    feat_ok, feat_fail = _features_pipeline_step(code_list)
    if not feat_ok:
        raise ValueError(f"建立特徵全部失敗，管線中止：{'；'.join(feat_fail[:3])}")

    # ---- 步驟③ 建立標籤 ----
    label_ok, label_fail = _labels_pipeline_step(code_list, label_version)
    if not label_ok:
        raise ValueError(f"建立標籤全部失敗，管線中止：{'；'.join(label_fail[:3])}")

    # ---- 步驟④ 建立並執行訓練 ----
    train_start = dates_payload["train_start"]
    train_end = dates_payload["train_end"]
    val_start = dates_payload["validation_start"]
    val_end = dates_payload["validation_end"]

    _existing_run = None
    if existing_run_id:
        from .models import ModelTrainingRun as _MTR
        _existing_run = _MTR.objects.filter(id=existing_run_id).first()
        if _existing_run is None:
            raise ValueError(f"找不到預先建立的訓練紀錄 id={existing_run_id}（可能已被清除），請重新啟動")
    run, launched_msg, run_status, metrics = _finalize_pipeline_run(
        stock_codes_text, model_type, label_version, n_repeats,
        hp, train_start, train_end, val_start, val_end,
        existing_run=_existing_run,
    )

    step_summary = (
        f"① 同步 {len(sync_ok)} 檔成功{('，失敗：' + '、'.join(sync_fail)) if sync_fail else ''}｜"
        f"② 特徵 {len(feat_ok)} 檔成功{('，失敗：' + '、'.join(feat_fail)) if feat_fail else ''}｜"
        f"③ 標籤 {len(label_ok)} 檔成功{('，失敗：' + '、'.join(label_fail)) if label_fail else ''}"
    )
    return {
        "success": True,
        "message": (
            f"一鍵管線完成（{split_mode}）：{step_summary}｜"
            f"④ 訓練 Train {train_start} ~ {train_end}｜Val {val_start} ~ {val_end}｜"
            f"版本 {run.model_version}｜{launched_msg}"
        ),
        "data": {
            "id": run.id,
            "model_version": run.model_version,
            "status": run_status,
            "metrics": metrics,
            "split_mode": split_mode,
            "pipeline_notes": pipeline_notes,
            "sync_fail": sync_fail,
            "feature_fail": feat_fail,
            "label_fail": label_fail,
            **dates_payload,
        },
    }


# ---------------------------------------------------------
# 模型預測
# ---------------------------------------------------------

def predict(stock_code, model_type=None):
    """
    載入「最近一次訓練完成」（可依 model_type 限定）的模型，對指定股票的最新一筆特徵做預測。

    流程：
    1. 找最新一筆 status="completed" 的 ModelTrainingRun（model_type 有值時只找該架構）
    2. 取該股票最新一筆 StockFeatureDaily
    3. 補上最近一筆已公告的月營收特徵（revenue_mom / revenue_yoy）
    4. 若特徵有缺值（代表滾動窗格天數還不夠），拒絕預測並說明原因
    5. 載入模型檔案，算出機率，寫入 StockPrediction

    回傳：dict，{"success": bool, "message": str, "data": {...}}
    """
    from .model_training import FEATURE_COLUMNS

    run_qs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
    if model_type:
        run_qs = run_qs.filter(model_type=model_type)
    latest_run = run_qs.order_by("-completed_at").first()
    if not latest_run:
        model_hint = f"（模型架構：{model_type}）" if model_type else ""
        return {
            "success": False,
            "message": f"尚未有訓練完成的模型{model_hint}，請先在「訓練/驗證區間設定」建立訓練紀錄並完成訓練",
            "data": None,
        }

    stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
    if not stock:
        return {"success": False, "message": f"找不到股票 {stock_code}，請先執行 sync_stock_data", "data": None}

    # 從最新一筆開始往前找「最近一筆所有特徵都完整」的資料。
    # 最新一筆往往因為滾動窗格天數不足、跨市場資料對齊時間差、月營收公告時間差而有缺值，
    # 直接拿最新一筆會導致預測失敗。往前找一筆特徵完整的資料，才能確保模型能正常預測。
    daily_field_names = [
        "ma20_slope", "ma60_slope", "rsi", "macd",
        "institutional_net_buy_ratio", "margin_usage_ratio",
        "excess_return_vs_benchmark", "beta_vs_benchmark",
        "usd_twd_change_rate", "sox_change_rate", "vix_level_percentile", "vix_change_rate",
        "inst_net_buy_20d", "inst_accel", "margin_mom",
        "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
        "rsi_diff_3d", "rsi_monthly", "macd_monthly",
        "sox_mom", "rs_vs_sox", "usd_twd", "vix",
    ]

    # ETF／指標型標的沒有月營收：月頻欄位對其「結構性不可得」，不列入完整性要求，
    # 改以 NaN 傳入模型（LightGBM / XGBoost 原生支援缺值）。
    has_monthly = StockFeatureMonthly.objects.filter(stock=stock).exists()

    import joblib

    from .model_artifacts import get_model_file
    bundle = joblib.load(get_model_file(latest_run))
    model = bundle["model"]
    feat_order = bundle.get("feature_columns") or FEATURE_COLUMNS

    feature_qs = StockFeatureDaily.objects.filter(stock=stock).order_by("-trade_date")

    # -------------------------------------------------------------
    # 序列模型（TFT）路徑：每個樣本是「最近 seq_len 個交易日」的特徵視窗，
    # 不能只送單列。收集「最新 seq_len 個特徵完整交易日」，與下方單列路徑
    # 使用完全相同的完整性判定（日頻欄位齊全；月營收僅在有月營收體系時要求），
    # 月頻特徵同樣用「最近一次已公告」的值對齊，避免 look-ahead bias。
    # -------------------------------------------------------------
    if bool(getattr(model, "requires_sequence", False)):
        seq_len = int(getattr(model, "seq_len", 30))
        collected = []  # 由新到舊：(daily_feat, monthly或None)
        for feat in feature_qs.iterator():
            if any(getattr(feat, col) is None for col in daily_field_names):
                continue
            m = None
            if has_monthly:
                m = (
                    StockFeatureMonthly.objects.filter(
                        stock=stock, effective_date__lte=feat.trade_date
                    )
                    .order_by("-effective_date")
                    .first()
                )
                if m is None or m.revenue_mom is None or m.revenue_yoy is None:
                    continue
            collected.append((feat, m))
            if len(collected) >= seq_len:
                break

        if len(collected) < seq_len:
            return {
                "success": False,
                "message": (
                    f"TFT 需要 {seq_len} 個特徵完整的交易日才能預測，"
                    f"目前只有 {len(collected)} 天，請確認已執行 build_features "
                    f"且資料筆數足夠"
                ),
                "data": None,
            }

        recs = []
        for f, mth in reversed(collected):   # 反轉成時間正序再組框
            row = {}
            for col in feat_order:
                if col == "revenue_mom":
                    v = mth.revenue_mom if mth else None
                elif col == "revenue_yoy":
                    v = mth.revenue_yoy if mth else None
                else:
                    v = getattr(f, col, None)
                row[col] = float(v) if v is not None else np.nan
            recs.append(row)

        Xs = pd.DataFrame(recs)[feat_order]
        Xs = Xs.apply(pd.to_numeric, errors="coerce").astype("float64")
        proba = float(model.predict_proba(Xs)[:, 1][-1])
        label = int(proba > 0.5)

        pred_date = collected[0][0].trade_date  # 最新一筆「特徵完整」的交易日
        StockPrediction.objects.update_or_create(
            stock=stock, trade_date=pred_date, model_version=latest_run.model_version,
            defaults={"predicted_probability": proba, "predicted_label": label},
        )
        return {
            "success": True,
            "message": f"預測完成（模型版本：{latest_run.model_version}，機率：{proba:.2%}）",
            "data": {
                "stock_code": stock_code,
                "trade_date": pred_date.isoformat(),
                "predicted_probability": proba,
                "predicted_label": label,
                "model_version": latest_run.model_version,
            },
        }

    latest_feature = None
    latest_monthly = None
    for feat in feature_qs.iterator():
        # 檢查日頻特徵是否完整
        if any(getattr(feat, col) is None for col in daily_field_names):
            continue
        # 月頻特徵只有在「該標的確實有月營收體系」時才要求完整
        m = None
        if has_monthly:
            m = (
                StockFeatureMonthly.objects.filter(stock=stock, effective_date__lte=feat.trade_date)
                .order_by("-effective_date")
                .first()
            )
            if m is None or m.revenue_mom is None or m.revenue_yoy is None:
                continue
        latest_feature = feat
        latest_monthly = m
        break

    if not latest_feature:
        hint = "" if has_monthly else "（此標的為 ETF／指標型，已自動略過月營收特徵）"
        return {
            "success": False,
            "message": f"找不到特徵完整的資料{hint}，請確認已執行 build_features 且資料筆數足夠（至少60個交易日以上）",
            "data": None,
        }

    row = {}
    for col in feat_order:
        if col == "revenue_mom":
            v = latest_monthly.revenue_mom if latest_monthly else None
        elif col == "revenue_yoy":
            v = latest_monthly.revenue_yoy if latest_monthly else None
        else:
            v = getattr(latest_feature, col, None)
        row[col] = float(v) if v is not None else None

    X = pd.DataFrame([row])[feat_order]
    # None/缺值 → NaN 並強制 float64，避免 object dtype 被 LightGBM/XGBoost 拒絕
    # （用 to_numeric 而非 fillna，避免 pandas 未來版本的 downcasting 行為變更警告）
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    # 結構性缺欄放行（與訓練端「整欄皆空自動剔除」對稱，NaN 餵給樹模型原生缺值處理）：
    # - ETF／指標型標的：月營收特徵結構性不可得 → 放行
    # - 個股：ETF 折溢價特徵結構性不可得（例如最新完成模型是以 ETF 資料訓練的）→ 同樣放行
    has_premium = StockFeatureDaily.objects.filter(
        stock=stock, etf_premium_discount__isnull=False
    ).exists()
    allowed_missing = set()
    if not has_monthly:
        allowed_missing |= {"revenue_mom", "revenue_yoy"}
    if not has_premium:
        allowed_missing |= {
            "etf_premium_discount", "etf_premium_ma5", "etf_premium_z20", "etf_premium_chg_5d",
        }
    missing_cols = [c for c in X.columns[X.isnull().any()] if c not in allowed_missing]
    if missing_cols:
        return {
            "success": False,
            "message": f"特徵有缺值（{', '.join(missing_cols)}），"
                        f"可能是資料筆數還不夠計算完整的滾動窗格特徵，請確認爬蟲資料是否足夠（至少60個交易日以上）",
            "data": None,
        }

    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba > 0.5)

    StockPrediction.objects.update_or_create(
        stock=stock, trade_date=latest_feature.trade_date, model_version=latest_run.model_version,
        defaults={"predicted_probability": proba, "predicted_label": label},
    )

    return {
        "success": True,
        "message": f"預測完成（模型版本：{latest_run.model_version}，機率：{proba:.2%}）",
        "data": {
            "stock_code": stock_code,
            "trade_date": latest_feature.trade_date.isoformat(),
            "predicted_probability": proba,
            "predicted_label": label,
            "model_version": latest_run.model_version,
        },
    }
