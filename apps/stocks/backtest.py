"""
盲測區策略回測（Strategy Backtest on the Blind Zone）——把「模型指標」翻譯成「金錢語言」。

目的（對應 MODEL_TUNING_GUIDE.md 開頭「訓練得好＝策略夏普提高」的標準）：
    AUC / PR-AUC 只說明「模型會排序」，這個模組回答評審真正的問題：
    「照你的模型買，扣掉成本後到底賺不賺錢？」

策略規則（與 X_Y_VARIABLE_DEFINITION.md 的標籤定義對齊）：
    1. 訊號日 t：該日特徵完整（與 services.predict 同一套完整性判定，重用
       model_training._load_dataset 的特徵組裝，含月營收 merge_asof 防作弊對齊），
       且模型機率 >= threshold（預設 0.5）。
    2. 進場：t 的「次一交易日」收盤價（還原收盤價）買進——訊號日收盤後才產生訊號，
       最早也只能次日成交，避免 look-ahead bias。
    3. 出場：進場後持有 hold_days（預設 20，對齊 Return20 的 20 個交易日）個交易日，
       到期日收盤賣出；期間若任一日「最低價 <= 進場價 × stop_loss（預設 0.95）」，
       視為當日觸價停損、以停損價成交（與進階雙關卡標籤的 95% 停損關卡一致；
       --stop-loss 0 可關閉）。
    4. 部位不重疊：持有期間出現的新訊號一律忽略（沒有槓桿、不分批）。
    5. 成本：手續費 0.1425%（買賣各一次）＋ 證券交易稅 0.3%（賣出），
       net = (1 - FEE - TAX) / (1 + FEE) × exit/entry - 1。

盲測區定義：
    ModelTrainingRun 只記錄 train/validation 四個日期；60/20/20 的最後 20%
    （validation_end 之後到資料最新日）就是「模型訓練時完全沒看過的盲測區」。
    若最新完成的 run 把驗證區延伸到了資料最新日（include_latest），盲測區會是空的——
    此時自動回溯找「有留盲測區」的最近一顆 run；全部都沒有時可用 --use-validation
    退回驗證區測試，但輸出會誠實標註「樣本內驗證區（非盲測），僅供參考」。

執行方式：
    python manage.py backtest_strategy 2330
    python manage.py backtest_strategy 2330 --model-type tft --threshold 0.55 --json report.json
"""
import datetime
import math
import os

import numpy as np
import pandas as pd

from .models import StockBasicInfo, StockPrice, ModelTrainingRun
from .model_training import _load_dataset

# 交易成本（台股一般股票）：手續費 0.1425% 買賣各一次、證交稅 0.3% 賣出時
FEE_RATE = 0.001425
TAX_RATE = 0.003

DEFAULT_THRESHOLD = 0.5     # 進場門檻（與決策看板的策略設定同一語意）
DEFAULT_HOLD_DAYS = 20      # 持有 20 個交易日（對齊 Return20）
DEFAULT_STOP_LOSS = 0.95    # 進場價 × 0.95 觸價停損（對齊進階雙關卡）；0 = 關閉
WARMUP_ROWS = 90            # 盲測區往前多載入的特徵列數（TFT 組 seq_len 視窗用）
TRADING_DAYS_PER_YEAR = 252


def _to_float_df(rows):
    """QuerySet values() → DataFrame，並把 Decimal 欄位轉 float。"""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    for col in df.columns:
        if col != "trade_date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


def fetch_price_frame(stock, start_date, end_date):
    """個股日線（含還原收盤價與最低價，回測與停損判定都用它）。"""
    return _to_float_df(
        StockPrice.objects.filter(
            stock=stock, trade_date__gte=start_date, trade_date__lte=end_date,
        ).values("trade_date", "open_price", "high_price", "low_price",
                 "close_price", "adjusted_close")
    )


def fetch_benchmark_frame(start_date, end_date, index_code="TWII"):
    """大盤指數（^TWII）收盤值，作為同期基準。"""
    from apps.market_data.models import MarketIndexRate

    df = _to_float_df(
        MarketIndexRate.objects.filter(
            index_code=index_code, trade_date__gte=start_date, trade_date__lte=end_date,
        ).values("trade_date", "close_value")
    )
    if not df.empty:
        df = df.rename(columns={"close_value": "close"})
    return df


def pick_run(model_type=None, model_version=None, stock_code=None,
             require_blind=True, use_validation=False):
    """
    選一顆 status=completed 的模型：
      - model_version 有值 → 精確指定
      - 否則由新到舊掃描同 model_type 的 run：
        * require_blind=True：取第一顆「validation_end 之後仍有行情資料」的 run
          （include_latest 重訓的模型驗證區吃滿資料、沒有盲測區，自動跳過）
        * require_blind=False：直接取最新一顆
    回傳 (run, blind_start, blind_end) 或 (None, 錯誤訊息, None)。
    """
    qs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
    if model_type:
        qs = qs.filter(model_type=model_type)
    if model_version:
        qs = qs.filter(model_version=model_version)
    runs = qs.order_by("-completed_at")[:20]

    missing = 0  # 檔案已不在磁碟上的 run 數（錯誤訊息中誠實回報）
    latest_px = StockPrice.objects.order_by("-trade_date").values_list("trade_date", flat=True).first()
    if latest_px is None:
        return None, "資料庫尚無任何股價資料，請先同步", None
    if stock_code:
        stock_px = StockPrice.objects.filter(
            stock__stock_code=stock_code).order_by("-trade_date")
        latest_stock = stock_px.values_list("trade_date", flat=True).first()
        if latest_stock is not None:
            latest_px = max(latest_px, latest_stock)

    from .model_artifacts import model_file_available

    for run in runs:
        if not model_file_available(run):
            missing += 1
            continue  # 模型檔遺失且資料庫無入庫備份（FILE_MISSING）的紀錄無法載入，跳過
        blind_start = run.validation_end + datetime.timedelta(days=1)
        if use_validation:
            return run, run.validation_start, run.validation_end
        if not require_blind:
            return run, blind_start, latest_px
        if blind_start <= latest_px:
            return run, blind_start, latest_px
    if require_blind:
        hint = ("（最近完成的訓練都把驗證區延伸到資料最新日，沒留下盲測區）" if runs else "")
        miss = (f"；另有 {missing} 顆模型的檔案已不在 trained_models/，已自動跳過"
                if missing else "")
        return None, (
            f"找不到「留有盲測區」的已完成模型{hint}{miss}。"
            "可改用 --use-validation 在驗證區測試（輸出會標註非盲測），"
            "或用較舊的 --model-version 指定一顆有留盲測區的模型。"
        ), None
    return None, "尚未有訓練完成的模型，請先執行 train_model 或一鍵重訓練", None


def score_signals(run, bundle, stock, blind_start, blind_end):
    """
    對單一股票在「暖機區＋盲測區」逐日算模型機率。

    暖機區：盲測區往前 WARMUP_ROWS 個有標籤的交易日——TFT 需要 seq_len 連續視窗，
    樹模型則只是多算不用。特徵組裝直接重用 model_training._load_dataset
    （與訓練端同一套月營收 merge_asof、缺值剔除邏輯，避免回測端另寫一套造成不一致）。

    回傳 (signals_df, used_cols, note)：
      signals_df：trade_date / proba（時間正序，僅含盲測區的日期）
    """
    import joblib

    model = bundle["model"]
    feat_order = bundle.get("feature_columns") or []

    warmup_start = blind_start - datetime.timedelta(days=int(WARMUP_ROWS * 1.6))
    merged, used_cols, _dropped = _load_dataset(
        warmup_start, blind_end, stock_codes=[stock.stock_code],
        label_version=run.label_version,
    )
    if merged.empty or not feat_order:
        return pd.DataFrame(columns=["trade_date", "proba"]), [], "盲測區內沒有特徵資料"

    merged["trade_date"] = pd.to_datetime(merged["trade_date"])
    merged = merged.sort_values("trade_date").reset_index(drop=True)

    X = merged.reindex(columns=feat_order)
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")

    if bool(getattr(model, "requires_sequence", False)):
        # 不含 stock_code 欄 → TFTClassifier 視為單一股票的連續序列
        proba = model.predict_proba(X)[:, 1]
        note = ""
    else:
        proba = model.predict_proba(X)[:, 1]
        note = ""

    sig = pd.DataFrame({"trade_date": merged["trade_date"].values, "proba": proba})
    # 只保留盲測區內的訊號（暖機列只作為視窗歷史，不進場）
    sig = sig[sig["trade_date"] >= pd.Timestamp(blind_start)].reset_index(drop=True)
    return sig, used_cols, note


def simulate_trades(signals, prices, threshold, hold_days, stop_loss):
    """
    非重疊部位模擬。

    signals：trade_date / proba（正序）；prices：trade_date / high / low / adjusted_close（正序）。
    回傳 (trades, equity, skipped_end)：
      trades：每筆 dict（訊號日、進出場、出場原因、毛利、淨利、持有日數）
      equity：每日權益曲線 DataFrame（trade_date / equity / in_market）
      skipped_end：持有窗超出資料末端而捨棄的訊號數
    """
    dates = prices["trade_date"].tolist()
    adj = prices["adjusted_close"].to_numpy(dtype="float64")
    lows = prices["low_price"].to_numpy(dtype="float64")
    idx_of = {d: i for i, d in enumerate(dates)}

    trades, skipped_end = [], 0
    daily_ret = np.zeros(len(dates), dtype="float64")
    in_market = np.zeros(len(dates), dtype="bool")

    cursor = 0  # 部位不重疊：出場日之前的新訊號全部略過
    for _, row in signals.iterrows():
        if row["proba"] < threshold:
            continue
        sig_ts = row["trade_date"]
        # 進場：訊號日的次一交易日收盤
        entry_i = next((i for i, d in enumerate(dates) if d > sig_ts), None)
        if entry_i is None:
            skipped_end += 1
            continue
        if entry_i < cursor:
            continue  # 持有期間的新訊號 → 忽略
        exit_i = entry_i + hold_days
        if exit_i >= len(dates):
            skipped_end += 1
            continue  # 持有窗不完整 → 不計入（不做一半的交易）

        entry_px = adj[entry_i]
        stop_px = entry_px * stop_loss if stop_loss and stop_loss > 0 else None

        exit_j, exit_px, reason = exit_i, adj[exit_i], "到期"
        if stop_px is not None:
            hit = next((k for k in range(entry_i + 1, exit_i + 1) if lows[k] <= stop_px), None)
            if hit is not None:
                exit_j, exit_px, reason = hit, stop_px, "停損"

        gross = exit_px / entry_px - 1.0
        net = (1 - FEE_RATE - TAX_RATE) / (1 + FEE_RATE) * (exit_px / entry_px) - 1.0
        trades.append({
            "signal_date": sig_ts.date().isoformat(),
            "entry_date": dates[entry_i].date().isoformat(),
            "entry_price": round(float(entry_px), 2),
            "exit_date": dates[exit_j].date().isoformat(),
            "exit_price": round(float(exit_px), 2),
            "reason": reason,
            "hold_days": exit_j - entry_i,
            "gross": gross,
            "net": net,
        })

        # 權益曲線：持有日套用還原收盤價日報酬，出場日以實際出場價結算
        daily_ret[entry_i] += 0.0  # 進場當天以收盤價成交，當日損益從次日起算
        for k in range(entry_i + 1, exit_j + 1):
            base = adj[k - 1] if k - 1 >= entry_i else entry_px
            px = adj[k]
            if k == exit_j:
                px = exit_px
            daily_ret[k] = px / base - 1.0
        in_market[entry_i: exit_j + 1] = True
        cursor = exit_j + 1  # 出場次日起才接受新訊號

    equity = np.cumprod(1.0 + daily_ret)
    eq = pd.DataFrame({
        "trade_date": pd.to_datetime(dates),
        "equity": equity,
        "in_market": in_market,
    })
    return trades, eq, skipped_end


def compute_metrics(trades, eq, prices, bench):
    """
    把交易紀錄與權益曲線彙整成績效指標。

    - 夏普：以回測區「每日策略報酬」計算（不在場＝0），年化 252；
      標準差為 0（例如零交易）時誠實回 None，不硬算。
    - 最大回撤：權益曲線相對歷史高點的最大跌幅。
    - 同期比較：個股買進持有（還原收盤價頭尾）與大盤 ^TWII 買進持有。
    """
    res = {"n_trades": len(trades),
           "n_stop": sum(1 for t in trades if t["reason"] == "停損")}
    if trades:
        nets = [t["net"] for t in trades]
        wins = sum(1 for n in nets if n > 0)
        res.update({
            "win_rate": wins / len(nets),
            "avg_net": float(np.mean(nets)),
            "best_net": float(max(nets)),
            "worst_net": float(min(nets)),
            "sum_net": float(np.sum(nets)),
            "avg_hold_days": float(np.mean([t["hold_days"] for t in trades])),
        })

    eq_ret = eq["equity"].pct_change().fillna(0.0) if len(eq) else pd.Series(dtype="float64")
    std = float(eq_ret.std(ddof=0)) if len(eq_ret) else 0.0
    res["total_return"] = float(eq["equity"].iloc[-1] - 1.0) if len(eq) else 0.0
    res["sharpe"] = (float(eq_ret.mean()) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
                     ) if std > 1e-12 else None
    if len(eq):
        res["max_drawdown"] = float((eq["equity"] / eq["equity"].cummax() - 1.0).min())
        res["exposure"] = float(eq["in_market"].mean())
    else:
        res["max_drawdown"], res["exposure"] = 0.0, 0.0

    res["stock_bnh"] = (float(prices["adjusted_close"].iloc[-1] /
                              prices["adjusted_close"].iloc[0] - 1.0)
                        if len(prices) >= 2 else None)
    res["bench_bnh"] = (float(bench["close"].iloc[-1] / bench["close"].iloc[0] - 1.0)
                        if bench is not None and len(bench) >= 2 else None)
    res["excess_vs_bnh"] = (res["total_return"] - res["stock_bnh"]
                            if res["stock_bnh"] is not None else None)
    return res


def run_backtest(stock_code, model_type=None, threshold=DEFAULT_THRESHOLD,
                 hold_days=DEFAULT_HOLD_DAYS, stop_loss=DEFAULT_STOP_LOSS,
                 model_version="", use_validation=False):
    """
    完整流程：選模型 → 載入 bundle → 算訊號 → 模擬交易 → 對比基準 → 彙整指標。
    回傳 result dict（模型資訊／期間／規則／metrics／trades），CLI 與未來 API 共用。
    """
    try:
        stock = StockBasicInfo.objects.get(stock_code=stock_code)
    except StockBasicInfo.DoesNotExist:
        raise ValueError(f"找不到股票代碼 {stock_code}，請先同步該股票的資料")

    auto_note = ""
    run, p_start, p_end = pick_run(
        model_type=model_type or None, model_version=model_version or None,
        stock_code=stock_code, use_validation=use_validation,
    )
    if run is None and not use_validation:
        # 沒有任何「留有盲測區且模型檔健在」的 run → 自動退回驗證區（輸出誠實標註非盲測）
        auto_note = ("最近完成的訓練都沒留下盲測區（驗證區已延伸至資料最新日），"
                     "已自動改在驗證區回測——模型看過這段資料，績效僅供參考、非盲測證據。")
        run, p_start, p_end = pick_run(
            model_type=model_type or None, model_version=model_version or None,
            stock_code=stock_code, use_validation=True,
        )
    if run is None:
        raise ValueError(p_start)

    import joblib

    from .model_artifacts import get_model_file
    bundle = joblib.load(get_model_file(run))

    prices = fetch_price_frame(stock, p_start, p_end)
    if prices.empty:
        raise ValueError("回測區間內沒有個股行情資料")

    sig, used_cols, note = score_signals(run, bundle, stock, p_start, p_end)
    trades, eq, skipped = simulate_trades(sig, prices, threshold, hold_days, stop_loss)
    bench = fetch_benchmark_frame(p_start, p_end)
    metrics = compute_metrics(trades, eq, prices, bench)

    return {
        "stock_code": stock_code,
        "stock_name": stock.stock_name,
        "model_type": run.model_type,
        "model_version": run.model_version,
        "train_period": [str(run.train_start), str(run.train_end)],
        "validation_period": [str(run.validation_start), str(run.validation_end)],
        "backtest_period": [str(p_start), str(p_end)],
        "is_blind": not bool(use_validation or auto_note),
        "params": {
            "threshold": threshold, "hold_days": hold_days, "stop_loss": stop_loss,
            "cost_roundtrip_pct": round((FEE_RATE * 2 + TAX_RATE) * 100, 4),
        },
        "note": "　".join(x for x in (auto_note, note) if x),
        "n_signals": int(len(sig)),
        "skipped_end": skipped,
        "metrics": metrics,
        "trades": trades,
    }
