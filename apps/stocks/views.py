from django.db.models import Min, Max
from django.http import JsonResponse, HttpResponseRedirect
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_POST
import json
from datetime import datetime as _dt

from .models import ModelTrainingRun, StockBasicInfo, StockPrice, StockPrediction, StockFeatureDaily
from . import services


def _tw_iso(dt):
    """把資料庫的 UTC 時間轉成台北時間的 ISO 字串（給前端直接顯示）。

    settings.py 設 USE_TZ=True，DateTimeField 讀出來是 UTC 時間；
    直接 isoformat() 會讓「歷史訓練紀錄」的訓練時間慢 8 小時，
    這裡統一以 settings.TIME_ZONE（Asia/Taipei）轉換後再輸出（+08:00）。
    """
    if not dt:
        return None
    # USE_TZ=True 時讀出的必為 aware datetime；保險起見 naive 時原樣輸出
    return timezone.localtime(dt).isoformat() if timezone.is_aware(dt) else dt.isoformat()


def _shap_summary_from_run(run):
    """從 ModelTrainingRun.diagnostics 抽出 SHAP 診斷摘要（model-stability 每輪用）。

    回傳 None 表示該 run 從未被 diagnose_model 診斷過（diagnostics 空白）；
    有診斷時回傳方法名稱、SHAP top1 特徵與佔比、三類誤導證據計數。
    """
    diag = run.diagnostics or {}
    d = diag.get("diagnosis") or {}
    if not diag or not d:
        return None
    imp = d.get("importance") or []
    top = imp[0] if imp else None
    counts = {"leak": 0, "direction": 0, "jagged": 0}
    for f in (d.get("findings") or []):
        t = f.get("type")
        if t in counts:
            counts[t] += 1
    return {
        "diagnosed": True,
        "method": d.get("method", ""),
        "diagnosed_at": diag.get("diagnosed_at", ""),
        "top_feature": (top or {}).get("feature"),
        "top_share": round(float(top["share"]) * 100, 2) if top and top.get("share") is not None else None,
        "findings": counts,
        "findings_total": len(d.get("findings") or []),
    }


def _spawn_background_pipeline(cli_args):
    """以「分離子行程」執行 manage.py run_pipeline ...，讓 HTTP 請求立即返回。

    訓練（尤其 TFT）遠超過 HTTP 請求合理時長，同步執行會讓 gunicorn worker
    因 timeout 被殺、前端收到空回應（Failed to execute 'json' on 'Response'）。
    改由背景子行程執行，stdout/stderr 導到 training_logs/ 下的日誌檔，
    訓練結果照常寫入 ModelTrainingRun（完成後重新整理頁面即可查看）。
    """
    import sys
    import uuid
    import subprocess
    from pathlib import Path
    from django.conf import settings

    base_dir = Path(settings.BASE_DIR)
    log_dir = base_dir / "training_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"pipeline_{_dt.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.log"
    with open(log_path, "ab") as logf:
        logf.write(f"===== {_dt.now():%Y-%m-%d %H:%M:%S} {' '.join(cli_args)} =====\n".encode("utf-8", "ignore"))
        logf.flush()
        proc = subprocess.Popen(
            [sys.executable, str(base_dir / "manage.py"), *cli_args],
            cwd=str(base_dir),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid, str(log_path)


def _training_in_progress():
    """是否已有訓練進行中（6 小時內建立、狀態不是 completed/failed 的訓練紀錄）。

    只看近 6 小時：若背景子行程隨容器重啟被殺、紀錄永遠停在 pending/training，
    舊紀錄不該永久卡死「只能同時一個訓練」的保護。
    """
    from datetime import timedelta

    from .model_training import mark_stale_runs_failed

    # 先清掉殭屍訓練（逾時無心跳），避免誤判「仍有訓練進行中」
    mark_stale_runs_failed()
    cutoff = timezone.now() - timedelta(hours=6)
    return (ModelTrainingRun.objects
            .exclude(status__in=["completed", "failed"])
            .filter(created_at__gte=cutoff)
            .exists())


def _pipeline_busy_response():
    return JsonResponse({
        "success": False,
        "message": "已有訓練正在執行中。請等它完成後再啟動新的訓練（可重新整理頁面，在「歷史訓練紀錄」查看目前狀態）。",
        "data": None,
    }, status=409)


def _background_started_response(model_type, pid, log_file):
    return JsonResponse({
        "success": True,
        "message": (f"已於背景開始訓練（{model_type}）：同步資料→建立特徵→建立標籤→訓練。"
                    "首次訓練需數分鐘（TFT 更久），完成後重新整理頁面即可在「歷史訓練紀錄」查看結果。"),
        "data": {"background": True, "model_type": model_type, "pid": pid, "log_file": log_file},
    })


def model_info(request):
    """
    列出支援的預測模型類型及其描述、特徵處理方式、超參數說明與全部特徵的中文描述。
    query parameter model_type 可選：lightgbm / xgboost / tft，限定回傳單一模型的資訊。
    供前端「本次訓練資訊卡」動態顯示。
    """
    from .model_training import MODEL_INFO, FEATURE_DESCRIPTIONS, FEATURE_CATEGORIES, FEATURE_COLUMNS
    model_type = request.GET.get("model_type", "").strip()
    models = {}
    for key, info in MODEL_INFO.items():
        if model_type and model_type != key:
            continue
        models[key] = {
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "features_note": info["features_note"],
            "is_sequence_model": info["is_sequence_model"],
            "training_time": info["training_time"],
            "hyperparam_descriptions": info["hyperparam_descriptions"],
        }
    return JsonResponse({
        "success": True,
        "data": {
            "models": models,
            "features": FEATURE_DESCRIPTIONS,
            "categories": [
                {"name": cat, "features": feats} for cat, feats in FEATURE_CATEGORIES
            ],
            "feature_columns": FEATURE_COLUMNS,
        },
    })


def predict_stock(request, stock_code):
    """「進行 AI 預測」按鈕對應的 API，呼叫 services.predict(stock_code)。"""
    try:
        model_type = request.GET.get("model_type") or None
        return JsonResponse(services.predict(stock_code, model_type))
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"預測失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def check_new_stock_data(request, stock_code):
    """
    「檢查是否有新資料」按鈕對應的 API。
    同步範圍：個股行情/籌碼/月營收 ＋ 跨市場資料（美金匯率、TWII/SOX/VIX 指數）
    ＋ ETF 淨值/折溢價（僅 ETF 標的）。
    跨市場欄位是日頻特徵的必要輸入，若不同步，近期特徵會因缺值而導致
    「預測基準日卡在過去、無法預測最新交易日」。
    """
    try:
        # ---- ETF 淨值/折溢價同步（非 ETF 會自動略過）----
        from apps.stocks.services import sync_etf_nav
        nav_result = sync_etf_nav(stock_code)
        nav_part = f"[淨值] {nav_result.get('message', '')}"

        # ---- 個股行情/籌碼/月營收同步 ----
        result = services.check_and_sync_stock_data(stock_code)

        # ---- 跨市場同步（特徵的必要相依來源）----
        cross_parts = [nav_part]
        try:
            from apps.fx_data.services import check_and_sync_fx_data
            fx = check_and_sync_fx_data()
            cross_parts.append(f"[美金] {fx.get('message', '')}")
        except Exception as exc:  # noqa: BLE001
            cross_parts.append(f"[美金] 同步失敗：{exc}")

        try:
            from apps.market_data.services import sync_all_indexes
            indexes = sync_all_indexes()
            for code, r in (indexes or {}).items():
                cross_parts.append(f"[{code}] {r.get('message', '')}")
        except Exception as exc:  # noqa: BLE001
            cross_parts.append(f"[指數] 同步失敗：{exc}")

        # ---- 因子表重算（若行情/籌碼有新資料）----
        # 監控面板讀的是 StockFeatureDaily；只同步原始資料不重算特徵，
        # 會造成「籌碼已入庫但面板仍是舊值」的落差。此處在確實有新增時自動重算日頻特徵。
        # 注意：services.check_and_sync_stock_data 回傳的鍵是 price / chip / revenue
        #（先前誤寫成 stock_price / stock_chip，導致此自動重算永遠不會觸發）。
        _synced_data = result.get("data") or {}
        _needs_feature = bool(
            (_synced_data.get("price") or {}).get("new_records")
            or (_synced_data.get("chip") or {}).get("new_records")
        )
        if _needs_feature:
            try:
                _feat_result = services.build_daily_features(stock_code)
                cross_parts.append(f"[特徵] {_feat_result.get('message', '')}")
            except Exception as exc:  # noqa: BLE001
                cross_parts.append(f"[特徵] 重算失敗：{exc}")

        base_msg = result.get("message", "")
        result["message"] = f"{base_msg}｜{'｜'.join(cross_parts)}" if cross_parts else base_msg
        if isinstance(result.get("data"), dict):
            result["data"]["cross_market"] = cross_parts
            if isinstance(nav_result.get("data"), dict):
                result["data"]["etf_nav"] = nav_result["data"]
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"檢查新資料失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)



def stock_detail(request, stock_code):
    """
    個股詳情頁。
    呈現：標題顯示 + K 線圖 (3-row layout: K線/機率/特徵監控)，參考原始 Streamlit 決策看板版。
    預測與策略參數已搬至「策略設定」頁面 (strategy_settings)。
    """
    from django.conf import settings as django_settings
    from apps.market_data.models import MarketIndexRate
    from apps.fx_data.models import UsdTwdRate

    try:
        stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
        if not stock:
            return JsonResponse({
                "success": False,
                "message": f"找不到股票 {stock_code}，請先執行「檢查是否有新資料」",
                "data": None,
            }, status=404)

        price_qs = StockPrice.objects.filter(stock=stock).order_by("-trade_date")[:250]
        price_series = [
            {
                "date": p.trade_date.isoformat(),
                "open": float(p.open_price),
                "high": float(p.high_price),
                "low": float(p.low_price),
                "close": float(p.adjusted_close),
                "volume": p.volume,
            }
            for p in reversed(price_qs)
        ]

        benchmark_code = django_settings.BENCHMARK_INDEX_CODE.replace("^", "")
        benchmark_qs = MarketIndexRate.objects.filter(index_code=benchmark_code).order_by("-trade_date")[:250]
        benchmark_series = [
            {"date": b.trade_date.isoformat(), "value": float(b.close_value)}
            for b in reversed(benchmark_qs)
        ]

        fx_qs = UsdTwdRate.objects.order_by("-rate_date")[:250]
        fx_series = [
            {"date": f.rate_date.isoformat(), "value": float(f.close_rate)}
            for f in reversed(fx_qs)
        ]

        prediction_qs = StockPrediction.objects.filter(stock=stock).order_by("-trade_date")[:250]
        prediction_series = [
            {
                "date": p.trade_date.isoformat(),
                "probability": p.predicted_probability,
                "label": p.predicted_label,
            }
            for p in reversed(prediction_qs)
        ]

        feature_fields = [
            "inst_net_buy_20d", "inst_accel", "margin_mom", "ma20_slope", "ma60_slope",
            "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
            "rsi_diff_3d", "rsi_monthly", "macd_monthly", "sox_mom", "rs_vs_sox",
            "usd_twd_change_rate", "vix_level_percentile",
        ]
        feature_qs = StockFeatureDaily.objects.filter(stock=stock).order_by("-trade_date")[:250]
        feature_series = [
            {field: getattr(f, field) for field in feature_fields} |
            {"date": f.trade_date.isoformat()}
            for f in reversed(feature_qs)
        ]

        # 每個模型架構的「最近一次訓練完成」資訊（版本、門檻、訓練/驗證區間）。
        # 決策看板必須「依最後訓練的模組與時間區段」顯示——因此不能用全域最後
        # 一筆紀錄（會與所選架構不一致），須逐架構取值，前端依下拉選擇的架構切換。
        model_runs = {}
        for _r in (
            ModelTrainingRun.objects.filter(status="completed")
            .exclude(model_file_path="")
            .order_by("model_type", "-completed_at")
        ):
            if _r.model_type not in model_runs:
                model_runs[_r.model_type] = {
                    "model_version": _r.model_version,
                    "threshold": float(getattr(_r, "threshold", None) or 0.5),
                    "train_start": _r.train_start.isoformat() if _r.train_start else None,
                    "train_end": _r.train_end.isoformat() if _r.train_end else None,
                    "validation_start": _r.validation_start.isoformat() if _r.validation_start else None,
                    "validation_end": _r.validation_end.isoformat() if _r.validation_end else None,
                }

        # 全域最近一次完成（作為 threshold 等舊有參考的 fallback）
        latest_run = (
            ModelTrainingRun.objects.filter(status="completed")
            .exclude(model_file_path="")
            .order_by("-completed_at")
            .first()
        )
        # 預設模型＝「AI 模型架構」下拉預設值（TFT）；未切換時面板就以它的
        # 最後訓練版本與區間顯示，確保「最後訓練的模型與時間區段」符合
        default_model = "tft"
        _default_run = model_runs.get(default_model) or None
        run_dates = None
        latest_model_version = None
        default_threshold = 0.5
        if _default_run:
            latest_model_version = _default_run["model_version"]
            run_dates = {
                "train_start": _default_run["train_start"],
                "train_end": _default_run["train_end"],
                "validation_start": _default_run["validation_start"],
                "validation_end": _default_run["validation_end"],
            }
            default_threshold = _default_run["threshold"]
        elif latest_run:
            default_threshold = float(getattr(latest_run, "threshold", None) or 0.5)

        context = {
            "stock_code": stock_code,
            "stock_name": stock.stock_name,
            "model_choices": ["lightgbm", "xgboost", "tft"],
            "model_runs": model_runs,
            "model_runs_json": json.dumps(model_runs),
            "default_threshold_pct": round(default_threshold * 100, 2),
            "latest_model_version": latest_model_version,
            "run_dates": run_dates,
            "run_dates_json": json.dumps(run_dates) if run_dates else "null",
            "chart_data": json.dumps({
                "stock_code": stock_code,
                "stock_name": stock.stock_name,
                "price": price_series,
                "benchmark": benchmark_series,
                "fx": fx_series,
                "predictions": prediction_series,
                "features": feature_series,
            }),
        }
        return render(request, "stocks/detail.html", context)

    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"頁面載入失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)
def chart_data(request, stock_code):
    """
    回傳單一股票畫圖所需的所有資料，一次回傳避免前端打多次 API：
    - price: OHLC + volume，供 K 線圖與量能圖使用
    - benchmark: 大盤（^TWII）走勢
    - fx: 美金匯率走勢
    - predictions: 歷史預測紀錄
    - features: 日頻特徵，供「特徵監控」圖表

    任何伺服器端例外都會被擷取並回傳 JSON 格式的錯誤（HTTP 500），而非 HTML 500 頁面。
    """
    from django.conf import settings as django_settings
    from apps.market_data.models import MarketIndexRate
    from apps.fx_data.models import UsdTwdRate

    try:
        stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
        if not stock:
            return JsonResponse({
                "success": False,
                "message": f"找不到股票 {stock_code}，請先執行「檢查是否有新資料」",
                "data": None,
            }, status=404)

        lookback_days = int(request.GET.get("lookback_days", 250))

        price_qs = StockPrice.objects.filter(stock=stock).order_by("-trade_date")[:lookback_days]
        price_series = [
            {
                "date": p.trade_date.isoformat(),
                "open": float(p.open_price),
                "high": float(p.high_price),
                "low": float(p.low_price),
                "close": float(p.adjusted_close),
                "volume": p.volume,
            }
            for p in reversed(price_qs)
        ]

        benchmark_code = django_settings.BENCHMARK_INDEX_CODE.replace("^", "")
        benchmark_qs = MarketIndexRate.objects.filter(index_code=benchmark_code).order_by("-trade_date")[:lookback_days]
        benchmark_series = [
            {"date": b.trade_date.isoformat(), "value": float(b.close_value)}
            for b in reversed(benchmark_qs)
        ]

        fx_qs = UsdTwdRate.objects.order_by("-rate_date")[:lookback_days]
        fx_series = [
            {"date": f.rate_date.isoformat(), "value": float(f.close_rate)}
            for f in reversed(fx_qs)
        ]

        prediction_qs = StockPrediction.objects.filter(stock=stock).order_by("-trade_date")[:lookback_days]
        prediction_series = [
            {
                "date": p.trade_date.isoformat(),
                "probability": p.predicted_probability,
                "label": p.predicted_label,
            }
            for p in reversed(prediction_qs)
        ]

        feature_fields = [
            "inst_net_buy_20d", "inst_accel", "margin_mom", "ma20_slope", "ma60_slope",
            "bias_ma20", "bias_ma60", "vol_ratio_5_20", "volatility_20d",
            "rsi_diff_3d", "rsi_monthly", "macd_monthly", "sox_mom", "rs_vs_sox",
            "usd_twd_change_rate", "vix_level_percentile",
        ]
        feature_qs = StockFeatureDaily.objects.filter(stock=stock).order_by("-trade_date")[:lookback_days]
        feature_series = [
            {field: getattr(f, field) for field in feature_fields} |
            {"date": f.trade_date.isoformat()}
            for f in reversed(feature_qs)
        ]

        # 歷史機率曲線＋買賣點（穿越制訊號）；失敗不影響其他圖表資料
        try:
            _sig_th = request.GET.get("signal_threshold")
            sig = services.compute_history_signals(
                stock_code, lookback_days,
                threshold_override=(float(_sig_th) if _sig_th else None),
                model_type=(request.GET.get("model_type") or None),
            )
        except Exception as sig_exc:  # noqa: BLE001
            sig = {"success": False, "message": str(sig_exc), "probs": [], "buys": [], "sells": []}

        return JsonResponse({
            "success": True,
            "message": "",
            "data": {
                "stock_code": stock_code,
                "stock_name": stock.stock_name,
                "price": price_series,
                "benchmark": benchmark_series,
                "fx": fx_series,
                "predictions": prediction_series,
                "features": feature_series,
                "probs": sig.get("probs", []),
                "signals": {"buys": sig.get("buys", []), "sells": sig.get("sells", [])},
                "signal_threshold": sig.get("threshold"),
                # 停利/停損門檻：供前端「AI 推論結果」卡片的持倉狀態判斷（停利賣出/停損賣出/續抱）
                "take_profit_pct": sig.get("take_profit_pct"),
                "stop_loss_pct": sig.get("stop_loss_pct"),
                "model_version": sig.get("model_version"),
                "signal_message": None if sig.get("success") else sig.get("message"),
            },
        })
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"圖表資料查詢失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)
@require_POST
def create_training_run(request):
    """
    前端「訓練設定」區塊送出訓練參數時呼叫這支 API。
    必要參數：train_start / train_end / validation_start / validation_end
    可選參數：
        model_type: "lightgbm"（預設）/ "xgboost" / "tft"（Temporal Fusion Transformer）
        hyperparams: {"n_estimators": ..., "learning_rate": ..., "num_leaves": ..., "max_depth": ...}
                     （TFT 為 seq_len / d_model / n_heads / num_layers / dropout / learning_rate / epochs / batch_size）
        stock_codes: 逗號分隔的股票代碼字串，留空代表用資料庫內全部股票
        label_version: \"advanced\"（預設）或 \"mvp\"
        n_repeats: 交叉驗證重複次數 (1~10，預設 1)
    """
    try:
        payload = json.loads(request.body)
        train_start = payload["train_start"]
        train_end = payload["train_end"]
        validation_start = payload["validation_start"]
        validation_end = payload["validation_end"]
    except (KeyError, json.JSONDecodeError):
        return JsonResponse({
            "success": False,
            "message": "缺少必要參數：train_start / train_end / validation_start / validation_end",
            "data": None,
        }, status=400)

    if train_end >= validation_start:
        return JsonResponse({
            "success": False,
            "message": "訓練結束日必須早於驗證起始日，避免資料洩漏",
            "data": None,
        }, status=400)

    model_type = payload.get("model_type", "lightgbm")
    if model_type not in ("lightgbm", "xgboost", "tft"):
        return JsonResponse({
            "success": False,
            "message": f"不支援的 model_type：{model_type}，可用值：lightgbm / xgboost / tft",
            "data": None,
        }, status=400)

    n_repeats = payload.get("n_repeats", 1)
    if not isinstance(n_repeats, int) or n_repeats < 1 or n_repeats > 10:
        return JsonResponse({
            "success": False,
            "message": f"n_repeats={n_repeats} 超出合理範圍（1~10 的整數）",
            "data": None,
        }, status=400)

    run = ModelTrainingRun.objects.create(
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        model_type=model_type,
        model_version=f"v_{model_type}_{train_start}_{train_end}_val_{validation_start}_{validation_end}",
        stock_codes=payload.get("stock_codes", ""),
                label_version=payload.get("label_version", "advanced"),
        hyperparams=payload.get("hyperparams", {}),
        n_repeats=n_repeats,
        status="pending",
    )
    from django.conf import settings as dsettings
    if dsettings.TRAINING_ASYNC:
        from .tasks import train_model_task
        train_model_task.delay(run.id)
        return JsonResponse({
            "success": True,
            "message": "已排入訓練佇列，請稍後重新整理訓練紀錄表格查看結果",
            "data": {
                "id": run.id,
                "model_version": run.model_version,
                "status": run.status,
            },
        })
    else:
        from .model_training import run_training
        training_result = run_training(run.id)
        run.refresh_from_db()
        return JsonResponse({
            "success": training_result["success"],
            "message": training_result["message"],
            "data": {
                "id": run.id,
                "model_version": run.model_version,
                "status": run.status,
                "metrics": training_result.get("data"),
            },
        })


@require_POST
def walkforward_retrain(request):
    """
    完整管線式重訓練（walk-forward retraining）：**同步資料 → 建立特徵 → 建立標籤 → 訓練**。
    不需要手動填日期：在同步完資料後以「目前資料庫的最新 60/20/20」自動切分，
    資料每天增長，切分點自動右移，模型永遠學到最新資料。

    Body(JSON，全部可選)：
        stock_codes: 逗號分隔字串，留空＝全部股票
        model_type / label_version / n_repeats / hyperparams：同 create_training_run
        lookback_years / include_latest：同 launch_walkforward_retrain
        dry_run: true 時只預覽步驟與切割日期，不真的同步/寫入/訓練

    真正訓練（dry_run=false）改為「背景子行程」執行：訓練（尤其 TFT）遠超過
    HTTP 請求合理時長，同步執行會讓 gunicorn worker 逾時被殺、前端收到空回應。
    本端點先做唯讀預檢（參數/切分驗證，錯誤立刻回 400），通過後立即返回
    「已於背景開始訓練」，結果照常寫入 ModelTrainingRun。
    """
    try:
        try:
            payload = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            payload = {}
        model_type = payload.get("model_type", "lightgbm")
        label_version = payload.get("label_version", "advanced")
        n_repeats = payload.get("n_repeats", 1)
        hyperparams = payload.get("hyperparams") or {}
        lookback_years = payload.get("lookback_years") or None
        include_latest = bool(payload.get("include_latest", False))
        dry_run = bool(payload.get("dry_run", False))

        # 唯讀預檢（與 dry-run 相同的驗證與切分預覽）：參數錯誤立刻回應，不用等背景行程
        result = services.launch_walkforward_retrain(
            stock_codes_text=payload.get("stock_codes", ""),
            model_type=model_type,
            label_version=label_version,
            n_repeats=n_repeats,
            hyperparams=hyperparams,
            dry_run=True,
            lookback_years=lookback_years,
            include_latest=include_latest,
        )
        if dry_run:
            return JsonResponse(result)
        if not result.get("success"):
            return JsonResponse(result, status=400)

        if model_type not in ("lightgbm", "xgboost", "tft"):
            return JsonResponse({"success": False, "message": f"不支援的模型類型：{model_type}", "data": None}, status=400)
        if _training_in_progress():
            return _pipeline_busy_response()
        try:
            n_repeats = int(n_repeats)
        except (TypeError, ValueError):
            return JsonResponse({"success": False, "message": f"n_repeats 必須是整數，收到：{n_repeats!r}", "data": None}, status=400)

        cli_args = [
            "run_pipeline", "--mode", "walkforward",
            "--stock-codes", str(payload.get("stock_codes", "")),
            "--model-type", model_type,
            "--label-version", label_version,
            "--n-repeats", str(n_repeats),
        ]
        if lookback_years:
            cli_args += ["--lookback-years", str(lookback_years)]
        if include_latest:
            cli_args += ["--include-latest"]
        if hyperparams:
            cli_args += ["--hyperparams", json.dumps(hyperparams, ensure_ascii=False)]

        pid, log_file = _spawn_background_pipeline(cli_args)
        return _background_started_response(model_type, pid, log_file)
    except ValueError as exc:
        return JsonResponse({"success": False, "message": str(exc), "data": None}, status=400)
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"滾動重訓練失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


@require_POST
def full_pipeline_train(request):
    """
    一鍵完整訓練管線（自選日期）：同步資料 → 建立特徵 → 建立標籤 → 建立並執行訓練。

    Body(JSON)：
        dates（必要）: {"train_start", "train_end", "validation_start", "validation_end"}
                → 完全依使用者自選的四個日期切分
        include_latest: true 時驗證區間延伸至最新交易日並注入 refit_on_all
        dry_run: true 時唯讀預覽（只回傳切分與步驟，不做任何同步/寫入/訓練）
        stock_codes / model_type / label_version / n_repeats / hyperparams：同 create_training_run

    註：已移除「全自動模式」（dates 缺略自動 60/20/20）——需要最新 60/20/20
        一鍵訓練請改用「一鍵重訓練」：POST /stocks/walkforward-retrain/

    真正訓練（dry_run=false）與「一鍵重訓練」相同改為背景子行程執行：
    先做唯讀預檢（dates 齊全性/切分驗證，錯誤立刻回 400），通過後立即返回。
    """
    try:
        try:
            payload = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            payload = {}
        dates = payload.get("dates") or None
        model_type = payload.get("model_type", "lightgbm")
        label_version = payload.get("label_version", "advanced")
        n_repeats = payload.get("n_repeats", 1)
        hyperparams = payload.get("hyperparams") or {}
        include_latest = bool(payload.get("include_latest", False))
        dry_run = bool(payload.get("dry_run", False))

        # 唯讀預檢：dates 缺略/格式錯誤/切分不合法等問題立刻回 400（與原本行為一致）
        result = services.launch_full_pipeline_training(
            stock_codes_text=payload.get("stock_codes", ""),
            model_type=model_type,
            label_version=label_version,
            n_repeats=n_repeats,
            hyperparams=hyperparams,
            include_latest=include_latest,
            dates=dates,
            dry_run=True,
        )
        if dry_run:
            return JsonResponse(result)
        if not result.get("success"):
            return JsonResponse(result, status=400)

        if model_type not in ("lightgbm", "xgboost", "tft"):
            return JsonResponse({"success": False, "message": f"不支援的模型類型：{model_type}", "data": None}, status=400)
        if _training_in_progress():
            return _pipeline_busy_response()
        try:
            n_repeats = int(n_repeats)
        except (TypeError, ValueError):
            return JsonResponse({"success": False, "message": f"n_repeats 必須是整數，收到：{n_repeats!r}", "data": None}, status=400)

        cli_args = [
            "run_pipeline", "--mode", "custom",
            "--stock-codes", str(payload.get("stock_codes", "")),
            "--model-type", model_type,
            "--label-version", label_version,
            "--n-repeats", str(n_repeats),
            "--train-start", str(dates.get("train_start", "")),
            "--train-end", str(dates.get("train_end", "")),
            "--validation-start", str(dates.get("validation_start", "")),
            "--validation-end", str(dates.get("validation_end", "")),
        ]
        if include_latest:
            cli_args += ["--include-latest"]
        if hyperparams:
            cli_args += ["--hyperparams", json.dumps(hyperparams, ensure_ascii=False)]

        pid, log_file = _spawn_background_pipeline(cli_args)
        return _background_started_response(model_type, pid, log_file)
    except ValueError as exc:
        return JsonResponse({"success": False, "message": str(exc), "data": None}, status=400)
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"一鍵管線失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def stock_signals(request, stock_code):
    """
    輕量端點：回傳指定門檻下的歷史機率曲線與買賣訊號。
    query：
        threshold: 0~1（選填；未帶則用模型門檻）——前端調整策略面板門檻時呼叫，
                   讓主圖 B/S 旗標即時跟隨新門檻重算
        days:      回看交易日數（預設 250）
        model_type: lightgbm / xgboost（選填；未帶則用最近一次完成的模型）
    """
    try:
        th = request.GET.get("threshold")
        days = int(request.GET.get("days", 250))
        result = services.compute_history_signals(
            stock_code, days,
            threshold_override=(float(th) if th else None),
            model_type=(request.GET.get("model_type") or None),
        )
        return JsonResponse({
            "success": result["success"],
            "message": result.get("message", ""),
            "data": {
                "probs": result.get("probs", []),
                "signals": {"buys": result.get("buys", []), "sells": result.get("sells", [])},
                "threshold": result.get("threshold"),
                "take_profit_pct": result.get("take_profit_pct"),
                "stop_loss_pct": result.get("stop_loss_pct"),
                "model_version": result.get("model_version"),
            },
        })
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"訊號查詢失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def list_training_runs(request):
    """列出所有訓練紀錄，支援分頁。"""
    try:
        # 先把「逾時無心跳」的殭屍訓練標記為 failed，讓表格反映真實狀態
        from .model_training import mark_stale_runs_failed
        mark_stale_runs_failed()
        runs = ModelTrainingRun.objects.all().order_by("-created_at")
        page_size = int(request.GET.get("page_size", 20))
        page_num = int(request.GET.get("page", 1))
        start = (page_num - 1) * page_size
        end = start + page_size

        runs_page = runs[start:end]
        results = [
            {
                "id": r.id,
                "train_start": r.train_start.isoformat() if r.train_start else None,
                "train_end": r.train_end.isoformat() if r.train_end else None,
                "validation_start": r.validation_start.isoformat() if r.validation_start else None,
                "validation_end": r.validation_end.isoformat() if r.validation_end else None,
                "model_type": r.model_type,
                "model_version": r.model_version,
                "status": r.status,
                "created_at": _tw_iso(r.created_at),
                "completed_at": _tw_iso(r.completed_at),
                "stock_codes": r.stock_codes,
                "label_version": r.label_version,
                "n_repeats": r.n_repeats,
                # 驗證指標與超參數：前端表格與趨勢圖都需要（缺這些會讓 AUC/PR-AUC/超參數欄空白）
                "auc": r.auc,
                "pr_auc": r.pr_auc,
                "auc_std": r.auc_std,
                "pr_auc_std": r.pr_auc_std,
                "hyperparams": r.hyperparams or {},
                "diagnostics": r.diagnostics or {},
                # 背景訓練心跳：running 狀態時可用 heartbeat_seconds_ago 判斷
                # 「還在算」（數值小且持續變動）或「已中斷」（超過 300 秒）
                "heartbeat_at": _tw_iso(r.heartbeat_at),
                "heartbeat_seconds_ago": (
                    int((timezone.now() - r.heartbeat_at).total_seconds())
                    if r.heartbeat_at else None
                ),
            }
            for r in runs_page
        ]

        has_next = end < runs.count()
        has_prev = page_num > 1
        return JsonResponse({
            "success": True,
            "message": "",
            "data": {
                "runs": results,
                "pagination": {
                    "page": page_num,
                    "page_size": page_size,
                    "total": runs.count(),
                    "has_next": has_next,
                    "has_prev": has_prev,
                },
            },
        })
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"查詢訓練紀錄失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def build_features(request, stock_code):
    """觸發特徵工程計算（日頻 + 月頻，同步模式）。"""
    from .services import build_daily_features, build_monthly_features
    daily_result = build_daily_features(stock_code)
    monthly_result = build_monthly_features(stock_code)
    # ETF／指標型標的沒有月營收屬「正常情況」而非錯誤：視為略過月頻特徵即可，
    # 否則一鍵管線會在「建立特徵」這步誤判失敗而中斷。
    monthly_benign = monthly_result["success"] or ("月營收" in monthly_result.get("message", ""))
    overall_success = daily_result["success"] and monthly_benign
    parts = [f"[日頻] {daily_result['message']}"]
    if monthly_result["success"]:
        parts.append(f"[月頻] {monthly_result['message']}")
    else:
        parts.append("[月頻] 已略過（此標的無月營收，ETF／指標型屬正常）")
    return JsonResponse({
        "success": overall_success,
        "message": "｜".join(parts),
        "data": {
            "daily": daily_result,
            "monthly": monthly_result,
        },
    })


def build_labels(request, stock_code):
    """觸發標籤計算（同步模式）。"""
    from .services import build_labels as _build_labels
    from django.conf import settings as dsettings
    use_advanced = getattr(dsettings, "USE_ADVANCED_LABEL_GATE", True)
    return JsonResponse(_build_labels(stock_code, use_advanced_gate=use_advanced))


def date_range(request):
    """
    回傳行情資料的最早/最新日期，供首頁「自動切分 60/20/20」按鈕使用。
    可選 query string：stock_code=<代碼>，有帶就只看該股票的資料範圍，沒帶則看全部股票。
    """
    try:
        import datetime

        stock_code = request.GET.get("stock_code", "").strip()
        qs = StockPrice.objects.all()
        if stock_code:
            stock = StockBasicInfo.objects.filter(stock_code=stock_code).first()
            if not stock:
                return JsonResponse({
                    "success": False,
                    "message": f"找不到股票 {stock_code}，請先執行「檢查是否有新資料」",
                    "data": None,
                }, status=404)
            qs = qs.filter(stock=stock)

        agg = qs.aggregate(min_date=Min("trade_date"), max_date=Max("trade_date"))
        if not agg["min_date"]:
            return JsonResponse({
                "success": False,
                "message": "資料庫尚無股價資料，請先執行「檢查是否有新資料」",
                "data": None,
            }, status=404)

        # 滾動視窗：lookback_years 有值時，回報的最早日期改為「最新日往前推 N 年」
        min_d, max_d = agg["min_date"], agg["max_date"]
        lookback_years = request.GET.get("lookback_years")
        if lookback_years:
            try:
                lb = float(lookback_years)
            except ValueError:
                lb = 0.0
            if lb > 0:
                candidate = max_d - datetime.timedelta(days=int(lb * 365))
                if candidate > min_d:
                    min_d = candidate

        return JsonResponse({
            "success": True,
            "message": "",
            "data": {
                "stock_code": stock_code,
                "min_date": min_d.isoformat(),
                "max_date": max_d.isoformat(),
            },
        })
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"查詢資料區間失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def strategy_settings(request, stock_code):
    """
    策略設定已整併進「決策看板」頁右欄，
    本路由僅保留相容性：直接轉址到決策看板。
    """
    return HttpResponseRedirect(f"/stocks/{stock_code}/")


def get_prediction_settings(request, stock_code):
    """取得當前股票的預測資訊，用於策略設定頁。可選 query：model_type=lightgbm/xgboost/tft。"""
    try:
        model_type = request.GET.get("model_type") or None
        return JsonResponse(services.predict(stock_code, model_type))
    except Exception as exc:
        return JsonResponse({
            "success": False,
            "message": f"查詢預測失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


# Walk-Forward 穩定性評估門檻（MODEL_TUNING_GUIDE.md 五-5）
WF_AUC_OK_THRESHOLD = 0.55   # 單輪 AUC ≥ 此值視為「有參考價值」的一輪
WF_AUC_DECLINE_LIMIT = 0.05  # 最遠輪 AUC 相對最初輪的衰退上限（超過視為不穩定）
WF_MIN_ROUNDS = 2            # 至少要有幾輪才做穩定性判斷


def feature_importance(request):
    """
    GET /stocks/feature-importance/?run_id=<id>&top_n=20
    回傳特徵重要性 Top N（MODEL_TUNING_GUIDE.md 五-4）：
    - 預設取「最近一次完成的樹模型訓練」且已記錄 importance 的紀錄
      （LightGBM＝split 使用次數、XGBoost＝gain 增益；TFT 無原生重要性不提供）
    - 每個特徵附 pct（佔全部重要性的百分比），供前端長條圖與「單一特徵獨大」警語
    """
    try:
        top_n = max(1, min(50, int(request.GET.get("top_n", 20))))
        run_id = request.GET.get("run_id")
        if run_id:
            run = ModelTrainingRun.objects.filter(id=run_id, status="completed").first()
        else:
            # 找最近一次「已記錄重要性」的樹模型訓練；
            # 欄位上線前訓練的舊紀錄沒有記錄 → 從已存的模型檔（joblib）補算一次並快取回 DB
            import os  # noqa: F401
            import joblib

            from .model_artifacts import get_model_file, model_file_available
            runs = list(ModelTrainingRun.objects.filter(
                status="completed", model_type__in=("lightgbm", "xgboost"),
            ).order_by("-completed_at")[:20])
            run = None
            for r in runs:
                if r.feature_importance:
                    run = r
                    break
                if model_file_available(r):
                    try:
                        bundle = joblib.load(get_model_file(r))
                        model_obj = bundle.get("model")
                        cols = bundle.get("feature_columns") or []
                        imp = getattr(model_obj, "feature_importances_", None)
                        if imp is not None and cols:
                            pairs = sorted(
                                zip(cols, [float(v) for v in imp]),
                                key=lambda x: x[1], reverse=True,
                            )
                            r.feature_importance = [
                                {"feature": c, "importance": round(float(v), 6)}
                                for c, v in pairs
                            ]
                            r.save(update_fields=["feature_importance"])
                            run = r
                            break
                    except Exception:
                        continue  # 模型檔讀取失敗就試下一筆
        if not run:
            return JsonResponse({
                "success": True,
                "data": {
                    "items": [], "total_features": 0, "warning": False, "warning_text": "",
                    "model_version": None, "model_type": None, "importance_type": None,
                    "message": "尚無含特徵重要性的訓練紀錄（樹模型訓練完成後才會記錄；TFT 不提供）",
                },
            })
        items = run.feature_importance or []
        total = sum((i.get("importance") or 0) for i in items) or 1.0
        out = [{
            "feature": i.get("feature"),
            "importance": i.get("importance"),
            "pct": round((i.get("importance") or 0) / total * 100, 2),
        } for i in items[:top_n]]
        top1_pct = out[0]["pct"] if out else 0.0
        warning = top1_pct >= 35.0
        return JsonResponse({"success": True, "data": {
            "run_id": run.id,
            "model_version": run.model_version,
            "model_type": run.model_type,
            "importance_type": ("split（LightGBM 預設：特徵被使用的次數）"
                                if run.model_type == "lightgbm"
                                else "gain（XGBoost 預設：特徵帶來的增益）"),
            "total_features": len(items),
            "items": out,
            "warning": warning,
            "warning_text": (
                "單一特徵佔比過高（≥35%）：模型可能過度依賴特定欄位（例如背下股票代碼），"
                "而非學到通用規律；建議檢查特徵組成或加強正則化。"
                if warning else ""
            ),
        }})
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"查詢特徵重要性失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def model_stability(request):
    """
    GET /stocks/model-stability/
    Walk-Forward 穩定性評估（MODEL_TUNING_GUIDE.md 五-5）：把「已完成且有 AUC」的訓練紀錄
    依 model_type 分組、依驗證期距訓練期月數排序，回傳每輪 AUC／PR-AUC 與整體結論：
    - 穩定性：足（≥2 輪、平均 AUC ≥ 0.55、且最遠輪相對最初輪衰退 ≤ 0.05）／不足
    - 參考價值百分比：AUC ≥ 0.55 的輪數佔比
    """
    try:
        runs = (ModelTrainingRun.objects
                .filter(status="completed")
                .exclude(auc__isnull=True))
        grouped = {}
        for r in runs:
            grouped.setdefault(r.model_type, []).append(r)
        reports = []
        for mtype, rs in grouped.items():
            rs.sort(key=lambda r: (r.months_from_train_end is None,
                                   r.months_from_train_end or 0))
            rounds = [{
                "run_id": r.id,
                "model_version": r.model_version,
                "train_start": r.train_start.isoformat(),
                "validation_start": r.validation_start.isoformat(),
                "validation_end": r.validation_end.isoformat(),
                "months_from_train_end": r.months_from_train_end,
                "auc": r.auc,
                "pr_auc": r.pr_auc,
                # SHAP 診斷摘要：該輪 run 被 diagnose_model 診斷過才會有值（None＝未診斷）
                "shap": _shap_summary_from_run(r),
            } for r in rs]
            aucs = [r.auc for r in rs]
            avg_auc = sum(aucs) / len(aucs)
            decline = (aucs[0] - aucs[-1]) if len(aucs) >= WF_MIN_ROUNDS else 0.0
            ok_count = sum(1 for a in aucs if a >= WF_AUC_OK_THRESHOLD)
            value_pct = round(ok_count / len(aucs) * 100, 1)
            stable = (len(aucs) >= WF_MIN_ROUNDS
                      and avg_auc >= WF_AUC_OK_THRESHOLD
                      and decline <= WF_AUC_DECLINE_LIMIT)
            verdict = (
                f"穩定性：{'足' if stable else '不足'}"
                f"（{len(aucs)} 輪，平均 AUC {avg_auc:.4f}，最遠輪相對最初輪衰退 {decline:.4f}）"
                f"｜參考價值 {value_pct}%（AUC ≥ {WF_AUC_OK_THRESHOLD} 的輪數佔比）"
            )
            reports.append({
                "model_type": mtype,
                "rounds_count": len(rounds),
                "avg_auc": round(avg_auc, 4),
                "decline": round(decline, 4),
                "value_pct": value_pct,
                "stable": stable,
                "verdict": verdict,
                "rounds": rounds,
            })
        reports.sort(key=lambda x: x["model_type"])
        return JsonResponse({"success": True, "data": {"reports": reports}})
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"穩定性評估失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def shap_diagnostics(request):
    """
    GET /stocks/shap-diagnostics/
    最近一次完成 SHAP 誤導特徵診斷的訓練紀錄（ModelTrainingRun.diagnostics 非空），
    供首頁「SHAP 誤導特徵診斷」卡片顯示：
    - method：TreeSHAP（樹模型）或逐股置換重要性（TFT）
    - importance：SHAP 重要性清單（feature／share 佔比%／mean_abs_shap 或 TFT 的 auc_drop）
    - findings：三類誤導證據（leak 獨大／direction 方向悖離／jagged 依賴曲線鋸齒）
    - correction：修正建議（剔除特徵／單調約束／人工步驟）
    - comparison：重訓後新舊 AUC 對比（情況 A／B／不顯著）
    完全沒有診斷紀錄時 success=True 且 data=None（前端顯示操作引導）。
    """
    try:
        run = (ModelTrainingRun.objects
               .exclude(diagnostics={})
               .order_by("-created_at")
               .first())
        if run is None:
            return JsonResponse({
                "success": True, "data": None,
                "message": ("尚無 SHAP 診斷紀錄：SHAP 診斷已納入訓練流程（訓練完成後自動執行，"
                            "超參數 auto_shap=false 可關閉）。此訊息代表尚未有新訓練，"
                            "或歷史舊紀錄尚未補診斷——可手動執行 "
                            "python manage.py diagnose_model <股票代碼> "
                            "--model-type lightgbm|xgboost|tft 補做，完成後報告會顯示在這裡。"),
            })
        diag = run.diagnostics or {}
        d = diag.get("diagnosis") or {}
        importance = []
        for row in (d.get("importance") or [])[:20]:
            value = row.get("mean_abs_shap", row.get("auc_drop"))
            importance.append({
                "feature": row.get("feature"),
                "share": round(float(row.get("share") or 0) * 100, 2),
                "value": (round(float(value), 6) if value is not None else None),
                "value_label": ("mean(|SHAP|)（平均絕對 SHAP 值）"
                                if "mean_abs_shap" in row else "AUC 降幅（置換後）"),
            })
        type_labels = {"leak": "獨大/作弊特徵", "direction": "方向悖離", "jagged": "依賴曲線鋸齒"}
        findings = [{
            "type": f.get("type"),
            "type_label": type_labels.get(f.get("type"), f.get("type")),
            "feature": f.get("feature"),
            "detail": f.get("detail"),
        } for f in (d.get("findings") or [])]
        correction = diag.get("correction") or {}
        comparison = diag.get("comparison") or {}
        return JsonResponse({"success": True, "data": {
            "run_id": run.id,
            "model_version": run.model_version,
            "model_type": run.model_type,
            "train_window": diag.get("train_window", ""),
            "val_window": diag.get("val_window", ""),
            "diagnosed_at": _tw_iso_str(diag.get("diagnosed_at")),
            "trigger": diag.get("trigger", ""),
            "method": d.get("method", ""),
            "n_features": d.get("n_features"),
            "n_samples": d.get("n_samples"),
            "importance": importance,
            "findings": findings,
            "correction": {
                "auto": correction.get("auto"),
                "exclude_features": correction.get("exclude_features") or [],
                "monotone_constraints": correction.get("monotone_constraints") or {},
                "manual_steps": correction.get("manual_steps") or [],
            },
            "comparison": {
                "status": comparison.get("status"),
                "old_auc": comparison.get("old_auc"),
                "new_auc": comparison.get("new_auc"),
                "delta_auc": comparison.get("delta_auc"),
                "case": comparison.get("case"),
                "case_readable": comparison.get("case_readable"),
            },
        }})
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({
            "success": False,
            "message": f"查詢 SHAP 診斷失敗：{type(exc).__name__} - {exc}",
            "data": None,
        }, status=500)


def _tw_iso_str(iso_str):
    """把診斷報告記錄的 UTC ISO 字串轉台北時間字串（解析失敗原樣回傳）。"""
    if not iso_str:
        return None
    try:
        dt = _dt.fromisoformat(iso_str)
    except (TypeError, ValueError):
        return iso_str
    return timezone.localtime(dt).isoformat() if timezone.is_aware(dt) else dt.isoformat()