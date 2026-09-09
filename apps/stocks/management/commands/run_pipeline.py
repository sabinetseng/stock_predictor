# -*- coding: utf-8 -*-
"""
背景執行一鍵訓練管線（供網頁按鈕以分離子行程啟動）。

背景：TFT／完整管線（同步→特徵→標籤→訓練）遠超過 HTTP 請求合理時長，
在 Render 免費方案上同步執行會讓 gunicorn worker 逾時被殺、前端收到空回應。
改由網頁層以 subprocess 啟動本指令，HTTP 請求立即返回，訓練照常寫入
ModelTrainingRun（完成後重新整理頁面即可查看）。

範例：
    python manage.py run_pipeline --mode walkforward --stock-codes 2330 --model-type tft
    python manage.py run_pipeline --mode custom --stock-codes 2330 \
        --train-start 2023-01-01 --train-end 2025-06-30 \
        --validation-start 2025-07-01 --validation-end 2026-06-30
"""
import json
import os
import time

from django.core.management.base import BaseCommand, CommandError

from apps.stocks.services import launch_walkforward_retrain, launch_full_pipeline_training


def _start_keep_alive():
    """Render 免費方案：15 分鐘沒有 HTTP 流量會休眠整個服務（背景訓練會被殺）。

    有 RENDER_EXTERNAL_URL 時，以 daemon 執行緒每 4 分鐘打一次輕量端點，
    讓平台持續偵測到流量，訓練期間服務不會被休眠。
    """
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not external_url:
        return None

    import threading
    import urllib.request

    def _ping_loop():
        while True:
            try:
                urllib.request.urlopen(external_url + "/stocks/model-info/", timeout=15)
            except Exception:  # noqa: BLE001
                pass  # ping 失敗不影響訓練
            time.sleep(240)

    thread = threading.Thread(target=_ping_loop, daemon=True)
    thread.start()
    return thread


class Command(BaseCommand):
    help = "背景執行一鍵訓練管線（同步資料→建立特徵→建立標籤→訓練）"

    def add_arguments(self, parser):
        parser.add_argument("--mode", default="walkforward", choices=["walkforward", "custom"],
                            help="walkforward＝最新資料 60/20/20；custom＝自選四個日期")
        parser.add_argument("--run-id", type=int, default=None,
                            help="網頁按鈕預先建立的 pending ModelTrainingRun id（沿用該筆紀錄執行）")
        parser.add_argument("--stock-codes", default="",
                            help="逗號分隔的股票代碼，留空代表全部股票")
        parser.add_argument("--model-type", default="lightgbm",
                            choices=["lightgbm", "xgboost", "tft"])
        parser.add_argument("--label-version", default="advanced", choices=["mvp", "advanced"])
        parser.add_argument("--n-repeats", type=int, default=1)
        parser.add_argument("--lookback-years", type=float, default=None,
                            help="滾動視窗年數（僅 walkforward 模式）")
        parser.add_argument("--include-latest", action="store_true",
                            help="驗證區間延伸至最新交易日，最終模型以全部資料重新擬合")
        parser.add_argument("--hyperparams", default="",
                            help="超參數 JSON 字串（選填）")
        parser.add_argument("--train-start", default=None)
        parser.add_argument("--train-end", default=None)
        parser.add_argument("--validation-start", default=None)
        parser.add_argument("--validation-end", default=None)

    def handle(self, *args, **options):
        # Render 免費方案防休眠：訓練期間持續製造 HTTP 流量
        if _start_keep_alive() is not None:
            self.stdout.write("keep-alive：已啟動（每 4 分鐘 ping，避免服務被休眠）")

        hyperparams = {}
        if options["hyperparams"]:
            try:
                hyperparams = json.loads(options["hyperparams"])
            except json.JSONDecodeError as exc:
                raise CommandError(f"hyperparams 不是合法 JSON：{exc}")

        common = dict(
            stock_codes_text=options["stock_codes"],
            model_type=options["model_type"],
            label_version=options["label_version"],
            n_repeats=options["n_repeats"],
            hyperparams=hyperparams,
        )

        if options["mode"] == "walkforward":
            result = launch_walkforward_retrain(
                lookback_years=options["lookback_years"],
                include_latest=options["include_latest"],
                dry_run=False,
                existing_run_id=options["run_id"],
                **common,
            )
        else:
            date_keys = ("train_start", "train_end", "validation_start", "validation_end")
            if not all(options[k] for k in date_keys):
                raise CommandError("custom 模式需要提供四個日期：--train-start/--train-end/"
                                   "--validation-start/--validation-end")
            result = launch_full_pipeline_training(
                dates={k: options[k] for k in date_keys},
                include_latest=options["include_latest"],
                existing_run_id=options["run_id"],
                **common,
            )

        self.stdout.write(self.style.SUCCESS(result.get("message", "")))
        if result.get("data"):
            self.stdout.write(str(result["data"])[:5000])
