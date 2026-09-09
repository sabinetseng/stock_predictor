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

from django.core.management.base import BaseCommand, CommandError

from apps.stocks.services import launch_walkforward_retrain, launch_full_pipeline_training


class Command(BaseCommand):
    help = "背景執行一鍵訓練管線（同步資料→建立特徵→建立標籤→訓練）"

    def add_arguments(self, parser):
        parser.add_argument("--mode", default="walkforward", choices=["walkforward", "custom"],
                            help="walkforward＝最新資料 60/20/20；custom＝自選四個日期")
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
                **common,
            )

        self.stdout.write(self.style.SUCCESS(result.get("message", "")))
        if result.get("data"):
            self.stdout.write(str(result["data"])[:5000])
