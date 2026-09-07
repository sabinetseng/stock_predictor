# -*- coding: utf-8 -*-
"""
滾動式重訓練管理指令（walk-forward retraining）。

用途：解決「時間推移、資料持續增加，但訓練區間停留在過去」的問題。
**完整管線式**：同步資料（行情/籌碼/月營收＋美金匯率＋TWII/SOX/VIX＋ETF淨值）
→ 建立特徵 → 建立標籤 → 以「目前資料庫的最新 60/20/20」自動切分並訓練，
可掛 Windows 排程任務 / cron 定期執行，讓模型持續學到最新資料。

範例：
    python manage.py walkforward_retrain --stock-codes 2330 --dry-run   # 只預覽步驟與切割日期
    python manage.py walkforward_retrain --stock-codes 2330,0050        # 完整管線（同步→特徵→標籤→訓練）
    python manage.py walkforward_retrain                                # 全部股票

標籤版本預設為進階雙關卡（超額報酬+停損），如需改用 MVP 單門檻請加 --label-version mvp
"""
from django.core.management.base import BaseCommand, CommandError

from apps.stocks.services import launch_walkforward_retrain


class Command(BaseCommand):
    help = "以目前最新資料的 60/20/20 自動切分並重訓練（可排程，滾動更新模型）"

    def add_arguments(self, parser):
        parser.add_argument("--stock-codes", default="",
                            help="逗號分隔的股票代碼，留空代表全部股票")
        parser.add_argument("--model-type", default="lightgbm",
                            choices=["lightgbm", "xgboost"])
        parser.add_argument("--label-version", default="advanced",
                            choices=["mvp", "advanced"],
                            help="標籤版本：advanced（預設，進階雙關卡）或 mvp")
        parser.add_argument("--n-repeats", type=int, default=1)
        parser.add_argument("--lookback-years", type=float, default=None,
                            help="滾動視窗年數（例 3＝只用近 3 年資料切分）；不填＝全部歷史")
        parser.add_argument("--include-latest", action="store_true",
                            help="驗證區間延伸至最新日，最終模型以全部資料重新擬合（學到最新市場）")
        parser.add_argument("--dry-run", action="store_true",
                            help="只計算並顯示切割日期，不建立紀錄、不訓練")

    def handle(self, *args, **options):
        try:
            result = launch_walkforward_retrain(
                stock_codes_text=options["stock_codes"],
                model_type=options["model_type"],
                label_version=options["label_version"],
                n_repeats=options["n_repeats"],
                lookback_years=options["lookback_years"],
                include_latest=options["include_latest"],
                dry_run=options["dry_run"],
            )
        except ValueError as exc:
            raise CommandError(str(exc))
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"重訓練失敗：{type(exc).__name__} - {exc}")

        self.stdout.write(self.style.SUCCESS(result["message"]))
        if result.get("data"):
            self.stdout.write(str(result["data"]))
