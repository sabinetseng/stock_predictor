"""
執行方式：
    python manage.py train_model --train-start 2018-01-01 --train-end 2024-12-31 \
        --val-start 2025-01-01 --val-end 2025-12-31

可選參數：
        --stocks 2330,2317          只用指定股票訓練，留空代表用資料庫內全部股票
    --mvp                       使用 MVP 單門檻標籤版本，預設為進階雙關卡版本
    --model-version v1_test     自訂模型版本名稱，留空會自動產生
    --model-type lightgbm       lightgbm（預設）、xgboost 或 tft（Temporal Fusion Transformer，需 PyTorch）
    --hyperparams '{"n_estimators": 300, "learning_rate": 0.03}'
                                 JSON 字串，只需要傳想覆蓋的欄位，沒傳的用系統預設值

執行前提：訓練與驗證區間都要先跑過 build_features 與 build_labels。
"""
import json

from django.core.management.base import BaseCommand
from apps.stocks.models import ModelTrainingRun
from apps.stocks import model_training


class Command(BaseCommand):
    help = "建立一筆 ModelTrainingRun 並執行模型訓練（LightGBM 或 XGBoost）"

    def add_arguments(self, parser):
        parser.add_argument("--train-start", required=True, type=str)
        parser.add_argument("--train-end", required=True, type=str)
        parser.add_argument("--val-start", required=True, type=str)
        parser.add_argument("--val-end", required=True, type=str)
        parser.add_argument("--stocks", type=str, default="", help="逗號分隔的股票代碼，留空代表全部")
        parser.add_argument("--mvp", action="store_true", help="使用 MVP 單門檻標籤版本（預設為進階雙關卡）")
        parser.add_argument("--model-version", type=str, default="")
        parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "tft"])
        parser.add_argument("--hyperparams", type=str, default="{}", help="JSON 字串，例如 '{\"n_estimators\": 300}'")
        parser.add_argument(
            "--n-repeats", type=int, default=1,
            help="重複訓練次數（不同亂數種子），統計上建議 >1 才能看出結果是否穩定，範圍1~10",
        )

    def handle(self, *args, **options):
        train_start = options["train_start"]
        train_end = options["train_end"]
        val_start = options["val_start"]
        val_end = options["val_end"]
        label_version = "mvp" if options["mvp"] else "advanced"
        model_type = options["model_type"]
        model_version = options["model_version"] or f"v_{model_type}_{train_start}_{train_end}_val_{val_start}_{val_end}"

        try:
            hyperparams = json.loads(options["hyperparams"])
        except json.JSONDecodeError:
            self.stdout.write(self.style.ERROR(f"--hyperparams 不是合法的 JSON 字串：{options['hyperparams']}"))
            return

        n_repeats = options["n_repeats"]
        if not (1 <= n_repeats <= 10):
            self.stdout.write(self.style.ERROR(f"--n-repeats={n_repeats} 超出合理範圍（1~10）"))
            return

        if train_end >= val_start:
            self.stdout.write(self.style.ERROR("訓練結束日必須早於驗證起始日，避免資料洩漏"))
            return

        run = ModelTrainingRun.objects.create(
            train_start=train_start, train_end=train_end,
            validation_start=val_start, validation_end=val_end,
            model_type=model_type, model_version=model_version,
            stock_codes=options["stocks"], label_version=label_version,
            hyperparams=hyperparams, n_repeats=n_repeats, status="pending",
        )

        self.stdout.write(f"開始訓練 {model_version}（模型：{model_type}，重複{n_repeats}次，標籤版本：{label_version}）...")
        result = model_training.run_training(run.id)

        if result["success"]:
            self.stdout.write(self.style.SUCCESS(result["message"]))
            self.stdout.write(f"實際使用的超參數: {result['data']['hyperparams']}")
            if result["data"].get("auc_std") is not None:
                self.stdout.write(
                    f"AUC 標準差: {result['data']['auc_std']:.4f}（重複{result['data']['n_repeats']}次的變異程度，"
                    f"越小代表結果越穩定可信）"
                )
        else:
            self.stdout.write(self.style.ERROR(result["message"]))
