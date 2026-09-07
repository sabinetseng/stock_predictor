"""
執行方式：
    python manage.py walk_forward_train --start-year 2018 --end-year 2025 --initial-train-years 3

會自動跑多輪訓練，驗證窗格逐年往後滑動，對應 PROJECT_PLAN.md 第 10.6 節的 walk-forward 設計：

    第1輪：訓練 2018-2020 → 驗證 2021
    第2輪：訓練 2018-2021 → 驗證 2022
    第3輪：訓練 2018-2022 → 驗證 2023
    ...以此類推，直到驗證年份超過 --end-year

每一輪都會建立一筆 ModelTrainingRun 並真的執行訓練，跑完可以到
Django Admin 的 Stocks > Model training runs 依 months_from_train_end 排序，
看驗證表現（AUC/PR-AUC）是否隨著時間拉遠而衰退，藉此判斷模型能不能撐到未來 5-10 年。

可選參數：
    --stocks 2330,2317          只用指定股票，留空代表用資料庫內全部股票
        --mvp                      使用 MVP 單門檻標籤版本，預設為進階雙關卡版本
    --initial-train-years 3     第一輪訓練用幾年的資料，預設3年

執行前提：每一年的資料都要先跑過 build_features 與 build_labels。
"""
import datetime

from django.core.management.base import BaseCommand
from apps.stocks.models import ModelTrainingRun
from apps.stocks import model_training


class Command(BaseCommand):
    help = "自動執行 walk-forward 滾動式回測，驗證模型是否能套用到未來更長時間"

    def add_arguments(self, parser):
        parser.add_argument("--start-year", required=True, type=int, help="整體資料的起始年份")
        parser.add_argument("--end-year", required=True, type=int, help="整體資料的結束年份（驗證跑到這年為止）")
        parser.add_argument("--initial-train-years", type=int, default=3, help="第一輪訓練用幾年的資料")
        parser.add_argument("--stocks", type=str, default="", help="逗號分隔的股票代碼，留空代表全部")
        parser.add_argument("--mvp", action="store_true", help="使用 MVP 單門檻標籤版本（預設為進階雙關卡）")
        parser.add_argument(
            "--n-repeats", type=int, default=1,
            help="每一輪重複訓練幾次（不同亂數種子），統計上建議 >1 才能看出結果穩不穩定，範圍1~10",
        )

    def handle(self, *args, **options):
        start_year = options["start_year"]
        end_year = options["end_year"]
        initial_train_years = options["initial_train_years"]
        label_version = "mvp" if options["mvp"] else "advanced"
        stock_codes = options["stocks"]
        n_repeats = options["n_repeats"]

        if not (1 <= n_repeats <= 10):
            self.stdout.write(self.style.ERROR(f"--n-repeats={n_repeats} 超出合理範圍（1~10）"))
            return

        first_val_year = start_year + initial_train_years
        if first_val_year > end_year:
            self.stdout.write(self.style.ERROR(
                f"--initial-train-years 太大，{start_year}+{initial_train_years}={first_val_year} "
                f"已經超過 --end-year={end_year}，沒有任何一輪可以跑"
            ))
            return

        round_number = 1
        val_year = first_val_year
        results_summary = []

        while val_year <= end_year:
            train_start = datetime.date(start_year, 1, 1)
            train_end = datetime.date(val_year - 1, 12, 31)
            val_start = datetime.date(val_year, 1, 1)
            val_end = datetime.date(val_year, 12, 31)
            model_version = f"walkforward_r{round_number}_train_{start_year}_{val_year-1}_val_{val_year}"

            self.stdout.write(
                f"\n=== 第 {round_number} 輪 === "
                f"訓練 {train_start}~{train_end} → 驗證 {val_start}~{val_end}"
            )

            run = ModelTrainingRun.objects.create(
                train_start=train_start, train_end=train_end,
                validation_start=val_start, validation_end=val_end,
                model_type="lightgbm", model_version=model_version,
                stock_codes=stock_codes, label_version=label_version, n_repeats=n_repeats,
                status="pending",
                notes=f"walk-forward 第{round_number}輪，自動產生",
            )

            result = model_training.run_training(run.id)
            if result["success"]:
                self.stdout.write(self.style.SUCCESS(f"  {result['message']}"))
                results_summary.append({
                    "round": round_number, "val_year": val_year,
                    "auc": result["data"]["auc"], "pr_auc": result["data"]["pr_auc"],
                })
            else:
                self.stdout.write(self.style.ERROR(f"  {result['message']}"))
                results_summary.append({"round": round_number, "val_year": val_year, "auc": None, "pr_auc": None})

            round_number += 1
            val_year += 1

        # 跑完印一個總表，方便直接在終端機看趨勢，不用開 Admin
        self.stdout.write("\n=== Walk-forward 總結（驗證表現隨時間拉遠的趨勢）===")
        self.stdout.write(f"{'輪次':<6}{'驗證年份':<10}{'AUC':<10}{'PR-AUC':<10}")
        for r in results_summary:
            auc_str = f"{r['auc']:.4f}" if r["auc"] is not None else "-"
            pr_auc_str = f"{r['pr_auc']:.4f}" if r["pr_auc"] is not None else "-"
            self.stdout.write(f"{r['round']:<6}{r['val_year']:<10}{auc_str:<10}{pr_auc_str:<10}")

        self.stdout.write(
            "\n提示：若 AUC 隨驗證年份拉遠而持續下滑，代表模型有 concept drift，"
            "需要考慮定期重新訓練（見 PROJECT_PLAN.md 第10.6節）"
        )
