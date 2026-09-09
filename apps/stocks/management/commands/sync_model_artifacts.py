# -*- coding: utf-8 -*-
"""把磁碟上存在、但尚未入庫的已完成訓練模型檔回補進資料庫（ModelArtifact）。

用法：
    python manage.py sync_model_artifacts            # 回補全部
    python manage.py sync_model_artifacts --dry-run  # 只統計，不寫入

背景：Render 免費方案磁碟暫態，模型檔會在重新部署／重啟時消失；
訓練完成時已自動入庫（model_training.py），本指令用於回補「入庫機制上線前」
或「入庫當下失敗」的舊訓練。
"""
from django.core.management.base import BaseCommand

from apps.stocks.model_artifacts import sync_artifacts_from_disk


class Command(BaseCommand):
    help = "把磁碟上的模型檔回補進資料庫（ModelArtifact 入庫備份）"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="只統計，不寫入")

    def handle(self, *args, **options):
        if options["dry_run"]:
            from apps.stocks.models import ModelArtifact, ModelTrainingRun

            runs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
            to_backfill = sum(
                1 for r in runs
                if not ModelArtifact.objects.filter(run=r).exists()
                and __import__("os").path.exists(r.model_file_path)
            )
            self.stdout.write(f"[dry-run] 可回補 {to_backfill} 筆（已完成訓練共 {runs.count()} 筆）")
            return

        backfilled, skipped, missing = sync_artifacts_from_disk()
        self.stdout.write(self.style.SUCCESS(
            f"模型檔入庫回補完成：入庫 {backfilled} 筆｜已有備份跳過 {skipped} 筆｜"
            f"磁碟與資料庫皆無檔案 {missing} 筆"
        ))
