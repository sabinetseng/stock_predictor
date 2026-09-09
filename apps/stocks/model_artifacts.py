# -*- coding: utf-8 -*-
"""模型檔案入庫（ModelArtifact）：讓訓練好的模型在重新部署／重啟後仍可使用。

Render 免費方案的磁碟是暫態的：trained_models/*.joblib 會在每次部署／重啟時
消失，導致「已完成」的訓練無法載入（預測／診斷／回測全部失效，只能重新訓練）。
解法：訓練完成時把模型檔位元組存進資料庫（PostgreSQL bytea），載入時若磁碟
檔案遺失，自動從資料庫還原到暫存快取目錄再交給 joblib.load。
"""
import os
import re
import tempfile
from pathlib import Path

from .models import ModelArtifact


def store_artifact(run, model_file_path):
    """把磁碟上的模型檔存進資料庫（訓練完成時呼叫；同 run 重複呼叫會更新內容）。"""
    with open(model_file_path, "rb") as f:
        content = f.read()
    artifact, _created = ModelArtifact.objects.update_or_create(
        run=run,
        defaults={
            "file_name": os.path.basename(model_file_path),
            "content": content,
            "size_bytes": len(content),
        },
    )
    return artifact


def model_file_available(run) -> bool:
    """磁碟上有模型檔，或資料庫有入庫備份，都算可用。"""
    if run.model_file_path and os.path.exists(run.model_file_path):
        return True
    return ModelArtifact.objects.filter(run=run).exists()


def get_model_file(run) -> str:
    """回傳「一定存在」的模型檔路徑：磁碟優先，遺失時從資料庫還原到暫存快取。

    還原目標放在系統暫存目錄（Render 重啟後本來就會重建，不需清理），
    以 run.id + model_version 命名避免不同訓練互相覆蓋。
    """
    if run.model_file_path and os.path.exists(run.model_file_path):
        return run.model_file_path

    artifact = ModelArtifact.objects.filter(run=run).first()
    if artifact is None:
        raise FileNotFoundError(
            f"模型檔案遺失且資料庫無備份（訓練紀錄 #{run.id}，{run.model_version}），請重新訓練"
        )

    cache_dir = Path(tempfile.gettempdir()) / "stock_model_artifacts"
    cache_dir.mkdir(exist_ok=True)
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.model_version)[:80]
    target = cache_dir / f"run_{run.id}_{safe_version}.joblib"
    if not target.exists() or target.stat().st_size != len(artifact.content):
        target.write_bytes(bytes(artifact.content))
    return str(target)


def sync_artifacts_from_disk():
    """把磁碟上存在、但尚未入庫的已完成訓練模型檔補進資料庫（回補既有訓練用）。

    回傳 (backfilled, skipped, missing)：
    - backfilled：這次成功入庫的筆數
    - skipped：資料庫已有備份、不需處理
    - missing：磁碟與資料庫都沒有模型檔（無法回補）
    """
    from .models import ModelTrainingRun

    backfilled = skipped = missing = 0
    runs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
    for run in runs:
        if ModelArtifact.objects.filter(run=run).exists():
            skipped += 1
            continue
        if run.model_file_path and os.path.exists(run.model_file_path):
            store_artifact(run, run.model_file_path)
            backfilled += 1
        else:
            missing += 1
    return backfilled, skipped, missing
