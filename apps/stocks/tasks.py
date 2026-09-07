"""
Celery 背景任務。

只有在 settings.TRAINING_ASYNC=True 且有 Redis + celery worker 在跑的情況下才會被用到，
預設（TRAINING_ASYNC=False）系統會直接同步呼叫 model_training.run_training，不會經過這裡。
"""
try:
    from celery import shared_task
except ImportError:
    shared_task = None


if shared_task:
    @shared_task
    def train_model_task(training_run_id):
        """
        背景執行模型訓練，對應 apps.stocks.model_training.run_training。
        呼叫方式（需要 Celery worker 正在運作）：
            train_model_task.delay(run.id)
        """
        from . import model_training
        return model_training.run_training(training_run_id)
