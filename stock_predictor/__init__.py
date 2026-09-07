# 讓 `celery -A stock_predictor worker` 這個指令可以找到 Celery app。
# 用 try/except 包起來是因為：沒裝 celery 套件、或還沒設定 Redis 時，
# Django 本身（跑網站、跑 manage.py 其他指令）完全不受影響，只有真的要用 Celery 時才需要它。
try:
    from .celery import app as celery_app
    __all__ = ("celery_app",)
except ImportError:
    pass
