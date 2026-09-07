"""
Celery 設定。

這個骨架寫好之後，訓練不用再卡在 API 請求裡等結果，而是丟給背景 worker 處理。
但要真的跑起來需要額外啟動兩個東西（這個容器環境沒有 Redis，沒辦法在這裡實際測試）：

1. Redis（Celery 預設用的訊息佇列，broker）
   Mac:     brew install redis && brew services start redis
   Windows: 到 https://github.com/microsoftarchive/redis/releases 下載安裝
   或直接用 Docker：docker run -d -p 6379:6379 redis

2. Celery worker（實際執行訓練任務的背景程式）
   在專案根目錄、虛擬環境啟用的狀態下另開一個終端機視窗執行：
   celery -A stock_predictor worker --loglevel=info

有了這兩個東西之後，apps/stocks/tasks.py 裡的 train_model_task 就能真的非同步執行。
在那之前，系統預設還是用 create_training_run view 裡「同步」執行的版本，不影響現有功能。
"""
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_predictor.settings")

app = Celery("stock_predictor")

# 從 Django settings.py 讀取以 CELERY_ 開頭的設定（見 settings.py 底部）
app.config_from_object("django.conf:settings", namespace="CELERY")

# 自動掃描每個 app 的 tasks.py
app.autodiscover_tasks()
