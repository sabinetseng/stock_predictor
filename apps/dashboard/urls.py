from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    # 首頁：股票代碼輸入、預測按鈕、檢查新資料按鈕都放在這頁
    path("", views.index, name="index"),
]
