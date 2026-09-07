from django.urls import path
from . import views

app_name = "sync"

urlpatterns = [
    # 「檢查是否有新資料」按鈕（股票）
    path("stock/<str:stock_code>/check/", views.check_stock_data, name="check_stock_data"),
    # 「檢查美金是否有新資料」按鈕
    path("fx/check/", views.check_fx_data, name="check_fx_data"),
    # 同步紀錄查詢（可選，供前端顯示最近同步狀態）
    path("logs/", views.recent_logs, name="recent_logs"),
]
