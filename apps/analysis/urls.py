from django.urls import path
from . import views

app_name = "analysis"

urlpatterns = [
    # 相關性分析前端頁面
    path("", views.correlation_page, name="correlation_page"),
    # 股票與美金 / 大盤的關聯分析圖表資料（JSON API）
    path("correlation/<str:stock_code>/", views.correlation, name="correlation"),
]
