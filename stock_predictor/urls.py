"""
專案主路由設定。
各 App 的路由分別寫在自己的 urls.py，這裡只負責 include。
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),

    # 首頁 / 整體視圖
    path("", include("apps.dashboard.urls")),

    # 股票資料、特徵、標籤、預測
    path("stocks/", include("apps.stocks.urls")),

    # 美金匯率資料、更新、關聯分析
    path("fx/", include("apps.fx_data.urls")),

    # 大盤與跨市場資料（^TWII、^SOX、^VIX）
    path("market/", include("apps.market_data.urls")),

    # 相關性分析、回測、報表
    path("analysis/", include("apps.analysis.urls")),

    # 共用資料同步 API（股票 / 美金「檢查是否有新資料」按鈕共用此路由前綴）
    path("sync/", include("apps.sync.urls")),
]
