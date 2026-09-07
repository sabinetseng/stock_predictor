from django.urls import path
from . import views

app_name = "fx_data"

urlpatterns = [
    # 「檢查美金是否有新資料」按鈕對應的 API
    path("check-new-data/", views.check_new_fx_data, name="check_new_data"),
    # 美金走勢圖表資料
    path("history/", views.fx_history, name="history"),
]
