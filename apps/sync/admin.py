from django.contrib import admin
from .models import DataSyncLog


@admin.register(DataSyncLog)
class DataSyncLogAdmin(admin.ModelAdmin):
    # 讓開發時可以直接在 Admin 列表看到最重要的欄位，方便監控同步狀況
    list_display = ("triggered_at", "source_type", "target_code", "status", "source_name")
    list_filter = ("source_type", "status")
    search_fields = ("target_code", "message")
