from django.contrib import admin
from . import models

# 先用最簡單的方式全部註冊，方便開發初期直接在 Admin 檢查資料是否正確入庫
admin.site.register(models.StockBasicInfo)
admin.site.register(models.StockPrice)
admin.site.register(models.StockChipData)
admin.site.register(models.StockMonthlyRevenue)
admin.site.register(models.StockFeatureDaily)
admin.site.register(models.StockFeatureMonthly)
admin.site.register(models.StockLabel)
admin.site.register(models.StockPrediction)


@admin.register(models.ModelTrainingRun)
class ModelTrainingRunAdmin(admin.ModelAdmin):
    # 方便在 Admin 直接比較不同訓練輪次的驗證表現，檢查是否隨時間衰退
    list_display = (
        "model_version", "train_start", "train_end",
        "validation_start", "validation_end",
        "status", "auc", "pr_auc", "months_from_train_end",
    )
    list_filter = ("status", "model_type")
    ordering = ("-created_at",)
