from django.db import models


class DataSyncLog(models.Model):
    """
    所有資料同步操作的紀錄（股票行情、籌碼、營收、美金匯率、跨市場指數）。
    每次「檢查是否有新資料」/「檢查美金是否有新資料」按鈕觸發的結果都寫在這裡，
    直接掛在 Django Admin 上即可監控，不需另外開發畫面。
    """

    SOURCE_CHOICES = [
        ("stock_price", "股票行情"),
        ("stock_chip", "股票籌碼面"),
        ("stock_revenue", "股票月營收"),
        ("usd_twd", "美金匯率"),
        ("market_index", "跨市場指數"),
        ("trading_calendar", "交易日行事曆"),
    ]

    STATUS_CHOICES = [
        ("success_new_data", "成功，有寫入新資料"),
        ("success_no_new_data", "成功，已是最新狀態"),
        ("failed", "失敗"),
    ]

    source_type = models.CharField(max_length=30, choices=SOURCE_CHOICES)
    target_code = models.CharField(
        max_length=20, blank=True, verbose_name="目標代碼，例如股票代碼；美金/大盤資料可留空"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    source_name = models.CharField(max_length=30, verbose_name="資料來源，例如 FinMind / yfinance / Yahoo Finance")
    message = models.TextField(blank=True, verbose_name="說明或錯誤訊息")
    triggered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "資料同步紀錄"
        ordering = ["-triggered_at"]

    def __str__(self):
        return f"[{self.triggered_at}] {self.source_type} - {self.status}"
