from django.db import models


class UsdTwdRate(models.Model):
    """
    美元兌台幣匯率，來源 Yahoo Finance（TWD=X），不需要 API Token。
    注意：匯率每天都有報價（含台股休市日），與股票資料合併時要用交易日曆對齊。
    """

    rate_date = models.DateField(unique=True, verbose_name="日期")
    close_rate = models.DecimalField(max_digits=10, decimal_places=4, verbose_name="收盤匯率")
    source_name = models.CharField(max_length=30, default="Yahoo Finance")
    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "美元兌台幣匯率"
        ordering = ["-rate_date"]
