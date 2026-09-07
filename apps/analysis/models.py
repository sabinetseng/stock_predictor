from django.db import models
from apps.stocks.models import StockBasicInfo


class CorrelationResult(models.Model):
    """
    股票與美金匯率 / 大盤的關聯分析結果，快取起來避免每次都重算。
    """

    METHOD_CHOICES = [
        ("pearson", "Pearson"),
        ("spearman", "Spearman"),
        ("lagged", "Lagged Correlation"),
        ("rolling", "Rolling Correlation"),
    ]

    stock = models.ForeignKey(StockBasicInfo, on_delete=models.CASCADE, related_name="correlation_results")
    compare_target = models.CharField(max_length=30, verbose_name="比較對象，例如 USD_TWD / TWII / SOX / VIX")
    method = models.CharField(max_length=20, choices=METHOD_CHOICES)
    lag_days = models.IntegerField(null=True, blank=True, verbose_name="lagged/rolling 分析用的天數")
    result_value = models.FloatField()
    calculated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "關聯分析結果"
        unique_together = ("stock", "compare_target", "method", "lag_days")
