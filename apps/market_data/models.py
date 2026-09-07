from django.db import models


class MarketIndexRate(models.Model):
    """
    跨市場指數資料：^TWII（大盤，計算個股相對強弱勢用）、
    ^SOX（費半）、^VIX（波動度/情緒指標）。
    來源 Yahoo Finance / FinMind，皆不需要 API Token。
    """

    INDEX_CHOICES = [
        ("TWII", "加權指數 ^TWII"),
        ("SOX", "費城半導體指數 ^SOX"),
        ("VIX", "波動度指數 ^VIX"),
    ]

    index_code = models.CharField(max_length=10, choices=INDEX_CHOICES)
    trade_date = models.DateField()
    close_value = models.DecimalField(max_digits=14, decimal_places=4)
    source_name = models.CharField(max_length=30)
    fetched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "跨市場指數資料"
        unique_together = ("index_code", "trade_date")


class TradingCalendar(models.Model):
    """
    台股交易日行事曆。
    用於精確定義「20 個交易日後」，不是日曆天 20 天。
    """

    calendar_date = models.DateField(unique=True)
    is_trading_day = models.BooleanField(default=True)

    class Meta:
        verbose_name = "台股交易日行事曆"
        ordering = ["calendar_date"]
