"""
執行方式：python manage.py sync_market_data

會依序同步 ^TWII（大盤）、^SOX（費半）、^VIX（波動度）三個指數，
以及台股交易日行事曆（來源：FinMind）。
"""
import datetime

from django.core.management.base import BaseCommand
from django.conf import settings
from apps.market_data import services


class Command(BaseCommand):
    help = "抓取 ^TWII / ^SOX / ^VIX 指數資料與交易日行事曆，並同步寫入資料庫"

    def handle(self, *args, **options):
        self.stdout.write("開始同步跨市場指數資料...")
        results = services.sync_all_indexes()

        for index_code, result in results.items():
            if result["success"]:
                self.stdout.write(self.style.SUCCESS(f"[{index_code}] {result['message']}"))
            else:
                self.stdout.write(self.style.ERROR(f"[{index_code}] {result['message']}"))

        self.stdout.write("開始同步交易日行事曆...")
        calendar_result = services.sync_trading_calendar(
            settings.TRAIN_DATE_START, datetime.date.today()
        )
        if calendar_result["success"]:
            self.stdout.write(self.style.SUCCESS(f"[交易日行事曆] {calendar_result['message']}"))
        else:
            self.stdout.write(self.style.ERROR(f"[交易日行事曆] {calendar_result['message']}"))
