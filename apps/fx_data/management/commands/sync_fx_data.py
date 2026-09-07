"""
執行方式：python manage.py sync_fx_data

會呼叫 apps.fx_data.services.check_and_sync_fx_data()，
自動判斷資料庫最後日期、抓取新資料、寫入 DataSyncLog。
"""
from django.core.management.base import BaseCommand
from apps.fx_data import services


class Command(BaseCommand):
    help = "抓取美元兌台幣（TWD=X）匯率，並同步寫入資料庫"

    def handle(self, *args, **options):
        self.stdout.write("開始同步美金匯率資料...")
        result = services.check_and_sync_fx_data()

        if result["success"]:
            self.stdout.write(self.style.SUCCESS(result["message"]))
        else:
            self.stdout.write(self.style.ERROR(result["message"]))
