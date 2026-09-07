"""
執行方式：python manage.py sync_stock_data 2330

會同步指定股票代碼的行情、籌碼面、月營收三種資料。
可一次帶多個股票代碼：python manage.py sync_stock_data 2330 2317 0050
"""
from django.core.management.base import BaseCommand
from apps.stocks import services


class Command(BaseCommand):
    help = "抓取指定股票代碼的行情/籌碼/月營收資料，並同步寫入資料庫"

    def add_arguments(self, parser):
        parser.add_argument("stock_codes", nargs="+", type=str, help="股票代碼，可帶多個，用空格分開")

    def handle(self, *args, **options):
        for stock_code in options["stock_codes"]:
            self.stdout.write(f"開始同步 {stock_code} ...")
            result = services.check_and_sync_stock_data(stock_code)

            for source, detail in result["data"].items():
                style = self.style.SUCCESS if detail["success"] else self.style.ERROR
                self.stdout.write(style(f"  [{source}] {detail['message']}"))
