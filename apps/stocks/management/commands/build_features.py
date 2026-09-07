"""
執行方式：python manage.py build_features 2330

會計算指定股票代碼的日頻特徵（技術面/籌碼面/相對大盤/跨市場總經）與月頻特徵（營收），
寫入 StockFeatureDaily / StockFeatureMonthly。

執行前提：該股票的行情/籌碼/月營收資料要先用 sync_stock_data 抓過，
大盤/SOX/VIX/美金資料要先用 sync_market_data / sync_fx_data 抓過。
"""
from django.core.management.base import BaseCommand
from apps.stocks import services


class Command(BaseCommand):
    help = "計算指定股票代碼的日頻/月頻特徵"

    def add_arguments(self, parser):
        parser.add_argument("stock_codes", nargs="+", type=str, help="股票代碼，可帶多個")

    def handle(self, *args, **options):
        for stock_code in options["stock_codes"]:
            self.stdout.write(f"開始計算 {stock_code} 的特徵...")

            daily_result = services.build_daily_features(stock_code)
            style = self.style.SUCCESS if daily_result["success"] else self.style.ERROR
            self.stdout.write(style(f"  [日頻特徵] {daily_result['message']}"))

            monthly_result = services.build_monthly_features(stock_code)
            style = self.style.SUCCESS if monthly_result["success"] else self.style.ERROR
            self.stdout.write(style(f"  [月頻特徵] {monthly_result['message']}"))
