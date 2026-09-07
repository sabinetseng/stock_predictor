"""
執行方式：
    python manage.py build_labels 2330                  # 進階雙關卡版本（預設）
    python manage.py build_labels 2330 --mvp           # MVP 單門檻版本

會計算 Return20 並產生標籤，寫入 StockLabel。
執行前提：該股票的還原股價資料要先用 sync_stock_data 抓過。
"""
from django.core.management.base import BaseCommand
from apps.stocks import services


class Command(BaseCommand):
    help = "計算指定股票代碼的 Return20 標籤（預設進階雙關卡版本）"

    def add_arguments(self, parser):
        parser.add_argument("stock_codes", nargs="+", type=str, help="股票代碼，可帶多個")
        parser.add_argument(
            "--mvp", action="store_true",
            help="使用 MVP 單門檻版本（預設為進階雙關卡）",
        )

    def handle(self, *args, **options):
        for stock_code in options["stock_codes"]:
            self.stdout.write(f"開始計算 {stock_code} 的標籤...")
            result = services.build_labels(stock_code, use_advanced_gate=not options["mvp"])
            style = self.style.SUCCESS if result["success"] else self.style.ERROR
            self.stdout.write(style(f"  {result['message']}"))
