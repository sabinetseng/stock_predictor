# -*- coding: utf-8 -*-
"""
匯入 ETF 歷史淨值 CSV（折溢價歷史回補的正式管道）。

執行方式：
    python manage.py import_etf_nav 0050 --csv nav_0050.csv
    python manage.py import_etf_nav 0056 --csv nav_0056.csv --source yuanta

CSV 格式（有表頭、逗號分隔，欄位名支援中英文）：
    date,nav[,close]
    日期,淨值[,收盤價]

close 可省略；缺漏時自動用資料庫的還原收盤價補算折溢價率。
淨值來源建議：投信官網（元大/國泰/富邦...）或證交所「投信基金 - ETF 淨值」
查詢頁下載的歷史資料，另存為 UTF-8 / Excel CSV 即可。

匯入後需重新執行「建立特徵」（build_features）才會產生 etf_premium_* 特徵。
"""
from django.core.management.base import BaseCommand, CommandError

from apps.stocks.services import import_etf_nav_csv


class Command(BaseCommand):
    help = "從 CSV 匯入 ETF 歷史淨值與折溢價（EtfNavData）"

    def add_arguments(self, parser):
        parser.add_argument("stock_codes", nargs="+", type=str, help="ETF 股票代碼，可帶多個")
        parser.add_argument("--csv", required=True, type=str, help="淨值 CSV 檔案路徑")
        parser.add_argument(
            "--source", type=str, default="csv_manual",
            help="來源標記（寫入 EtfNavData.source_name），預設 csv_manual",
        )

    def handle(self, *args, **options):
        for stock_code in options["stock_codes"]:
            self.stdout.write(f"開始匯入 {stock_code} 的淨值...")
            result = import_etf_nav_csv(stock_code, options["csv"], options["source"])
            style = self.style.SUCCESS if result["success"] else self.style.ERROR
            self.stdout.write(style(f"  {result['message']}"))
            if not result["success"]:
                raise CommandError(result["message"])