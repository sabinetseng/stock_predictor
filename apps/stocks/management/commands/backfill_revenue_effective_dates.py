"""
backfill_revenue_effective_dates — 把歷史月營收的 announcement_date 回填為「保守生效日」。

背景（look-ahead bias 防護）：
    FinMind 的月營收 date 欄位實測為「次月 1 日」（月級粒度），不是每家公司真實公告日。
    依 MOPS 營業額資訊申報規定，上市/上櫃公司應於「每月 10 日前」公告上月營收，
    且 MOPS 各介面均不提供歷史真實公告日（實測 t05st10_ifrs / t21sc03 / OpenAPI t187ap05_L）。
    因此以「次月 10 日」為特徵生效日，保證模型任何一天拿到的營收都是真實世界已公告的。

這個指令會：
    1. 把 StockMonthlyRevenue.announcement_date 更新為 effective_announcement_date() 的結果
    2. 刪除該股票的 StockFeatureMonthly 全部列，用新公告日重建（避免殘留舊生效日的幽靈列）
    3. --verify-mops N：抓最近 N 個月 MOPS 官方彙總檔（t21sc03，上市股），
       交叉驗證 FinMind 營收數值；上櫃股 MOPS 無同格式檔，會標記「無法驗證」

用法：
    python manage.py backfill_revenue_effective_dates --dry-run
    python manage.py backfill_revenue_effective_dates
    python manage.py backfill_revenue_effective_dates --verify-mops 6
"""
import datetime

from django.core.management.base import BaseCommand

from apps.stocks.models import StockBasicInfo, StockMonthlyRevenue, StockFeatureMonthly
from apps.stocks.services import (
    effective_announcement_date,
    build_monthly_features,
    fetch_mops_month_revenue_summary,
)


class Command(BaseCommand):
    help = "把歷史月營收 announcement_date 回填為保守生效日（次月10日），並重建月頻特徵表。"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run", action="store_true",
            help="只顯示將異動的內容，不寫入資料庫",
        )
        parser.add_argument(
            "--stocks", default="",
            help="只處理指定股票代碼（逗號分隔）；留空=處理全部有營收的股票",
        )
        parser.add_argument(
            "--verify-mops", type=int, default=0, metavar="N",
            help="用 MOPS 官方彙總檔交叉驗證最近 N 個月的營收數值（僅上市股）",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        stock_filter = [s.strip() for s in options["stocks"].split(",") if s.strip()]

        rev_qs = StockMonthlyRevenue.objects.select_related("stock").order_by("stock__stock_code", "revenue_month")
        if stock_filter:
            rev_qs = rev_qs.filter(stock__stock_code__in=stock_filter)

        # ---- 1) 計算每列的新生效日 ----
        changed_by_stock = {}
        unchanged = 0
        rows_to_update = []
        for r in rev_qs:
            new_date = effective_announcement_date(r.revenue_month, r.announcement_date)
            if new_date != r.announcement_date:
                rows_to_update.append((r, new_date))
                changed_by_stock.setdefault(r.stock.stock_code, []).append((r, new_date))
            else:
                unchanged += 1

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"共 {len(rows_to_update)} 筆需更新、{unchanged} 筆已是生效日"
        ))
        for code, pairs in sorted(changed_by_stock.items()):
            first, last = pairs[0], pairs[-1]
            self.stdout.write(
                f"  {code}: {len(pairs)} 筆，{first[0].revenue_month} "
                f"{first[0].announcement_date} -> {first[1]} ... "
                f"{last[0].revenue_month} {last[0].announcement_date} -> {last[1]}"
            )

        # ---- 2) 寫入（dry-run 則跳過）----
        if dry_run:
            self.stdout.write(self.style.WARNING("dry-run：未寫入任何資料"))
        else:
            for r, new_date in rows_to_update:
                r.announcement_date = new_date
                r.save(update_fields=["announcement_date"])

            # 重建特徵表：update_or_create 只會 upsert「新生效日」的列，
            # 舊生效日的幽靈列必須刪除，否則舊的次月1日生效列殘留 → 偷看照舊
            stocks = StockBasicInfo.objects.filter(
                stock_code__in=changed_by_stock.keys())
            for stock in stocks:
                deleted, _ = StockFeatureMonthly.objects.filter(stock=stock).delete()
                result = build_monthly_features(stock.stock_code)
                self.stdout.write(self.style.SUCCESS(
                    f"  {stock.stock_code}: 刪除舊特徵 {deleted} 列，重建 {result.get('rows_written', 0)} 列"
                ))
            self.stdout.write(self.style.SUCCESS("回填完成"))

        # ---- 3) MOPS 官方數值交叉驗證 ----
        n_verify = options["verify_mops"]
        if n_verify > 0:
            self._verify_with_mops(n_verify, stock_filter, dry_run)

    def _verify_with_mops(self, n_months, stock_filter, dry_run):
        # 取 DB 中最新的 N 個營收月份
        months = sorted({
            r["revenue_month"] for r in StockMonthlyRevenue.objects.values("revenue_month")
        }, reverse=True)[:n_months]
        self.stdout.write(self.style.MIGRATE_HEADING(f"MOPS 交叉驗證最近 {len(months)} 個月"))
        for m in months:
            roc_year = m.year - 1911
            summary = fetch_mops_month_revenue_summary(roc_year, m.month, market="sii")
            if summary is None:
                self.stdout.write(self.style.WARNING(
                    f"  {m}: MOPS 彙總檔無法取得（網路失敗或非上市市場合併區間），跳過"
                ))
                continue
            rows = StockMonthlyRevenue.objects.filter(revenue_month=m)
            if stock_filter:
                rows = rows.filter(stock__stock_code__in=stock_filter)
            for r in rows:
                mops_val = summary.get(r.stock.stock_code)
                if mops_val is None:
                    # 上櫃（tii）不在 mopsov 的 sii 彙總檔內
                    self.stdout.write(self.style.WARNING(
                        f"  {m} {r.stock.stock_code}: MOPS 上市彙總檔無此代號（可能為上櫃），無法驗證"
                    ))
                    continue
                if r.revenue_amount is None:
                    self.stdout.write(self.style.WARNING(
                        f"  {m} {r.stock.stock_code}: DB revenue_amount 為空（尚未公布的佔位列），無法驗證"
                    ))
                    continue
                db_thousand = round(float(r.revenue_amount) / 1000)
                ok = (mops_val == db_thousand)
                self.stdout.write(
                    f"  {m} {r.stock.stock_code}: DB={db_thousand:,} 仟元 vs MOPS={mops_val:,} 仟元 "
                    f"{'OK' if ok else 'XX 不一致'}"
                )
