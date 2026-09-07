"""
backtest_strategy — 盲測區策略回測 CLI。

用法：
    python manage.py backtest_strategy 2330
    python manage.py backtest_strategy 2330 --model-type tft --threshold 0.55
    python manage.py backtest_strategy 2330 --use-validation      # 驗證區（非盲測，僅供參考）
    python manage.py backtest_strategy 2330 --json report.json    # 另存完整 JSON 報告
"""
import json

from django.core.management.base import BaseCommand, CommandError

from apps.stocks.backtest import (
    DEFAULT_HOLD_DAYS, DEFAULT_STOP_LOSS, DEFAULT_THRESHOLD, run_backtest,
)


class Command(BaseCommand):
    help = ("把盲測區的模型機率變成一筆筆交易：勝率／夏普／最大回撤／對比買進持有"
            "（含手續費 0.1425%×2 與證交稅 0.3%）")

    def add_arguments(self, parser):
        parser.add_argument("stock_code", type=str, help="股票代碼，例如 2330 或 0050")
        parser.add_argument("--model-type", default=None,
                            choices=["lightgbm", "xgboost", "tft"],
                            help="限定模型架構；留空＝挑最近一顆留有盲測區的已完成模型")
        parser.add_argument("--model-version", default="",
                            help="精確指定 ModelTrainingRun.model_version")
        parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                            help="進場門檻（預設 0.5）")
        parser.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS,
                            help="持有交易日數（預設 20，對齊 Return20）")
        parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS,
                            help="停損比例（預設 0.95＝進場價 95%；0 表示關閉）")
        parser.add_argument("--use-validation", action="store_true",
                            help="改在驗證區回測（模型看過的資料，輸出會標註僅供參考）")
        parser.add_argument("--json", dest="json_path", default="",
                            help="把完整報告（含每筆交易）另存為 JSON")

    def handle(self, *args, **opts):
        try:
            r = run_backtest(
                opts["stock_code"], model_type=opts["model_type"],
                threshold=opts["threshold"], hold_days=opts["hold_days"],
                stop_loss=opts["stop_loss"], model_version=opts["model_version"],
                use_validation=opts["use_validation"],
            )
        except ValueError as exc:
            raise CommandError(str(exc))

        m = r["metrics"]
        tag = "盲測區" if r["is_blind"] else "驗證區（模型看過的資料，僅供參考）"
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"=== {r['stock_code']} {r['stock_name']}｜{tag}策略回測 ==="))
        self.stdout.write(f"模型：{r['model_type']} · {r['model_version']}")
        self.stdout.write(
            f"訓練 {r['train_period'][0]} ~ {r['train_period'][1]}｜"
            f"驗證 {r['validation_period'][0]} ~ {r['validation_period'][1]}")
        self.stdout.write(
            f"回測區間：{r['backtest_period'][0]} ~ {r['backtest_period'][1]}｜"
            f"訊號日 {r['n_signals']} 天")
        p = r["params"]
        self.stdout.write(
            f"規則：機率>={p['threshold']} 次日收盤進場｜持有 {p['hold_days']} 日｜"
            f"停損 {p['stop_loss'] if p['stop_loss'] > 0 else '關閉'}｜"
            f"成本約 {p['cost_roundtrip_pct']}%（來回）")
        if r["note"]:
            self.stdout.write(self.style.WARNING("備註：" + r["note"]))

        self.stdout.write("")
        if not r["trades"]:
            reason = ("模型機率皆未達門檻" if r["n_signals"] == 0
                      else "有訊號日但都無法完整出場")
            self.stdout.write(self.style.WARNING(
                f"回測區內沒有產生可計入績效的交易（{reason}）——可調低 --threshold 再試"))
        else:
            self.stdout.write(f"交易紀錄（{len(r['trades'])} 筆）：")
            for i, t in enumerate(r["trades"], 1):
                self.stdout.write(
                    f"  #{i} 訊號 {t['signal_date']} → 進 {t['entry_date']} @{t['entry_price']}"
                    f" → 出 {t['exit_date']} @{t['exit_price']}"
                    f"（{t['reason']}，{t['hold_days']} 日）淨 {t['net'] * 100:+.2f}%")
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS("績效摘要："))
            self.stdout.write(
                f"  勝率 {m['win_rate'] * 100:.1f}%｜平均每筆淨報酬 {m['avg_net'] * 100:+.2f}%｜"
                f"最好 {m['best_net'] * 100:+.2f}%／最差 {m['worst_net'] * 100:+.2f}%")
            sharpe = (f"{m['sharpe']:.2f}" if m["sharpe"] is not None
                      else "無法計算（日報酬變異為 0）")
            self.stdout.write(
                f"  總淨報酬（複利）{m['total_return'] * 100:+.2f}%｜年化夏普 {sharpe}｜"
                f"最大回撤 {m['max_drawdown'] * 100:.2f}%")
            self.stdout.write(
                f"  停損 {m['n_stop']} 次｜平均持有 {m['avg_hold_days']:.0f} 日｜"
                f"在場比例 {m['exposure'] * 100:.0f}%")
            if m["stock_bnh"] is not None:
                line = f"  同期買進持有：個股 {m['stock_bnh'] * 100:+.2f}%"
                if m["bench_bnh"] is not None:
                    line += f"｜大盤 {m['bench_bnh'] * 100:+.2f}%"
                line += (f"｜策略 - 個股買進持有 "
                         f"{m['excess_vs_bnh'] * 100:+.2f} 百分點")
                self.stdout.write(line)
        if r["skipped_end"]:
            self.stdout.write(self.style.WARNING(
                f"（另有 {r['skipped_end']} 筆訊號因持有窗超出資料末端而捨棄，未計入績效）"))

        if opts["json_path"]:
            with open(opts["json_path"], "w", encoding="utf-8") as fh:
                json.dump(r, fh, ensure_ascii=False, indent=2, default=str)
            self.stdout.write(self.style.SUCCESS(f"完整報告已寫入 {opts['json_path']}"))
