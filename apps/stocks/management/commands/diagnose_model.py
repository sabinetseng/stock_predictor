"""
誤導特徵診斷與重訓（SHAP 黃金流程）——MODEL_TUNING_GUIDE.md「SHAP 誤導特徵診斷」的實作。

流程：訓練 → AUC → 用 SHAP／置換重要性抓「誤導特徵」 → 給修正建議（剔除／單調約束／轉換）
      → 可選「帶修正旗標重訓」 → 新舊 AUC 對比（情況 A／B）。

執行方式：
    python manage.py diagnose_model 2330 --model-type lightgbm          # 只診斷
    python manage.py diagnose_model 2330 --model-type lightgbm --auto --retrain
    python manage.py diagnose_model 0050 --model-type tft --auto --retrain --time-budget 600
    python manage.py diagnose_model 2330 --exclude "rsi,macd" --mono "volatility_20d:-1" --retrain

說明：
    --auto        自動套用所有抓到的誤導特徵修正（剔獨大特徵＋加單調約束；鋸齒只能給人工建議）
    --exclude     手動指定要剔除的特徵（逗號分隔）
    --mono        手動指定單調約束（格式 f:+1,g:-1）
    --retrain     帶修正旗標重訓，並產出「新舊 AUC 對比」（情況 A／B 分析）
    兩棵樹用 TreeSHAP；TFT 是 PyTorch 序列模型吃不了 TreeExplainer，用「逐股置換重要性」等價替代。
    評估資料一律用該次訓練的「驗證區間」，輸出會誠實標註這不是盲測。
"""
from django.core.management.base import BaseCommand

from apps.stocks import diagnostics
from apps.stocks.diagnostics import TFT_TIME_BUDGET


class Command(BaseCommand):
    help = "對已完成模型跑 SHAP／置換重要性診斷，可選帶修正重訓並比對新舊 AUC"

    def add_arguments(self, parser):
        parser.add_argument("stock_code", nargs="?", type=str, default="",
                            help="股票代碼，留空代表全部股票範圍內的模型")
        parser.add_argument("--model-type", type=str, default="",
                            choices=["", "lightgbm", "xgboost", "tft"],
                            help="限制診斷的模型架構（預設：最新一顆不分架構）")
        parser.add_argument("--model-version", type=str, default="",
                            help="指定要診斷的模型版本名稱（預設：最新一顆）")
        parser.add_argument("--auto", action="store_true",
                            help="自動套用所有修正建議（剔獨大特徵＋加單調約束）")
        parser.add_argument("--exclude", type=str, default="",
                            help="手動指定剔除的特徵，逗號分隔（會與 auto 結果合併）")
        parser.add_argument("--mono", type=str, default="",
                            help="手動指定單調約束，格式 f:+1,g:-1（僅樹模型生效）")
        parser.add_argument("--retrain", action="store_true",
                            help="帶修正旗標重訓，並比對新舊 AUC（情況 A／B 分析）")
        parser.add_argument("--new-version", type=str, default="",
                            help="重訓版本的模型版本名稱（預設：原版本名稱加 _corrected）")
        parser.add_argument("--time-budget", type=int, default=TFT_TIME_BUDGET,
                            help="TFT 置換重要性的時間預算（秒，預設 900）；樹模型不花時間不受影響")
        parser.add_argument("--mono-cap", type=int, default=8,
                            help="auto 模式下單調約束最多套用幾個特徵（依 |rho| 取最顯著）")

    def handle(self, *args, **options):
        exclude = [s.strip() for s in options["exclude"].split(",") if s.strip()]
        mono = {}
        for pair in [s.strip() for s in options["mono"].split(",") if s.strip()]:
            if ":" not in pair:
                self.stdout.write(self.style.ERROR(f"--mono 需為 f:+1 格式：{pair}"))
                return
            feat, sgn = pair.split(":", 1)
            try:
                mono[feat.strip()] = int(sgn)
            except (TypeError, ValueError):
                self.stdout.write(self.style.ERROR(f"--mono 方向只能是 +1 或 -1：{pair}"))
                return

        report, err = diagnostics.run_diagnosis(
            model_type=options["model_type"] or None,
            model_version=options["model_version"] or None,
            stock_code=options["stock_code"],
            auto=options["auto"],
            exclude=exclude or None,
            mono=mono or None,
            retrain=options["retrain"],
            new_version=options["new_version"] or None,
            time_budget=options["time_budget"],
            mono_cap=options["mono_cap"],
        )
        if err:
            self.stdout.write(self.style.ERROR(err))
            return
        self._print_report(report)

    def _print_report(self, report):
        w = self.stdout.write
        d = report["diagnosis"]
        ok = self.style.SUCCESS
        warn = self.style.WARNING
        w("")
        w("=" * 74)
        w("  SHAP 誤導特徵診斷報告")
        w("=" * 74)
        w(f"  模型      : {report['model_type']} / {report['model_version']} (run #{report['run_id']})")
        w(f"  訓練區間  : {report['train_window']}   驗證區間: {report['val_window']}")
        w(f"  股票範圍  : {report['stock_codes']}   標籤版本: {report['label_version']}")
        w(f"  方法      : {d['method']}")
        w(f"  樣本數    : {d.get('n_samples', '—')}   特徵數: {d['n_features']}")
        w(f"  評估資料  : {report['evaluation']}")
        w("-" * 74)
        w("  重要性 Top 8：")
        for r in d.get("importance", [])[:8]:
            if "auc_drop" in r:
                w("    %-22s auc_drop=%s  share=%6.2f%%"
                  % (r["feature"], r.get("auc_drop"), r.get("share", 0) * 100))
            else:
                w("    %-22s share=%6.2f%%  mean_abs_shap=%.5f"
                  % (r["feature"], r["share"] * 100, r.get("mean_abs_shap", 0)))
        w("-" * 74)
        findings = d.get("findings", [])
        if findings:
            w(warn("  誤導特徵（%d 筆）：" % len(findings)))
            for f_ in findings:
                w("    [%s] %-22s %s → 建議: %s"
                  % (f_["type"], f_["feature"], f_.get("detail", ""), f_["action"]))
        else:
            w(ok("  未發現明顯誤導特徵（維持現況即可）。"))
        c = report["correction"]
        if c["exclude_features"] or c["monotone_constraints"] or c["manual_steps"]:
            w("-" * 74)
            w("  修正計畫：")
            if c["exclude_features"]:
                w("    剔除特徵   : %s" % ", ".join(c["exclude_features"]))
            if c["monotone_constraints"]:
                w("    單調約束   : %s" % ", ".join(
                    f"{k}:{v:+d}" for k, v in c["monotone_constraints"].items()))
            for step in c["manual_steps"]:
                w("    人工處理   : %s" % step)
        comp = report["comparison"]
        if comp["status"] == "compared":
            w("-" * 74)
            w("  新舊 AUC 對比（情況 A／B）：")
            old_auc = "N/A" if comp["old_auc"] is None else f"{comp['old_auc']:.4f}"
            w("    舊    : %s（AUC=%s）" % (comp["old_model_version"], old_auc))
            new_auc = "N/A" if comp["new_auc"] is None else f"{comp['new_auc']:.4f}"
            w("    新    : run #%s %s（AUC=%s）" % (
                comp["new_run_id"], comp["new_model_version"], new_auc))
            if comp["delta_auc"] is not None:
                w("    ΔAUC : %s（%s）" % (
                    f"{comp['delta_auc']:+.4f}",
                    "顯著" if comp["significant"] else "不顯著"))
                if comp["case"] in ("A", "B"):
                    w("    → %s" % comp["case_readable"])
        w("=" * 74)
        w(ok("  完整報告已存回 ModelTrainingRun.diagnostics（run #%s）" % report["run_id"]))
        w("")