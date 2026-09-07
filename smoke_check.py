"""
專案煙霧測試（Smoke Check）——一鍵健檢資料層／模型層／API 層／前端模板。

執行方式：
    python smoke_check.py          # 任何 FAIL 以非 0 結束碼離開，方便掛排程或 CI
"""
import os
import re
import sys
import json

# Windows 主控台預設 cp950，無法編碼部分中文（如「幂」），會讓 print() 拋
# UnicodeEncodeError 並誤觸外層 try/except 造成假 FAIL——強制以 UTF-8 輸出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 舊版 Python 無 reconfigure 時略過
    pass

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_predictor.settings")
import django

django.setup()

from django.test import Client  # noqa: E402

PASS, WARN = "\x1b[32mPASS\x1b[0m", "\x1b[33mWARN\x1b[0m"
FAILS = []


def check(name, ok, detail="", warn_only=False):
    tag = PASS if ok else (WARN if warn_only else "FAIL\x1b[0m")
    print(f"[{tag}] {name}" + (f"  -> {detail}" if detail else ""))
    if not ok and not warn_only:
        FAILS.append(name)


print("=" * 72)
print("A. 資料層")
print("=" * 72)
from apps.stocks.models import StockFeatureDaily, ModelTrainingRun, EtfNavData  # noqa: E402
from apps.stocks.model_training import FEATURE_COLUMNS, DEFAULT_HYPERPARAMS  # noqa: E402

check("FEATURE_COLUMNS 共 32 個特徵", len(FEATURE_COLUMNS) == 32, f"實際 {len(FEATURE_COLUMNS)}")
check("DEFAULT_HYPERPARAMS 含三種模型",
      {"lightgbm", "xgboost", "tft"} <= set(DEFAULT_HYPERPARAMS.keys()))
check("ETF 淨值資料存在", EtfNavData.objects.count() > 0,
      f"{EtfNavData.objects.count()} 筆")

etf_feat = StockFeatureDaily.objects.filter(stock__stock_code="0050").exclude(
    etf_premium_discount__isnull=True).count()
check("0050 折溢價特徵已計算", etf_feat > 0, f"{etf_feat} 列有值")

print("=" * 72)
print("B. 模型層")
print("=" * 72)
import joblib  # noqa: E402
import apps.stocks.services as services  # noqa: E402

for mtype in ["lightgbm", "xgboost", "tft"]:
    run = (ModelTrainingRun.objects.filter(status="completed", model_type=mtype)
           .exclude(model_file_path="").order_by("-completed_at").first())
    if run is None:
        check(f"{mtype}: 存在已完成訓練", False, "無任何完成紀錄", warn_only=True)
        continue
    ok_file = os.path.exists(run.model_file_path)
    check(f"{mtype}: 模型檔存在（{run.model_version[:40]}）", ok_file)
    if ok_file:
        b = joblib.load(run.model_file_path)
        model = b["model"]
        n_f = len(b.get("feature_columns") or [])
        seq_flag = bool(getattr(model, "requires_sequence", False))
        check(f"{mtype}: bundle 完整（特徵 {n_f}、序列={seq_flag}）",
              n_f > 0 and seq_flag == (mtype == "tft"))

r_0050 = services.predict("0050")
check("services.predict('0050')（ETF 序列路徑）成功", bool(r_0050.get("success")),
      str(r_0050.get("message"))[:60])
r_2330 = services.predict("2330")
d_2330 = r_2330.get("data") or {}  # success=False 時 data 為 None，用 or {} 避免崩潰
check("services.predict('2330')（個股路徑）成功", bool(r_2330.get("success")),
      f"機率={d_2330.get('predicted_probability')}｜{str(r_2330.get('message'))[:60]}")

print("=" * 72)
print("C/D. API 層與模板（Django test Client，HOST=localhost）")
print("=" * 72)
client = Client(HTTP_HOST="localhost")


def jget(url, name, want_success=True):
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        check(name, False, repr(exc))
        return {}
    if resp.status_code != 200:
        check(name, False, f"http {resp.status_code}")
        return {}
    data = resp.json()
    check(name, data.get("success") is want_success or not want_success,
          str(data.get("message"))[:70])
    return data


home = client.get("/")
check("GET / 首頁 200", home.status_code == 200)
if home.status_code == 200:
    c = home.content.decode("utf-8", "ignore")
    check("首頁：預測模型預設 TFT", 'value="tft" selected' in c)
    check("首頁：TFT 參數區塊初始顯示", '<div id="tft-params">' in c)
    check("首頁：LightGBM 區塊初始隱藏", '<div id="lightgbm-params">' not in c)
    check("首頁：最新訓練結果摘要卡存在", "training-summary-content" in c)
    check("首頁：自選日期模式＋一鍵管線按鈕在位",
          all(s in c for s in ["manual-dates-toggle", "btn-full-pipeline-manual",
                               "btn-walkforward-retrain", "full-pipeline-train"]))
    check("首頁：全自動一鍵訓練按鈕已移除",
          "btn-full-pipeline-auto" not in c and "全自動一鍵訓練" not in c)
    check("首頁：流程按鈕為一般數字（無圈圈字元）",
          all(s in c for s in ["1. 同步個股資料", "6. 進行 AI 預測", "5. 訓練 / 驗證區間設定"])
          and not any(0x2460 <= ord(ch) <= 0x2468 for ch in c))
    check("首頁：莫蘭迪綠主色在位", "#5e7561" in c)
    check("首頁：相關性分析／策略設定按鈕已移除",
          "btn-correlation-analysis" not in c and "btn-strategy-settings" not in c)

    # --- 無 emoji 小 icon + 新排版（checkbox-row / dates-mode-hint / 超參數列式） ---
    _emoji_pat = re.compile(
        "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u231B\u23F3]")
    _found_h = _emoji_pat.findall(c)
    check("首頁：無 emoji 小 icon", not _found_h, f"殘留 {_found_h[:8]}")
    check("首頁：自選日期 checkbox 使用 checkbox-row 排版（同「訓練納入最新資料」）",
          'class="checkbox-row"' in c and "manual-dates-toggle" in c)
    check("首頁：日期模式說明行（dates-mode-hint）在位", "dates-mode-hint" in c)
    check("首頁：超參數區改用 feature-list 列式排版（同特徵清單）",
          'class="feature-list" id="info-hyperparams"' in c
          and "hyperparam-item" not in c and "hyperparam-list" not in c)
    check("首頁：資訊卡三區可摺疊（預設收合，body 帶 hidden）",
          c.count('class="info-collapse-header"') == 3
          and 'id="info-model-body" hidden' in c
          and 'id="info-hyperparams-body" hidden' in c
          and 'id="info-features-body" hidden' in c)
    check("首頁：模型名稱徽章固定於摺疊標題列（收合仍顯示模型名稱）",
          'id="model-info-summary"' in c and "collapse-arrow" in c)
    check("首頁：舊「這次訓練會用到哪些資料與參數」details 卡已移除（併入資訊卡）",
          "training-data-explanation" not in c
          and "這次訓練會用到哪些資料與參數" not in c)
    check("首頁：資訊卡已併入訓練目標（Y）定義、特徵組成與資料範圍說明",
          "訓練目標（Y）" in c and "X_Y_VARIABLE_DEFINITION.md" in c
          and "資料範圍：只使用上方" in c)
    check("首頁：一鍵按鈕文字無 emoji",
          all(s in c for s in ["自動切分（60/20/20）", "一鍵重訓練（最新 60/20/20，含同步／特徵／標籤）",
                               "自選日期一鍵訓練"]))

detail = client.get("/stocks/0050/")
check("GET /stocks/0050/ 決策看板 200", detail.status_code == 200)
if detail.status_code == 200:
    dc = detail.content.decode("utf-8", "ignore")
    check("決策看板：AI 模型架構預設 TFT", '<option value="tft" selected>' in dc)
    check("決策看板：莫蘭迪綠主色在位", "#5e7561" in dc)
    _emoji_pat2 = re.compile(
        "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u231B\u23F3]")
    _found_d = _emoji_pat2.findall(dc)
    check("決策看板：無 emoji 小 icon", not _found_d, f"殘留 {_found_d[:8]}")
    check("決策看板：含各架構最後訓練資訊（MODEL_RUNS）", "MODEL_RUNS = {" in dc)
    check("決策看板：訓練/驗證區間元素在位",
          "model-train-range" in dc and "model-val-range" in dc)
    check("決策看板：AI 推論結果卡含賣出狀態機（停利/停損/續抱/賣出訊號）",
          all(s in dc for s in ("建議停利賣出 (TAKE PROFIT)", "建議停損賣出 (STOP LOSS)",
                                "持倉觀察中 (HOLD)", "賣出訊號觸發 (SELL)")))
    check("決策看板：停利/停損門檻取自後端（take_profit_pct/stop_loss_pct）",
          "CHART_DATA.take_profit_pct" in dc and "CHART_DATA.stop_loss_pct" in dc)

cd = jget("/stocks/0050/chart-data/", "chart-data（0050）回傳成功")
if cd.get("data"):
    d = cd["data"]
    check("chart-data：價格序列非空", len(d.get("price") or []) > 0)
    check("chart-data：含停利/停損門檻（take_profit_pct/stop_loss_pct）",
          isinstance(d.get("take_profit_pct"), (int, float))
          and isinstance(d.get("stop_loss_pct"), (int, float)),
          f"停利 +{round(d.get('take_profit_pct', 0) * 100, 1)}%／停損 -{round(d.get('stop_loss_pct', 0) * 100, 1)}%")
jget("/stocks/2330/chart-data/", "chart-data（2330）回傳成功")

dr = jget("/stocks/date-range/?stock_code=0050", "date-range API 成功")
if dr.get("data"):
    check("date-range：min/max 齊備",
          bool(dr["data"].get("min_date")) and bool(dr["data"].get("max_date")))

tr = jget("/stocks/training-runs/?page_size=5", "training-runs 列表 API 成功")
runs = (tr.get("data") or {}).get("runs") or []
check("training-runs：序列化含 auc/pr_auc/hyperparams／diagnostics 欄位",
      bool(runs) and all(k in runs[0] for k in ("auc", "pr_auc", "hyperparams", "diagnostics")),
      f"{len(runs)} 筆")
_created0 = (runs[0].get("created_at") or "") if runs else ""
check("training-runs：created_at 為台北時間（UTC+8）",
      bool(_created0) and _created0.endswith("+08:00"),
      _created0[:35] or "無紀錄")

mi = jget("/stocks/model-info/", "model-info API 成功")
mi_data = (mi.get("data") or {})
check("model-info：模型三種 + 特徵 32 + 分類 9",
      set(mi_data.get("models") or {}) == {"lightgbm", "xgboost", "tft"}
      and len((mi_data.get("features") or {})) == 32
      and len((mi_data.get("categories") or [])) == 9,
      f"models={list((mi_data.get('models') or {}).keys())}")
mi_lgb = jget("/stocks/model-info/?model_type=lightgbm", "model-info 依模型過濾")
check("model-info：model_type 過濾生效",
      list(((mi_lgb.get("data") or {}).get("models") or {}).keys()) == ["lightgbm"])

# 特徵重要性（五-4）與 Walk-Forward 穩定性評估（五-5）
fi = jget("/stocks/feature-importance/", "feature-importance API 成功")
check("feature-importance：回傳 items 與模型資訊欄位",
      "items" in (fi.get("data") or {}) and "model_version" in (fi.get("data") or {}))
ms = jget("/stocks/model-stability/", "model-stability API 成功")
check("model-stability：回傳 reports（依模型類型分組）",
      isinstance((ms.get("data") or {}).get("reports"), list))
_ms_rounds = [r for rep in ((ms.get("data") or {}).get("reports") or [])
              for r in (rep.get("rounds") or [])]
check("model-stability：每輪含 shap 診斷摘要欄位（未診斷為 null）",
      bool(_ms_rounds) and all("shap" in r for r in _ms_rounds),
      f"{len(_ms_rounds)} 輪，{sum(1 for r in _ms_rounds if r.get('shap'))} 輪有 SHAP")
check("ModelTrainingRun 含 feature_importance 欄位", hasattr(ModelTrainingRun, "feature_importance"))

# SHAP 誤導特徵診斷（diagnose_model 產出的報告；diagnosed runs 才有資料）
sd = jget("/stocks/shap-diagnostics/", "shap-diagnostics API 成功")
_sd_data = sd.get("data")
check("shap-diagnostics：success 且結構正確（無診斷時 data=null）",
      sd.get("success") is True and (_sd_data is None or isinstance(_sd_data, dict)))
if _sd_data:
    check("shap-diagnostics：含 importance/findings/correction/comparison",
          all(k in _sd_data for k in ("importance", "findings", "correction", "comparison"))
          and bool(_sd_data.get("importance")),
          f"run #{_sd_data.get('run_id')} importance={len(_sd_data.get('importance') or [])} 筆")
else:
    check("shap-diagnostics：尚無診斷紀錄時顯示引導訊息",
          bool(sd.get("message")), (sd.get("message") or "")[:30])

# SHAP 診斷已納入訓練流程（run_training 完成後自動執行，auto_shap=false 可關閉）
import inspect as _inspect  # noqa: E402
from apps.stocks import diagnostics as _diag_mod  # noqa: E402
from apps.stocks import model_training as _mt_mod  # noqa: E402
check("訓練流程已納入 SHAP 自動診斷（run_training 呼叫 attach_auto_diagnosis）",
      hasattr(_diag_mod, "attach_auto_diagnosis")
      and "attach_auto_diagnosis" in _inspect.getsource(_mt_mod.run_training))
check("SHAP 自動診斷可用超參數 auto_shap 關閉（run_training 讀取旗標）",
      "auto_shap" in _inspect.getsource(_mt_mod.run_training),
      f"AUTO_TFT_TIME_BUDGET={getattr(_diag_mod, 'AUTO_TFT_TIME_BUDGET', '?')} 秒")

# 一鍵完整訓練管線已改為「僅自選日期」：dates 缺略必須回 400（全自動模式已移除）
# ——最新 60/20/20 一鍵訓練改由「一鍵重訓練」（walkforward-retrain）提供。
fp = client.post("/stocks/full-pipeline-train/", data=json.dumps({"dry_run": True}),
                 content_type="application/json")
try:
    fpj = fp.json()
    check("full-pipeline 無 dates 被拒絕（全自動模式已移除，應 400）",
          fp.status_code == 400 and not fpj.get("success"),
          str(fpj.get("message"))[:70])
except Exception:
    check("full-pipeline 無 dates 回傳 JSON", False, f"http {fp.status_code}")

wf = client.post("/stocks/walkforward-retrain/", data=json.dumps({"dry_run": True}),
                 content_type="application/json")
try:
    wfj = wf.json()
    wd = wfj.get("data") or {}
    check("walkforward-retrain dry-run（最新 60/20/20）成功",
          wf.status_code == 200 and wfj.get("success")
          and all(wd.get(k) for k in ("train_start", "train_end",
                                      "validation_start", "validation_end")),
          str(wfj.get("message"))[:70])
    check("walkforward-retrain dry-run 預覽完整管線步驟（同步/特徵/標籤/訓練）",
          wd.get("split_mode") == "最新資料 60/20/20"
          and (wd.get("steps") or []) == ["同步資料", "建立特徵", "建立標籤", "訓練"],
          str(wd.get("steps")))
except Exception:
    check("walkforward-retrain dry-run 回傳 JSON", False, f"http {wf.status_code}")

fp2 = client.post("/stocks/full-pipeline-train/", data=json.dumps({
    "dry_run": True,
    "dates": {"train_start": "2024-01-01", "train_end": "2024-06-30",
              "validation_start": "2024-07-01", "validation_end": "2024-09-30"},
}), content_type="application/json")
try:
    fp2j = fp2.json()
    check("full-pipeline dry-run（自選日期）成功",
          fp2.status_code == 200 and fp2j.get("success") and (fp2j.get("data") or {}).get("split_mode") == "自選日期",
          str(fp2j.get("message"))[:70])
except Exception:
    check("full-pipeline dry-run（自選日期）回傳 JSON", False, f"http {fp2.status_code}")

# 自選日期模式必須擋下 train_end >= validation_start（資料洩漏防護）
fp3 = client.post("/stocks/full-pipeline-train/", data=json.dumps({
    "dry_run": True,
    "dates": {"train_start": "2024-01-01", "train_end": "2024-08-30",
              "validation_start": "2024-07-01", "validation_end": "2024-09-30"},
}), content_type="application/json")
try:
    fp3j = fp3.json()
    check("full-pipeline 自選日期洩漏防護（train_end >= val_start 應 400）",
          fp3.status_code == 400 and not fp3j.get("success"),
          str(fp3j.get("message"))[:70])
except Exception:
    check("full-pipeline 洩漏防護回傳 JSON", False, f"http {fp3.status_code}")

jget("/stocks/0050/signals/?threshold=3&days=60&model_type=tft", "signals API 成功")
jget("/stocks/0050/prediction-settings/?model_type=tft", "prediction-settings API 成功")
ss = client.get("/stocks/0050/strategy-settings/")
check("strategy-settings 可達（200 或轉址決策看板）",
      ss.status_code in (200, 301, 302), f"http {ss.status_code}")

lbl = client.post("/stocks/0050/build-labels/", data="{}",
                  content_type="application/json")
try:
    lj = lbl.json()
    check("POST build-labels（幂等重算）成功", lbl.status_code == 200 and lj.get("success"),
          str(lj.get("message"))[:70])
except Exception as _lbl_exc:
    check("POST build-labels 回傳 JSON", False,
          f"http {lbl.status_code}｜exc={repr(_lbl_exc)[:150]}｜body={lbl.content.decode('utf-8','ignore')[:120]}")

corr = client.get("/analysis/correlation/0050/")
try:
    cj = corr.json()
    corr_ok = corr.status_code == 200 and cj.get("success") is not False
except Exception:
    corr_ok = corr.status_code == 200
check("analysis/correlation 相關性分析 API", corr_ok,
      f"http {corr.status_code}", warn_only=True)

print("=" * 72)
if FAILS:
    print(f"\u274c 健檢未通過：{len(FAILS)} 項 FAIL -> {FAILS}")
    sys.exit(1)
print("\u2705 全部檢查通過（或僅 WARN）。")

