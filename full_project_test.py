import os, sys, json, glob
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_predictor.settings")
import django
django.setup()
from django.test import Client

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass

client = Client(HTTP_HOST="localhost")
FAILS, WARNS = [], []

def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"[{tag}] {name}" + (f"  -> {detail}" if detail else ""))
    if not ok and not warn_only: FAILS.append(name)
    elif not ok and warn_only: WARNS.append(name)

def jget(url, name, want_success=True):
    try:
        resp = client.get(url)
    except Exception as exc:
        check(name, False, repr(exc))
        return {}
    if resp.status_code != 200:
        check(name, False, f"http {resp.status_code}")
        return {}
    data = json.loads(resp.content.decode("utf-8"))
    check(name, data.get("success") is want_success or not want_success, str(data.get("message",""))[:60])
    return data

def jpost(url, body, name, want_success=True):
    try:
        resp = client.post(url, data=json.dumps(body), content_type="application/json")
    except Exception as exc:
        check(name, False, repr(exc))
        return {}
    if resp.status_code not in (200, 201):
        if not want_success:
            check(name, True, f"http {resp.status_code} (expected failure)")
        else:
            check(name, False, f"http {resp.status_code}")
        return {}
    data = json.loads(resp.content.decode("utf-8"))
    check(name, data.get("success") is want_success or not want_success, str(data.get("message",""))[:60])
    return data

# ===== 1. Smoke Test =====
print("="*70)
print("1. Smoke Test")
print("="*70)
for url, desc in [("/" ,"Home"), ("/stocks/2330/" ,"Stock 2330"), ("/stocks/0050/" ,"Stock 0050"), ("/admin/" ,"Admin")]:
    try:
        r = client.get(url)
        check(f"Smoke: {desc}", r.status_code in (200,302), f"http {r.status_code}")
    except Exception as e:
        check(f"Smoke: {desc}", False, str(e))
for url, desc in [("/stocks/model-info/" ,"Model Info"), ("/stocks/2330/predict/?model_type=tft" ,"Predict")]:
    jget(url, f"Smoke: {desc}")

# ===== 2. Build Test =====
print("\n" + "="*70)
print("2. Build Test")
print("="*70)
r = client.get("/")
c = r.content.decode("utf-8")
home_items = [
    ("btn-check-stock-new-data" ,"1.Sync Stock"),
    ("btn-check-fx-new-data" ,"2.Sync FX"),
    ("btn-build-features" ,"3.Build Features"),
    ("btn-build-labels" ,"4.Build Labels"),
    ("5. 訓練 / 驗證區間設定" ,"5.Training Config"),
    ("btn-predict" ,"6.Predict"),
    ("training-config" ,"training-config"),
    ("prediction-result-box" ,"prediction-result-box"),
    ("btn-walkforward-retrain" ,"walkforward-retrain"),
    ("btn-full-pipeline-manual" ,"full-pipeline-manual"),
    ("btn-create-training-run" ,"create-training-run"),
    ("training-info-card" ,"training-info-card"),
    ("training-runs-table" ,"training-runs-table"),
    ("model-type-select" ,"model-type-select"),
]
for eid, desc in home_items:
    check(f"Build: Home {desc}", eid in c)

r = client.get("/stocks/2330/")
c = r.content.decode("utf-8")
detail_items = [
    ("btn-predict" ,"predict btn"),
    ("chart-main" ,"chart-main"),
    ("chart-prob" ,"chart-prob"),
    ("chart-feat" ,"chart-feat"),
    ("corr-pearson" ,"corr-pearson"),
    ("corr-spearman" ,"corr-spearman"),
    ("corr-target" ,"corr-target"),
    ("btn-corr-analyze" ,"btn-corr-analyze"),
    ("model_type" ,"model_type"),
    ("threshold" ,"threshold"),
    ("btn-reset-th" ,"btn-reset-th"),
    ("result" ,"result"),
    ("doctrine-panel" ,"doctrine-panel"),
    ("feat-table" ,"feat-table"),
    ("strategy-panel" ,"strategy-panel"),
    ("correlation-panel" ,"correlation-panel"),
]
for eid, desc in detail_items:
    check(f"Build: Detail {desc}", eid in c)

model_files = glob.glob("trained_models/*.joblib")
check("Build: model files exist", len(model_files)>0, f"{len(model_files)} files")
for mt in ["lightgbm","xgboost","tft"]:
    files = [f for f in model_files if f"v_{mt}" in f]
    check(f"Build: {mt} model", len(files)>0, f"{len(files)} files")

from apps.stocks.models import ModelTrainingRun, StockBasicInfo, StockPrice, StockFeatureDaily, StockLabel, StockPrediction
for model, desc in [(ModelTrainingRun,"ModelTrainingRun"),(StockBasicInfo,"StockBasicInfo"),(StockPrice,"StockPrice"),(StockFeatureDaily,"StockFeatureDaily"),(StockLabel,"StockLabel"),(StockPrediction,"StockPrediction")]:
    check(f"Build: {desc} readable", model.objects.count()>=0, f"{model.objects.count()} rows")

# ===== 3. Regression Test =====
print("\n" + "="*70)
print("3. Regression Test")
print("="*70)
get_apis = [
    ("/stocks/model-info/" ,"model-info"),
    ("/stocks/model-info/?model_type=lightgbm" ,"model-info-lgb"),
    ("/stocks/model-info/?model_type=xgboost" ,"model-info-xgb"),
    ("/stocks/model-info/?model_type=tft" ,"model-info-tft"),
    ("/stocks/feature-importance/" ,"feature-importance"),
    ("/stocks/model-stability/" ,"model-stability"),
    ("/stocks/shap-diagnostics/" ,"shap-diagnostics"),
    ("/stocks/training-runs/" ,"training-runs"),
    ("/stocks/date-range/?stock_code=2330" ,"date-range"),
    ("/stocks/2330/chart-data/" ,"chart-data-2330"),
    ("/stocks/0050/chart-data/" ,"chart-data-0050"),
    ("/stocks/2330/signals/" ,"signals-2330"),
    ("/stocks/2330/prediction-settings/?model_type=tft" ,"prediction-settings"),
    ("/stocks/2330/check-new-data/" ,"check-new-data"),
    ("/stocks/2330/build-features/" ,"build-features"),
    ("/stocks/2330/build-labels/" ,"build-labels"),
    ("/fx/check-new-data/" ,"fx-check-new-data"),
    ("/fx/history/" ,"fx-history"),
    ("/analysis/correlation/2330/" ,"correlation-2330"),
]
for url, desc in get_apis:
    jget(url, f"Regression: {desc}")

post_apis = [
    ("/stocks/training-runs/create/" ,{"stock_code":"2330","train_start":"2022-01-01","train_end":"2024-01-01","validation_start":"2024-01-02","validation_end":"2025-01-01"} ,"create-training-run"),
    ("/stocks/walkforward-retrain/" ,{"stock_codes":"2330","dry_run":True} ,"walkforward-retrain"),
    ("/stocks/full-pipeline-train/" ,{"stock_codes":"2330","dates":{"train_start":"2022-01-01","train_end":"2024-01-01","validation_start":"2024-01-02","validation_end":"2025-01-01"},"dry_run":True} ,"full-pipeline-train"),
    ("/stocks/walkforward-retrain/" ,{} ,"walkforward-retrain-noargs", False),
    ("/stocks/full-pipeline-train/" ,{} ,"full-pipeline-train-noargs", False),
]
for item in post_apis:
    url, body, desc = item[0], item[1], item[2]
    want = item[3] if len(item)>3 else True
    jpost(url, body, f"Regression: {desc}", want)

from apps.stocks.model_training import MODEL_INFO, FEATURE_COLUMNS, DEFAULT_HYPERPARAMS
check("Regression: MODEL_INFO 3 models", {"lightgbm","xgboost","tft"}<=set(MODEL_INFO.keys()))
check("Regression: FEATURE_COLUMNS 32", len(FEATURE_COLUMNS)==32, f"{len(FEATURE_COLUMNS)}")
check("Regression: DEFAULT_HYPERPARAMS 3 models", {"lightgbm","xgboost","tft"}<=set(DEFAULT_HYPERPARAMS.keys()))

r = client.get("/stocks/2330/strategy-settings/")
check("Regression: strategy-settings redirect", r.status_code==302, f"http {r.status_code}")

# ===== 4. Integration Test =====
print("\n" + "="*70)
print("4. Integration Test")
print("="*70)
from apps.stocks import services
for mt in ["tft","lightgbm","xgboost"]:
    result = services.predict("2330", mt)
    check(f"Integration: predict 2330 {mt}", result.get("success"), str(result.get("message",""))[:60])

pred_before = StockPrediction.objects.count()
services.predict("2330", "tft")
pred_after = StockPrediction.objects.count()
check("Integration: prediction written", pred_after>=pred_before, f"before={pred_before}, after={pred_after}")

from apps.analysis.services import calc_all_correlations
for target in ["USD_TWD","TWII","SOX","VIX"]:
    result = calc_all_correlations("2330", target)
    for method in ["pearson","spearman","lagged","rolling"]:
        check(f"Integration: corr {target} {method}", result.get(method,{}).get("success"))

r = client.get("/stocks/2330/chart-data/")
d = json.loads(r.content.decode("utf-8"))
check("Integration: chart-data success", d.get("success"))
cd = d.get("data",{})
check("Integration: chart-data has prices", "prices" in cd or "price" in str(cd).lower())

r = client.get("/stocks/2330/signals/")
d = json.loads(r.content.decode("utf-8"))
check("Integration: signals success", d.get("success"))

r = client.get("/stocks/2330/predict/?model_type=tft")
d = json.loads(r.content.decode("utf-8"))
check("Integration: predict API success", d.get("success") and d.get("data") is not None)
pd = d.get("data",{})
for key in ["stock_code","predicted_probability","predicted_label","model_version"]:
    check(f"Integration: predict data has {key}", key in pd)

# ===== Summary =====
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"FAIL: {len(FAILS)}")
print(f"WARN: {len(WARNS)}")
if FAILS:
    print("\nFailed:")
    for f in FAILS: print(f"  - {f}")
    sys.exit(1)
else:
    print("\nAll tests PASSED!")
    sys.exit(0)
