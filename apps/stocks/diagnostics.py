"""
誤導特徵診斷（SHAP Golden Loop）——「訓練 → AUC → 抓誤導特徵 → 修正 → 重訓 → 比對」的診斷端。

目的（對應 MODEL_TUNING_GUIDE.md 的調參核心）：
    特徵重要性只說「模型用了誰」，這個模組回答更關鍵的問題：
    「模型是不是被某些特徵誤導了？」三類誤導各有對應的修正動作：
      1. 作弊/獨大特徵（重要性佔比 >= 35%，與首頁過擬合警語同一門檻）→ 剔除後重訓
      2. 依賴曲線鋸齒狀（模型在硬背噪訊）→ 建議分箱/平滑/對數轉換（人工處理）
      3. 方向違反領域直覺（SHAP 方向與 MONO_HINTS 相反）→ 加單調約束後重訓

三種模型的診斷方法（模型能力不同，方法不同）：
    lightgbm / xgboost：TreeSHAP（精確、快）＋方向健檢＋依賴曲線震盪指數
    tft（PyTorch 序列模型，吃不了 TreeExplainer）：逐特徵置換重要性
        （把驗證段的特徵值打亂 → 看 AUC 掉多少；掉越多＝模型越依賴它）

評估資料：該次訓練的「驗證區間」（與 run.auc 同一窗口、同一標籤版本、
同一套 _load_dataset 特徵組裝），並非盲測區——診斷的是「模型學到了什麼」，
不是「未來表現」，輸出中會誠實標註。

執行方式（見 management/commands/diagnose_model.py）：
    python manage.py diagnose_model 2330 --model-type lightgbm
    python manage.py diagnose_model 2330 --model-type tft --auto --retrain

自動診斷（Section D）：SHAP 診斷已納入訓練流程——run_training 在訓練完成後
    自動呼叫 attach_auto_diagnosis 把報告存進 run.diagnostics（預設開啟，
    超參數 auto_shap=false 可關閉；TFT 時間預算 AUTO_TFT_TIME_BUDGET=60 秒，
    可用超參數 shap_time_budget 覆蓋）。手動指令保留給「歷史舊紀錄補診斷」
    與「--auto --retrain 帶修正重訓比對」兩種情境。
"""
import os
import warnings

import numpy as np
import pandas as pd

from django.utils import timezone

from .models import ModelTrainingRun
from .model_training import _load_dataset, run_training

# ---- 門檻（與既有機制對齊，避免另立標準）----
LEAK_SHARE = 0.35   # 單一特徵重要性佔比 >= 35% → 疑似作弊/獨大（同首頁過擬合警語門檻）
JAGGED_OI = 0.5     # 依賴曲線震盪指數 >= 0.5 → 硬背噪訊，建議分箱/平滑
DIR_ALPHA = 0.05    # 方向健檢 Spearman 的顯著門檻
DIR_RHO = 0.05      # 方向健檢的最小相關強度（低於此視為模型幾乎沒在用方向性）
OSC_BINS = 12       # 依賴曲線分箱數（震盪指數用）
OSC_MIN_BIN = 8     # 每箱最少樣本數，不足就不評（避免小樣本假鋸齒）

# 領域直覺方向（+1＝特徵越大越好買點、-1＝越大越危險）——來源：小白名詞手冊／X_Y_VARIABLE_DEFINITION.md。
# 只列「方向近乎公理」的特徵；RSI／乖離率／匯率／Beta／ETF 折溢價等屬非單調或方向不明，
# 不做方向健檢、也不建議加約束（硬約束反而傷害模型）。
MONO_HINTS = {
    "ma20_slope": 1, "ma60_slope": 1, "macd": 1, "macd_monthly": 1,
    "institutional_net_buy_ratio": 1, "inst_net_buy_20d": 1, "inst_accel": 1,
    "revenue_mom": 1, "revenue_yoy": 1,
    "excess_return_vs_benchmark": 1, "rs_vs_sox": 1, "sox_mom": 1, "sox_change_rate": 1,
    "margin_usage_ratio": -1, "volatility_20d": -1,
    "vix": -1, "vix_level_percentile": -1, "vix_change_rate": -1,
}


def _stock_list(run):
    """ModelTrainingRun.stock_codes（逗號分隔字串）→ list；空值代表全市場（None）。"""
    codes = [s.strip() for s in (run.stock_codes or "").split(",") if s.strip()]
    return codes or None


def pick_diag_run(model_type=None, model_version=None, stock_code=""):
    """
    選一顆要診斷的已完成模型（與 backtest.pick_run 不同的點：不需要留盲測區——
    診斷評估的是驗證區間）。回傳 (run, None) 或 (None, 錯誤訊息)。
    """
    qs = ModelTrainingRun.objects.filter(status="completed").exclude(model_file_path="")
    if model_type:
        qs = qs.filter(model_type=model_type)
    if model_version:
        qs = qs.filter(model_version=model_version)
    if stock_code:
        qs = qs.filter(stock_codes__contains=stock_code)
    for run in qs.order_by("-completed_at", "-id")[:20]:
        if run.model_file_path and os.path.exists(run.model_file_path):
            return run, None
    return None, (
        "找不到可診斷的已完成模型（模型檔已不在 trained_models/ 的紀錄會自動跳過）。"
        "請先用 train_model／一鍵重訓練完成一次訓練。"
    )


def _safe_auc(y, proba):
    """驗證集只有單一類別時誠實回 None（與訓練端同一原則：不硬算）。"""
    if y is None or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, proba))


def _oscillation(x_vals, shap_vals, n_bins=OSC_BINS, min_bin=OSC_MIN_BIN):
    """
    依賴曲線震盪指數：把特徵值分位分箱、取每箱平均 SHAP，
    「相鄰箱平均差的平均」除以曲線總幅度。0＝單調平滑；越大越鋸齒（硬背噪訊）。
    樣本不足時回 None（不誣賴特徵）。
    """
    xs = pd.Series(x_vals).rank(method="first")
    try:
        bins = pd.qcut(xs, q=n_bins, duplicates="drop")
    except ValueError:
        return None
    grp = pd.Series(np.asarray(shap_vals)).groupby(bins.values, observed=True).agg(["mean", "count"])
    grp = grp[grp["count"] >= min_bin]
    if len(grp) < 4:
        return None
    m = grp["mean"].to_numpy(dtype="float64")
    rng = float(np.max(m) - np.min(m))
    if rng <= 1e-12:
        return None
    return float(np.mean(np.abs(np.diff(m))) / rng)


def _direction_findings(val_df, cols, shap_matrix):
    """
    方向健檢：對 MONO_HINTS 裡「方向近乎公理」的特徵，檢查 SHAP 貢獻方向是否相反。
    Spearman(特徵值, 該特徵的 SHAP 值)：顯著（p < DIR_ALPHA）、有一定強度（|rho| >= DIR_RHO）
    且方向與領域直覺相反 → 判定「誤導」，建議加單調約束重訓。
    RSI／乖離率／匯率等非單調特徵不在 MONO_HINTS，自然跳過（硬約束反而傷模型）。
    """
    from scipy.stats import spearmanr
    out = []
    for i, c in enumerate(cols):
        hint = MONO_HINTS.get(c)
        if hint is None:
            continue
        xv = pd.to_numeric(val_df[c], errors="coerce").to_numpy(dtype="float64")
        sv = np.asarray(shap_matrix)[:, i]
        ok = np.isfinite(xv) & np.isfinite(sv)
        if ok.sum() < 50 or np.nanstd(xv[ok]) < 1e-9 or np.nanstd(sv[ok]) < 1e-9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 常數/近常數輸入的 ConstantInputWarning 已在上行排除
            rho, p = spearmanr(xv[ok], sv[ok])
        if rho is None or p is None or not np.isfinite(rho) or not np.isfinite(p):
            continue
        if p < DIR_ALPHA and abs(rho) >= DIR_RHO and (rho * hint) < 0:
            out.append({
                "feature": c, "hint": int(hint),
                "rho": round(float(rho), 3), "p_value": round(float(p), 4),
            })
    return out


def diagnose_trees(run):
    """
    LightGBM / XGBoost 診斷：TreeSHAP（精確、快）算每個特徵的平均 |SHAP|，
    產出三類誤導證據：leak（獨大/作弊）、direction（方向違反領域直覺）、
    jagged（依賴曲線鋸齒狀＝硬背噪訊）。評估資料＝該次訓練的驗證區間。
    """
    try:
        import joblib
        import shap
    except ImportError as exc:
        raise RuntimeError("診斷需要 shap 套件：pip install shap") from exc

    # shap 0.52 對 LightGBM 二元分類的 TreeExplainer 會噴「output has changed to a list」提醒，
    # 這是已知且已處理的行為（下面 sv 解包處已相容 list），純噪音，抑制掉讓輸出乾淨。
    warnings.filterwarnings("ignore", category=UserWarning, module="shap")

    bundle = joblib.load(run.model_file_path)
    model = bundle["model"]
    cols = [str(c) for c in bundle.get("feature_columns", [])]
    if not cols:
        raise RuntimeError("模型檔缺少 feature_columns，無法診斷（請重新訓練一次）")

    val_df, _, _ = _load_dataset(
        run.validation_start, run.validation_end, _stock_list(run), run.label_version,
    )
    val_df = val_df.dropna(subset=cols)
    if val_df.empty:
        raise RuntimeError("驗證區間組不出資料（請確認該區間已 build_features／build_labels）")

    X = val_df[cols]
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype="float64")
    if sv.ndim == 3:  # 某些 shap 版本回傳 (n, m, 2)
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    total = float(mean_abs.sum())
    shares = mean_abs / total if total > 1e-12 else np.zeros_like(mean_abs)
    ranked = sorted(
        [{"feature": c, "mean_abs_shap": round(float(m), 6), "share": round(float(s), 4)}
         for c, m, s in zip(cols, mean_abs, shares)],
        key=lambda t: -t["mean_abs_shap"],
    )

    findings = []
    for r in ranked:
        if r["share"] >= LEAK_SHARE:
            findings.append({
                "type": "leak", "feature": r["feature"],
                "detail": "SHAP 重要性佔比 %.1f%% >= %.0f%%，疑似獨大/作弊特徵"
                          % (r["share"] * 100, LEAK_SHARE * 100),
                "action": "exclude",
            })
    for v in _direction_findings(val_df, cols, sv):
        findings.append({
            "type": "direction", "feature": v["feature"],
            "detail": "SHAP 方向違反領域直覺（預期 %+d，實際 rho=%+.3f, p=%.4f）"
                      % (v["hint"], v["rho"], v["p_value"]),
            "action": "monotone", "hint": v["hint"], "rho": v["rho"], "p_value": v["p_value"],
        })
    for r in ranked[:12]:  # 只檢查前 12 大特徵的依賴曲線（其餘貢獻太小，鋸齒無意義）
        oi = _oscillation(
            pd.to_numeric(X[r["feature"]], errors="coerce").to_numpy(dtype="float64"),
            sv[:, cols.index(r["feature"])],
        )
        if oi is not None and oi >= JAGGED_OI:
            findings.append({
                "type": "jagged", "feature": r["feature"],
                "detail": "依賴曲線震盪指數 %.2f >= %.1f，模型可能在硬背噪訊"
                          % (oi, JAGGED_OI),
                "action": "transform",
            })

    return {
        "method": "TreeSHAP + 方向健檢 + 依賴曲線震盪指數",
        "n_samples": int(len(val_df)),
        "n_features": len(cols),
        "importance": ranked,
        "findings": findings,
    }


TFT_TIME_BUDGET = 900  # TFT 置換重要性總時間預算（秒），超過就誠實標記未測完的特徵


def diagnose_tft(run, time_budget=TFT_TIME_BUDGET):
    """
    TFT 診斷：PyTorch 序列模型吃不了 TreeExplainer，改用「置換重要性」當等價替代——
    把驗證段某特徵的值逐股打亂（保留分布、破壞與標籤的對應），看 AUC 掉多少：
    掉越多＝模型越依賴該特徵。評估窗與訓練端同一套組法（訓練尾歷史 + 驗證段）。
    每股各自打亂（不跨股票混合），避免製造不存在的跨股特徵值。
    """
    import time as _time
    import joblib

    bundle = joblib.load(run.model_file_path)
    model = bundle["model"]
    cols = [str(c) for c in bundle.get("feature_columns", [])]
    if not cols:
        raise RuntimeError("模型檔缺少 feature_columns，無法診斷")

    stocks = _stock_list(run)
    tr_df, _, _ = _load_dataset(run.train_start, run.train_end, stocks, run.label_version)
    va_df, _, _ = _load_dataset(run.validation_start, run.validation_end, stocks, run.label_version)
    tr_df = tr_df.dropna(subset=cols)
    va_df = va_df.dropna(subset=cols)
    if tr_df.empty or va_df.empty:
        raise RuntimeError("訓練尾／驗證區間組不出完整特徵的資料，無法組視窗")

    # 與 run_training 完全相同的組窗法：訓練尾當歷史（_pos 負數）、驗證段為評估對象（_pos>=0）
    meta = ["stock_code", "trade_date", "label"]
    _tr = tr_df[meta + cols].copy()
    _va = va_df[meta + cols].copy()
    _tr["_pos"] = -1 - np.arange(len(_tr))
    _va["_pos"] = np.arange(len(_va))
    src = pd.concat([_tr, _va], ignore_index=True).sort_values(["stock_code", "trade_date"])
    y_val = _va.sort_values("_pos")["label"].to_numpy()
    val_pos_mask = src["_pos"] >= 0

    base_proba = model.predict_proba(src)[:, 1]
    base_auc = _safe_auc(y_val, base_proba[val_pos_mask.to_numpy()])
    if base_auc is None:
        raise RuntimeError("驗證段只有單一類別，AUC 無法定義，置換重要性不成立")

    # 逐股的驗證段列索引（打亂只在股內進行）
    val_idx_by_stock = {
        code: g.index.to_numpy()
        for code, g in src[val_pos_mask].groupby("stock_code")
    }

    rows, skipped = [], []
    t0 = _time.time()
    for i, c in enumerate(cols):
        elapsed = _time.time() - t0
        est = elapsed / i if i else 0.0
        if i and est * (len(cols) - i) > time_budget:
            skipped = cols[i:]
            break
        rng = np.random.RandomState(42 + i)
        dfp = src.copy()
        for idx in val_idx_by_stock.values():
            dfp.loc[idx, c] = rng.permutation(dfp.loc[idx, c].to_numpy())
        proba = model.predict_proba(dfp)[:, 1]
        auc_perm = _safe_auc(y_val, proba[val_pos_mask.to_numpy()])
        drop = (base_auc - auc_perm) if auc_perm is not None else 0.0
        rows.append({"feature": c, "auc_drop": round(max(0.0, float(drop)), 6)})

    total_drop = sum(r["auc_drop"] for r in rows)
    for r in rows:
        r["share"] = round(r["auc_drop"] / total_drop, 4) if total_drop > 1e-12 else 0.0
    rows.sort(key=lambda t: -t["auc_drop"])

    findings = []
    for r in rows:
        if r["share"] >= LEAK_SHARE:
            findings.append({
                "type": "leak", "feature": r["feature"],
                "detail": "置換重要性佔比 %.1f%% >= %.0f%%，疑似獨大/作弊特徵"
                          % (r["share"] * 100, LEAK_SHARE * 100),
                "action": "exclude",
            })
    return {
        "method": "逐股置換重要性（TFT 無 TreeSHAP，等價替代）",
        "n_samples": int(val_pos_mask.sum()),
        "n_features": len(cols),
        "base_auc": round(float(base_auc), 4),
        "tested_features": len(rows),
        "skipped_features": skipped,
        "time_budget_seconds": int(time_budget),
        "importance": rows,
        "findings": findings,
    }


# ==================== Section C：修正組裝 → 重訓 → 新舊 AUC 對比 ====================

def build_correction(diag, auto=False, exclude=None, mono=None, mono_cap=8):
    """
    把診斷 findings 組裝成可執行的修正參數（交給 run_training 的控制旗標）：
        exclude_features＝剔除作弊/獨大特徵（findings 中 type=leak 的 action）
        monotone_constraints＝{特徵: +1/-1}（findings 中 type=direction 的 action）
        jagged（type=transform）無法自動修——只給人工建議（分箱/平滑/對數）。
    auto=True：全部照 findings 自動套用（單調約束最多 mono_cap 個，依 |rho| 取最顯著）。
    手動模式：--exclude "a,b" 與 --mono "f:+1,g:-1"，會與 auto 結果合併並驗證特徵名。
    """
    known = {r["feature"] for r in diag.get("importance", [])}
    excl, mono_map, manual_steps = [], {}, []
    if auto:
        for f_ in diag.get("findings", []):
            if f_["type"] == "leak" and f_["feature"] not in excl:
                excl.append(f_["feature"])
            elif f_["type"] == "direction":
                mono_map[f_["feature"]] = int(f_["hint"])
            elif f_["type"] == "jagged":
                manual_steps.append(
                    "特徵 %s 依賴曲線鋸齒狀（無法自動修）：建議分箱／平滑／對數轉換後重算特徵"
                    % f_["feature"]
                )
        if len(mono_map) > mono_cap:  # 依 |rho| 由大到小保留最顯著的前 mono_cap 個
            ranked = sorted(
                (f_ for f_ in diag.get("findings", []) if f_["type"] == "direction"),
                key=lambda f_: -abs(f_.get("rho", 0.0)),
            )
            mono_map = {f_["feature"]: int(f_["hint"]) for f_ in ranked[:mono_cap]}
    for raw in (exclude or []):
        if raw not in known:
            raise ValueError("--exclude 的特徵不在模型特徵清單內：%s" % raw)
        if raw not in excl:
            excl.append(raw)
    for feat, sgn in (mono or {}).items():
        if feat not in known:
            raise ValueError("--mono 的特徵不在模型特徵清單內：%s" % feat)
        if sgn not in (1, -1):
            raise ValueError("--mono 方向只能是 +1 或 -1：%s" % feat)
        mono_map[feat] = int(sgn)
    return {
        "exclude_features": sorted(excl),
        "monotone_constraints": {k: mono_map[k] for k in sorted(mono_map)},
        "manual_steps": manual_steps,
        "auto": bool(auto),
    }


def _fmt_auc(run):
    if run.auc is None:
        return "N/A"
    if run.auc_std:
        return f"{run.auc:.4f} ± {run.auc_std:.4f}"
    return f"{run.auc:.4f}"


def compare_runs(old_run, new_run=None):
    """
    情況 A／B 分析（MODEL_TUNING_GUIDE.md「SHAP 誤導特徵診斷」）：
        情況 A：AUC 略降（顯著）→ 成功移除過擬合/資料洩漏，舊分數是虛胖，新模型線上更安全
        情況 B：AUC 顯著提升 → 去噪成功，分類能力本質性變強
        中性：變化落在 ±（舊標準差＋新標準差）內，不足以判斷論。
    只回傳資料，不自己做決定；「significant」的判定用兩次訓練標準差總合當顯著門檻。
    """
    out = {
        "old_run_id": old_run.id,
        "old_model_version": old_run.model_version,
        "old_auc": old_run.auc,
        "old_auc_std": old_run.auc_std,
        "old_train_window": f"{old_run.train_start}~{old_run.train_end}",
        "old_val_window": f"{old_run.validation_start}~{old_run.validation_end}",
        "status": "diagnosed_only",
    }
    if new_run is None:
        return out
    new_auc, new_std = new_run.auc, new_run.auc_std
    out.update({
        "new_run_id": new_run.id,
        "new_model_version": new_run.model_version,
        "new_auc": new_auc,
        "new_auc_std": new_std,
        "new_train_window": f"{new_run.train_start}~{new_run.train_end}",
        "new_val_window": f"{new_run.validation_start}~{new_run.validation_end}",
    })
    if old_run.auc is None or new_auc is None:
        out["status"] = "auc_na"
        out["case"] = "unknown"
        out["case_readable"] = "新舊 AUC 任一為 None（驗證集單一類別），無法對比"
        out["delta_auc"] = None
        out["significant"] = False
        return out
    delta = new_auc - old_run.auc
    sig = abs(delta) > ((old_run.auc_std or 0.0) + (new_std or 0.0) + 1e-9)
    case = "B" if (delta >= 0 and sig) else ("A" if (delta < 0 and sig) else "neutral")
    out.update({
        "delta_auc": round(float(delta), 4),
        "significant": bool(sig),
        "case": case,
        "case_readable": {
            "A": "情況 A：AUC 顯著下降 → 移除誤導特徵後模型更安全"
                 "（舊高 AUC 多來自過擬合/資料洩漏，新模型線下稍低但線上更穩）",
            "B": "情況 B：AUC 顯著提升 → 去噪成功，模型分類能力實際變強",
            "neutral": "變動不顯著（在 ±標準差範圍內），需更多證據再下結論",
        }[case],
        "status": "compared",
    })
    return out


def run_diagnosis(model_type=None, model_version=None, stock_code="",
                  auto=False, exclude=None, mono=None, retrain=False,
                  new_version=None, time_budget=TFT_TIME_BUDGET, mono_cap=8):
    """
    黃金流程主流程（management/commands/diagnose_model.py 呼叫）：
        1. pick_diag_run 挑一顆已完成模型
        2. 樹模型→TreeSHAP；TFT→置換重要性，抓三類誤導證據
        3. build_correction 組修正參數（exclude / monotone / manual）
        4. （可選）建新 run、帶修正旗標重訓（沿用相同區間/股票/標籤）
        5. compare_runs 新舊 AUC 對比 → 整份報告存回 ModelTrainingRun.diagnostics
    回傳 (report_dict, error_msg)；error_msg 為 None 表示成功。
    """
    run, err = pick_diag_run(
        model_type=model_type, model_version=model_version, stock_code=stock_code,
    )
    if err:
        return None, err
    diag = (
        diagnose_trees(run)
        if run.model_type in ("lightgbm", "xgboost")
        else diagnose_tft(run, time_budget=time_budget)
    )
    try:
        correction = build_correction(diag, auto=auto, exclude=exclude, mono=mono, mono_cap=mono_cap)
    except ValueError as exc:
        return None, str(exc)

    report = {
        "run_id": run.id,
        "model_type": run.model_type,
        "model_version": run.model_version,
        "train_window": f"{run.train_start}~{run.train_end}",
        "val_window": f"{run.validation_start}~{run.validation_end}",
        "label_version": run.label_version,
        "stock_codes": run.stock_codes or "全部股票",
        "diagnosed_at": timezone.now().isoformat(),
        "evaluation": "驗證區間（非盲測）：目的是檢查模型『學到了什麼』，不代表未來表現。",
        "diagnosis": diag,
    }

    new_run = None
    if retrain and (correction["exclude_features"] or correction["monotone_constraints"]):
        hps = {k: v for k, v in (run.hyperparams or {}).items()}
        hps.pop("refit_on_all", None)  # 診斷重訓＝一般切分，不做全量重擬合
        hps["exclude_features"] = correction["exclude_features"]
        if correction["monotone_constraints"]:
            hps["monotone_constraints"] = correction["monotone_constraints"]
        new_run = ModelTrainingRun.objects.create(
            train_start=run.train_start, train_end=run.train_end,
            validation_start=run.validation_start, validation_end=run.validation_end,
            model_type=run.model_type,
            model_version=new_version or f"{run.model_version}_corrected",
            stock_codes=run.stock_codes, label_version=run.label_version,
            hyperparams=hps, n_repeats=max(run.n_repeats or 1, 1),
            status="pending",
        )
        result = run_training(new_run.id)
        if not result["success"]:
            return None, f"重訓失敗：{result['message']}"
        new_run.refresh_from_db()

    comparison = compare_runs(run, new_run)
    report["correction"] = {
        "auto": correction["auto"],
        "exclude_features": correction["exclude_features"],
        "monotone_constraints": correction["monotone_constraints"],
        "manual_steps": correction["manual_steps"],
    }
    report["comparison"] = comparison

    run.diagnostics = report
    run.save(update_fields=["diagnostics"])
    if new_run is not None:
        new_run.diagnostics = report
        new_run.save(update_fields=["diagnostics"])
    return report, None


# ==================== Section D：訓練流程自動診斷 ====================

# 訓練流程內建診斷的 TFT 時間預算（秒）：手動 diagnose_model 上限 900 秒，
# 自動診斷只給 60 秒（置換重要性量力而為，超過就誠實標記未測完的特徵），
# 避免拖慢訓練太多；建立訓練時可用超參數 shap_time_budget 覆蓋。
AUTO_TFT_TIME_BUDGET = 60


def attach_auto_diagnosis(run, time_budget=None):
    """
    訓練流程自動診斷（model_training.run_training 完成後呼叫）：
    對剛訓練完成的 run 跑 SHAP（樹模型 TreeSHAP）／逐股置換重要性（TFT），
    把診斷報告（三類誤導證據＋SHAP 重要性＋修正建議）存進 run.diagnostics。

    與手動 run_diagnosis 的差異：不重訓、不做新舊 AUC 對比（沒有「修正前」可比），
    comparison 固定 status=diagnosed_only（前端會顯示「僅診斷、未重訓」）。

    回傳 None 表示成功；回傳錯誤訊息字串表示失敗（不寫入 diagnostics，
    呼叫端只把失敗原因記在 run.notes，訓練結果不受影響）。
    """
    if run.model_type not in ("lightgbm", "xgboost", "tft"):
        return f"未知模型類型 {run.model_type}，跳過自動診斷"
    if not run.model_file_path or not os.path.exists(run.model_file_path):
        return "模型檔不存在，無法自動診斷"
    try:
        diag = (
            diagnose_trees(run)
            if run.model_type in ("lightgbm", "xgboost")
            else diagnose_tft(run, time_budget=time_budget or AUTO_TFT_TIME_BUDGET)
        )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__} - {exc}"
    try:
        correction = build_correction(diag, auto=False)
    except ValueError:
        # build_correction 只在手動參數錯誤時 raise；保險起見失敗就給空建議
        correction = {"auto": False, "exclude_features": [],
                      "monotone_constraints": {}, "manual_steps": []}
    report = {
        "run_id": run.id,
        "model_type": run.model_type,
        "model_version": run.model_version,
        "train_window": f"{run.train_start}~{run.train_end}",
        "val_window": f"{run.validation_start}~{run.validation_end}",
        "label_version": run.label_version,
        "stock_codes": run.stock_codes or "全部股票",
        "diagnosed_at": timezone.now().isoformat(),
        "trigger": "auto（訓練完成後自動執行；超參數 auto_shap=false 可關閉）",
        "evaluation": "驗證區間（非盲測）：目的是檢查模型『學到了什麼』，不代表未來表現。",
        "diagnosis": diag,
        "correction": {
            "auto": correction["auto"],
            "exclude_features": correction["exclude_features"],
            "monotone_constraints": correction["monotone_constraints"],
            "manual_steps": correction["manual_steps"],
        },
        "comparison": compare_runs(run, None),
    }
    run.diagnostics = report
    run.save(update_fields=["diagnostics"])
    return None

# === [APPEND] ===
