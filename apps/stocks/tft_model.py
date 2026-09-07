"""
TFT（Temporal Fusion Transformer）模型實作：輕量版、純 PyTorch。

設計說明：
- 目的與 LightGBM / XGBoost 相同：吃同一組日頻特徵（StockFeatureDaily + 月頻營收），
  對「標籤日」做二元分類（該日之後 Return20 是否超過門檻），輸出突破機率。
- 與樹模型的根本差異：TFT 是序列模型，每個樣本是「最近 seq_len 個交易日的特徵視窗」，
  而不是單日一行。因此訓練/推論端都會以股票為單位重建時間窗
  （由 run_training 及 services.predict 分別負責備妥帶 stock_code / trade_date 的框）。
- 架構（Bryan Lim et al., 2021 的精簡版，本專案未使用靜態協變數，故省略
  Static Covariate Encoders；股票間差異交由「跨股票混合訓練」天然處理）：
    1. Variable Selection Network（VSN）：學習每個時間步哪些特徵重要，軟性選擇特徵。
    2. LSTM 編碼器：擷取視窗內的時序動態（多層，層間 dropout）。
    3. Gated Residual Network（GRN）：非線性轉換 + GLU 閘門 + LayerNorm 跳躍連接，
       可以學到近似恆等映射（深而穩定）。
    4. 多頭自我注意力：讓視窗內任意兩天直接互動（捕捉長距離依賴）。
    5. 輸出頭：GRN + GLU 輸出閘門 + Linear -> logit（二元分類，sigmoid 即機率）。
- 不平衡標籤：BCEWithLogitsLoss(pos_weight=neg/pos)，對應樹模型的 is_unbalance /
  scale_pos_weight，不需 SMOTE。
- 早停：以驗證集 AUC 為準（兩類都存在時）；否則退回驗證 loss；無驗證集則跑滿 epochs
  （「訓練納入最新資料」的最終重擬合走這條路）。
- 缺值處理：訓練期統計每個特徵的平均值（NaN 忽略），推論時 NaN 先補該平均值再標準化；
  常數特徵的 std 保底為 1，避免除以零。
"""
import random
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TFT 需要 PyTorch。請先安裝（CPU 版即可）："
        "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc


class _GRN(nn.Module):
    """Gated Residual Network：ELU 非線性 + GLU 閘門 + LayerNorm 跳躍連接。"""

    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.skip = nn.Linear(dim, dim)  # 同維跳躍也過一層線性，初期更穩定
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.elu(self.fc1(x)))
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(h * g + self.skip(x))


class _VSN(nn.Module):
    """
    Variable Selection Network：對每個時間步的特徵向量做「軟性特徵選擇」。
    - sel：把展平特徵映射成各特徵的選擇權重（softmax 後總和為 1）
    - proj[i]：第 i 個特徵（1 維）各自的非線性投影（類似特徵專屬 embedding）
    輸出再經過 grn_out（其內建 LayerNorm 跳躍連接）穩定梯度。
    """

    def __init__(self, n_feat, d_model, dropout):
        super().__init__()
        self.n_feat = n_feat
        self.proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_feat)])
        self.sel = nn.Linear(n_feat, n_feat)
        self.grn_out = _GRN(d_model, dropout)

    def forward(self, x):                                  # x: (B, T, F)
        bsz, seq_len, n_f = x.shape
        flat = x.reshape(bsz * seq_len, n_f)
        w = torch.softmax(self.sel(flat), dim=-1)          # (B*T, F) 特徵權重
        h = torch.zeros(bsz * seq_len, self.proj[0].out_features, device=x.device)
        for i in range(n_f):
            h = h + w[:, i:i + 1] * F.elu(self.proj[i](flat[:, i:i + 1]))
        return self.grn_out(h).view(bsz, seq_len, -1)


class _TFTBackbone(nn.Module):
    """TFT 主幹：VSN -> LSTM(多層) -> 全局情境加成 -> 自注意力 -> GRN/GLU -> logit。"""

    def __init__(self, n_feat, d_model, n_heads, num_layers, dropout):
        super().__init__()
        self.vsn = _VSN(n_feat, d_model, dropout)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 全局可學習情境向量（替代 TFT 的 static enrichment，因本專案無靜態協變數）
        self.context = nn.Parameter(torch.zeros(d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.post = _GRN(d_model, dropout)
        self.head_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):                                  # x: (B, T, F)
        v = self.vsn(x)
        h, _ = self.lstm(v)
        h = h + self.context
        a, _ = self.attn(h, h, h, need_weights=False)
        g = self.post(a)
        g = g * torch.sigmoid(self.head_gate(g))           # GLU 輸出閘門
        return self.out(g[:, -1, :]).squeeze(-1)           # 只取最後一天 -> logit


class TFTClassifier:
    """
    sklearn 風格包裝器，讓 run_training / services.predict 幾乎不用改：

    - fit_frame(train_df, val_df, feature_cols)：
      train_df / val_df 需含 stock_code、trade_date 欄（_load_dataset 的輸出本來就有）。
      內部逐股重建「連續 seq_len 日視窗」，視窗標籤＝最後一天的 label。
      val_df=None 表示不做早停、跑滿 epochs（用於最終重擬合）。
    - predict_proba(X_df)：X_df 為時間正序的特徵框。
      * 含 stock_code 欄 -> 逐股分組各算各的視窗（run_training 批次驗證用）
      * 不含 stock_code 欄 -> 整框視為單一股票的連續序列
        （compute_history_signals / services.predict 的推論端就是單股正序框）
      不足 seq_len 列的前段無法成窗，回傳 base_rate_（訓練期正樣本率）中性機率，
      讓回傳長度永遠等於輸入列數，呼叫端不需特判。
    - requires_sequence = True：services.predict 載入 bundle 後檢查此旗標，
      改送入最近 seq_len 日完整特徵視窗而非單列。
    """

    requires_sequence = True

    def __init__(self, seq_len=30, d_model=32, n_heads=4, num_layers=2,
                 dropout=0.1, learning_rate=0.003, epochs=30, batch_size=128,
                 random_state=42, early_stopping_patience=5, max_seconds=None):
        self.seq_len = max(2, int(seq_len))
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.early_stopping_patience = int(early_stopping_patience)
        # 訓練時間上限（秒）：None＝不限；超過就在 epoch 邊界提前結束（MODEL_TUNING_GUIDE.md 六-2）
        self.max_seconds = max_seconds
        # fit 後填入
        self.feature_columns_ = None
        self.model_ = None
        self.classes_ = [0, 1]
        self.base_rate_ = 0.5
        self.scaler_mean_ = None     # np.ndarray (F,)
        self.scaler_scale_ = None    # np.ndarray (F,)
        self.scaler_fill_ = None     # NaN 補值用的欄平均
        self.train_windows_ = 0

    # ---------------- 內部工具 ----------------
    def _iter_series(self, df):
        """
        把輸入框切成一個個「單一股票、時間正序」的子序列。
        yield (絕對位置索引 ndarray, 子 DataFrame)。推論端框架本來就正序且同股連續，
        這裡不再重排，僅分組；同股不連續的罕見情況由呼叫端保證。
        """
        if "stock_code" in df.columns:
            for _, grp in df.groupby("stock_code", sort=False):
                yield grp.index.to_numpy(), grp
        else:
            yield df.index.to_numpy(), df

    def _windows_from(self, sub_df, with_label=False):
        """從時間正序的單股子框取所有滑動視窗；回傳 (wins, end_positions[, labels])。"""
        s = self.seq_len
        if len(sub_df) < s:
            return None, None, None
        arr = sub_df[self.feature_columns_].to_numpy(dtype="float64")
        ends = np.arange(s - 1, len(arr))
        wins = np.stack([arr[i - s + 1: i + 1] for i in ends])
        labels = None
        if with_label:
            labels = sub_df["label"].to_numpy(dtype="float32")[ends]
        return wins, ends, labels

    def _prep(self, wins):
        """NaN 補欄平均 -> 標準化。wins: (N, T, F) float64 -> float32 tensor。"""
        z = np.where(np.isnan(wins), self.scaler_fill_, wins)
        z = (z - self.scaler_mean_) / self.scaler_scale_
        return torch.from_numpy(z.astype("float32"))

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ---------------- 訓練 ----------------
    def fit_frame(self, train_df, val_df, feature_cols):
        self.feature_columns_ = list(feature_cols)
        self._seed_everything(self.random_state)

        def _collect_wins(df):
            wins_l, labels_l = [], []
            if "stock_code" in df.columns:
                grouped = df.sort_values(
                    ["stock_code", "trade_date"]).groupby("stock_code", sort=False)
            else:
                grouped = [(None, df)]
            for _, sub in grouped:
                w, _, lab = self._windows_from(sub, with_label=True)
                if w is not None:
                    wins_l.append(w)
                    labels_l.append(lab)
            return wins_l, labels_l

        train_wins, train_labels = _collect_wins(train_df)
        if not train_wins:
            raise ValueError(
                f"TFT 需要每檔股票至少 {self.seq_len} 個交易日的連續特徵才能切視窗，"
                f"目前訓練區間資料不足"
            )
        wins_np = np.concatenate(train_wins, axis=0)          # (N, T, F)
        y_np = np.concatenate(train_labels, axis=0)           # (N,)
        self.train_windows_ = int(len(y_np))

        # 標準化統計量（只看訓練窗，忽略 NaN，常數欄 std 保底）
        flat = wins_np.reshape(-1, len(feature_cols))
        mean = np.nanmean(flat, axis=0)
        scale = np.nanstd(flat, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        fill = np.where(np.isnan(mean), 0.0, mean)
        self.scaler_mean_, self.scaler_scale_, self.scaler_fill_ = mean, scale, fill

        Xtr = self._prep(wins_np)
        ytr = torch.from_numpy((y_np > 0.5).astype("float32"))

        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        pos_w = torch.tensor([n_neg / n_pos]) if n_pos > 0 else None  # 不平衡標籤
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self.model_ = _TFTBackbone(
            n_feat=len(feature_cols), d_model=self.d_model, n_heads=self.n_heads,
            num_layers=self.num_layers, dropout=self.dropout,
        )
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)

        # 驗證集視窗（可為空：refit_on_all 最終重擬合時就不做早停、跑滿 epochs）
        val_X = val_y = None
        if val_df is not None:
            vw, vl = _collect_wins(val_df)
            if vw:
                val_X = self._prep(np.concatenate(vw, axis=0))
                val_y = np.concatenate(vl, axis=0)

        best_score, best_state, bad_epochs = -np.inf, None, 0
        g = torch.Generator().manual_seed(self.random_state)
        t0 = time.time()
        for _epoch in range(max(1, self.epochs)):
            # 訓練時間上限：超過就提前結束——有驗證集時用目前最佳狀態，無驗證集（refit）時用當下權重
            if self.max_seconds is not None and (time.time() - t0) > self.max_seconds:
                break
            self.model_.train()
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(Xtr), self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            if val_X is not None:
                # ---- 早停評估：以驗證 AUC 為主；驗證集只有一類時退回負 logloss ----
                from sklearn.metrics import roc_auc_score
                self.model_.eval()
                with torch.no_grad():
                    logits = self.model_(val_X).numpy()
                both = len(set(val_y.tolist())) > 1
                score = (roc_auc_score(val_y, logits) if both else
                         -float(np.mean(np.logaddexp(0.0, logits) - val_y * logits)))
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in
                                  self.model_.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs > self.early_stopping_patience:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.base_rate_ = float(n_pos / max(1.0, n_pos + n_neg))
        return self

    # ---------------- 推論 ----------------
    def predict_proba(self, X_df):
        """
        回傳 shape=(len(X_df), 2) 的機率矩陣（欄位順序對齊 classes_）。
        每一列的機率來自「該列往前 seq_len 天」的特徵視窗；
        該股不足 seq_len 列的前段回傳 base_rate_ 中性機率。
        """
        if getattr(self, "model_", None) is None:
            raise RuntimeError("TFT 尚未訓練完成，不能直接推論")
        out = np.full(len(X_df), self.base_rate_, dtype="float64")
        for positions, sub in self._iter_series(X_df):
            wins, ends, _ = self._windows_from(sub)
            if wins is None:
                continue
            with torch.no_grad():
                p = torch.sigmoid(self.model_(self._prep(wins))).numpy()
            out[positions[ends]] = p
        return np.column_stack([1.0 - out, out])

    def predict(self, X_df):
        return (self.predict_proba(X_df)[:, 1] >= 0.5).astype(int)


