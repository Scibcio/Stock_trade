"""
--------------------------------------------
SIGNAL PIPELINE - MODEL B: CNN-LSTM (PyTorch)
--------------------------------------------

A 1D-CNN -> LSTM classifier on 60-day sequences of the LEAN feature set
(config.LEAN_FEATURES). Captures temporal/path patterns the tabular XGBoost can't.
PyTorch (native Windows CUDA) for the RTX 5070.

Anti-leakage discipline (same rigor as Model A):
  - sequences are built PER TICKER (a window never spans two stocks)
  - the MinMaxScaler is fit on the TRAIN fold ONLY, then applied to test
  - windows are assigned to train/test by their END date (+ purge), so no
    training label straddles the boundary
Outputs an (uncalibrated) win probability per row; ensemble.py calibrates.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import config

SEED = 42


# ----------------------------------
# SEQUENCE BUILDING  (torch-free, testable)
# ----------------------------------

def make_sequences(df: pd.DataFrame,
                   feature_cols: list,
                   seq_length: int = config.SEQ_LENGTH):
    """
    Slide a seq_length window over each ticker SEPARATELY (never across tickers).
    Each window is labelled/dated by its LAST day. Returns
    (X[n, seq, n_feat], y, end_dates, end_tickers).
    """
    fcols = list(feature_cols)
    Xs, ys, dts, tks = [], [], [], []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        if len(g) <= seq_length:
            continue
        feats = g[fcols].to_numpy("float32")
        win = sliding_window_view(feats, seq_length, axis=0).transpose(0, 2, 1)  # (W, seq, F)
        end = slice(seq_length - 1, len(g))
        Xs.append(win)
        ys.append(g["Target_Label"].to_numpy("int8")[end])
        dts.append(g["date"].to_numpy()[end])
        tks.append(np.full(win.shape[0], tkr))
    if not Xs:
        return (np.empty((0, seq_length, len(fcols)), "float32"),
                np.empty(0, "int8"), np.array([]), np.array([]))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(dts), np.concatenate(tks))


def fit_scaler(train_df: pd.DataFrame, feature_cols: list):
    # fit ONLY on train rows - the core anti-leakage step
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler().fit(train_df[list(feature_cols)].to_numpy("float32"))


def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    df = df.copy()
    df[list(feature_cols)] = scaler.transform(df[list(feature_cols)].to_numpy("float32"))
    return df


# ----------------------------------
# NETWORK
# ----------------------------------

def build_model(n_features: int):
    # Conv1d -> BatchNorm -> MaxPool -> Dropout -> LSTM -> Dropout -> Linear
    import torch
    import torch.nn as nn

    class QuantLSTM(nn.Module):
        def __init__(self, n_feat):
            super().__init__()
            self.conv = nn.Conv1d(n_feat, 64, kernel_size=5, padding=2)
            self.bn = nn.BatchNorm1d(64)
            self.pool = nn.MaxPool1d(2)
            self.drop1 = nn.Dropout(0.3)
            self.lstm = nn.LSTM(64, 64, batch_first=True)
            self.drop2 = nn.Dropout(0.3)
            self.head = nn.Linear(64, 1)

        def forward(self, x):                       # x: (B, seq, F)
            z = x.transpose(1, 2)                    # (B, F, seq) for Conv1d
            z = self.pool(torch.relu(self.bn(self.conv(z))))
            z = self.drop1(z).transpose(1, 2)        # (B, seq', 64)
            out, _ = self.lstm(z)
            return self.head(self.drop2(out[:, -1, :])).squeeze(1)   # logits (B,)

    return QuantLSTM(n_features)


# ----------------------------------
# TRAIN / PREDICT
# ----------------------------------

def _device(device=None):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def predict_lstm(model, X, device=None, batch_size=8192) -> np.ndarray:
    import torch
    dev = _device(device)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs) if outs else np.empty(0, "float32")


def train_lstm(X_train, y_train, epochs: int = 15, batch_size: int = 2048,
               device=None, val_frac: float = 0.15):
    """
    Adam + BCEWithLogitsLoss (pos_weight for class imbalance) + early stopping.
    Deterministic seed for reproducibility. Chronological train/val split.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = _device(device)

    cut = int(len(X_train) * (1 - val_frac))
    Xtr = torch.as_tensor(X_train[:cut], dtype=torch.float32)
    ytr = torch.as_tensor(y_train[:cut], dtype=torch.float32)
    Xva, yva = X_train[cut:], y_train[cut:]

    pos = max(float((y_train[:cut] == 1).sum()), 1.0)
    pos_weight = torch.tensor([float((y_train[:cut] == 0).sum()) / pos], device=dev)

    model = build_model(X_train.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    best, best_state, patience, bad = float("inf"), None, 3, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb.to(dev)), yb.to(dev)).backward()
            opt.step()
        vloss = _val_loss(model, Xva, yva, loss_fn, dev)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def _val_loss(model, X, y, loss_fn, dev, batch_size=8192) -> float:
    import torch
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
            yb = torch.as_tensor(y[i:i + batch_size], dtype=torch.float32, device=dev)
            tot += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    return tot / max(n, 1)


# ----------------------------------
# WALK-FORWARD  (mirrors pipeline.walk_forward, LSTM path)
# ----------------------------------

def walk_forward(pooled: pd.DataFrame, epochs: int = 15, device=None):
    from sklearn.metrics import roc_auc_score

    feat = config.LEAN_FEATURES
    oof, metrics = [], []

    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=config.PURGE_DAYS)).isoformat()
        train_df = pooled[pooled["date"] <= cut]
        if train_df.empty:
            continue

        scaler = fit_scaler(train_df, feat)                     # TRAIN only
        X, y, dts, tks = make_sequences(apply_scaler(pooled, feat, scaler), feat)
        tr = dts <= cut
        te = (dts >= f["test_start"]) & (dts <= f["test_end"])
        if tr.sum() == 0 or te.sum() == 0:
            continue

        model = train_lstm(X[tr], y[tr], epochs=epochs, device=device)
        prob = predict_lstm(model, X[te], device=device)

        oof.append(pd.DataFrame({"date": dts[te], "ticker": tks[te],
                                 "Target_Label": y[te], "fold": f["fold"], "p_lstm": prob}))
        auc = roc_auc_score(y[te], prob)
        metrics.append({"fold": f["fold"], "auc": round(float(auc), 4), "test_rows": int(te.sum())})
        print(f"    fold {f['fold']:>2}  test {int(te.sum()):>7,}  AUC {auc:.4f}")

    return pd.concat(oof, ignore_index=True), pd.DataFrame(metrics)
