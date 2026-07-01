"""
Gate tests for model_lstm.py (Phase 3, Model B).
The sequence-construction tests are the security layer: they prove the two
leakage vectors specific to a sequence model can't happen. They run without
torch; the training smoke test skips until torch is installed.
"""

import numpy as np
import pandas as pd
import pytest

import config
import model_lstm

FEAT = config.LEAN_FEATURES

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _panel(n_a=120, n_b=90):
    # two tickers, so we can prove sequences never mix them
    rng = np.random.default_rng(0)

    def one(tkr, n):
        return pd.DataFrame({
            "date": pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"),
            "ticker": tkr,
            **{c: rng.normal(size=n).astype("float32") for c in FEAT},
            "Target_Label": rng.integers(0, 2, n).astype("int8"),
        })

    return pd.concat([one("AAA", n_a), one("BBB", n_b)], ignore_index=True)


def test_sequences_are_per_ticker():
    # count must equal the per-ticker window counts - a cross-ticker slide would exceed it
    seq = 10
    X, y, dts, tks = model_lstm.make_sequences(_panel(120, 90), FEAT, seq_length=seq)
    assert len(X) == (120 - seq + 1) + (90 - seq + 1)
    assert X.shape[1] == seq and X.shape[2] == len(FEAT)
    assert len(X) == len(y) == len(dts) == len(tks)


def test_no_lookahead_in_sequences():
    # a window's contents must not change when future rows are deleted
    seq = 10
    panel = _panel(120, 90)
    full = model_lstm.make_sequences(panel, FEAT, seq_length=seq)
    target_date = full[2][40]                       # a window end date in ticker AAA
    part = model_lstm.make_sequences(panel[panel["date"] <= target_date], FEAT, seq_length=seq)
    fi = np.where((full[2] == target_date) & (full[3] == "AAA"))[0][0]
    pi = np.where((part[2] == target_date) & (part[3] == "AAA"))[0][0]
    assert np.allclose(full[0][fi], part[0][pi])


def test_scaler_fits_train_only():
    # scaler stats come from train rows only; later test rows may scale outside [0, 1]
    panel = _panel(200, 200)
    train = panel[panel["date"] < "2020-06-01"]
    scaler = model_lstm.fit_scaler(train, FEAT)
    scaled_train = model_lstm.apply_scaler(train, FEAT, scaler)[FEAT].to_numpy()
    scaled_full = model_lstm.apply_scaler(panel, FEAT, scaler)[FEAT].to_numpy()
    assert scaled_train.min() >= -1e-6 and scaled_train.max() <= 1 + 1e-6
    assert scaled_full.max() > 1 + 1e-9 or scaled_full.min() < -1e-9   # test rows unconstrained


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_train_predict_smoke():
    # learnable sequence signal -> held-out AUC beats random; probs in [0, 1]
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    n, seq, f = 2000, 20, len(FEAT)
    X = rng.normal(size=(n, seq, f)).astype("float32")
    y = (X[:, -1, 0] > 0).astype("int8")                    # label = sign of last step, feature 0
    model = model_lstm.train_lstm(X[:1500], y[:1500], epochs=8, batch_size=256, device="cpu")
    p = model_lstm.predict_lstm(model, X[1500:], device="cpu")
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert roc_auc_score(y[1500:], p) > 0.7
