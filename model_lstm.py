"""
--------------------------------------------
SIGNAL PIPELINE - MODEL B: CNN-LSTM (PyTorch)
--------------------------------------------

A 1D-CNN -> LSTM classifier on 60-day sequences of the LEAN feature set.
Captures temporal / path patterns the tabular model can't see.

PyTorch (not TensorFlow) on purpose: native Windows CUDA and proper RTX 5070
(Blackwell) support, so no more CuDNN-LSTM bypass hacks.

Outputs a win probability (0..1) per (ticker, date).
"""

import numpy as np
import pandas as pd

import config


def build_sequences(df: pd.DataFrame,
                    feature_cols: list[str],
                    seq_length: int = config.SEQ_LENGTH):
    """
    Slide a window of seq_length days over the feature frame.
    Returns (X, y): X shape (n, seq_length, n_features); y = label at each window end.
    """
    raise NotImplementedError


def build_model(n_features: int):
    """
    Build the CNN-LSTM:
        Conv1d -> BatchNorm -> MaxPool -> Dropout
        -> LSTM -> Dropout -> Linear -> sigmoid
    (PyTorch port of the FYP Keras architecture.)
    """
    import torch                # lazy import - module stays importable before torch is installed
    import torch.nn as nn
    raise NotImplementedError


def train_lstm(X_train, y_train, X_val, y_val, class_weights=None):
    """Train the network (Adam, BCE loss, early stopping). Returns the fitted model."""
    import torch
    raise NotImplementedError


def predict_lstm(model, X) -> np.ndarray:
    """Return win-probability (0..1) for each sequence."""
    raise NotImplementedError
