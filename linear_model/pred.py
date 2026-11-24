"""
Final prediction script for the softmax (multinomial logistic regression)
model. Loads the frozen weights + fitted feature-preprocessor state from
softmax_model.npz and predicts using only numpy/pandas -- no sklearn.

Usage:
    from pred import predict_all
    predict_all("example_test_set.csv")
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402  (path insert above must run first)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "softmax_model.npz")
_state = None


def _load_state():
    global _state
    if _state is None:
        npz = np.load(_MODEL_PATH, allow_pickle=False)
        state = features.npz_kwargs_to_state(npz)
        state["W"] = npz["W"].astype(np.float64)
        state["b"] = npz["b"].astype(np.float64)
        state["classes"] = np.array([str(c) for c in npz["classes"]])
        _state = state
    return _state


def _predict_labels(df):
    state = _load_state()
    X, _ = features.transform_features(df, state)
    logits = X @ state["W"].T + state["b"]
    idx = np.argmax(logits, axis=1)
    return [str(state["classes"][i]) for i in idx]


def predict(row):
    """Predicts the sport for a single row (a dict-like / pandas Series)."""
    return _predict_labels(pd.DataFrame([row]))[0]


def predict_df(df):
    """Makes predictions for every row in an already-loaded DataFrame."""
    return _predict_labels(df)


def predict_all(filename):
    """Makes predictions for every row in the CSV file named by filename."""
    df = pd.read_csv(filename)
    return predict_df(df)


if __name__ == "__main__":
    print(predict_all(sys.argv[1])[:10])
