"""Final prediction script for the MLP. Uses only numpy and pandas.

Loads the weights and fitted preprocessor state from mlp_model.npz.

Usage:
    from pred import predict_all
    predict_all("example_test_set.csv")
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features  # noqa: E402

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp_model.npz")
_cache = None


def _load():
    global _cache
    if _cache is None:
        npz = np.load(_PATH, allow_pickle=False)
        s = json.loads(str(npz["prep_json"]))
        prep = features.Preprocessor(min_df=s["min_df"], max_vocab=s["max_vocab"],
                                     top_colours=s["top_colours"],
                                     text_fields=tuple(s["text_fields"]),
                                     use_tfidf=s["use_tfidf"])
        prep.medians = s["medians"]
        prep.vocab = s["vocab"]
        prep.idf = {k: np.array(v) for k, v in s["idf"].items()}
        prep.colours = s["colours"]
        prep.pay_cap = s["pay_cap"]
        prep.names = s["names"]
        prep.mean_ = npz["prep_mean"]
        prep.std_ = npz["prep_std"]
        n = int(npz["n_layers"])
        layers = [(npz[f"W{i}"], npz[f"b{i}"]) for i in range(n)]
        classes = [str(c) for c in npz["classes"]]
        _cache = (prep, layers, classes)
    return _cache


def logits_df(df):
    prep, layers, _ = _load()
    h = prep.transform(df)
    for i, (W, b) in enumerate(layers):
        h = h @ W.T + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)     # ReLU (dropout is off at prediction time)
    return h


def predict_df(df):
    _, _, classes = _load()
    return [classes[i] for i in np.argmax(logits_df(df), axis=1)]


def predict(row):
    return predict_df(pd.DataFrame([row]))[0]


def predict_all(filename):
    """Make predictions for every row in the CSV file `filename`."""
    return predict_df(pd.read_csv(filename))


if __name__ == "__main__":
    print(predict_all(sys.argv[1])[:10])
