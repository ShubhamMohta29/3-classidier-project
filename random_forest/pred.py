"""Load the exported random forest and predict in CSV row order."""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from . import features
else:
    import features

MODEL_PATH = Path(__file__).resolve().with_name("forest_model.npz")


def project_text(X, components):
    """Keep structured columns and optionally reduce the word features."""
    if components.shape[1] == 0:
        return X
    n_structured = X.shape[1] - components.shape[1]
    return np.column_stack((X[:, :n_structured], X[:, n_structured:] @ components.T))


@lru_cache(maxsize=1)
def _load_model():
    with np.load(MODEL_PATH, allow_pickle=False) as model:
        state = features.npz_kwargs_to_state(model)
        arrays = {key: model[key] for key in (
            "classes", "components", "roots", "left", "right",
            "feature", "threshold", "probabilities",
        )}
    return state, arrays


def _forest_probabilities(X, model):
    # sklearn converts inputs to float32 before comparing to float64 thresholds.
    X = np.asarray(X, dtype=np.float32)
    rows = np.arange(len(X))
    left, right = model["left"], model["right"]
    feature, threshold = model["feature"], model["threshold"]
    leaf_probabilities = model["probabilities"]
    probabilities = np.zeros((len(X), len(model["classes"])))

    for root in model["roots"]:
        nodes = np.full(len(X), root, dtype=np.int32)
        active = rows[left[nodes] >= 0]
        while active.size:
            branch = nodes[active]
            go_left = X[active, feature[branch]] <= threshold[branch]
            nodes[active] = np.where(go_left, left[branch], right[branch])
            active = active[left[nodes[active]] >= 0]
        probabilities += leaf_probabilities[nodes]
    return probabilities / len(model["roots"])


def predict_df(df):
    """Predict one sport per input row."""
    if df.empty:
        return []
    state, model = _load_model()
    X, _ = features.transform_features(df, state)
    probabilities = _forest_probabilities(project_text(X, model["components"]), model)
    return model["classes"][probabilities.argmax(axis=1)].tolist()


def predict_all(filename):
    """Return one sport name per row in the provided CSV file."""
    return predict_df(pd.read_csv(filename))


if __name__ == "__main__":
    print(predict_all(sys.argv[1])[:10])
