"""Tune on the team's grouped folds, evaluate once, and export the forest."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

# Each worker fits independent forests. Avoid nested BLAS parallelism.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold

if __package__:
    from . import features, pred
else:
    import features
    import pred

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "csc311_project_training_202609.csv"
SPLITS_FILE = HERE.parent / "splits.csv"
SEED = 311
TEXT_KEYS = ["min_df", "vocab_cap", "weighting"]
MODEL_KEYS = ["representation", "min_samples_leaf", "max_features", "max_depth", "n_estimators"]
TEXT_GRID = list(product((2, 3, 5), (100, 300), ("tfidf", "count")))
BASE_GRID = list(product(("words", "svd"), (1, 2, 5, 10), ("sqrt", "0.1", "0.3"), (0,), (96,)))
CLASSES = ["Baseball", "Basketball", "Soccer"]


def load_data():
    data = pd.read_csv(DATA_FILE)
    splits = pd.read_csv(SPLITS_FILE)
    data = data.merge(splits, on=features.ID_COL, how="left", validate="many_to_one")
    if set(data["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected each respondent to be assigned to train, val, or test.")
    return data


def iter_cv_folds(pool):
    # Match the first repeat in linear_model/train_softmax.py. Its other
    # repeats permute rows but reuse the same respondent memberships.
    order = np.random.default_rng(SEED).permutation(len(pool))
    shuffled = pool.iloc[order]
    groups = shuffled[features.ID_COL]
    for train, val in GroupKFold(n_splits=5).split(shuffled, groups=groups):
        yield shuffled.iloc[train], shuffled.iloc[val]


def fit_text_projection(X, state, representation="svd"):
    """Fit the text blocks specified by the representation."""
    if representation == "words":
        return np.empty((0, 0))
    names = state["feature_names"]
    text_width = sum(len(state[f"text_vocab_{name}"]) for name, _ in features.TEXT_COLUMNS)
    text = X[:, len(names) - text_width:]
    blocks = []
    start = 0
    for name, _ in features.TEXT_COLUMNS:
        stop = start + len(state[f"text_vocab_{name}"])
        width = stop - start
        if name == "song" and representation.startswith("no_song"):
            basis = np.empty((0, width))
        elif width <= 8 or representation == "no_song_words" or (
            name == "song" and representation == "song_words"
        ):
            basis = np.eye(width)
        else:
            basis = TruncatedSVD(n_components=8, random_state=SEED).fit(
                text[:, start:stop]
            ).components_
        block = np.zeros((len(basis), text_width))
        block[:, start:stop] = basis
        blocks.append(block)
        start = stop
    return np.vstack(blocks)


def fit_forest(X, y, config, *, oob_score=False):
    _, leaf, max_features, depth, n_trees = config
    forest = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=depth or None,
        min_samples_leaf=leaf,
        max_features=max_features if max_features == "sqrt" else float(max_features),
        random_state=SEED,
        n_jobs=1,
        oob_score=oob_score,
    )
    return forest.fit(X, y)


def score(y, probabilities, labels):
    guesses = np.asarray(labels)[probabilities.argmax(axis=1)]
    return {
        "acc": accuracy_score(y, guesses),
        "f1": f1_score(y, guesses, labels=labels, average="macro", zero_division=0),
        "log_loss": log_loss(y, probabilities, labels=labels),
    }


def fit_cv_unit(unit):
    text_params, fold, train, val, configs = unit
    state = features.fit_features(train, **dict(zip(TEXT_KEYS, text_params)))
    X_train, _ = features.transform_features(train, state)
    X_val, _ = features.transform_features(val, state)
    y_train, y_val = train[features.LABEL_COL], val[features.LABEL_COL]

    designs = {}
    for representation in dict.fromkeys(config[0] for config in configs):
        components = fit_text_projection(X_train, state, representation)
        designs[representation] = (
            pred.project_text(X_train, components), pred.project_text(X_val, components),
        )

    records = []
    for config in configs:
        train_design, val_design = designs[config[0]]
        forest = fit_forest(train_design, y_train, config)
        records.append((
            *text_params, *config, fold,
            *score(y_val, forest.predict_proba(val_design), forest.classes_).values(),
        ))
    return records


def run_cv(folds, grid, workers):
    units = [
        (text_params, fold, train, val, configs)
        for text_params, configs in grid.items()
        for fold, (train, val) in enumerate(folds, 1)
    ]
    records = []
    if workers == 1:
        for unit in units:
            records.extend(fit_cv_unit(unit))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(fit_cv_unit, units):
                records.extend(result)
    return pd.DataFrame(records, columns=TEXT_KEYS + MODEL_KEYS + ["fold", "acc", "f1", "log_loss"])


def summarize(scores):
    return (
        scores.groupby(TEXT_KEYS + MODEL_KEYS, as_index=False)
        .agg(
            mean_acc=("acc", "mean"),
            std_acc=("acc", "std"),
            mean_f1=("f1", "mean"),
            mean_log_loss=("log_loss", "mean"),
        )
        .sort_values(["mean_acc", "mean_f1"], ascending=False, kind="stable")
    )


def refinement_grid(summary):
    """Expand the top three settings per representation across depth/tree count."""
    grid = {}
    shortlist = summary.groupby("representation", sort=False).head(3)
    for row in shortlist[TEXT_KEYS + MODEL_KEYS].itertuples(index=False, name=None):
        text_params = row[:len(TEXT_KEYS)]
        representation, leaf, max_features, _, _ = row[len(TEXT_KEYS):]
        for depth, n_trees in product((8, 16, 0), (96, 192)):
            if (depth, n_trees) != (0, 96):
                config = (representation, leaf, max_features, depth, n_trees)
                grid.setdefault(text_params, []).append(config)
    return grid


def fit_final(data, text_params, config, *, oob_score=False):
    state = features.fit_features(data, **dict(zip(TEXT_KEYS, text_params)))
    X, _ = features.transform_features(data, state)
    components = fit_text_projection(X, state, config[0])
    design = pred.project_text(X, components)
    forest = fit_forest(design, data[features.LABEL_COL], config, oob_score=oob_score)
    return state, components, forest, design


def save_model(destination, state, components, forest):
    arrays = features.state_to_npz_kwargs(state)
    arrays["classes"] = np.asarray(forest.classes_, dtype=str)
    arrays["components"] = components
    trees = [estimator.tree_ for estimator in forest.estimators_]
    roots = np.cumsum([0, *(tree.node_count for tree in trees[:-1])], dtype=np.int32)
    arrays["roots"] = roots
    for key, attr in (("left", "children_left"), ("right", "children_right")):
        children = [getattr(tree, attr) for tree in trees]
        arrays[key] = np.concatenate([
            np.where(indices < 0, -1, indices + root)
            for indices, root in zip(children, roots)
        ]).astype(np.int32)
    arrays["feature"] = np.concatenate([tree.feature for tree in trees]).astype(np.int32)
    arrays["threshold"] = np.concatenate([tree.threshold for tree in trees])
    values = np.concatenate([tree.value[:, 0, :] for tree in trees])
    arrays["probabilities"] = values / values.sum(axis=1, keepdims=True)
    np.savez_compressed(destination, **arrays)


def export_model(state, components, forest):
    save_model(pred.MODEL_PATH, state, components, forest)
    pred._load_model.cache_clear()
    print(f"Saved {pred.MODEL_PATH.name} ({pred.MODEL_PATH.stat().st_size:,} bytes).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    started = perf_counter()
    data = load_data()
    pool = data[data["split"].isin(["train", "val"])]
    test = data[data["split"] == "test"]
    folds = list(iter_cv_folds(pool))
    print(f"NumPy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}")
    print(f"CV pool: {len(pool)} rows / {pool[features.ID_COL].nunique()} respondents")
    print(f"Locked test: {len(test)} rows / {test[features.ID_COL].nunique()} respondents")
    print(f"Validation rows per fold: {[len(val) for _, val in folds]}")

    print("Stage 1: text settings, representation, minimum leaf size, and max features.", flush=True)
    base = run_cv(folds, dict.fromkeys(TEXT_GRID, BASE_GRID), args.workers)
    print(f"Stage 1 complete: {len(base)} fits in {perf_counter() - started:.1f}s", flush=True)
    print("Stage 2: depth and tree count for the top three settings per representation.", flush=True)
    refined = run_cv(folds, refinement_grid(summarize(base)), args.workers)
    scores = pd.concat([base.assign(stage=1), refined.assign(stage=2)], ignore_index=True)
    summary = summarize(scores)
    print(f"{len(summary)} configurations, {len(scores)} fits in {perf_counter() - started:.1f}s.")
    print(f"Top 10 configurations:\n{summary.head(10).to_string(index=False)}")
    scores.to_csv(HERE / "cv_scores.csv", index=False)

    best = summary.iloc[0]
    text_params = tuple(best[TEXT_KEYS])
    config = tuple(best[MODEL_KEYS])
    print(f"Selected: {dict(zip(TEXT_KEYS + MODEL_KEYS, (*text_params, *config)))}")
    chosen = (scores[TEXT_KEYS + MODEL_KEYS] == best[TEXT_KEYS + MODEL_KEYS]).all(axis=1)
    print(f"Selected fold metrics:\n{scores.loc[chosen, ['fold', 'acc', 'f1', 'log_loss']].to_string(index=False)}")

    print("Locked-test fit:")
    state, components, forest, _ = fit_final(pool, text_params, config)
    X_test, _ = features.transform_features(test, state)
    y_test = test[features.LABEL_COL]
    probabilities = forest.predict_proba(pred.project_text(X_test, components))
    guesses = forest.classes_[probabilities.argmax(axis=1)]
    for metric, value in score(y_test, probabilities, CLASSES).items():
        print(f"Locked-test {metric}: {value:.6f}")
    print(f"Confusion matrix, rows=true and columns=predicted, order={CLASSES}:")
    print(confusion_matrix(y_test, guesses, labels=CLASSES))
    print(classification_report(y_test, guesses, labels=CLASSES, digits=4, zero_division=0))

    print("Deployment fit:")
    state, components, forest, design = fit_final(data, text_params, config)
    export_model(state, components, forest)
    _, exported = pred._load_model()
    np.testing.assert_allclose(
        pred._forest_probabilities(design, exported), forest.predict_proba(design),
        rtol=0, atol=1e-12,
    )
    assert pred.predict_df(data.drop(columns=features.LABEL_COL)) == forest.predict(design).tolist()
    print(f"Export verified against scikit-learn on all {len(data)} rows.")


if __name__ == "__main__":
    main()
