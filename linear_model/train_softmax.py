"""
Trains and tunes the softmax (multinomial logistic regression) model for the
3-class sport classifier, using sklearn. Produces softmax_model.npz, the
frozen weights + fitted feature-preprocessor state that pred.py loads at
prediction time with zero sklearn dependency.

Usage:
    python train_softmax.py
"""
import concurrent.futures as cf
import os
import time
import warnings

# Must be set before numpy/scipy/sklearn are imported in each worker process
# (ProcessPoolExecutor uses "spawn" on Windows, so every worker re-imports
# them from scratch). Without this, each process's BLAS backend spawns its
# own thread pool, and that oversubscription -- num_workers processes each
# multithreaded -- was enough to exhaust the system's page-file commit limit
# with a 12-worker pool; one thread per process per worker is sufficient
# since the parallelism here is already at the process level.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold

import features as F

DATA_FILE = "../csc311_project_training_202609.csv"
SPLITS_FILE = "../splits.csv"
MODEL_OUT = "softmax_model.npz"

TEXT_GRID = [
    {"min_df": min_df, "vocab_cap": vocab_cap, "weighting": weighting}
    for min_df in (2, 3, 5)
    for vocab_cap in (100, 300)
    for weighting in ("tfidf", "count")
]
C_GRID = np.logspace(-4, 1, 13)
PENALTIES = ("l1", "l2")
MAX_ITER = 5000

# os.cpu_count() is 12 on this machine, but measured directly (see
# training_run.log history): with BLAS threads pinned to 1 per process
# (above), 8 workers is the actual throughput sweet spot -- 12 workers hits
# CPU/cache contention and is slower *in total wall time* than 8 despite
# using more processes, and also leaves less memory headroom. Re-measure
# with the smoke-test snippet in this file's history before changing this
# on a different machine.
MAX_WORKERS = 8

N_SPLITS = 5
N_REPEATS = 3
SEED0 = 311


def load_data():
    df = pd.read_csv(DATA_FILE)
    splits = pd.read_csv(SPLITS_FILE)
    df = df.merge(splits, on=F.ID_COL, how="left")
    if df["split"].isna().any():
        raise ValueError("some respondents in the training CSV have no split assignment")
    return df


def iter_cv_folds(pool_df, seed0=SEED0, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Yields (train_fold_df, val_fold_df) pairs: n_repeats independent
    respondent shufflings x n_splits grouped folds each."""
    groups = pool_df[F.ID_COL].to_numpy()
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed0 + repeat)
        perm = rng.permutation(len(pool_df))
        df_p = pool_df.iloc[perm].reset_index(drop=True)
        g_p = groups[perm]
        gkf = GroupKFold(n_splits=n_splits)
        for tr_idx, va_idx in gkf.split(df_p, groups=g_p):
            yield df_p.iloc[tr_idx], df_p.iloc[va_idx]


def fit_logreg(X, y, C, penalty, max_iter=MAX_ITER):
    solver = "lbfgs" if penalty == "l2" else "saga"
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X, y)
    n_warn = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return clf, n_warn


def _fit_one_unit(unit):
    """Worker for one (text_params, fold) unit: fits the feature state once
    and reuses it across every (C, penalty) model config on that fold. Runs
    in its own process (see run_cv) -- SAGA at weak regularization is slow
    enough in this p~n regime that the full grid needs real parallelism
    across folds/text-configs to finish in a practical amount of time."""
    text_params, fold_id, train_fold, val_fold, c_grid, penalties = unit
    state = F.fit_features(train_fold, **text_params)
    X_tr, _ = F.transform_features(train_fold, state)
    X_va, _ = F.transform_features(val_fold, state)
    y_tr = train_fold[F.LABEL_COL].to_numpy()
    y_va = val_fold[F.LABEL_COL].to_numpy()

    records = []
    for penalty in penalties:
        for C in c_grid:
            clf, n_warn = fit_logreg(X_tr, y_tr, C=C, penalty=penalty)
            preds = clf.predict(X_va)
            records.append({
                **text_params,
                "C": C,
                "penalty": penalty,
                "fold": fold_id,
                "acc": accuracy_score(y_va, preds),
                "f1": f1_score(y_va, preds, average="macro"),
                "n_warn": n_warn,
            })
    return records


def run_cv(pool_df, text_grid=TEXT_GRID, c_grid=C_GRID, penalties=PENALTIES, max_workers=None):
    """Grid search, restructured so feature extraction (expensive, depends
    only on text_params) runs once per (text_params, fold) and is reused
    across every (C, penalty) model config on that fold. Each (text_params,
    fold) unit is independent, so units are distributed across worker
    processes (defaults to MAX_WORKERS, not all CPU cores -- see its
    comment)."""
    units = [
        (text_params, fold_id, train_fold, val_fold, c_grid, penalties)
        for text_params in text_grid
        for fold_id, (train_fold, val_fold) in enumerate(iter_cv_folds(pool_df))
    ]
    max_workers = max_workers or MAX_WORKERS
    print(f"Running {len(units)} (text_params, fold) units on {max_workers} workers...")

    records = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for i, unit_records in enumerate(pool.map(_fit_one_unit, units), 1):
            records.extend(unit_records)
            print(f"  [{i}/{len(units)}] units done", flush=True)
    return pd.DataFrame.from_records(records)


def summarize(results):
    group_cols = ["min_df", "vocab_cap", "weighting", "C", "penalty"]
    summary = (
        results.groupby(group_cols)
        .agg(
            mean_acc=("acc", "mean"),
            std_acc=("acc", "std"),
            mean_f1=("f1", "mean"),
            n_warn=("n_warn", "sum"),
        )
        .reset_index()
        .sort_values(["mean_acc", "mean_f1"], ascending=False)
    )
    return summary


def evaluate_once(clf, X, y, label):
    preds = clf.predict(X)
    print(f"\n{label} accuracy:  {accuracy_score(y, preds):.4f}")
    print(f"{label} macro-F1:  {f1_score(y, preds, average='macro'):.4f}")
    print(f"{label} log-loss:  {log_loss(y, clf.predict_proba(X), labels=clf.classes_):.4f}")
    print(f"{label} confusion matrix (rows=true, cols=pred), classes={list(clf.classes_)}")
    print(confusion_matrix(y, preds, labels=clf.classes_))
    print(classification_report(y, preds))


def main():
    df = load_data()
    pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"CV pool: {len(pool)} rows / {pool[F.ID_COL].nunique()} respondents")
    print(f"Locked test: {len(test)} rows / {test[F.ID_COL].nunique()} respondents")

    t0 = time.time()
    results = run_cv(pool)
    print(f"\nGrid search: {len(results)} fits in {time.time() - t0:.1f}s")

    summary = summarize(results)
    print("\nTop 10 configs by mean CV accuracy:")
    print(summary.head(10).to_string(index=False))

    best = summary.iloc[0]
    best_text = {
        "min_df": int(best["min_df"]),
        "vocab_cap": int(best["vocab_cap"]),
        "weighting": best["weighting"],
    }
    best_model = {"C": float(best["C"]), "penalty": best["penalty"]}
    print(f"\nBest config: {best_text} {best_model}")

    # Retrain on train+val, evaluate ONCE on the locked test split.
    state_tv = F.fit_features(pool, **best_text)
    X_tv, _ = F.transform_features(pool, state_tv)
    y_tv = pool[F.LABEL_COL].to_numpy()
    clf_tv, _ = fit_logreg(X_tv, y_tv, **best_model)

    X_test, _ = F.transform_features(test, state_tv)
    y_test = test[F.LABEL_COL].to_numpy()
    evaluate_once(clf_tv, X_test, y_test, "Locked test")

    # Final refit on ALL data for the deployed pred.py artifact.
    state_full = F.fit_features(df, **best_text)
    X_full, _ = F.transform_features(df, state_full)
    y_full = df[F.LABEL_COL].to_numpy()
    clf_full, _ = fit_logreg(X_full, y_full, **best_model)

    npz_kwargs = F.state_to_npz_kwargs(state_full)
    npz_kwargs["W"] = clf_full.coef_.astype(np.float64)
    npz_kwargs["b"] = clf_full.intercept_.astype(np.float64)
    npz_kwargs["classes"] = np.array(clf_full.classes_, dtype="<U16")
    np.savez(MODEL_OUT, **npz_kwargs)
    print(f"\nSaved {MODEL_OUT}")


if __name__ == "__main__":
    main()
