"""Joint search over the TEXT settings and the MLP settings.

The logistic regression and random forest searched the same 12 text options
(min_df {2,3,5} x vocab cap {100,300} x TF-IDF vs counts). This does the same
for the MLP: every text option is crossed with 40 random MLP configurations
(the round-2 ranges), each scored on the same 5 grouped folds (first
shuffling, 1 seed). Criterion: mean CV accuracy, macro-F1 tie-break.

Saves NEW files only (existing results are untouched):
    mlp_text_tuning_scores.csv, mlp_text_tuning_summary.csv

Usage:  python tune_mlp_text.py            (about 30-40 minutes)
        python tune_mlp_text.py --n-configs 3   (quick test)
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

import mlp_lib as L
from features import Preprocessor

TEXT_GRID = {"min_df": [2, 3, 5], "max_vocab": [100, 300],
             "weighting": ["tfidf", "count"]}
GRID = {                      # round-2 MLP ranges
    "hidden_layers": [1, 2, 3],
    "hidden_width": [16, 32, 64, 128],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-3, 1e-2, 3e-2, 1e-1],
    "dropout": [0.0, 0.3, 0.5, 0.6],
}


def prepare_fold_text(train_df, val_df, text):
    prep = Preprocessor(min_df=text["min_df"], max_vocab=text["max_vocab"],
                        use_tfidf=(text["weighting"] == "tfidf")).fit(train_df)
    return (prep.transform(train_df), L.labels_to_idx(train_df),
            prep.transform(val_df), L.labels_to_idx(val_df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    df = L.load_data()
    pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    fold_dfs = list(L.iter_cv_folds(pool, n_repeats=1))

    combos = list(itertools.product(*GRID.values()))
    rng = np.random.default_rng(L.SEED0)
    chosen = [combos[i] for i in rng.permutation(len(combos))[:args.n_configs]]
    texts = [dict(zip(TEXT_GRID.keys(), t))
             for t in itertools.product(*TEXT_GRID.values())]

    records, t0 = [], time.time()
    for ti, text in enumerate(texts):
        folds = [prepare_fold_text(tr, va, text) for tr, va in fold_dfs]
        for ci, combo in enumerate(chosen):
            cfg = dict(zip(GRID.keys(), combo))
            for fi, (Xtr, ytr, Xva, yva) in enumerate(folds):
                res = [L.train_model(Xtr, ytr, Xva, yva, seed=L.SEED0 + s, **cfg)[1]
                       for s in range(args.seeds)]
                records.append({"text": ti, **text, "config": ci, **cfg, "fold": fi,
                                "acc": np.mean([r["val_acc"] for r in res]),
                                "f1": np.mean([r["val_f1"] for r in res]),
                                "loss": np.mean([r["val_loss"] for r in res]),
                                "best_epoch": np.mean([r["best_epoch"] for r in res])})
        part = pd.DataFrame(records)
        cur = part[part.text == ti].groupby("config").acc.mean()
        print(f"[text {ti + 1}/{len(texts)}] {text}  best acc={cur.max():.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        part.to_csv("mlp_text_tuning_scores.csv", index=False)   # safe to stop early

    scores = pd.DataFrame(records)
    keys = list(TEXT_GRID.keys()) + list(GRID.keys())
    summary = (scores.groupby(["text", "config"] + keys)
               .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"),
                    mean_f1=("f1", "mean"), mean_best_epoch=("best_epoch", "mean"))
               .reset_index().sort_values(["mean_acc", "mean_f1"], ascending=False))
    summary.to_csv("mlp_text_tuning_summary.csv", index=False)
    print("\nTop 10 (text setting + MLP setting):")
    print(summary.head(10).to_string(index=False))
    print("\nBest MLP config per text setting:")
    print(summary.groupby("text").head(1).sort_values("text")[
        list(TEXT_GRID.keys()) + ["mean_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()
