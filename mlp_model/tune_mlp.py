"""Random search for the MLP hyperparameters (proposal section 2.4).

Search space: layers {1,2} x width {32,64,128}; Adam lr {1e-3, 3e-4};
weight decay {1e-4,1e-3,1e-2}; dropout {0,0.3,0.5}. 40 random configs, each
scored on the same grouped 5-fold CV folds as the other models.
Criterion: mean CV accuracy, macro-F1 as tie-break.

Usage:  python tune_mlp.py [--n-configs 40] [--seeds 1]
Writes: mlp_tuning_scores.csv (per fold), mlp_tuning_summary.csv (per config)
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

import mlp_lib as L

GRID = {
    "hidden_layers": [1, 2, 3],
    "hidden_width": [16, 32, 64, 128],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-3, 1e-2, 3e-2, 1e-1],
    "dropout": [0.0, 0.3, 0.5, 0.6],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1, help="seeds averaged per fold")
    args = ap.parse_args()

    df = L.load_data()
    pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    print(f"CV pool: {len(pool)} rows / {pool['id'].nunique()} respondents")

    # Preprocessing does not depend on hyperparameters: do it once per fold.
    folds = [L.prepare_fold(tr, va) for tr, va in L.iter_cv_folds(pool, n_repeats=1)]

    combos = list(itertools.product(*GRID.values()))
    rng = np.random.default_rng(L.SEED0)
    chosen = [combos[i] for i in rng.permutation(len(combos))[:args.n_configs]]

    records, t0 = [], time.time()
    for ci, combo in enumerate(chosen):
        cfg = dict(zip(GRID.keys(), combo))
        for fi, (Xtr, ytr, Xva, yva) in enumerate(folds):
            accs, f1s, losses, eps = [], [], [], []
            for s in range(args.seeds):
                _, info = L.train_model(Xtr, ytr, Xva, yva, seed=L.SEED0 + s, **cfg)
                accs.append(info["val_acc"]); f1s.append(info["val_f1"])
                losses.append(info["val_loss"]); eps.append(info["best_epoch"])
            records.append({"config": ci, **cfg, "fold": fi,
                            "acc": np.mean(accs), "f1": np.mean(f1s),
                            "loss": np.mean(losses), "best_epoch": np.mean(eps)})
        part = pd.DataFrame(records)
        cur = part[part.config == ci]
        print(f"[{ci + 1}/{len(chosen)}] {cfg}  acc={cur.acc.mean():.4f} "
              f"f1={cur.f1.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)

    scores = pd.DataFrame(records)
    scores.to_csv("mlp_tuning_scores.csv", index=False)
    keys = list(GRID.keys())
    summary = (scores.groupby(["config"] + keys)
               .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"),
                    mean_f1=("f1", "mean"), mean_loss=("loss", "mean"),
                    mean_best_epoch=("best_epoch", "mean"))
               .reset_index().sort_values(["mean_acc", "mean_f1"], ascending=False))
    summary.to_csv("mlp_tuning_summary.csv", index=False)
    print("\nTop 10 configs:")
    print(summary.head(10).to_string(index=False))
    print("\nCheck: if the best value of a parameter is at the edge of its range,")
    print("extend GRID for that parameter and rerun (proposal section 2.4).")


if __name__ == "__main__":
    main()
