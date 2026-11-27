"""Final MLP evaluation and export (proposal sections 2.5 and 2.6).

1. Evaluate the best config (from mlp_tuning_summary.csv, or --config) on the
   shared CV folds: 3 respondent shufflings x 5 folds, averaged over 3 seeds
   -> 15 fold scores -> mlp_cv_scores.csv.
2. Retrain on train+val (fixed epochs = median best epoch from CV) and
   evaluate ONCE on the locked test split.
3. Refit on all rows and write mlp_model.npz for pred.py.

Usage:  python train_final.py
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import mlp_lib as L
from features import Preprocessor

KEYS = ["hidden_layers", "hidden_width", "lr", "weight_decay", "dropout"]


def best_config(path="mlp_tuning_summary.csv"):
    row = pd.read_csv(path).iloc[0]
    return {"hidden_layers": int(row.hidden_layers), "hidden_width": int(row.hidden_width),
            "lr": float(row.lr), "weight_decay": float(row.weight_decay),
            "dropout": float(row.dropout)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help='JSON, e.g. \'{"hidden_layers":1,"hidden_width":128,'
                    '"lr":0.001,"weight_decay":0.0001,"dropout":0.5}\'')
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    cfg = json.loads(args.config) if args.config else best_config()
    print("Config:", cfg)

    df = L.load_data()
    pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    # ---- 1. 15-fold CV scores ----
    rows, epochs = [], []
    for k, (tr, va) in enumerate(L.iter_cv_folds(pool, n_repeats=args.repeats)):
        Xtr, ytr, Xva, yva = L.prepare_fold(tr, va)
        res = [L.train_model(Xtr, ytr, Xva, yva, seed=L.SEED0 + s, **cfg)[1]
               for s in range(args.seeds)]
        rows.append({"repeat": k // 5, "fold": k % 5,
                     "acc": np.mean([r["val_acc"] for r in res]),
                     "f1": np.mean([r["val_f1"] for r in res]),
                     "loss": np.mean([r["val_loss"] for r in res])})
        epochs += [r["best_epoch"] for r in res]
        print(f"fold {k + 1}/{5 * args.repeats}: acc={rows[-1]['acc']:.4f} "
              f"f1={rows[-1]['f1']:.4f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv("mlp_cv_scores.csv", index=False)
    print(f"\nMLP CV accuracy: {cv.acc.mean():.4f} +/- {cv.acc.std():.4f} "
          f"(n={len(cv)} folds)  macro-F1: {cv.f1.mean():.4f}  loss: {cv.loss.mean():.4f}")
    n_epochs = int(np.median(epochs))
    print(f"Median best epoch: {n_epochs}")

    # ---- 2. train+val -> locked test (once) ----
    prep = Preprocessor().fit(pool)
    model, _ = L.train_model(prep.transform(pool), L.labels_to_idx(pool),
                             fixed_epochs=n_epochs, **cfg)
    import torch
    with torch.no_grad():
        pred = model(torch.tensor(prep.transform(test), dtype=torch.float32)).argmax(1).numpy()
    y = L.labels_to_idx(test)
    print("\nLocked test accuracy:", round(accuracy_score(y, pred), 4))
    print("Locked test macro-F1:", round(f1_score(y, pred, average="macro"), 4))
    print("Confusion matrix (rows=true, cols=pred), classes =", L.CLASSES)
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    print(classification_report(y, pred, target_names=L.CLASSES))

    # ---- 3. refit on all rows, export for pred.py ----
    prep_all = Preprocessor().fit(df)
    model_all, _ = L.train_model(prep_all.transform(df), L.labels_to_idx(df),
                                 fixed_epochs=n_epochs, **cfg)
    print("Saved", L.export_npz(model_all, prep_all))


if __name__ == "__main__":
    main()
