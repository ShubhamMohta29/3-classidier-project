"""Final evaluation + export for the best (text setting, MLP setting) found by
tune_mlp_text.py. Same protocol as train_final.py, but saves NEW files:
    mlp_cv_scores_text.csv, mlp_model_text.npz

If the result is better and you want to use it, rename mlp_model_text.npz to
mlp_model.npz (pred.py reads the text setting from the file, so nothing else
changes).

Usage:  python train_final_text.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

import mlp_lib as L
from features import Preprocessor


def make_prep(text):
    return Preprocessor(min_df=text["min_df"], max_vocab=text["max_vocab"],
                        use_tfidf=(text["weighting"] == "tfidf"))


def main():
    row = pd.read_csv("mlp_text_tuning_summary.csv").iloc[0]
    text = {"min_df": int(row.min_df), "max_vocab": int(row.max_vocab),
            "weighting": str(row.weighting)}
    cfg = {"hidden_layers": int(row.hidden_layers), "hidden_width": int(row.hidden_width),
           "lr": float(row.lr), "weight_decay": float(row.weight_decay),
           "dropout": float(row.dropout)}
    print("Text setting:", text)
    print("MLP config:  ", cfg)

    df = L.load_data()
    pool = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    rows, epochs = [], []
    for k, (tr, va) in enumerate(L.iter_cv_folds(pool, n_repeats=3)):
        prep = make_prep(text).fit(tr)
        Xtr, ytr = prep.transform(tr), L.labels_to_idx(tr)
        Xva, yva = prep.transform(va), L.labels_to_idx(va)
        res = [L.train_model(Xtr, ytr, Xva, yva, seed=L.SEED0 + s, **cfg)[1]
               for s in range(3)]
        rows.append({"repeat": k // 5, "fold": k % 5,
                     "acc": np.mean([r["val_acc"] for r in res]),
                     "f1": np.mean([r["val_f1"] for r in res]),
                     "loss": np.mean([r["val_loss"] for r in res])})
        epochs += [r["best_epoch"] for r in res]
        print(f"fold {k + 1}/15: acc={rows[-1]['acc']:.4f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv("mlp_cv_scores_text.csv", index=False)
    print(f"\nMLP (text-tuned) CV accuracy: {cv.acc.mean():.4f} +/- {cv.acc.std():.4f} "
          f"macro-F1: {cv.f1.mean():.4f}   (previous fixed-text MLP: 0.8978 +/- 0.0362)")
    n_epochs = int(np.median(epochs))
    print("Median best epoch:", n_epochs)

    import torch
    prep = make_prep(text).fit(pool)
    model, _ = L.train_model(prep.transform(pool), L.labels_to_idx(pool),
                             fixed_epochs=n_epochs, **cfg)
    with torch.no_grad():
        pred = model(torch.tensor(prep.transform(test), dtype=torch.float32)).argmax(1).numpy()
    y = L.labels_to_idx(test)
    print("\nLocked test accuracy:", round(accuracy_score(y, pred), 4))
    print("Locked test macro-F1:", round(f1_score(y, pred, average="macro"), 4))
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    print(classification_report(y, pred, target_names=L.CLASSES))

    prep_all = make_prep(text).fit(df)
    model_all, _ = L.train_model(prep_all.transform(df), L.labels_to_idx(df),
                                 fixed_epochs=n_epochs, **cfg)
    print("Saved", L.export_npz(model_all, prep_all, path="mlp_model_text.npz"))


if __name__ == "__main__":
    main()
