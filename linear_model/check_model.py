"""
Quick sanity check for softmax_model.npz: runs pred.py's prediction logic
against real data and reports accuracy / confusion matrix / per-class report.

CAVEAT: softmax_model.npz was refit on ALL rows (train+val+test) as the
final deployed artifact (see train_softmax.py's "Final refit on ALL data"
step). So evaluating it against the test split here is optimistic / not a
true generalization estimate -- the model has already seen those exact rows.
Use this script as an end-to-end smoke test (does prediction run, is the
confusion matrix roughly diagonal), not as a fresh accuracy number. The
honest held-out accuracy is the "Locked test" block in training_run.log,
computed from a model trained only on train+val, before the final refit.

Usage:
    python check_model.py
    python check_model.py --all
    python check_model.py --data path/to/other.csv
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import features as F
import pred

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = os.path.join(_HERE, "..", "csc311_project_training_202609.csv")
SPLITS_FILE = os.path.join(_HERE, "..", "splits.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_FILE)
    parser.add_argument("--all", action="store_true", help="evaluate all rows instead of just the locked test split")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if not args.all:
        splits = pd.read_csv(SPLITS_FILE)
        df = df.merge(splits, on=F.ID_COL, how="left")
        df = df[df["split"] == "test"].reset_index(drop=True)
        print(f"Evaluating on locked test split: {len(df)} rows")
    else:
        print(f"Evaluating on ALL rows (optimistic -- model was refit on this data): {len(df)} rows")

    y_true = df[F.LABEL_COL].to_numpy()
    y_pred = pred.predict_df(df)

    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    classes = sorted(set(y_true) | set(y_pred))
    print(f"\nConfusion matrix (rows=true, cols=pred), classes={classes}")
    print(confusion_matrix(y_true, y_pred, labels=classes))
    print()
    print(classification_report(y_true, y_pred, labels=classes))


if __name__ == "__main__":
    main()
