"""
Generates the team's shared respondent-level train/val/test split and writes
it to splits.csv. Run this once; splits.csv is then committed to the repo so
every teammate's code joins against the same, pinned split instead of each
recomputing a random permutation locally (which can silently disagree across
machines/library versions/preprocessing order).

Usage:
    python make_splits.py
"""
import numpy as np
import pandas as pd

DATA_FILE = "csc311_project_training_202609.csv"
OUT_FILE = "splits.csv"
ID_COL = "id"
SPLIT_COL = "split"
LABEL_COL = "label"
SEED = 311
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def make_splits(ids, seed=SEED, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """
    Assigns each unique respondent id to train/val/test.

    Sorting before permuting makes the result independent of the row order
    the ids happen to appear in (CSV order, pandas version, any filtering
    someone did beforehand) -- only the seed and the frac values matter.
    """
    unique_ids = np.sort(np.unique(ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_ids)

    n = len(perm)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test

    train_ids = perm[:n_train]
    val_ids = perm[n_train:n_train + n_val]
    test_ids = perm[n_train + n_val:]
    return train_ids, val_ids, test_ids


def report_label_balance(df, out):
    """Sanity check: label balance should be preserved across splits since
    every respondent answers once per sport."""
    merged = df.merge(out, on=ID_COL, how="left")
    print(merged.groupby(SPLIT_COL)[LABEL_COL].value_counts().unstack(fill_value=0))


def main():
    df = pd.read_csv(DATA_FILE)
    train_ids, val_ids, test_ids = make_splits(df[ID_COL])

    rows = (
        [(i, "train") for i in train_ids]
        + [(i, "val") for i in val_ids]
        + [(i, "test") for i in test_ids]
    )
    out = pd.DataFrame(rows, columns=[ID_COL, SPLIT_COL]).sort_values(ID_COL).reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"Wrote {OUT_FILE}: {len(out)} respondents "
          f"(train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")

    report_label_balance(df, out)


if __name__ == "__main__":
    main()
