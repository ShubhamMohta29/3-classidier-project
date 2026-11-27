"""Shared helpers for the MLP: data loading, CV folds, training, export.

Only train_model() needs PyTorch. Everything the deployed pred.py needs is
saved to mlp_model.npz so prediction can run with numpy/pandas only.
"""
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from features import Preprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "..", "csc311_project_training_202609.csv")
SPLITS_FILE = os.path.join(HERE, "..", "splits.csv")
MODEL_OUT = os.path.join(HERE, "mlp_model.npz")

CLASSES = ["Baseball", "Basketball", "Soccer"]
LABEL_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}
SEED0 = 311          # same base seed as the linear and forest trainers
PATIENCE = 20
MAX_EPOCHS = 300
BATCH_SIZE = 32


def load_data():
    df = pd.read_csv(DATA_FILE)
    splits = pd.read_csv(SPLITS_FILE)
    df = df.merge(splits, on="id", how="left")
    if df["split"].isna().any():
        raise ValueError("Some rows have no entry in splits.csv")
    return df


def iter_cv_folds(pool_df, seed0=SEED0, n_splits=5, n_repeats=1):
    """Identical to the linear / forest trainers: n_repeats shufflings x
    n_splits respondent-grouped folds."""
    groups = pool_df["id"].to_numpy()
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed0 + repeat)
        perm = rng.permutation(len(pool_df))
        shuffled = pool_df.iloc[perm].reset_index(drop=True)
        g = groups[perm]
        for tr, va in GroupKFold(n_splits=n_splits).split(shuffled, groups=g):
            yield shuffled.iloc[tr], shuffled.iloc[va]


def labels_to_idx(df):
    return df["label"].map(LABEL_TO_INDEX).to_numpy()


def prepare_fold(train_df, val_df):
    """Fit the preprocessor on the training part only, then transform both."""
    prep = Preprocessor().fit(train_df)
    return (prep.transform(train_df), labels_to_idx(train_df),
            prep.transform(val_df), labels_to_idx(val_df))


def build_model(n_in, hidden_width, hidden_layers, dropout):
    from torch import nn
    layers, width = [], n_in
    for _ in range(hidden_layers):
        layers += [nn.Linear(width, hidden_width), nn.ReLU(), nn.Dropout(p=dropout)]
        width = hidden_width
    layers.append(nn.Linear(width, 3))
    return nn.Sequential(*layers)


def train_model(X_tr, y_tr, X_val=None, y_val=None, *, hidden_width=128,
                hidden_layers=1, dropout=0.5, lr=1e-3, weight_decay=1e-4,
                seed=SEED0, fixed_epochs=None):
    """Train one MLP.

    * With X_val: early stopping (patience 20 on validation loss), best
      weights restored. Returns (model, info) with val accuracy / loss / F1.
    * With fixed_epochs and no X_val: train exactly that many epochs (used for
      the final refit, where there is no held-out set).
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = build_model(X_tr.shape[1], hidden_width, hidden_layers, dropout)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                        shuffle=True, generator=gen)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    have_val = X_val is not None
    if have_val:
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)
    n_epochs = fixed_epochs if (fixed_epochs and not have_val) else MAX_EPOCHS
    best_loss, best_epoch, best_w, bad = float("inf"), 0, None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if have_val:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xv), yv).item()
            if vl < best_loss:
                best_loss, best_epoch, best_w, bad = vl, epoch, deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if have_val:
        model.load_state_dict(best_w)
    model.eval()
    info = {"best_epoch": best_epoch if have_val else n_epochs}
    if have_val:
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1).numpy()
        info.update(val_loss=best_loss,
                    val_acc=float((pred == y_val).mean()),
                    val_f1=float(f1_score(y_val, pred, labels=[0, 1, 2],
                                          average="macro", zero_division=0)))
    return model, info


# ---------------------------------------------------------------------------
# Export: torch model + fitted Preprocessor -> one .npz usable with numpy only
# ---------------------------------------------------------------------------
def export_npz(model, prep, path=MODEL_OUT):
    from torch import nn
    linears = [m for m in model if isinstance(m, nn.Linear)]
    out = {"n_layers": np.array(len(linears)),
           "classes": np.array(CLASSES, dtype="<U16"),
           "prep_mean": prep.mean_.astype(np.float64),
           "prep_std": prep.std_.astype(np.float64)}
    for i, lin in enumerate(linears):
        out[f"W{i}"] = lin.weight.detach().numpy().astype(np.float64)
        out[f"b{i}"] = lin.bias.detach().numpy().astype(np.float64)
    state = {
        "min_df": prep.min_df, "max_vocab": prep.max_vocab,
        "top_colours": prep.top_colours, "text_fields": list(prep.text_fields),
        "use_tfidf": prep.use_tfidf, "medians": prep.medians,
        "vocab": prep.vocab, "idf": {k: list(map(float, v)) for k, v in prep.idf.items()},
        "colours": prep.colours, "pay_cap": prep.pay_cap, "names": prep.names,
    }
    out["prep_json"] = np.array(json.dumps(state))
    np.savez(path, **out)
    return path
