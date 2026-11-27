"""Sanity check: pred.py (numpy) must give the same outputs as the PyTorch
model rebuilt from the same saved weights, and we report accuracy.

NOTE: mlp_model.npz is refit on ALL rows, so accuracy here is optimistic
(the honest estimate is the "Locked test" block of train_final's output).

Usage:  python check_model.py
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

import mlp_lib as L
import pred


def main():
    df = L.load_data()
    test = df[df["split"] == "test"].reset_index(drop=True)

    npz = np.load(L.MODEL_OUT, allow_pickle=False)
    n = int(npz["n_layers"])
    mods = []
    for i in range(n):
        lin = torch.nn.Linear(npz[f"W{i}"].shape[1], npz[f"W{i}"].shape[0])
        lin.weight.data = torch.tensor(npz[f"W{i}"])
        lin.bias.data = torch.tensor(npz[f"b{i}"])
        mods += [lin] + ([torch.nn.ReLU()] if i < n - 1 else [])
    net = torch.nn.Sequential(*mods).double().eval()

    prep, _, _ = pred._load()
    X = prep.transform(test)
    with torch.no_grad():
        torch_logits = net(torch.tensor(X, dtype=torch.float64)).numpy()
    np_logits = pred.logits_df(test)
    print("max |torch - numpy| logits:", np.abs(torch_logits - np_logits).max())
    assert np.allclose(torch_logits, np_logits, atol=1e-8), "numpy forward pass differs!"

    y_pred = pred.predict_df(test)
    y_true = test["label"].to_numpy()
    print("Accuracy (optimistic, model saw these rows):", round(accuracy_score(y_true, y_pred), 4))
    print(confusion_matrix(y_true, y_pred, labels=L.CLASSES))
    print("OK: numpy and torch predictions agree.")


if __name__ == "__main__":
    main()
