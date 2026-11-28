import torch
from torch import nn
import pandas as pd

import numpy as np
from sklearn.model_selection import GroupKFold

from copy import deepcopy
from sklearn.metrics import f1_score

from features import Preprocessor
from torch.utils.data import DataLoader, TensorDataset

def iter_cv_folds(pool_df, seed0=311, n_splits=5, n_repeats=3):
    groups = pool_df["id"].to_numpy()

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed0 + repeat)
        perm = rng.permutation(len(pool_df))
        shuffled_df = pool_df.iloc[perm].reset_index(drop=True)
        shuffled_groups = groups[perm]

        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(
            shuffled_df, groups=shuffled_groups
        ):
            yield shuffled_df.iloc[train_idx], shuffled_df.iloc[val_idx]

def train_one_fold(
    train_df, val_df, seed=311, hidden_width=64, dropout=0.0,
    learning_rate=0.001, weight_decay=0.0, hidden_layers=1,
):
    torch.manual_seed(seed)
    prep = Preprocessor().fit(train_df)
    X_train = prep.transform(train_df)
    X_val = prep.transform(val_df)

    print("Training input shape:", X_train.shape)
    print("Validation input shape:", X_val.shape)
    print("First five training labels:", train_df["label"].head().tolist())

    layers = []
    input_width = X_train.shape[1]

    for _ in range(hidden_layers):
        layers.extend([
            nn.Linear(input_width, hidden_width),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ])
        input_width = hidden_width

    layers.append(nn.Linear(input_width, 3))
    model = nn.Sequential(*layers)
    label_to_index = {"Baseball": 0, "Basketball": 1, "Soccer": 2}

    train_inputs = torch.tensor(X_train, dtype=torch.float32)
    val_inputs = torch.tensor(X_val, dtype=torch.float32)

    train_targets = torch.tensor(
        train_df["label"].map(label_to_index).to_numpy(), dtype=torch.long
    )
    val_targets = torch.tensor(
        val_df["label"].map(label_to_index).to_numpy(), dtype=torch.long
    )

    train_data = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    patience = 20

    for epoch in range(1, 301):
        model.train()
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            scores = model(batch_inputs)
            loss = loss_fn(scores, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(val_inputs)
            val_loss = loss_fn(val_scores, val_targets).item()
            predictions = val_scores.argmax(dim=1)
            val_accuracy = (predictions == val_targets).float().mean().item()

        print(
            f"Epoch {epoch:02d} | "
            f"Validation loss: {val_loss:.3f} | "
            f"Validation accuracy: {val_accuracy:.1%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        predictions = model(val_inputs).argmax(dim=1)
        restored_accuracy = (predictions == val_targets).float().mean().item()
    
    macro_f1 = f1_score(
        val_targets.cpu().numpy(),
        predictions.cpu().numpy(),
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    print(
        f"Restored epoch {best_epoch} | "
        f"Validation loss: {best_val_loss:.3f} | "
        f"Validation accuracy: {restored_accuracy:.1%}"
    )

    return {
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_accuracy": restored_accuracy,
        "val_macro_f1": macro_f1,
    }

if __name__ == "__main__":
    df = pd.read_csv("csc311_project_training_202609.csv")
    splits = pd.read_csv("splits.csv")

    split_by_id = splits.set_index("id")["split"]
    row_split = df["id"].map(split_by_id)

    if row_split.isna().any():
        raise ValueError("Some survey responses have no student ID match in splits.csv.")

    pool_df = df.loc[row_split.isin(["train", "val"])].reset_index(drop=True)
    folds = list(iter_cv_folds(pool_df, n_repeats=1))
    fold_results = []

    for fold_number, (fold_train, fold_val) in enumerate(folds, start=1):
        print(f"\nFold {fold_number}/{len(folds)}")
        result = train_one_fold(
            fold_train,
            fold_val,
            seed=311,
            hidden_width=128,
            dropout=0.5,
            learning_rate=0.001,
            weight_decay=0.0001,
            hidden_layers=1,
        )
        fold_results.append(result)

    print("\nSelected MLP results")
    for fold_number, result in enumerate(fold_results, start=1):
        print(
            f"Fold {fold_number} | Best epoch: {result['best_epoch']} | "
            f"Loss: {result['val_loss']:.3f} | "
            f"Accuracy: {result['val_accuracy']:.2%}"
        )

    accuracies = [result["val_accuracy"] for result in fold_results]
    print(f"Mean 5-fold validation accuracy: {np.mean(accuracies):.2%}")

    macro_f1_scores = [result["val_macro_f1"] for result in fold_results]
    print(f"Mean 5-fold macro-F1: {np.mean(macro_f1_scores):.2%}")