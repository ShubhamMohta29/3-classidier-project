"""
This Python file provides some useful code for reading the training file
"csc311_project_training_202609.csv". You may adapt this code as you see fit.
However, keep in mind that the code provided does only basic feature
transformations to build a rudimentary kNN model in sklearn. Not all features
are considered in this code, and you should consider those features! Use this
code where appropriate, but don't stop here!
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = "csc311_project_training_202609.csv"

EASE_COL = "How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy."
LIKERT_COLS = [
    "Success in this sport depends heavily on strategy and planning.",
    "Players must be comfortable with a high risk of injury.",
    "Individual brilliance can outweigh teamwork in this sport.",
]
SEASON_COL = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}


def make_features(df):
    """Builds a numeric feature matrix from a small subset of the questions."""
    features = pd.DataFrame(index=df.index)

    # Numeric rating (non-numeric answers become NaN)
    features["ease"] = pd.to_numeric(df[EASE_COL], errors="coerce")

    # Likert scale questions (e.g. "Agree" -> 4)
    for i, col in enumerate(LIKERT_COLS):
        features[f"likert_{i}"] = df[col].map(LIKERT_MAP)

    # Multi-select question: one binary feature per season
    # (e.g. "Spring,Summer" -> Winter=0, Spring=1, Summer=1, Fall=0)
    seasons = df[SEASON_COL].fillna("").str.split(",")
    for season in SEASONS:
        features[season] = seasons.apply(lambda s: int(season in s))

    return features


def split_by_student(ids, val_frac=0.2, test_frac=0.3, seed=42):
    """
    Splits the unique student ids into train/val/test groups. All three rows
    from the same student end up in the same split, so the model is never
    tested on a student it has already seen during training.
    """
    rng = np.random.default_rng(seed)
    students = rng.permutation(ids.unique())
    n_test = int(len(students) * test_frac)
    n_val = int(len(students) * val_frac)
    test_ids = students[:n_test]
    val_ids = students[n_test:n_test + n_val]
    train_ids = students[n_test + n_val:]
    return ids.isin(train_ids), ids.isin(val_ids), ids.isin(test_ids)


def main():
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)

    # 2. Build features and drop rows with missing values
    X = make_features(df)
    y = df["label"]
    keep = X.notna().all(axis=1)
    X, y, ids = X[keep].values, y[keep].values, df["id"][keep]

    # 3. Split Data by student (50% Train, 20% Val, 30% Test)
    train, val, test = split_by_student(ids)
    X_train, y_train = X[train], y[train]
    X_val, y_val = X[val], y[val]
    X_test, y_test = X[test], y[test]
    print(f"Data Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 4. Find Best k using Validation Set
    best_k = 1
    best_val_acc = 0
    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_acc = knn.score(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
    print(f"Best k found: {best_k}")

    # 5. Apply to Test Set and Report Accuracies
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)

    print(f"Training Accuracy: {final_knn.score(X_train, y_train):.4f}")
    print(f"Validation Accuracy: {final_knn.score(X_val, y_val):.4f}")
    print(f"Test Accuracy: {final_knn.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
