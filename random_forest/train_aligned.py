"""Train the forest using the report's shared text grid and linear CV protocol."""
import argparse
from io import BytesIO
from itertools import product
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from . import train_forest as training
from . import pred

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

FOREST_GRID = list(product(
    ("words",), (1, 2, 5, 10), ("sqrt", "0.1", "0.3"), (8, 16, 0), (96, 192),
))
KEYS = training.TEXT_KEYS + training.MODEL_KEYS
SUBMISSION_LIMIT = 10_000_000


def pick(scores, settings):
    return scores.loc[(scores[KEYS] == settings[KEYS]).all(axis=1)].sort_values("fold")


def evaluate_text(folds, winner, selected, workers):
    """Compare targeted text changes with the already selected word-level forest."""
    text = tuple(winner[training.TEXT_KEYS])
    _, *forest_params = tuple(winner[training.MODEL_KEYS])
    variants = ["song_words", "no_song_words"]
    grid = {text: [(variant, *forest_params) for variant in variants]}
    scores = training.run_cv(folds, grid, workers)
    scores = pd.concat([selected, scores], ignore_index=True)
    scores.to_csv(training.HERE / "aligned_text_checks.csv", index=False)
    print("Fixed forest settings, mean CV accuracy by representation:")
    print(scores.groupby("representation").acc.mean().to_string())
    paired = scores.pivot(index="fold", columns="representation", values="acc")
    print(f"Song ablation: {(paired.words > paired.no_song_words).sum()}/{len(folds)} runs "
          f"improved with songs; mean difference {(paired.words - paired.no_song_words).mean():+.6f}.")


def evaluate_holdout(pool, holdout, text, settings):
    state, components, forest, _ = training.fit_final(pool, text, settings, oob_score=True)
    print(f"Development out-of-bag accuracy: {forest.oob_score_:.6f}")
    X, _ = training.features.transform_features(holdout, state)
    labels = holdout[training.features.LABEL_COL]
    probabilities = forest.predict_proba(pred.project_text(X, components))
    guesses = forest.classes_[probabilities.argmax(axis=1)]
    print("Holdout metrics for the selected settings (previously scored split):")
    for metric, value in training.score(labels, probabilities, forest.classes_).items():
        print(f"  {metric}: {value:.6f}")
    print(f"Confusion matrix, rows=true and columns=predicted, order={list(forest.classes_)}:")
    print(confusion_matrix(labels, guesses, labels=forest.classes_))
    print(classification_report(labels, guesses, labels=forest.classes_, digits=6, zero_division=0))


def export_winner(data, text, settings):
    state, components, forest, design = training.fit_final(data, text, settings)
    script_size = sum((training.HERE / name).stat().st_size for name in ("pred.py", "features.py"))
    with BytesIO() as archive:
        training.save_model(archive, state, components, forest)
        size = archive.tell() + script_size
        if size >= SUBMISSION_LIMIT:
            raise ValueError(f"Prediction files total {size:,} bytes, over the course limit.")
        archive.seek(0)
        with np.load(archive, allow_pickle=False) as exported:
            np.testing.assert_allclose(
                pred._forest_probabilities(design, exported),
                forest.predict_proba(design), rtol=0, atol=1e-12,
            )
        with TemporaryDirectory(dir=pred.MODEL_PATH.parent) as folder:
            staged = Path(folder) / pred.MODEL_PATH.name
            staged.write_bytes(archive.getvalue())
            staged.replace(pred.MODEL_PATH)
    pred._load_model.cache_clear()
    assert pred.predict_df(data.drop(columns=training.features.LABEL_COL)) == forest.predict(design).tolist()
    print(f"Export verified on all {len(data)} rows. Prediction files total {size:,} bytes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    started = perf_counter()
    data = training.load_data()
    pool = data.loc[data["split"].isin(("train", "val"))].reset_index(drop=True)
    folds = list(training.iter_cv_folds(pool, n_repeats=3))
    groups = [frozenset(val[training.features.ID_COL]) for _, val in folds]
    print(f"{len(folds)} shared validation runs covering {len(set(groups))} distinct respondent groups.")
    grid = dict.fromkeys(training.TEXT_GRID, FOREST_GRID)
    print(f"{len(grid)} text settings × {len(FOREST_GRID)} forest settings.", flush=True)
    scores = training.run_cv(folds, grid, args.workers, show_progress=True)
    summary = training.summarize(scores)
    scores.to_csv(training.HERE / "aligned_cv_scores.csv", index=False)
    summary.to_csv(training.HERE / "aligned_cv_summary.csv", index=False)
    print(f"{len(summary)} configurations, {len(scores)} fold scores in {perf_counter() - started:.1f}s.")
    print(summary.head(10).to_string(index=False))

    winner = summary.iloc[0]
    text = tuple(winner[training.TEXT_KEYS])
    settings = tuple(winner[training.MODEL_KEYS])
    print(f"Selected configuration: {dict(zip(KEYS, (*text, *settings)))}")
    evaluate_text(folds, winner, pick(scores, winner), args.workers)
    evaluate_holdout(pool, data.loc[data["split"] == "test"], text, settings)
    export_winner(data, text, settings)


if __name__ == "__main__":
    main()
