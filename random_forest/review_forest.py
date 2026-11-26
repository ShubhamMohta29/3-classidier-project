"""Compare the proposal and song-focused forests on repeated grouped CV."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from itertools import product
import os
from time import perf_counter

from . import train_forest as training
from . import pred

import numpy as np
import pandas as pd

PROPOSAL_TEXT = (3, 300, "tfidf")
ORIGINAL_TEXT = (5, 100, "tfidf")
ORIGINAL_FOREST = ("words", 1, "0.1", 16, 96)
PARAMETERS = training.TEXT_KEYS + training.MODEL_KEYS
SUBMISSION_LIMIT = 10_000_000
SCRIPTS_SIZE = sum((training.HERE / name).stat().st_size for name in ("pred.py", "features.py"))


def repeated_folds(pool):
    """Keep the shared first pass, then shuffle respondent membership twice."""
    yield from training.iter_cv_folds(pool)
    groups = pool[training.features.ID_COL]
    ids = np.sort(groups.unique())
    for seed in (312, 313):
        shuffled_ids = np.random.default_rng(seed).permutation(ids)
        for validation_ids in np.array_split(shuffled_ids, 5):
            validation = groups.isin(validation_ids)
            yield pool.loc[~validation], pool.loc[validation]


def proposal_grid():
    grid = list(product(("words",), (1, 2, 5, 10), ("sqrt", "0.1", "0.3"),
                        (5, 10, 20, 30, 0), (100, 300, 500)))
    sampled = np.random.default_rng(training.SEED).choice(len(grid), size=60, replace=False)
    return {PROPOSAL_TEXT: [grid[index] for index in sampled]}


def proposal_extension(best):
    """Try one step beyond each expandable edge in the best proposal setting."""
    representation, leaf, features, depth, trees = best
    smaller_or_larger = {
        (0, 10): 15, (1, "sqrt"): "0.03", (1, "0.3"): "0.6",
        (2, 5): 3, (2, 30): 45, (3, 100): 50, (3, 500): 750,
    }
    options = []
    for i, value in enumerate((leaf, features, depth, trees)):
        edge = smaller_or_larger.get((i, value))
        options.append([value] if edge is None else [value, edge])
    return [(representation, *candidate) for candidate in product(*options)
            if (representation, *candidate) != best]


def alternatives_grid():
    previous_scores = pd.read_csv(training.HERE / "cv_scores.csv", dtype={"max_features": str})
    previous_summary = training.summarize(previous_scores)
    shortlist = previous_summary.groupby("representation", sort=False).head(3)
    grid = {ORIGINAL_TEXT: [ORIGINAL_FOREST]}
    for row in shortlist[PARAMETERS].itertuples(index=False, name=None):
        text, config = row[:3], row[3:]
        configs = grid.setdefault(text, [])
        if config not in configs:
            configs.append(config)

    # Expand the previous winner's text cutoff and tree count at the edges.
    for min_df, vocab_cap in product((5, 8), (60, 100)):
        text = (min_df, vocab_cap, "tfidf")
        grid.setdefault(text, []).extend(
            ("words", 1, "0.1", 16, trees)
            for trees in (48, 96)
            if (text, trees) != (ORIGINAL_TEXT, 96)
        )

    # Retain sparse song words while reducing competing food and feeling columns.
    for min_df, vocab_cap, weighting in product((3, 5), (100, 300), ("count", "tfidf")):
        text = (min_df, vocab_cap, weighting)
        grid.setdefault(text, []).extend(
            ("song_words", 1, max_features, 16, 96)
            for max_features in ("0.1", "0.3", "0.6", "1.0")
        )
    # Isolate the effect of searching more features without changing the input.
    grid[ORIGINAL_TEXT].extend(("words", 1, value, 16, 96) for value in ("0.3", "0.6", "1.0"))
    return grid


def development_check(unit):
    pool, text, config = unit
    state, components, forest, _ = training.fit_final(pool, text, config, oob_score=True)
    with BytesIO() as archive:
        training.save_model(archive, state, components, forest)
        total_bytes = archive.tell() + SCRIPTS_SIZE
    return (*text, *config, forest.oob_score_, total_bytes)


def run_development_checks(pool, grids, workers):
    units = [(pool, text, config) for grid in grids for text, configs in grid.items()
             for config in configs]
    if workers == 1:
        records = list(map(development_check, units))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(development_check, units))
    checks = pd.DataFrame(records, columns=PARAMETERS + ["oob_acc", "submission_bytes"])
    checks.to_csv(training.HERE / "review_checks.csv", index=False)
    return checks


def evaluate_song_ablation(folds, winner, selected_scores, workers):
    representation, leaf, max_features, depth, trees = tuple(winner[training.MODEL_KEYS])
    representations = {"words": "no_song_words", "song_words": "no_song_svd", "svd": "no_song_svd"}
    config = (representations[representation], leaf, max_features, depth, trees)
    text = tuple(winner[training.TEXT_KEYS])
    ablation = training.run_cv(folds, {text: [config]}, workers)
    paired = selected_scores[["fold", "acc", "f1", "log_loss"]].merge(
        ablation[["fold", "acc", "f1", "log_loss"]], on="fold", suffixes=("_full", "_without_song"),
    )
    paired.to_csv(training.HERE / "song_ablation.csv", index=False)
    print(f"Same settings without song: {ablation.acc.mean():.6f} mean CV accuracy. "
          f"Paired change with song: {(paired.acc_full - paired.acc_without_song).mean():+.6f}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--export", action="store_true", help="Replace the deployment model with the CV winner.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    started = perf_counter()
    data = training.load_data()
    pool = data[data["split"].isin(["train", "val"])]
    folds = list(repeated_folds(pool))
    memberships = [frozenset(val[training.features.ID_COL]) for _, val in folds]
    assert len(set(memberships)) == 15
    assert all(set(train[training.features.ID_COL]).isdisjoint(val[training.features.ID_COL])
               for train, val in folds)
    print(f"Seed {training.SEED}, 15 distinct validation folds, {len(pool)} development rows.", flush=True)

    proposed = proposal_grid()
    proposal_scores = training.run_cv(folds, proposed, args.workers).assign(search="proposal")
    best_proposal = training.summarize(proposal_scores).iloc[0]
    print(f"Proposal random search complete ({perf_counter() - started:.1f}s).", flush=True)
    extension = proposal_extension(tuple(best_proposal[training.MODEL_KEYS]))
    extensions = {PROPOSAL_TEXT: extension} if extension else {}
    alternatives = alternatives_grid()
    score_sets = [proposal_scores]
    for name, grid in (("proposal_extension", extensions), ("alternative", alternatives)):
        if grid:
            score_sets.append(training.run_cv(folds, grid, args.workers).assign(search=name))
            print(f"{name} CV complete ({perf_counter() - started:.1f}s).", flush=True)
    scores = pd.concat(score_sets, ignore_index=True)
    scores.to_csv(training.HERE / "review_cv_scores.csv", index=False)

    checks = run_development_checks(pool, (proposed, extensions, alternatives), args.workers)
    summary = training.summarize(scores).merge(checks, on=PARAMETERS, validate="one_to_one")
    origins = scores[PARAMETERS + ["search"]].drop_duplicates()
    summary = summary.merge(origins, on=PARAMETERS, validate="one_to_one")
    summary.to_csv(training.HERE / "review_summary.csv", index=False)
    eligible = summary[summary.submission_bytes < SUBMISSION_LIMIT]
    print(f"{len(summary)} configurations, {len(scores)} CV fits, "
          f"submission sizes {checks.submission_bytes.min():,}–{checks.submission_bytes.max():,} bytes.")
    print(f"Best per search:\n{eligible.groupby('search', sort=False).head(1).to_string(index=False)}")
    print(f"Best song-focused model:\n{eligible[eligible.representation == 'song_words'].head(1).to_string(index=False)}")
    winner = eligible.iloc[0]
    print(f"Overall CV winner:\n{winner.to_string()}")
    selected = (scores[PARAMETERS] == winner[PARAMETERS]).all(axis=1)
    evaluate_song_ablation(folds, winner, scores.loc[selected], args.workers)

    if args.export:
        text = tuple(winner[training.TEXT_KEYS])
        config = tuple(winner[training.MODEL_KEYS])
        state, components, forest, design = training.fit_final(data, text, config)
        with BytesIO() as archive:
            training.save_model(archive, state, components, forest)
            if archive.tell() + SCRIPTS_SIZE >= SUBMISSION_LIMIT:
                raise ValueError("Full-data model exceeds the prediction submission limit.")
            archive.seek(0)
            with np.load(archive, allow_pickle=False) as exported:
                np.testing.assert_allclose(
                    pred._forest_probabilities(design, exported), forest.predict_proba(design),
                    rtol=0, atol=1e-12,
                )
            pred.MODEL_PATH.write_bytes(archive.getvalue())
        pred._load_model.cache_clear()
        assert pred.predict_df(data.drop(columns=training.features.LABEL_COL)) == forest.predict(design).tolist()
        print(f"Export verified against scikit-learn on all {len(data)} rows.")


if __name__ == "__main__":
    main()
