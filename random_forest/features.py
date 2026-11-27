"""
Survey feature engineering copied from linear_model/features.py so the
prediction files can be shipped together. Training and prediction both use
this copy. Fit state on training rows with fit_features and apply it to new
rows with transform_features.
"""
import re

import numpy as np
import pandas as pd

# Column names
ID_COL = "id"
EASE_COL = (
    "How easy do you think it would be for someone with no prior experience "
    "to start playing this sport recreationally? Rate it on a scale from 0 "
    "to 10, where 0 means extremely difficult and 10 means extremely easy."
)
LIKERT_COLS = [
    "Success in this sport depends heavily on strategy and planning.",
    "Players must be comfortable with a high risk of injury.",
    "Individual brilliance can outweigh teamwork in this sport.",
]
PLACE_COL = "Where might you play this sport?"
SEASON_COL = (
    "During which seasons would you be most likely to play this sport "
    "recreationally in Toronto?"
)
MUSIC_COL = "What types of music do you most associate with this sport?"
FOOD_COL = (
    "What foods and drinks do you associate with watching this sport in the "
    "place or culture you know best?"
)
COLOUR_COL = "What colour do you associate with this sport?"
SONG_COL = "Which song do you associate with this sport?"
FEELING_COL = "How do you feel when you watch this sport?"
PRICE_COL = (
    "How much would you be willing to pay for a decent seat at a "
    "professional competition of this sport?"
)
LABEL_COL = "label"

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

# Closed checkbox vocabularies -- verified exhaustively against every value
# actually present in the training CSV. Fixed code constants, not fitted
# state: a rare option vanishing from one small CV fold's training partition
# must not change the feature space.
PLACE_VOCAB = (
    "At home or nearby",
    "Community recreation facility",
    "Indoor facility",
    "Others",
    "Outdoor public space",
    "Private sports facility",
    "School or university facility",
)
SEASON_VOCAB = ("Fall", "Spring", "Summer", "Winter")
MUSIC_VOCAB = (
    "Classical or jazz",
    "Country",
    "Electronic or dance",
    "Folk or traditional",
    "Hip-hop or rap",
    "Latin",
    "No particular type",
    "Other",
    "Pop",
    "Rock",
)

# (short name, column, vocab) for the three multi-select columns, in the
# fixed order they appear in the output feature matrix. Adding a future
# multi-select column means adding one entry here, not editing existing code.
MULTISELECT_COLUMNS = (
    ("place", PLACE_COL, PLACE_VOCAB),
    ("season", SEASON_COL, SEASON_VOCAB),
    ("music", MUSIC_COL, MUSIC_VOCAB),
)

# (short name, column) for the three free-text TF-IDF columns, in output order.
TEXT_COLUMNS = (
    ("food", FOOD_COL),
    ("song", SONG_COL),
    ("feeling", FEELING_COL),
)

STOPWORDS = frozenset(
    """
    a an the and or but if then else so to of in on at by for with without
    from as is are was were be been being this that these those it its it's
    i you he she we they me him her us them my your his its our their mine
    yours his hers ours theirs do does did doing have has had having not no
    nor can could will would shall should may might must about above after
    again against all am any because before below between both down during
    each few further here how into more most once only other over own same
    some such than too very what when where which while who whom why
    """.split()
)

COLOUR_CANONICAL = (
    "red", "blue", "green", "yellow", "orange", "white", "black", "brown",
    "purple", "gold", "pink", "grey", "maroon", "silver", "navy", "teal",
    "beige", "violet", "turquoise", "cyan", "magenta",
)
COLOUR_SYNONYMS = {
    "golden": "gold", "balck": "black", "gray": "grey", "navy blue": "navy",
    "dark blue": "blue", "light blue": "blue", "sky blue": "blue",
    "royal blue": "blue", "forest green": "green", "dark green": "green",
    "bright green": "green", "dark red": "red", "crimson": "red",
}
COLOUR_NONANSWER_PHRASES = frozenset({
    "n/a", "na", "none", "not sure", "no idea", "idk", "i don't know",
    "i dont know", "no particular colour", "no particular color",
    "no colour", "no color", "#name?", "unsure",
})

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
SCALE_WORDS = {"hundred": 100, "thousand": 1000}
REFUSAL_PHRASES = (
    "not willing", "unwilling", "not really willing", "not interested",
    "no interest", "nothing", "not much", "not a fan", "not very",
)

# Dispatch table (Simple Factory): which concrete weighting scheme to apply
# to a raw term-count vector. Small, stable set selected by a string key at
# fit time -- an unrecognized key is a programming error, so it raises
# rather than silently falling back to a default.
_WEIGHTING_FUNCS = {
    "tfidf": lambda tf, idf: tf * idf,
    "count": lambda tf, idf: tf,
}

_PRICE_LOG_CLIP_QUANTILE = 0.99


def _apply_weighting(tf, idf, weighting):
    try:
        fn = _WEIGHTING_FUNCS[weighting]
    except KeyError:
        raise ValueError(
            f"unknown weighting: {weighting!r} (expected one of "
            f"{sorted(_WEIGHTING_FUNCS)})"
        ) from None
    return fn(tf, idf)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def _tokenize(text):
    if pd.isna(text):
        return []
    words = re.findall(r"[a-z']+", str(text).lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


# ---------------------------------------------------------------------------
# Multi-select -> multi-hot
# ---------------------------------------------------------------------------
def _multihot(series, vocab):
    option_sets = series.fillna("").astype(str).str.split(",").apply(
        lambda toks: {t.strip() for t in toks if t.strip()}
    )
    columns = [
        option_sets.apply(lambda s: 1.0 if opt in s else 0.0).to_numpy()
        for opt in vocab
    ]
    return np.column_stack(columns) if columns else np.empty((len(series), 0))


# ---------------------------------------------------------------------------
# Colour normalization
# ---------------------------------------------------------------------------
def _normalize_colour(raw):
    """Returns a canonical colour name, 'OTHER' (answered but unmappable),
    or None (missing / explicit non-answer)."""
    if pd.isna(raw):
        return None
    s = str(raw).replace("\xa0", " ").strip().lower().rstrip(". ")
    if s == "" or any(
        s == p or s.startswith(p + " ") or s.startswith(p + ",")
        for p in COLOUR_NONANSWER_PHRASES
    ):
        return None

    best_pos, best_colour = None, None
    for word, canonical in COLOUR_SYNONYMS.items():
        m = re.search(rf"\b{re.escape(word)}\b", s)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_colour = m.start(), canonical
    for colour in COLOUR_CANONICAL:
        m = re.search(rf"\b{re.escape(colour)}\b", s)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_colour = m.start(), colour

    return best_colour if best_colour is not None else "OTHER"


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------
# The U+FFFD alternative below is deliberate, not mangled encoding: some raw
# CSV answers contain a literal replacement character where an en-dash
# should be (e.g. "CAD $80�$120"), confirmed against the real data --
# without it, those rows silently fall through to the generic 2-number mean
# path instead of being recognized as an explicit range.
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|—|�|to|and)\s*(\d+(?:\.\d+)?)"
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS_RE = re.compile(r"(?<=\d)[ ,](?=\d{3}(?:\D|$))")


def _parse_spelled_number(s):
    words = re.findall(r"[a-z]+", s)
    total, found = 0, False
    for w in words:
        if w in NUMBER_WORDS:
            total += NUMBER_WORDS[w]
            found = True
        elif w in SCALE_WORDS:
            total = (total or 1) * SCALE_WORDS[w]
            found = True
    return float(total) if found else None


def _parse_price(raw):
    """Returns a parsed dollar amount, or None if unparseable/missing."""
    if pd.isna(raw):
        return None
    s = str(raw).replace("\xa0", " ").strip().lower()
    s = _THOUSANDS_RE.sub("", s)

    range_match = _RANGE_RE.search(s)
    if range_match:
        a, b = float(range_match.group(1)), float(range_match.group(2))
        return (a + b) / 2.0

    nums = _NUM_RE.findall(s)
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) >= 2:
        # No explicit range keyword matched; approximate with the mean of
        # the first two numbers mentioned, ignoring any further numbers in
        # a caveat clause (documented simplification -- see project plan).
        return (float(nums[0]) + float(nums[1])) / 2.0

    spelled = _parse_spelled_number(s)
    if spelled is not None:
        return spelled

    if any(p in s for p in REFUSAL_PHRASES) or re.search(r"\bzero\b", s):
        return 0.0

    return None


# ---------------------------------------------------------------------------
# Feature name construction (single source of truth for column order/count)
# ---------------------------------------------------------------------------
def _build_feature_names(state):
    names = ["ease"]
    names += [f"likert_{i}" for i in range(len(LIKERT_COLS))]
    for short_name, _col, vocab in MULTISELECT_COLUMNS:
        names += [f"{short_name}__{opt}" for opt in vocab]
    names += [f"colour__{c}" for c in state["colour_top10"]]
    names += ["colour__other", "colour__missing"]
    names += ["price", "price__missing"]
    for short_name, _col in TEXT_COLUMNS:
        vocab = state[f"text_vocab_{short_name}"]
        names += [f"{short_name}__{w}" for w in vocab]
    return names


# ---------------------------------------------------------------------------
# fit_features / transform_features
# ---------------------------------------------------------------------------
def fit_features(df, min_df=3, vocab_cap=300, weighting="tfidf"):
    """
    Fits all preprocessing statistics on the rows in df (and ONLY those
    rows -- call this separately per CV fold / per final train set to avoid
    leakage). Returns a plain dict ("state") that transform_features can
    later apply, unchanged, to any other rows.
    """
    if weighting not in _WEIGHTING_FUNCS:
        raise ValueError(
            f"unknown weighting: {weighting!r} (expected one of "
            f"{sorted(_WEIGHTING_FUNCS)})"
        )

    state = {"min_df": min_df, "vocab_cap": vocab_cap, "weighting": weighting}

    # --- ease ---
    ease = pd.to_numeric(df[EASE_COL], errors="coerce")
    state["ease_median"] = float(ease.median())

    # --- likert ---
    likert_medians = []
    for col in LIKERT_COLS:
        mapped = df[col].map(LIKERT_MAP)
        likert_medians.append(float(mapped.median()))
    state["likert_median"] = np.array(likert_medians)

    # --- colour: top-10 by training-split frequency, alpha tiebreak ---
    resolved = df[COLOUR_COL].map(_normalize_colour)
    counts = resolved[~resolved.isin([None, "OTHER"])].value_counts()
    top10 = sorted(counts.index.tolist(), key=lambda c: (-counts[c], c))[:10]
    state["colour_top10"] = top10

    # --- price ---
    parsed = df[PRICE_COL].map(_parse_price)
    parseable = parsed.dropna()
    price_median = float(parseable.median()) if len(parseable) else 0.0
    state["price_median_dollars"] = price_median
    clip_value = float(np.quantile(np.log1p(parseable.to_numpy(dtype=float)), _PRICE_LOG_CLIP_QUANTILE)) \
        if len(parseable) else float(np.log1p(price_median))
    state["price_log_clip"] = clip_value

    # --- text vocab/idf ---
    n_docs = len(df)
    for short_name, col in TEXT_COLUMNS:
        doc_tokens = df[col].map(_tokenize)
        doc_freq = {}
        for toks in doc_tokens:
            for w in set(toks):
                doc_freq[w] = doc_freq.get(w, 0) + 1
        kept = [w for w, c in doc_freq.items() if c >= min_df]
        kept.sort(key=lambda w: (-doc_freq[w], w))
        kept = kept[:vocab_cap]
        kept.sort()  # deterministic alphabetical indexing
        idf = np.array(
            [np.log((1.0 + n_docs) / (1.0 + doc_freq[w])) + 1.0 for w in kept]
        )
        state[f"text_vocab_{short_name}"] = kept
        state[f"text_idf_{short_name}"] = idf

    state["feature_names"] = _build_feature_names(state)

    # Standardize every column of the assembled matrix (not just the numeric
    # ones) -- multi-hot, colour and TF-IDF columns have very different
    # scales from each other too, and L2/SAGA are scale-sensitive to all of
    # them, not just ease/Likert/price. One mean/std vector over the whole
    # feature matrix, fit on these rows only, replaces the per-block scaling
    # that used to live in a few of the blocks below.
    X_raw = _transform_raw(df, state)
    feature_mean = X_raw.mean(axis=0)
    feature_std = X_raw.std(axis=0, ddof=0)
    feature_std[feature_std == 0] = 1.0
    state["feature_mean"] = feature_mean
    state["feature_std"] = feature_std
    return state


def _transform_raw(df, state):
    """Builds the imputed/encoded feature matrix without the final global
    standardization -- shared by fit_features (to compute feature_mean/std)
    and transform_features (which applies them)."""
    n = len(df)
    blocks = []

    # --- ease ---
    ease = pd.to_numeric(df[EASE_COL], errors="coerce").fillna(state["ease_median"])
    blocks.append(ease.to_numpy(dtype=float).reshape(-1, 1))

    # --- likert ---
    likert_cols = []
    for i, col in enumerate(LIKERT_COLS):
        mapped = df[col].map(LIKERT_MAP).fillna(state["likert_median"][i])
        likert_cols.append(mapped.to_numpy(dtype=float))
    blocks.append(np.column_stack(likert_cols))

    # --- multi-select ---
    for _short_name, col, vocab in MULTISELECT_COLUMNS:
        blocks.append(_multihot(df[col], vocab))

    # --- colour ---
    top10 = state["colour_top10"]
    resolved = df[COLOUR_COL].map(_normalize_colour)
    onehot = np.zeros((n, len(top10)))
    other_flag = np.zeros((n, 1))
    missing_flag = np.zeros((n, 1))
    for i, val in enumerate(resolved):
        if val is None:
            missing_flag[i, 0] = 1.0
        elif val in top10:
            onehot[i, top10.index(val)] = 1.0
        else:
            other_flag[i, 0] = 1.0
    blocks.append(onehot)
    blocks.append(other_flag)
    blocks.append(missing_flag)

    # --- price --- (log1p + clip; global standardization happens later)
    parsed = df[PRICE_COL].map(_parse_price).astype(float)
    missing_flag = parsed.isna().to_numpy(dtype=float).reshape(-1, 1)
    filled = parsed.fillna(state["price_median_dollars"]).to_numpy(dtype=float)
    log_vals = np.clip(np.log1p(filled), None, state["price_log_clip"]).reshape(-1, 1)
    blocks.append(log_vals)
    blocks.append(missing_flag)

    # --- text --- (row-wise L2 normalization stays here: it's about
    # document length, not column scale, so it's independent of the global
    # column standardization applied afterward)
    for short_name, col in TEXT_COLUMNS:
        vocab = state[f"text_vocab_{short_name}"]
        idf = state[f"text_idf_{short_name}"]
        index = {w: i for i, w in enumerate(vocab)}
        mat = np.zeros((n, len(vocab)))
        for row_i, text in enumerate(df[col]):
            for w in _tokenize(text):
                j = index.get(w)
                if j is not None:
                    mat[row_i, j] += 1.0
        mat = _apply_weighting(mat, idf, state["weighting"])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        blocks.append(mat)

    return np.column_stack(blocks) if blocks else np.empty((n, 0))


def transform_features(df, state):
    """Applies previously-fitted state to df. Never mutates state."""
    X_raw = _transform_raw(df, state)
    X = (X_raw - state["feature_mean"]) / state["feature_std"]
    return X, _build_feature_names(state)


# ---------------------------------------------------------------------------
# Serialization (Adapter: in-memory state dict <-> flat .npz arrays)
# ---------------------------------------------------------------------------
_STRING_LIST_KEYS = (
    "colour_top10", "feature_names",
    "text_vocab_food", "text_vocab_song", "text_vocab_feeling",
)
_SCALAR_KEYS = (
    "ease_median", "price_median_dollars", "price_log_clip",
)
_ARRAY_KEYS = (
    "likert_median",
    "text_idf_food", "text_idf_song", "text_idf_feeling",
    "feature_mean", "feature_std",
)


def state_to_npz_kwargs(state):
    """Translates a fitted state dict into a flat dict of numpy arrays
    suitable for np.savez(..., **kwargs)."""
    kwargs = {}
    for key in _STRING_LIST_KEYS:
        kwargs[key] = np.array(state[key], dtype="<U80")
    for key in _SCALAR_KEYS:
        kwargs[key] = np.array(state[key], dtype=np.float64)
    for key in _ARRAY_KEYS:
        kwargs[key] = np.asarray(state[key], dtype=np.float64)
    kwargs["weighting"] = np.array([state["weighting"]], dtype="<U16")
    kwargs["min_df"] = np.array(state["min_df"], dtype=np.int64)
    kwargs["vocab_cap"] = np.array(state["vocab_cap"], dtype=np.int64)
    return kwargs


def npz_kwargs_to_state(npz):
    """Reconstructs a state dict from an np.load(...) NpzFile (or any
    mapping of the same keys/arrays produced by state_to_npz_kwargs)."""
    state = {}
    for key in _STRING_LIST_KEYS:
        state[key] = [str(s) for s in npz[key]]
    for key in _SCALAR_KEYS:
        state[key] = float(npz[key])
    for key in _ARRAY_KEYS:
        state[key] = np.asarray(npz[key], dtype=np.float64)
    state["weighting"] = str(npz["weighting"][0])
    state["min_df"] = int(npz["min_df"])
    state["vocab_cap"] = int(npz["vocab_cap"])
    return state
