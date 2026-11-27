"""
Shared preprocessing for the CSC311 sports classifier.

Uses only the standard library, numpy and pandas, so the exact same code can
be reused inside pred.py (which may not import sklearn).

Usage:
    split = split_by_student(df["id"])           # train / val / test masks
    prep  = Preprocessor().fit(df[split.train])  # learn vocab, medians, scales
    X     = prep.transform(df)                   # numeric matrix
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column names
# --------------------------------------------------------------------------
EASE = ("How easy do you think it would be for someone with no prior experience "
        "to start playing this sport recreationally? Rate it on a scale from 0 to "
        "10, where 0 means extremely difficult and 10 means extremely easy.")
LIKERT = {
    "strategy": "Success in this sport depends heavily on strategy and planning.",
    "injury": "Players must be comfortable with a high risk of injury.",
    "individual": "Individual brilliance can outweigh teamwork in this sport.",
}
PLACE = "Where might you play this sport?"
SEASON = "During which seasons would you be most likely to play this sport recreationally in Toronto?"
MUSIC = "What types of music do you most associate with this sport?"
FOOD = ("What foods and drinks do you associate with watching this sport in the "
        "place or culture you know best?")
COLOUR = "What colour do you associate with this sport?"
SONG = "Which song do you associate with this sport?"
FEEL = "How do you feel when you watch this sport?"
PAY = ("How much would you be willing to pay for a decent seat at a professional "
       "competition of this sport?")

SHORT = {EASE: "ease", **{v: f"likert_{k}" for k, v in LIKERT.items()},
         PLACE: "place", SEASON: "season", MUSIC: "music", FOOD: "food",
         COLOUR: "colour", SONG: "song", FEEL: "feeling", PAY: "pay"}

LIKERT_MAP = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3,
              "Agree": 4, "Strongly agree": 5}

# Options of the three multi-select questions (fixed by the survey)
MULTI = {
    PLACE: ["Indoor facility", "Outdoor public space", "School or university facility",
            "Community recreation facility", "Private sports facility",
            "At home or nearby", "Others"],
    SEASON: ["Winter", "Spring", "Summer", "Fall"],
    MUSIC: ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance", "Latin",
            "Country", "Folk or traditional", "Classical or jazz", "Other",
            "No particular type"],
}

MISSING_TOKENS = {"", "n/a", "na", "-", "--", "none", "null", "nan", "?", "."}
CLASSES = ["Baseball", "Basketball", "Soccer"]


# --------------------------------------------------------------------------
# Cleaning helpers
# --------------------------------------------------------------------------
def normalize_missing(s: pd.Series) -> pd.Series:
    """Map null, empty string, 'n/a', '-', etc. to NaN."""
    s = s.astype("string").str.strip()
    return s.mask(s.isna() | s.str.lower().isin(MISSING_TOKENS))


_ZERO = re.compile(r"(\bzero\b|\bnothing\b|not willing|not really willing|unwilling|"
                   r"no willing|would not pay|wouldn'?t pay|won'?t go|not pay|pay me|"
                   r"^none\b|^no i don|don'?t watch|forget it)", re.I)
_WORD_NUM = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
             "five": 5, "hundred": 100}


def _words_to_number(t: str):
    """'Twenty five dollars' -> 25; returns None if no number words."""
    words = re.findall(r"[a-z]+", t.lower())
    if "hundreds" in words or "couple" in words:   # too vague -> unparseable
        return None
    vals = [_WORD_NUM[w] for w in words if w in _WORD_NUM]
    if not vals:
        return None
    total = 0
    for v in vals:
        total = total * v if v == 100 and total else total + v
    return float(total)
_NUM = re.compile(r"\d[\d,\s]*(?:\.\d+)?")


def parse_price(text) -> float:
    """
    Convert a free-text price answer into dollars.
      '$100'           -> 100       '200 CAD' -> 200      '$1,000' -> 1000
      '100-200'        -> 150 (midpoint of a range)
      'Not willing'    -> 0         'Zero'    -> 0
      'very willing'   -> NaN (no number and no 'zero' phrase -> unparseable)
    """
    if text is None or (isinstance(text, float) and np.isnan(text)) or pd.isna(text):
        return np.nan
    t = str(text).strip()
    nums = []
    for m in _NUM.findall(t):
        m = m.strip().replace(",", "")
        # "10 000" -> 10000, but "100 200" (two numbers) must stay separate
        parts = m.split()
        if len(parts) == 2 and len(parts[1]) == 3:
            m = "".join(parts)
        else:
            m = parts[0]
        try:
            nums.append(float(m))
        except ValueError:
            pass
    if not nums:
        if _ZERO.search(t) or t.lower() in {"none", "no"}:
            return 0.0
        w = _words_to_number(t)
        return w if w is not None else np.nan
    is_range = len(nums) >= 2 and re.search(r"\d\s*(?:-|–|to)\s*\$?\s*\d", t)
    return float(np.mean(nums[:2])) if is_range else nums[0]


def tokenize(text) -> list:
    """Lower-case, split on non-letters, drop 1-letter tokens and stop words."""
    if pd.isna(text):
        return []
    toks = re.findall(r"[a-z]+", str(text).lower())
    return [w for w in toks if len(w) > 1 and w not in STOP]


STOP = set("""a an and the of to in for with or on at by it is be this that my
i me we you they he she as are was were so if but not no do does would like
some any just very really more most much from about into than then sport
there their its also can will one""".split())

COLOUR_SYNONYMS = {"grey": "gray", "golden": "gold", "navy": "blue",
                   "dark blue": "blue", "light blue": "blue", "sky blue": "blue",
                   "dark green": "green", "light green": "green"}


def clean_colour(s: pd.Series) -> pd.Series:
    s = normalize_missing(s).str.lower().str.strip(" .!")
    return s.replace(COLOUR_SYNONYMS)


# --------------------------------------------------------------------------
# Train / val / test split grouped by student
# --------------------------------------------------------------------------
@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_by_student(ids: pd.Series, val_frac=0.15, test_frac=0.15, seed=311) -> Split:
    """All three rows of a respondent land in the same split."""
    rng = np.random.default_rng(seed)
    students = rng.permutation(ids.unique())
    n_test, n_val = int(len(students) * test_frac), int(len(students) * val_frac)
    test_ids = students[:n_test]
    val_ids = students[n_test:n_test + n_val]
    train_ids = students[n_test + n_val:]
    return Split(ids.isin(train_ids).values, ids.isin(val_ids).values,
                 ids.isin(test_ids).values)


# --------------------------------------------------------------------------
# Preprocessor: everything learned from data is learned in fit() only
# --------------------------------------------------------------------------
@dataclass
class Preprocessor:
    min_df: int = 3                 # keep words appearing in >= min_df train rows
    max_vocab: int = 300            # per text field
    top_colours: int = 10
    text_fields: tuple = (FOOD, FEEL, SONG)
    use_tfidf: bool = True
    # learned state
    medians: dict = field(default_factory=dict)
    vocab: dict = field(default_factory=dict)
    idf: dict = field(default_factory=dict)
    colours: list = field(default_factory=list)
    pay_cap: float = 0.0
    mean_: np.ndarray = None
    std_: np.ndarray = None
    names: list = field(default_factory=list)

    # ---- raw (unscaled) feature blocks ----
    def _numeric(self, df):
        out = pd.DataFrame(index=df.index)
        out["ease"] = pd.to_numeric(df[EASE], errors="coerce")
        for k, col in LIKERT.items():
            out[f"likert_{k}"] = normalize_missing(df[col]).map(LIKERT_MAP).astype(float)
        out["log_pay"] = np.log1p(df[PAY].map(parse_price).clip(lower=0))
        return out

    def _multi_hot(self, df):
        out = pd.DataFrame(index=df.index)
        for col, opts in MULTI.items():
            sel = normalize_missing(df[col]).fillna("").str.split(",")
            for o in opts:
                out[f"{SHORT[col]}={o}"] = sel.apply(lambda s, o=o: int(o in s))
        return out

    def fit(self, df):
        num = self._numeric(df)
        self.medians = num.median().to_dict()
        self.pay_cap = float(num["log_pay"].quantile(0.99))
        vc = clean_colour(df[COLOUR]).value_counts()
        self.colours = list(vc.index[: self.top_colours])
        n = len(df)
        for col in self.text_fields:
            docs = df[col].map(lambda t: set(tokenize(t)))
            dfreq = pd.Series([w for d in docs for w in d]).value_counts()
            dfreq = dfreq[dfreq >= self.min_df].iloc[: self.max_vocab]
            self.vocab[col] = list(dfreq.index)
            self.idf[col] = np.log((1 + n) / (1 + dfreq.values)) + 1
        X = self._raw(df)
        self.names = list(X.columns)
        self.mean_ = X.mean().values
        self.std_ = X.std(ddof=0).replace(0, 1).values
        return self

    def _raw(self, df):
        num = self._numeric(df)
        miss = num.isna().astype(int).add_suffix("_missing")[["ease_missing", "log_pay_missing"]]
        num = num.fillna(self.medians)
        num["log_pay"] = num["log_pay"].clip(upper=self.pay_cap)
        blocks = [num, miss, self._multi_hot(df)]

        col = clean_colour(df[COLOUR]).astype(object).where(lambda s: s.notna(), None)
        colour = pd.DataFrame({f"colour={c}": (col == c).astype(int) for c in self.colours},
                              index=df.index)
        colour["colour=other"] = ((~col.isin(self.colours)) & col.notna()).astype(int)
        colour["colour_missing"] = col.isna().astype(int)
        blocks.append(colour)

        for c in self.text_fields:
            toks = df[c].map(tokenize)
            idx = {w: i for i, w in enumerate(self.vocab[c])}
            M = np.zeros((len(df), len(idx)))
            for r, ts in enumerate(toks):
                for w in ts:          # unseen words are simply ignored
                    if w in idx:
                        M[r, idx[w]] += 1
            if self.use_tfidf:
                M = M * self.idf[c]
                norms = np.linalg.norm(M, axis=1, keepdims=True)
                M = M / np.where(norms == 0, 1, norms)
            blocks.append(pd.DataFrame(M, index=df.index,
                                       columns=[f"{SHORT[c]}:{w}" for w in self.vocab[c]]))
        return pd.concat(blocks, axis=1)

    def transform(self, df, scale=True):
        X = self._raw(df).values.astype(float)
        return (X - self.mean_) / self.std_ if scale else X
