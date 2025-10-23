from __future__ import annotations
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from dataclasses import dataclass
from typing import Dict, Set
import pandas as pd

# TOKENIZATION
_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")

def tokenize_upper(s: str) -> list[str]:
    """
    Tokenize to UPPERCASE to match LM dictionary convention.
    Headlines are short, so a simple regex works well.
    """
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return [m.group(0).upper() for m in _WORD_RE.finditer(s)]

# LM LEXICON AND METRIC
@dataclass(frozen=True)
class LMLex:
    negative: Set[str]
    uncertainty: Set[str]
    litigious: Set[str]
    constraining: Set[str]
    positive: Set[str]

def load_lm_lexicon_csv(path: str | Path) -> LMLex:
    """
    Load the LM Master Dictionary CSV (or a CSV derived from it) and build sets for the
    categories we care about. Column names vary a bit across releases; we handle common cases.

    Expected columns (case-insensitive):
      - word
      - negative
      - uncertainty
      - litigious
      - constraining
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}  # case-insensitive map

    def need(name: str) -> str:
        if name not in cols:
            raise ValueError(f"LM column missing: {name!r}. Found columns: {list(df.columns)}")
        return cols[name]

    c_word  = need("word")
    c_neg   = need("negative")
    c_unc   = need("uncertainty")
    c_lit   = need("litigious")
    c_con   = need("constraining")
    c_pos   = need("positive")

    def pick(colname: str) -> Set[str]:
        # LM marks membership with 1 (sometimes >0). Keep uppercase to match tokenize_upper().
        return set(df.loc[df[colname] > 0, c_word].astype(str).str.upper())

    return LMLex(
        negative=pick(c_neg),
        uncertainty=pick(c_unc),
        litigious=pick(c_lit),
        constraining=pick(c_con),
        positive=pick(c_pos),
    )

def add_lm_metrics(df: pd.DataFrame, lm: LMLex) -> pd.DataFrame:
    """
    Add LM counts, shares, and binary flags per title.
    - counts: how many tokens from the title are in each LM category
    - shares: counts / total tokens in title (avoid divide-by-zero via max(1, n_tokens))
    - flags : 1 if any token from that category appears in the title
    """
    def score_title(title: str) -> Dict[str, int | float]:
        toks = tokenize_upper(title)
        n = max(1, len(toks))
        c_neg  = sum(tok in lm.negative     for tok in toks)
        c_unc  = sum(tok in lm.uncertainty  for tok in toks)
        c_lit  = sum(tok in lm.litigious    for tok in toks)
        c_con  = sum(tok in lm.constraining for tok in toks)
        c_pos  = sum(tok in lm.positive    for tok in toks)
        return {
            "lm_tokens": n,
            "lm_neg": c_neg,
            "lm_uncert": c_unc,
            "lm_litig": c_lit,
            "lm_constr": c_con,
            "lm_pos": c_pos,
            "lm_neg_share": c_neg / n,
            "lm_uncert_share": c_unc / n,
            "lm_litig_share": c_lit / n,
            "lm_constr_share": c_con / n,
            "lm_pos_share": c_pos / n,
            "lm_has_neg": int(c_neg > 0),
            "lm_has_uncert": int(c_unc > 0),
            "lm_has_litig": int(c_lit > 0),
            "lm_has_constr": int(c_con > 0),
            "lm_has_pos": int(c_pos > 0),
        }

    lm_df = df["title"].map(score_title).apply(pd.Series)
    return pd.concat([df, lm_df], axis=1)

#VADER
_vader = SentimentIntensityAnalyzer()  # vaderSentiment.readthedocs.io

def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER sentiment scores:
      - vader_neg, vader_neu, vader_pos
      - vader_compound (normalized polarity in [-1, 1])
    """
    s = df["title"].astype(str).map(_vader.polarity_scores).apply(pd.Series)
    s = s.rename(columns={"neg": "vader_neg", "neu": "vader_neu", "pos": "vader_pos", "compound": "vader_compound"})
    return pd.concat([df, s], axis=1)

#GENERAL HELPERS
def overall_tone_summary(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Return one-row summary with interpretable, comparable metrics.
    """
    out = pd.DataFrame({
        "n_items": [len(df_scored)],
        "vader_compound_mean": [df_scored["vader_compound"].mean()],
        "vader_compound_median": [df_scored["vader_compound"].median()],
        "lm_negative_headline_pct": [df_scored["lm_has_neg"].mean()],       
        "lm_uncertain_headline_pct": [df_scored["lm_has_uncert"].mean()],
        "lm_positive_headline_pct": [df_scored["lm_has_pos"].mean()],
        "lm_neg_share_mean": [df_scored["lm_neg_share"].mean()],
        "lm_uncert_share_mean": [df_scored["lm_uncert_share"].mean()],
        "lm_pos_share_mean": [df_scored["lm_pos_share"].mean()],
        "lm_litig_share_mean": [df_scored["lm_litig_share"].mean()],
        "lm_constr_share_mean": [df_scored["lm_constr_share"].mean()],
    })
    return out


def top_lm_terms(df_scored: pd.DataFrame, lm: LMLex, top_n: int = 25) -> Dict[str, pd.Series]:
    """
    Show which LM dictionary words most often triggered each category in your corpus.
    This is great for QA and interpretability (you can spot junky tokens or domain specifics).
    """
    counters = {
        "negative": Counter(),
        "uncertainty": Counter(),
        "litigious": Counter(),
        "constraining": Counter(),
        "positive": Counter(),
    }

    for title in df_scored["title"].astype(str):
        toks = tokenize_upper(title)
        for tok in toks:
            if tok in lm.negative:     
                counters["negative"][tok] += 1
            if tok in lm.uncertainty:  
                counters["uncertainty"][tok] += 1
            if tok in lm.litigious:    
                counters["litigious"][tok] += 1
            if tok in lm.constraining: 
                counters["constraining"][tok] += 1
            if tok in lm.positive:    
                counters["positive"][tok] += 1

    # Convert to sorted Series for convenience
    return {k: pd.Series(v).sort_values(ascending=False).head(top_n) for k, v in counters.items()}

def add_lm_and_vader(df: pd.DataFrame, lm_csv_path: str | Path) -> pd.DataFrame:
    """
    Convenience wrapper:
      - loads LM (from your CSV),
      - adds LM metrics and VADER polarity to df,
      - returns a new DataFrame with all features.
    """
    lm = load_lm_lexicon_csv(lm_csv_path)
    df1 = add_lm_metrics(df.copy(), lm)
    df2 = add_vader(df1)
    return df2
