from __future__ import annotations
import re
from typing import Dict, List, Iterable
import numpy as np
import pandas as pd
import html
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer


# NORMALIZATION HELPERS
_SOURCE_TRAILER_RE = re.compile(r"\s*[–—-]\s+[A-Z][\w .&’'-]{2,}$")  # strip ' — Reuters' or ' - Bloomberg'

def fix_unicode(s: str) -> str:
    """Unescape HTML, normalize to Unicode NFC, trim/collapse spaces."""
    s = html.unescape(s or "")
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_trailing_source(s: str) -> str:
    """Remove trailing '— Publisher' or ' - Publisher' if present."""
    return _SOURCE_TRAILER_RE.sub("", s).strip()

def normalize_title(title: str) -> str:
    """One-pass normalization for headlines."""
    t = fix_unicode(title)
    t = strip_trailing_source(t)
    t = t.lower()
    return t

# VECTORIZATION
def build_count_matrix(titles: List[str],
                       min_df: int | float = 2,
                       max_df: int | float = 0.7,
                       max_features: int | None = 20000):
    """
    Returns:
      vectorizer, sparse count matrix X (docs x terms), vocabulary index (terms in order)
    """
    # CountVectorizer: bag-of-words COUNTS (not TF-IDF), good for frequency tables.
    # - english stop-words: quick, decent baseline for headlines
    # - ngram_range=(1,2): unigrams + bigrams
    # - min_df / max_df: drop singletons and boilerplate-common tokens
    vect = CountVectorizer(
        lowercase=False,          # we already lowercased in normalize_title
        stop_words="english",     # built-in English list  (you can pass a custom list)
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=min_df,
        max_df=max_df,
        max_features=max_features
        # token_pattern default is (?u)\b\w\w+\b — 2+ word chars (kept)
    )
    X = vect.fit_transform(titles)
    vocab = pd.Index(vect.get_feature_names_out(), name="term")
    return vect, X, vocab

#AGGREGATION
def summarize_terms(X, vocab: pd.Index, top_n: int = 30):
    """
    Produce:
      - overall top unigrams
      - overall top bigrams
      - tidy table (term, ngram, count, share)
    """

    # total counts per term across all docs
    term_counts = pd.Series(np.asarray(X.sum(axis=0)).ravel(), index=vocab, name="count").sort_values(ascending=False)
    total_tokens = int(term_counts.sum()) or 1
    term_df = term_counts.to_frame()
    term_df["share"] = term_df["count"] / total_tokens

    # split by ngram size (space present => bigram+)
    is_bigram_plus = term_df.index.str.contains(r"\s")
    uni = term_df.loc[~is_bigram_plus].head(top_n).reset_index()
    bi  = term_df.loc[is_bigram_plus].head(top_n).reset_index()

    # annotate ngram type for the tidy table
    tidy = term_df.reset_index()
    tidy["ngram"] = tidy["term"].str.contains(r"\s").map(lambda b: "bigram+" if b else "unigram")

    return {
        "top_unigrams": uni,         # columns: term, count, share
        "top_bigrams": bi,           # columns: term, count, share
        "tidy_all_terms": tidy       # columns: term, count, share, ngram
    }

def keyword_frequency(records: Iterable[Dict], top_n: int = 30):
    """
    End-to-end:
      records -> normalized titles -> count matrix -> top terms

    Returns dict with:
      - 'top_unigrams'     (DataFrame: term, count, share)
      - 'top_bigrams'      (DataFrame: term, count, share)
      - 'tidy_all_terms'   (DataFrame: term, count, share, ngram)
      - 'df'               (the normalized dataframe of items)
    """
    # Load into DataFrame and keep only non-empty titles
    df = pd.DataFrame(records, columns=["date", "title", "url"]).copy()
    df = df[df["title"].astype(str).str.strip().ne("")].reset_index(drop=True)

    # Normalize titles
    df["title_norm"] = df["title"].astype(str).map(normalize_title)

    # Vectorize (counts)
    vect, X, vocab = build_count_matrix(df["title_norm"].tolist())

    # Summaries
    out = summarize_terms(X, vocab, top_n=top_n)
    out["df"] = df
    return out

