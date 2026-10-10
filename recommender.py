"""
recommender.py — AI-based recommendation engine for MovieFlix.

Uses TF-IDF + cosine similarity on movie metadata (overview, genres, keywords)
combined with weighted preference scoring (mood, genre, style, language, year,
rating, runtime) and optional session-level feedback (likes/dislikes).

Scores are normalised to [0, 1] and labelled to explain each recommendation.
No dummy data is used. All input comes from TMDB API responses.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    GENRE_MAP,
    MOOD_GENRE_MAP,
    STYLE_KEYWORD_MAP,
    RELEASE_PERIODS,
    DURATION_OPTIONS,
)
from tmdb_client import safe_get_title, safe_get_year


# ─── Text preparation ─────────────────────────────────────────────────────────

def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _build_doc(movie: dict) -> str:
    """Combine multiple metadata fields into a single text document for TF-IDF."""
    parts = []
    title = safe_get_title(movie)
    parts.append(_clean(title))

    overview = _clean(movie.get("overview", ""))
    parts.append(overview)

    genre_ids = movie.get("genre_ids") or []
    if not genre_ids:
        genre_ids = [g.get("id", 0) for g in movie.get("genres", [])]
    genre_names = " ".join(GENRE_MAP.get(gid, "") for gid in genre_ids)
    parts.append(_clean(genre_names))

    # keywords from details endpoint
    kw_data = movie.get("keywords", {})
    if isinstance(kw_data, dict):
        kws = kw_data.get("keywords") or kw_data.get("results") or []
        kw_text = " ".join(k.get("name", "") for k in kws)
        parts.append(_clean(kw_text))

    lang = movie.get("original_language", "")
    parts.append(lang)

    return " ".join(p for p in parts if p)


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def _genre_score(movie: dict, selected_genres: list[str]) -> float:
    """1.0 if any selected genre matches, else 0.0. Partial overlap scores proportionally."""
    if not selected_genres:
        return 0.0
    genre_ids = movie.get("genre_ids") or [g.get("id", 0) for g in movie.get("genres", [])]
    movie_genres = {GENRE_MAP.get(gid, "") for gid in genre_ids}
    if not movie_genres:
        return 0.0
    hits = sum(1 for g in selected_genres if g in movie_genres)
    return min(hits / max(len(selected_genres), 1), 1.0)


def _mood_score(movie: dict, selected_moods: list[str]) -> float:
    if not selected_moods:
        return 0.0
    genre_ids = set(
        movie.get("genre_ids") or [g.get("id", 0) for g in movie.get("genres", [])]
    )
    doc = _build_doc(movie)
    total = 0.0
    for mood in selected_moods:
        mapping = MOOD_GENRE_MAP.get(mood, {})
        genre_hits = sum(1 for gid in mapping.get("genres", []) if gid in genre_ids)
        kw_hits = sum(1 for kw in mapping.get("keywords", []) if kw in doc)
        total += (genre_hits / max(len(mapping.get("genres", [1])), 1) * 0.6
                  + kw_hits / max(len(mapping.get("keywords", [1])), 1) * 0.4)
    return min(total / len(selected_moods), 1.0)


def _style_score(movie: dict, selected_styles: list[str]) -> float:
    if not selected_styles:
        return 0.0
    doc = _build_doc(movie)
    scores = []
    for style in selected_styles:
        kws = STYLE_KEYWORD_MAP.get(style, [])
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in doc)
        scores.append(min(hits / len(kws), 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _language_score(movie: dict, language_code: str | None) -> float:
    if not language_code:
        return 0.0
    return 1.0 if movie.get("original_language") == language_code else 0.0


def _year_score(movie: dict, year_from: int | None, year_to: int | None) -> float:
    if year_from is None and year_to is None:
        return 0.0
    year = safe_get_year(movie)
    if year is None:
        return 0.0
    if year_from and year > year_to:
        return 0.0
    if year_to and year > year_to:
        return 0.0
    if year_from and year < year_from:
        return 0.0
    return 1.0


def _rating_score(movie: dict, min_rating: float) -> float:
    if min_rating <= 0:
        return 0.0
    rating = movie.get("vote_average", 0.0) or 0.0
    vote_count = movie.get("vote_count", 0) or 0
    if vote_count < 20:
        return 0.0
    return 1.0 if rating >= min_rating else max(0.0, 1.0 - (min_rating - rating) / 5.0)


def _runtime_score(movie: dict, runtime_min: int, runtime_max: int) -> float:
    if runtime_min == 0 and runtime_max >= 999:
        return 0.0
    rt = movie.get("runtime") or 0
    if rt <= 0:
        return 0.0
    if runtime_min <= rt <= runtime_max:
        return 1.0
    return 0.0


def _feedback_score(movie_id: int, liked_ids: set[int], disliked_ids: set[int]) -> float:
    if movie_id in disliked_ids:
        return -1.0
    if movie_id in liked_ids:
        return 0.3
    return 0.0


def _tfidf_score(movie_idx: int, query_vec: np.ndarray, tfidf_matrix) -> float:
    """Cosine similarity between movie and user query vector."""
    movie_vec = tfidf_matrix[movie_idx]
    sim = cosine_similarity(query_vec, movie_vec)[0][0]
    return float(sim)


# ─── Main recommendation engine ───────────────────────────────────────────────

def build_explanation(
    component_scores: dict[str, float],
    movie: dict,
    genre_names: list[str],
) -> str:
    """Generate a short human-readable reason for recommending this movie."""
    reasons = []

    if component_scores.get("tfidf", 0) > 0.1:
        reasons.append("matches your description")
    if component_scores.get("genre", 0) > 0.5:
        genres_str = ", ".join(genre_names[:2]) if genre_names else ""
        if genres_str:
            reasons.append(f"fits the {genres_str} genre")
    if component_scores.get("mood", 0) > 0.4:
        reasons.append("matches your mood")
    if component_scores.get("style", 0) > 0.3:
        reasons.append("aligns with your preferred style")
    if component_scores.get("language", 0) > 0:
        reasons.append("in your preferred language")
    if component_scores.get("year", 0) > 0:
        reasons.append("from your preferred era")
    rating = movie.get("vote_average", 0.0) or 0.0
    if rating >= 7.5:
        reasons.append(f"highly rated ({rating:.1f}/10)")

    if not reasons:
        reasons.append("popular on TMDB")
    return "Recommended because it " + ", and it ".join(reasons) + "."


def recommend(
    candidates: list[dict],
    preferences: dict,
    liked_ids: set[int] | None = None,
    disliked_ids: set[int] | None = None,
    n: int = 20,
) -> list[dict]:
    """
    Score and rank candidate movies according to user preferences.

    preferences keys (all optional):
        genres      list[str]       genre names
        moods       list[str]       mood labels from MOOD_GENRE_MAP
        styles      list[str]       style labels from STYLES
        language    str | None      TMDB language code
        year_range  tuple(int,int)  (from, to) years
        min_rating  float
        runtime     tuple(int,int)  (min, max) minutes
        description str             free-text query
    """
    if liked_ids is None:
        liked_ids = set()
    if disliked_ids is None:
        disliked_ids = set()

    prefs = preferences or {}
    genres = prefs.get("genres") or []
    moods = prefs.get("moods") or []
    styles = prefs.get("styles") or []
    language = prefs.get("language")
    year_range = prefs.get("year_range") or (None, None)
    min_rating = float(prefs.get("min_rating") or 0.0)
    runtime = prefs.get("runtime") or (0, 999)
    description = (prefs.get("description") or "").strip()

    year_from, year_to = year_range
    rt_min, rt_max = runtime

    # Determine which components are actually active
    has_description = bool(description)
    has_genre = bool(genres)
    has_mood = bool(moods)
    has_style = bool(styles)
    has_language = bool(language)
    has_year = year_from is not None or year_to is not None
    has_rating = min_rating > 0
    has_runtime = rt_min > 0 or rt_max < 999

    active = sum([has_description, has_genre, has_mood, has_style,
                  has_language, has_year, has_rating, has_runtime])

    # Weights (only active components contribute)
    W = {
        "tfidf": 0.25 if has_description else 0.0,
        "genre": 0.25 if has_genre else 0.0,
        "mood": 0.20 if has_mood else 0.0,
        "style": 0.10 if has_style else 0.0,
        "language": 0.08 if has_language else 0.0,
        "year": 0.06 if has_year else 0.0,
        "rating_filter": 0.03 if has_rating else 0.0,
        "runtime_filter": 0.03 if has_runtime else 0.0,
    }

    total_w = sum(W.values())
    if total_w > 0:
        W = {k: v / total_w for k, v in W.items()}
    else:
        # No preferences at all → sort by TMDB popularity + rating
        ranked = sorted(
            [m for m in candidates if m.get("id") not in disliked_ids],
            key=lambda m: (
                (m.get("vote_average") or 0) * min(m.get("vote_count") or 0, 1000) / 1000
                + (m.get("popularity") or 0) / 1000
            ),
            reverse=True,
        )
        for i, m in enumerate(ranked[:n]):
            genre_ids = m.get("genre_ids") or [g.get("id", 0) for g in m.get("genres", [])]
            gnames = [GENRE_MAP.get(gid, "") for gid in genre_ids if gid in GENRE_MAP]
            m["_match_score"] = round(
                min((m.get("vote_average") or 0) / 10.0 * 0.7
                    + min((m.get("popularity") or 0) / 300, 1.0) * 0.3, 1.0), 3
            )
            m["_explanation"] = "Trending and highly rated on TMDB."
            m["_genre_names"] = gnames
        return ranked[:n]

    # Build TF-IDF matrix if description is present
    tfidf_matrix = None
    query_vec = None
    if has_description:
        docs = [_build_doc(m) for m in candidates]
        if docs:
            try:
                vec = TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                tfidf_matrix = vec.fit_transform(docs)
                query_vec = vec.transform([description.lower()])
            except Exception:
                tfidf_matrix = None
                query_vec = None

    scored = []
    for idx, movie in enumerate(candidates):
        mid = movie.get("id")
        if mid in disliked_ids:
            continue

        comp: dict[str, float] = {}

        if tfidf_matrix is not None and query_vec is not None:
            comp["tfidf"] = _tfidf_score(idx, query_vec, tfidf_matrix)
        else:
            comp["tfidf"] = 0.0

        comp["genre"] = _genre_score(movie, genres)
        comp["mood"] = _mood_score(movie, moods)
        comp["style"] = _style_score(movie, styles)
        comp["language"] = _language_score(movie, language)
        comp["year"] = _year_score(movie, year_from, year_to)
        comp["rating_filter"] = _rating_score(movie, min_rating)
        comp["runtime_filter"] = _runtime_score(movie, rt_min, rt_max)

        raw_score = sum(W[k] * comp[k] for k in W)

        # Hard filter: exclude if below min rating (when specified)
        if has_rating:
            rating = movie.get("vote_average") or 0.0
            vote_count = movie.get("vote_count") or 0
            if vote_count >= 20 and rating < min_rating - 0.5:
                continue

        # Hard filter: exclude wrong year range (when specified)
        if has_year:
            year = safe_get_year(movie)
            if year is not None:
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue

        # Apply feedback adjustment
        fb = _feedback_score(mid, liked_ids, disliked_ids)
        final_score = max(0.0, min(1.0, raw_score + fb))

        # Build genre names for display
        genre_ids = movie.get("genre_ids") or [g.get("id", 0) for g in movie.get("genres", [])]
        gnames = [GENRE_MAP.get(gid, "") for gid in genre_ids if gid in GENRE_MAP]

        movie["_match_score"] = round(final_score, 3)
        movie["_explanation"] = build_explanation(comp, movie, gnames)
        movie["_component_scores"] = comp
        movie["_genre_names"] = gnames

        scored.append(movie)

    scored.sort(key=lambda m: m["_match_score"], reverse=True)
    return scored[:n]


# ─── Analytics helpers ────────────────────────────────────────────────────────

def build_analytics_df(movies: list[dict]) -> pd.DataFrame:
    """Convert a list of TMDB movie dicts into a DataFrame suitable for analytics."""
    rows = []
    for m in movies:
        genre_ids = m.get("genre_ids") or [g.get("id", 0) for g in m.get("genres", [])]
        genre_names = [GENRE_MAP.get(gid, "") for gid in genre_ids if gid in GENRE_MAP]
        year = safe_get_year(m)
        rows.append({
            "id": m.get("id"),
            "title": safe_get_title(m),
            "vote_average": m.get("vote_average") or np.nan,
            "vote_count": m.get("vote_count") or 0,
            "popularity": m.get("popularity") or np.nan,
            "year": year,
            "genres": genre_names,
            "genres_str": ", ".join(genre_names),
            "overview": m.get("overview", ""),
            "match_score": m.get("_match_score", np.nan),
            "media_type": m.get("media_type", "movie"),
        })
    return pd.DataFrame(rows)
