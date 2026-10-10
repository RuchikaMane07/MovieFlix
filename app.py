"""
app.py — MovieFlix AI Movie Recommendation System
Main Streamlit application.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    GENRE_MAP,
    MOOD_GENRE_MAP,
    STYLES,
    LANGUAGES,
    RELEASE_PERIODS,
    DURATION_OPTIONS,
    CONTENT_PREFERENCES,
    DEFAULT_CANDIDATE_COUNT,
    MAX_RECOMMENDATIONS,
    PALETTE_DARK,
    PALETTE_LIGHT,
    TMDB_MOVIE_PAGE,
)
from recommender import recommend, build_analytics_df
from tmdb_client import (
    check_api_key,
    fetch_trending,
    fetch_popular,
    fetch_discover,
    search_movies,
    fetch_movie_details,
    fetch_similar,
    fetch_search_suggestions,
    poster_url,
    backdrop_url,
    safe_get_title,
    safe_get_year,
    get_genre_names,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieFlix · AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Session state defaults ───────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "theme": "dark",
        "nav": "discover",            # discover | analytics | search
        "preferences": {},
        "candidates": [],
        "recommendations": [],
        "liked": set(),
        "disliked": set(),
        "detail_movie": None,
        "detail_media_type": "movie",
        "search_query": "",
        "search_results": [],
        "sort_by": "Best Match",
        "last_fetch_label": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ─── Theme helper ─────────────────────────────────────────────────────────────
def palette() -> dict:
    return PALETTE_DARK if st.session_state.theme == "dark" else PALETTE_LIGHT


def inject_css() -> None:
    p = palette()
    is_dark = st.session_state.theme == "dark"
    css = f"""
<style>
/* ── Global reset ── */
html, body, [class*="css"] {{
    font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
}}
.stApp {{
    background-color: {p['bg']};
    color: {p['text']};
}}
section[data-testid="stSidebar"] {{
    background-color: {p['surface']};
    border-right: 1px solid {p['border']};
}}
/* ── Buttons ── */
.stButton > button {{
    background: {p['accent']};
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.45rem 1.1rem;
    font-weight: 600;
    font-size: 0.85rem;
    transition: opacity 0.2s, transform 0.15s;
    cursor: pointer;
}}
.stButton > button:hover {{
    opacity: 0.88;
    transform: translateY(-1px);
}}
.stButton > button:active {{
    transform: translateY(0);
}}
/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > textarea {{
    background-color: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    color: {p['text']};
    padding: 0.5rem 0.8rem;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {{
    border-color: {p['accent']};
    box-shadow: 0 0 0 2px {p['accent']}33;
    outline: none;
}}
/* ── Multiselect ── */
.stMultiSelect [data-baseweb="tag"] {{
    background-color: {p['accent']}33 !important;
    color: {p['accent']} !important;
    border: 1px solid {p['accent']}66 !important;
    border-radius: 20px !important;
}}
.stMultiSelect [data-baseweb="tag"] span {{
    color: {p['accent']} !important;
}}
/* ── Selectbox ── */
.stSelectbox > div > div {{
    background-color: {p['surface2']};
    border: 1px solid {p['border']};
    border-radius: 8px;
    color: {p['text']};
}}
/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {p['accent']};
}}
/* ── Expander ── */
details[data-testid="stExpander"] {{
    background-color: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 10px;
    padding: 0.5rem;
}}
details[data-testid="stExpander"] summary {{
    color: {p['text']};
    font-weight: 600;
}}
/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 12px;
    padding: 1rem;
}}
[data-testid="stMetricLabel"] {{
    color: {p['muted']} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {p['text']} !important;
    font-size: 1.5rem !important;
    font-weight: 700;
}}
/* ── Dividers ── */
hr {{
    border: none;
    border-top: 1px solid {p['border']};
    margin: 1rem 0;
}}
/* ── Custom card ── */
.movie-card {{
    background: {p['surface']};
    border: 1px solid {p['border']};
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}}
.movie-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px {p['accent']}22;
    border-color: {p['accent']}88;
}}
.movie-card img {{
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}}
.movie-card-body {{
    padding: 0.75rem;
}}
.movie-card-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {p['text']};
    margin: 0 0 0.25rem 0;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.movie-card-meta {{
    font-size: 0.75rem;
    color: {p['muted']};
    margin-bottom: 0.35rem;
}}
.genre-chip {{
    display: inline-block;
    background: {p['accent']}22;
    color: {p['accent']};
    border: 1px solid {p['accent']}44;
    border-radius: 20px;
    padding: 0.1rem 0.5rem;
    font-size: 0.68rem;
    font-weight: 600;
    margin: 0.1rem 0.1rem 0.1rem 0;
    letter-spacing: 0.02em;
}}
.match-score {{
    display: inline-block;
    background: {'#7c6af722' if is_dark else '#5b4de822'};
    color: {p['accent']};
    border: 1px solid {p['accent']}55;
    border-radius: 20px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
}}
.tmdb-rating {{
    color: {p['warning']};
    font-weight: 700;
    font-size: 0.82rem;
}}
.overview-text {{
    font-size: 0.78rem;
    color: {p['muted']};
    line-height: 1.45;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}}
.explanation-text {{
    font-size: 0.72rem;
    color: {p['success']};
    font-style: italic;
    margin-top: 0.4rem;
    line-height: 1.3;
}}
/* ── Hero section ── */
.hero-section {{
    background: linear-gradient(135deg,
        {p['surface']} 0%,
        {p['surface2']} 100%);
    border: 1px solid {p['border']};
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}}
.hero-title {{
    font-size: 2.8rem;
    font-weight: 800;
    color: {p['text']};
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}}
.hero-accent {{
    color: {p['accent']};
}}
.hero-tagline {{
    font-size: 1.1rem;
    color: {p['muted']};
    margin: 0;
}}
/* ── Section header ── */
.section-header {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {p['text']};
    margin: 1.2rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {p['accent']}55;
}}
/* ── Skeleton ── */
.skeleton {{
    background: linear-gradient(90deg,
        {p['surface']} 25%,
        {p['surface2']} 50%,
        {p['surface']} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 10px;
}}
@keyframes shimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}
/* ── Nav pill ── */
.nav-pill {{
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: background 0.2s;
}}
/* ── Alert boxes ── */
.alert-info {{
    background: {p['accent']}18;
    border: 1px solid {p['accent']}44;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: {p['text']};
    font-size: 0.85rem;
    margin: 0.5rem 0;
}}
.alert-success {{
    background: {p['success']}18;
    border: 1px solid {p['success']}44;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: {p['text']};
    font-size: 0.85rem;
}}
.alert-warn {{
    background: {p['warning']}18;
    border: 1px solid {p['warning']}44;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: {p['text']};
    font-size: 0.85rem;
}}
.alert-error {{
    background: {p['error']}18;
    border: 1px solid {p['error']}44;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: {p['text']};
    font-size: 0.85rem;
}}
/* ── Detail modal overlay ── */
.detail-backdrop {{
    border-radius: 16px;
    overflow: hidden;
    position: relative;
    min-height: 200px;
    background-size: cover;
    background-position: center;
}}
.detail-backdrop-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(to right,
        {p['surface']}f0 30%,
        {p['surface']}88 70%,
        transparent 100%);
}}
/* ── Footer ── */
.footer {{
    text-align: center;
    font-size: 0.75rem;
    color: {p['muted']};
    border-top: 1px solid {p['border']};
    padding: 1.2rem 0 0.5rem 0;
    margin-top: 2rem;
}}
/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {p['surface']}; }}
::-webkit-scrollbar-thumb {{ background: {p['border']}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {p['muted']}; }}
/* ── Plotly chart container ── */
.js-plotly-plot {{ border-radius: 12px; overflow: hidden; }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ─── Chart helpers ────────────────────────────────────────────────────────────
def chart_layout(title: str = "") -> dict:
    p = palette()
    return dict(
        title=dict(text=title, font=dict(color=p["text"], size=15, family="SF Pro Display, Segoe UI, system-ui")),
        paper_bgcolor=p["chart_paper"],
        plot_bgcolor=p["chart_bg"],
        font=dict(color=p["chart_font"], family="SF Pro Display, Segoe UI, system-ui"),
        xaxis=dict(
            gridcolor=p["chart_grid"],
            zerolinecolor=p["chart_grid"],
            tickfont=dict(color=p["muted"]),
            title_font=dict(color=p["muted"]),
        ),
        yaxis=dict(
            gridcolor=p["chart_grid"],
            zerolinecolor=p["chart_grid"],
            tickfont=dict(color=p["muted"]),
            title_font=dict(color=p["muted"]),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor=p["surface2"],
            font_color=p["text"],
            bordercolor=p["border"],
        ),
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    p = palette()
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:1rem 0 0.5rem 0;'>"
            f"<span style='font-size:2rem;'>🎬</span><br>"
            f"<span style='font-size:1.2rem;font-weight:800;color:{p['text']};'>MovieFlix</span><br>"
            f"<span style='font-size:0.72rem;color:{p['muted']};'>AI Movie Recommendations</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Theme toggle
        st.markdown(
            f"<div style='font-size:0.78rem;color:{p['muted']};text-transform:uppercase;"
            f"letter-spacing:0.06em;font-weight:600;margin-bottom:0.5rem;'>Theme</div>",
            unsafe_allow_html=True,
        )
        theme_col1, theme_col2, theme_col3 = st.columns(3)
        with theme_col1:
            if st.button("🌙 Dark", use_container_width=True,
                         type="primary" if st.session_state.theme == "dark" else "secondary"):
                st.session_state.theme = "dark"
                st.rerun()
        with theme_col2:
            if st.button("☀️ Light", use_container_width=True,
                         type="primary" if st.session_state.theme == "light" else "secondary"):
                st.session_state.theme = "light"
                st.rerun()
        with theme_col3:
            system_theme = _detect_system_theme()
            if st.button("💻 Auto", use_container_width=True,
                         type="primary" if st.session_state.theme == "system" else "secondary"):
                st.session_state.theme = system_theme
                st.rerun()
        st.divider()

        # Navigation
        st.markdown(
            f"<div style='font-size:0.78rem;color:{p['muted']};text-transform:uppercase;"
            f"letter-spacing:0.06em;font-weight:600;margin-bottom:0.5rem;'>Navigation</div>",
            unsafe_allow_html=True,
        )
        nav_items = [
            ("🎯 Discover", "discover"),
            ("🔍 Search", "search"),
            ("📊 Analytics", "analytics"),
        ]
        for label, key in nav_items:
            is_active = st.session_state.nav == key
            if st.button(
                label,
                use_container_width=True,
                type="primary" if is_active else "secondary",
                key=f"nav_{key}",
            ):
                st.session_state.nav = key
                if key == "analytics" and not st.session_state.candidates:
                    _load_default_candidates()
                st.rerun()

        st.divider()

        # Preferences panel (only in Discover mode)
        if st.session_state.nav == "discover":
            render_preferences_sidebar()
            st.divider()

        # Footer attribution
        st.markdown(
            f"<div style='font-size:0.68rem;color:{p['muted']};text-align:center;'>"
            f"This product uses the <a href='https://www.themoviedb.org/documentation/api' "
            f"style='color:{p['accent']};text-decoration:none;'>TMDB API</a> but is not endorsed "
            f"or certified by TMDB.</div>",
            unsafe_allow_html=True,
        )


def _detect_system_theme() -> str:
    """Heuristic: return 'dark'. Streamlit has no JS bridge, so dark is the safe default."""
    return "dark"


# ─── Preferences sidebar panel ────────────────────────────────────────────────
def render_preferences_sidebar() -> None:
    p = palette()
    st.markdown(
        f"<div style='font-size:0.78rem;color:{p['muted']};text-transform:uppercase;"
        f"letter-spacing:0.06em;font-weight:600;margin-bottom:0.5rem;'>Preferences</div>",
        unsafe_allow_html=True,
    )
    prefs = st.session_state.preferences

    # Description / free text
    desc = st.text_area(
        "Describe what you want",
        value=prefs.get("description", ""),
        placeholder="e.g. A mind-bending sci-fi thriller with plot twists",
        height=70,
        key="pref_desc",
    )

    # Mood
    moods_all = list(MOOD_GENRE_MAP.keys())
    sel_moods = st.multiselect(
        "Mood",
        options=moods_all,
        default=prefs.get("moods", []),
        key="pref_moods",
    )

    # Genre
    genres_all = sorted(GENRE_MAP.values())
    sel_genres = st.multiselect(
        "Genre",
        options=genres_all,
        default=prefs.get("genres", []),
        key="pref_genres",
    )

    # Advanced Filters
    with st.expander("⚙️ Advanced Filters", expanded=False):
        # Style
        sel_styles = st.multiselect(
            "Movie Style",
            options=STYLES,
            default=prefs.get("styles", []),
            key="pref_styles",
        )

        # Language
        lang_names = list(LANGUAGES.keys())
        default_lang = "Any"
        if prefs.get("language"):
            rev = {v: k for k, v in LANGUAGES.items()}
            default_lang = rev.get(prefs.get("language"), "Any")
        sel_lang_name = st.selectbox(
            "Language",
            options=lang_names,
            index=lang_names.index(default_lang),
            key="pref_lang",
        )
        sel_lang = LANGUAGES[sel_lang_name]

        # Release period
        period_names = list(RELEASE_PERIODS.keys())
        default_period = "Any Era"
        sel_period = st.selectbox(
            "Release Period",
            options=period_names,
            index=period_names.index(default_period),
            key="pref_period",
        )
        year_from, year_to = RELEASE_PERIODS[sel_period]

        # Minimum rating
        min_rating = st.slider(
            "Minimum TMDB Rating",
            min_value=0.0,
            max_value=9.0,
            value=float(prefs.get("min_rating", 0.0)),
            step=0.5,
            key="pref_rating",
            help="0 = no filter",
        )

        # Duration
        dur_names = list(DURATION_OPTIONS.keys())
        sel_dur = st.selectbox(
            "Movie Duration",
            options=dur_names,
            index=0,
            key="pref_dur",
        )
        rt_min, rt_max = DURATION_OPTIONS[sel_dur]

        # Content preference
        sel_content = st.selectbox(
            "Content Type",
            options=CONTENT_PREFERENCES,
            index=0,
            key="pref_content",
        )
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Get Recs", use_container_width=True, key="apply_prefs"):
            st.session_state.preferences = {
                "description": desc,
                "moods": sel_moods,
                "genres": sel_genres,
                "styles": sel_styles,
                "language": sel_lang,
                "year_range": (year_from, year_to),
                "min_rating": min_rating,
                "runtime": (rt_min, rt_max),
                "content": sel_content,
            }
            _fetch_and_recommend()
            st.rerun()
    with col2:
        if st.button("🗑 Clear", use_container_width=True, key="clear_prefs"):
            st.session_state.preferences = {}
            st.session_state.recommendations = []
            st.session_state.candidates = []
            st.rerun()


# ─── Data loading ─────────────────────────────────────────────────────────────
def _determine_media_types(content_pref: str) -> list[str]:
    if content_pref == "Movies only":
        return ["movie"]
    if content_pref == "TV Shows only":
        return ["tv"]
    return ["movie"]  # default to movie for simplicity in discover


def _fetch_and_recommend() -> None:
    prefs = st.session_state.preferences
    content_pref = prefs.get("content", "Any")
    media_types = _determine_media_types(content_pref)

    genres_selected = prefs.get("genres") or []
    moods_selected = prefs.get("moods") or []
    language = prefs.get("language")
    year_range = prefs.get("year_range") or (None, None)
    min_rating = float(prefs.get("min_rating") or 0.0)

    # Build genre_ids from selected genre names
    rev_genre = {v: k for k, v in GENRE_MAP.items()}
    genre_ids = tuple(rev_genre[g] for g in genres_selected if g in rev_genre)

    # Add mood genres to discovery
    mood_genre_ids = set()
    for mood in moods_selected:
        from config import MOOD_GENRE_MAP as MGM
        mood_genre_ids.update(MGM.get(mood, {}).get("genres", []))
    all_genre_ids = tuple(set(genre_ids) | mood_genre_ids)

    candidates: list[dict] = []
    media_type = media_types[0]

    pages_needed = math.ceil(DEFAULT_CANDIDATE_COUNT / 20)

    if all_genre_ids or language or year_range[0] or year_range[1] or min_rating:
        for page in range(1, pages_needed + 1):
            results = fetch_discover(
                media_type=media_type,
                genre_ids=all_genre_ids,
                language=language,
                year_from=year_range[0],
                year_to=year_range[1],
                min_rating=min_rating,
                page=page,
            )
            candidates.extend(results)
            if len(results) < 20:
                break
    else:
        # No strong filters → use trending + popular
        candidates.extend(fetch_trending(media_type, "week"))
        candidates.extend(fetch_popular(media_type, 1))
        candidates.extend(fetch_popular(media_type, 2))
        candidates.extend(fetch_popular(media_type, 3))

    # Deduplicate by id
    seen = set()
    unique = []
    for m in candidates:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(m)

    st.session_state.candidates = unique
    st.session_state.last_fetch_label = f"{len(unique)} movies fetched"
    st.session_state.recommendations = recommend(
        unique,
        prefs,
        liked_ids=st.session_state.liked,
        disliked_ids=st.session_state.disliked,
        n=MAX_RECOMMENDATIONS,
    )


def _load_default_candidates() -> None:
    """Load trending + popular movies when no preferences are set."""
    candidates = []
    candidates.extend(fetch_trending("movie", "week"))
    candidates.extend(fetch_popular("movie", 1))
    candidates.extend(fetch_popular("movie", 2))
    candidates.extend(fetch_popular("movie", 3))
    candidates.extend(fetch_popular("movie", 4))

    seen = set()
    unique = []
    for m in candidates:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(m)

    st.session_state.candidates = unique
    st.session_state.last_fetch_label = f"{len(unique)} trending & popular movies"
    if not st.session_state.recommendations:
        st.session_state.recommendations = recommend(
            unique, {}, n=MAX_RECOMMENDATIONS
        )


# ─── Movie card renderer ─────────────────────────────────────────────────────
def render_movie_card(movie: dict, show_match: bool = True) -> None:
    p = palette()
    title = safe_get_title(movie)
    year = safe_get_year(movie) or ""
    rating = movie.get("vote_average") or 0.0
    vote_count = movie.get("vote_count") or 0
    overview = movie.get("overview", "No overview available.")
    genre_names = movie.get("_genre_names") or get_genre_names(
        movie.get("genre_ids") or [g.get("id", 0) for g in movie.get("genres", [])]
    )
    match_score = movie.get("_match_score")
    explanation = movie.get("_explanation", "")
    mid = movie.get("id")
    media_type = movie.get("media_type", "movie")
    if "title" in movie and "media_type" not in movie:
        media_type = "movie"

    # Poster
    p_url = poster_url(movie.get("poster_path"), "medium")

    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

        # Poster image
        if p_url:
            st.markdown(
                f'<img src="{p_url}" alt="{title} poster" '
                f'style="width:100%;aspect-ratio:2/3;object-fit:cover;display:block;border-radius:14px 14px 0 0;">',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="width:100%;aspect-ratio:2/3;display:flex;align-items:center;'
                f'justify-content:center;background:{p["surface2"]};color:{p["muted"]};'
                f'font-size:2.5rem;border-radius:14px 14px 0 0;">🎬</div>',
                unsafe_allow_html=True,
            )

        # Card body
        genre_chips = "".join(
            f'<span class="genre-chip">{g}</span>'
            for g in genre_names[:3]
        )
        rating_stars = "⭐" if rating >= 7.0 else ("👍" if rating >= 5.5 else "")
        match_pct = f"{int(match_score * 100)}%" if match_score is not None else ""

        st.markdown(
            f"""<div class="movie-card-body">
  <div class="movie-card-title" title="{title}">{title}</div>
  <div class="movie-card-meta">
    {year} &nbsp;·&nbsp;
    <span class="tmdb-rating">{rating_stars} {rating:.1f}/10</span>
    &nbsp;({vote_count:,} votes)
  </div>
  <div style="margin-bottom:0.4rem;">{genre_chips}</div>
  {'<span class="match-score">🎯 Match: ' + match_pct + '</span>' if show_match and match_score is not None else ''}
  <div class="overview-text" style="margin-top:0.4rem;">{overview}</div>
  {'<div class="explanation-text">' + explanation + '</div>' if explanation else ''}
</div>""",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Action buttons
        b1, b2, b3 = st.columns(3)
        with b1:
            liked = mid in st.session_state.liked
            if st.button("👍" if not liked else "💚", key=f"like_{mid}_{media_type}",
                         use_container_width=True, help="Like this movie"):
                if liked:
                    st.session_state.liked.discard(mid)
                else:
                    st.session_state.liked.add(mid)
                    st.session_state.disliked.discard(mid)
        with b2:
            disliked = mid in st.session_state.disliked
            if st.button("👎" if not disliked else "🚫", key=f"dislike_{mid}_{media_type}",
                         use_container_width=True, help="Dislike this movie"):
                if disliked:
                    st.session_state.disliked.discard(mid)
                else:
                    st.session_state.disliked.add(mid)
                    st.session_state.liked.discard(mid)
        with b3:
            if st.button("ℹ️ Details", key=f"detail_{mid}_{media_type}",
                         use_container_width=True):
                st.session_state.detail_movie = mid
                st.session_state.detail_media_type = media_type
                st.rerun()


# ─── Movie detail view ────────────────────────────────────────────────────────
def render_movie_detail() -> None:
    p = palette()
    mid = st.session_state.detail_movie
    media_type = st.session_state.detail_media_type

    with st.spinner("Loading movie details…"):
        details = fetch_movie_details(mid, media_type)

    if not details:
        st.markdown(
            '<div class="alert-error">Could not load movie details. Please try again.</div>',
            unsafe_allow_html=True,
        )
        if st.button("← Back"):
            st.session_state.detail_movie = None
            st.rerun()
        return

    title = safe_get_title(details)
    overview = details.get("overview", "No overview available.")
    rating = details.get("vote_average") or 0.0
    vote_count = details.get("vote_count") or 0
    runtime = details.get("runtime") or 0
    release_date = details.get("release_date") or details.get("first_air_date") or "N/A"
    orig_lang = details.get("original_language", "N/A").upper()
    genres = [g.get("name", "") for g in details.get("genres", [])]
    poster = poster_url(details.get("poster_path"), "large")
    backdrop = backdrop_url(details.get("backdrop_path"), "large")
    tmdb_url = f"{TMDB_MOVIE_PAGE}/{mid}"

    if st.button("← Back to results", key="back_from_detail"):
        st.session_state.detail_movie = None
        st.rerun()

    st.divider()

    # Backdrop with overlay
    if backdrop:
        st.markdown(
            f'<div style="border-radius:16px;overflow:hidden;margin-bottom:1.5rem;">'
            f'<img src="{backdrop}" style="width:100%;height:300px;object-fit:cover;display:block;">'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Main content
    col_img, col_info = st.columns([1, 2], gap="large")

    with col_img:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown(
                f'<div style="aspect-ratio:2/3;background:{p["surface2"]};border-radius:12px;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:4rem;">🎬</div>',
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(
            f'<h1 style="color:{p["text"]};margin:0 0 0.5rem 0;font-size:2rem;font-weight:800;">{title}</h1>',
            unsafe_allow_html=True,
        )

        # Genre chips
        genre_chips = "".join(f'<span class="genre-chip">{g}</span>' for g in genres)
        st.markdown(genre_chips, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("⭐ TMDB Rating", f"{rating:.1f}/10")
        with m2:
            st.metric("🗳️ Votes", f"{vote_count:,}")
        with m3:
            st.metric("⏱️ Runtime", f"{runtime} min" if runtime else "N/A")
        with m4:
            st.metric("🌐 Language", orig_lang)

        st.markdown(
            f'<div style="margin:0.8rem 0;font-size:0.85rem;color:{p["muted"]};">📅 Released: {release_date}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:0.95rem;color:{p["text"]};line-height:1.6;">{overview}</div>',
            unsafe_allow_html=True,
        )

        # Find movie in recommendations for explanation
        explanation = ""
        for m in st.session_state.recommendations:
            if m.get("id") == mid:
                explanation = m.get("_explanation", "")
                break
        if explanation:
            st.markdown(
                f'<div class="alert-info" style="margin-top:0.8rem;">💡 {explanation}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<a href="{tmdb_url}" target="_blank" style="background:{p["accent"]};color:#fff;'
            f'padding:0.5rem 1.2rem;border-radius:8px;text-decoration:none;font-weight:600;'
            f'font-size:0.85rem;">🔗 View on TMDB</a>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Similar movies
    st.markdown('<div class="section-header">Similar Movies</div>', unsafe_allow_html=True)
    with st.spinner("Loading similar movies…"):
        similar = fetch_similar(mid, media_type)

    if similar:
        cols = st.columns(min(len(similar[:6]), 6))
        for i, sim_movie in enumerate(similar[:6]):
            with cols[i]:
                sim_title = safe_get_title(sim_movie)
                sim_year = safe_get_year(sim_movie) or ""
                sim_rating = sim_movie.get("vote_average") or 0.0
                sim_poster = poster_url(sim_movie.get("poster_path"), "small")
                if sim_poster:
                    st.image(sim_poster, use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="aspect-ratio:2/3;background:{p["surface2"]};'
                        f'border-radius:8px;display:flex;align-items:center;'
                        f'justify-content:center;font-size:2rem;">🎬</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="font-size:0.75rem;font-weight:600;color:{p["text"]};">'
                    f'{sim_title}</div>'
                    f'<div style="font-size:0.68rem;color:{p["muted"]};">'
                    f'{sim_year} · ⭐{sim_rating:.1f}</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Details", key=f"sim_detail_{sim_movie.get('id')}",
                             use_container_width=True):
                    st.session_state.detail_movie = sim_movie.get("id")
                    st.session_state.detail_media_type = sim_movie.get("media_type", "movie")
                    st.rerun()
    else:
        st.markdown(
            '<div class="alert-warn">No similar movies found in TMDB data.</div>',
            unsafe_allow_html=True,
        )


# ─── Discover page ────────────────────────────────────────────────────────────
def render_discover() -> None:
    p = palette()

    # Hero
    st.markdown(
        f"""<div class="hero-section">
  <div class="hero-title">🎬 Movie<span class="hero-accent">Flix</span></div>
  <div class="hero-tagline">Your AI-powered movie discovery companion.<br>
  Set your mood, pick your genre, and let the magic happen.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Load defaults if nothing fetched yet
    if not st.session_state.candidates:
        with st.spinner("🍿 Loading trending movies…"):
            _load_default_candidates()

    recs = st.session_state.recommendations
    has_prefs = bool(st.session_state.preferences)

    # Sort controls
    sort_col, info_col = st.columns([2, 3])
    with sort_col:
        sort_options = ["Best Match", "Highest Rating", "Most Popular", "Newest"]
        st.session_state.sort_by = st.selectbox(
            "Sort by", sort_options,
            index=sort_options.index(st.session_state.sort_by),
            key="sort_select",
        )
    with info_col:
        fetch_info = st.session_state.get("last_fetch_label", "")
        liked_count = len(st.session_state.liked)
        disliked_count = len(st.session_state.disliked)
        if fetch_info:
            st.markdown(
                f'<div style="padding-top:1.6rem;font-size:0.78rem;color:{p["muted"]};">'
                f'📦 {fetch_info}'
                + (f' &nbsp;·&nbsp; 👍 {liked_count}' if liked_count else '')
                + (f' &nbsp;·&nbsp; 👎 {disliked_count}' if disliked_count else '')
                + f'</div>',
                unsafe_allow_html=True,
            )

    # Apply sort
    sort_by = st.session_state.sort_by
    if recs:
        if sort_by == "Highest Rating":
            recs_display = sorted(recs, key=lambda m: m.get("vote_average") or 0.0, reverse=True)
        elif sort_by == "Most Popular":
            recs_display = sorted(recs, key=lambda m: m.get("popularity") or 0.0, reverse=True)
        elif sort_by == "Newest":
            recs_display = sorted(recs, key=lambda m: safe_get_year(m) or 0, reverse=True)
        else:
            recs_display = sorted(recs, key=lambda m: m.get("_match_score") or 0.0, reverse=True)
    else:
        recs_display = []

    if not recs_display:
        st.markdown(
            f"""<div class="alert-info">
  <b>No recommendations yet.</b><br>
  Use the <b>Preferences</b> panel on the left to set your mood and genres, then click <b>Get Recs</b>.<br>
  Or just browse with no filters — we'll show you what's trending on TMDB.
</div>""",
            unsafe_allow_html=True,
        )
        if st.button("🎲 Show Me Trending Movies", key="load_trending"):
            with st.spinner("Loading…"):
                _load_default_candidates()
            st.rerun()
        return

    # Show recommendations header
    title_label = "🎯 Recommended for You" if has_prefs else "🔥 Trending on TMDB"
    st.markdown(f'<div class="section-header">{title_label} ({len(recs_display)} results)</div>',
                unsafe_allow_html=True)

    if has_prefs and st.button("🔄 Refresh Recommendations", key="refresh_recs"):
        with st.spinner("Recalculating…"):
            _fetch_and_recommend()
        st.rerun()

    # Grid of cards — 4 columns
    cols_per_row = 4
    for row_start in range(0, len(recs_display), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for col_i, movie in enumerate(recs_display[row_start:row_start + cols_per_row]):
            with cols[col_i]:
                render_movie_card(movie, show_match=has_prefs)


# ─── Search page ─────────────────────────────────────────────────────────────
def render_search() -> None:
    p = palette()
    st.markdown('<div class="hero-section" style="padding:1.5rem;">'
                f'<div class="hero-title" style="font-size:1.8rem;">🔍 Search Movies</div>'
                f'<div class="hero-tagline">Find any movie in the TMDB database</div>'
                f'</div>', unsafe_allow_html=True)

    query = st.text_input(
        "Search for a movie",
        value=st.session_state.search_query,
        placeholder="e.g. Inception, The Dark Knight, Parasite…",
        key="search_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_clicked = st.button("🔍 Search", use_container_width=True, key="search_btn")
    with col2:
        if st.session_state.search_results and st.button("✕ Clear Results",
                                                          use_container_width=True,
                                                          key="clear_search"):
            st.session_state.search_query = ""
            st.session_state.search_results = []
            st.rerun()

    if search_clicked and query.strip():
        st.session_state.search_query = query
        with st.spinner("Searching TMDB…"):
            results = search_movies(query)
        filtered = [
            r for r in results.get("results", [])
            if r.get("media_type") in ("movie", "tv", None)
               and (r.get("poster_path") or r.get("backdrop_path"))
        ]
        st.session_state.search_results = filtered

    results = st.session_state.search_results

    if not results and st.session_state.search_query:
        st.markdown(
            f'<div class="alert-warn">No results found for "<b>{st.session_state.search_query}</b>". '
            f'Try a different search term.</div>',
            unsafe_allow_html=True,
        )
        return

    if results:
        total = len(results)
        st.markdown(
            f'<div class="section-header">Results for "{st.session_state.search_query}" ({total} found)</div>',
            unsafe_allow_html=True,
        )
        cols_per_row = 4
        for row_start in range(0, min(len(results), 20), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for col_i, movie in enumerate(results[row_start:row_start + cols_per_row]):
                with cols[col_i]:
                    render_movie_card(movie, show_match=False)


# ─── Analytics page ───────────────────────────────────────────────────────────
def render_analytics() -> None:
    p = palette()

    st.markdown(
        f'<div class="hero-section" style="padding:1.5rem;">'
        f'<div class="hero-title" style="font-size:1.8rem;">📊 Analytics Dashboard</div>'
        f'<div class="hero-tagline">Real data from TMDB API responses &mdash; no fabricated values</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    candidates = st.session_state.candidates
    recommendations = st.session_state.recommendations

    if not candidates:
        with st.spinner("Loading data for analytics…"):
            _load_default_candidates()
        candidates = st.session_state.candidates

    if not candidates:
        st.markdown(
            '<div class="alert-error">No data available. Please check your API connection.</div>',
            unsafe_allow_html=True,
        )
        return

    df = build_analytics_df(candidates)
    recs_df = build_analytics_df(recommendations) if recommendations else pd.DataFrame()

    # Source info
    import datetime
    fetch_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f'<div style="font-size:0.78rem;color:{p["muted"]};margin-bottom:1rem;">'
        f'📦 Analysing <b>{len(df)}</b> movies from TMDB API &nbsp;·&nbsp; '
        f'Retrieved at <b>{fetch_time}</b> &nbsp;·&nbsp; '
        f'<em>All charts are calculated from this retrieved dataset.</em></div>',
        unsafe_allow_html=True,
    )

    # F. Summary metrics
    st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)
    _render_summary_metrics(df, p)

    st.divider()

    # A. Rating distribution
    st.markdown('<div class="section-header">A. Rating Distribution</div>', unsafe_allow_html=True)
    _render_rating_distribution(df, p)

    st.divider()

    # B. Genre distribution
    st.markdown('<div class="section-header">B. Genre Distribution</div>', unsafe_allow_html=True)
    _render_genre_distribution(df, p)

    st.divider()

    # C. Release year trends
    st.markdown('<div class="section-header">C. Release Year Trends</div>', unsafe_allow_html=True)
    _render_year_trends(df, p)

    st.divider()

    # D. Popularity vs Rating
    st.markdown('<div class="section-header">D. Popularity vs. Rating</div>', unsafe_allow_html=True)
    _render_popularity_scatter(df, p)

    st.divider()

    # E. Recommendation comparison
    if not recs_df.empty:
        st.markdown('<div class="section-header">E. Recommendation Comparison</div>',
                    unsafe_allow_html=True)
        _render_recommendation_comparison(recs_df, p)
        st.divider()

    # G. User preference analysis
    prefs = st.session_state.preferences
    if prefs.get("genres") and not recs_df.empty:
        st.markdown('<div class="section-header">G. Preference vs. Result Genre Analysis</div>',
                    unsafe_allow_html=True)
        _render_preference_analysis(prefs, recs_df, p)
        st.divider()


def _render_summary_metrics(df: pd.DataFrame, p: dict) -> None:
    valid_ratings = df["vote_average"].dropna()
    valid_ratings = valid_ratings[valid_ratings > 0]

    n_movies = len(df)
    avg_rating = valid_ratings.mean() if not valid_ratings.empty else None
    best_movie_row = df.loc[df["vote_average"].idxmax()] if not valid_ratings.empty else None
    best_movie_name = best_movie_row["title"] if best_movie_row is not None else "N/A"
    best_movie_rating = best_movie_row["vote_average"] if best_movie_row is not None else None

    all_genres: list[str] = []
    for genres in df["genres"]:
        all_genres.extend(genres)
    n_genres = len(set(all_genres))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🎬 Movies Analysed", n_movies)
    with c2:
        st.metric("⭐ Avg TMDB Rating",
                  f"{avg_rating:.2f}/10" if avg_rating is not None else "N/A")
    with c3:
        st.metric(
            "🏆 Highest Rated",
            best_movie_name[:18] + "…" if len(best_movie_name) > 18 else best_movie_name,
            delta=f"⭐ {best_movie_rating:.1f}" if best_movie_rating else None,
        )
    with c4:
        st.metric("🎭 Distinct Genres", n_genres)


def _render_rating_distribution(df: pd.DataFrame, p: dict) -> None:
    valid = df["vote_average"].dropna()
    valid = valid[valid > 0]

    if valid.empty:
        st.markdown('<div class="alert-warn">No rating data available.</div>',
                    unsafe_allow_html=True)
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid,
        nbinsx=20,
        marker_color=p["accent"],
        opacity=0.85,
        name="Movies",
        hovertemplate="Rating: %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **chart_layout(f"TMDB Rating Distribution — {len(valid)} movies"),
        xaxis_title="TMDB Vote Average",
        yaxis_title="Number of Movies",
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'Distribution of <b>vote_average</b> values across {len(valid)} movies in the retrieved dataset. '
        f'Movies with fewer than 20 votes are included but may not reflect audience consensus.</div>',
        unsafe_allow_html=True,
    )


def _render_genre_distribution(df: pd.DataFrame, p: dict) -> None:
    from collections import Counter

    all_genres: list[str] = []
    for genres in df["genres"]:
        all_genres.extend(genres)

    if not all_genres:
        st.markdown('<div class="alert-warn">No genre data available.</div>',
                    unsafe_allow_html=True)
        return

    counts = Counter(all_genres)
    genre_df = pd.DataFrame(counts.items(), columns=["Genre", "Count"])
    genre_df = genre_df.sort_values("Count", ascending=False)

    fig = px.bar(
        genre_df,
        x="Genre",
        y="Count",
        color="Count",
        color_continuous_scale=[[0, p["surface2"]], [1, p["accent"]]],
        template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
    )
    fig.update_layout(
        **chart_layout(f"Genre Distribution — {len(df)} movies"),
        xaxis_title="Genre",
        yaxis_title="Number of Movies",
        coloraxis_showscale=False,
        xaxis_tickangle=-35,
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y} movies<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'Genre counts may overlap: a movie can belong to multiple genres. '
        f'Total genre assignments: <b>{len(all_genres)}</b> across <b>{len(df)}</b> movies.</div>',
        unsafe_allow_html=True,
    )


def _render_year_trends(df: pd.DataFrame, p: dict) -> None:
    year_data = df["year"].dropna().astype(int)
    year_data = year_data[year_data > 1900]

    if year_data.empty:
        st.markdown('<div class="alert-warn">No release year data available.</div>',
                    unsafe_allow_html=True)
        return

    year_counts = year_data.value_counts().reset_index()
    year_counts.columns = ["Year", "Count"]
    year_counts = year_counts.sort_values("Year")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_counts["Year"],
        y=year_counts["Count"],
        mode="lines+markers",
        line=dict(color=p["accent"], width=2.5),
        marker=dict(color=p["accent2"], size=6),
        fill="tozeroy",
        fillcolor=p["accent"] + "22",
        hovertemplate="Year: %{x}<br>Movies: %{y}<extra></extra>",
        name="Movies",
    ))
    fig.update_layout(
        **chart_layout(f"Movies by Release Year — {len(year_data)} movies with valid dates"),
        xaxis_title="Release Year",
        yaxis_title="Number of Movies",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'Shows the release year distribution of the <b>{len(year_data)}</b> movies in the retrieved dataset. '
        f'This is <em>not</em> a representation of all movies released globally — '
        f'only the movies currently fetched by MovieFlix.</div>',
        unsafe_allow_html=True,
    )


def _render_popularity_scatter(df: pd.DataFrame, p: dict) -> None:
    scatter_df = df[
        df["popularity"].notna() & df["vote_average"].notna() &
        (df["popularity"] > 0) & (df["vote_average"] > 0)
    ].copy()

    if scatter_df.empty:
        st.markdown('<div class="alert-warn">Insufficient data for scatter plot.</div>',
                    unsafe_allow_html=True)
        return

    fig = px.scatter(
        scatter_df,
        x="popularity",
        y="vote_average",
        hover_name="title",
        hover_data={"vote_count": True, "popularity": ":.1f", "vote_average": ":.2f"},
        color="vote_average",
        color_continuous_scale=[[0, p["surface2"]], [0.5, p["accent"]], [1, p["accent2"]]],
        template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
    )
    fig.update_layout(
        **chart_layout(f"Popularity vs. Rating — {len(scatter_df)} movies"),
        xaxis_title="TMDB Popularity Score (platform-specific metric)",
        yaxis_title="TMDB Vote Average",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'<b>TMDB Popularity</b> is a proprietary platform metric influenced by page views, '
        f'watchlist additions, and vote activity — it is not the same as audience rating.</div>',
        unsafe_allow_html=True,
    )


def _render_recommendation_comparison(recs_df: pd.DataFrame, p: dict) -> None:
    top = recs_df.dropna(subset=["match_score"]).nlargest(15, "match_score")
    if top.empty:
        st.markdown('<div class="alert-warn">No recommendation data to compare.</div>',
                    unsafe_allow_html=True)
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="TMDB Rating (÷10)",
        x=top["title"],
        y=top["vote_average"] / 10.0,
        marker_color=p["accent2"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>TMDB Rating: %{customdata:.1f}/10<extra></extra>",
        customdata=top["vote_average"],
    ))
    fig.add_trace(go.Bar(
        name="Match Score (0–1)",
        x=top["title"],
        y=top["match_score"],
        marker_color=p["accent"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Match Score: %{y:.0%}<extra></extra>",
    ))
    fig.update_layout(
        **chart_layout(f"Top Recommendations — Match Score vs. TMDB Rating"),
        barmode="group",
        xaxis_tickangle=-35,
        yaxis_title="Score (0–1 scale)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=p["text"]),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'<b>Match Score</b> is an internally calculated preference-fit score (0–1). '
        f'<b>TMDB Rating</b> is the official TMDB vote average, normalised to 0–1 for comparison. '
        f'These are distinct metrics.</div>',
        unsafe_allow_html=True,
    )


def _render_preference_analysis(prefs: dict, recs_df: pd.DataFrame, p: dict) -> None:
    from collections import Counter
    selected_genres = prefs.get("genres") or []
    if not selected_genres:
        return

    rec_genres: list[str] = []
    for g_list in recs_df["genres"]:
        rec_genres.extend(g_list)

    if not rec_genres:
        st.markdown('<div class="alert-warn">No genre data in recommendations.</div>',
                    unsafe_allow_html=True)
        return

    rec_counts = Counter(rec_genres)
    combined = set(selected_genres) | set(rec_counts.keys())

    pref_vals = [1 if g in selected_genres else 0 for g in sorted(combined)]
    rec_vals = [rec_counts.get(g, 0) for g in sorted(combined)]
    labels = sorted(combined)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Your Selected Genres",
        x=labels,
        y=pref_vals,
        marker_color=p["success"],
        opacity=0.8,
    ))
    rec_max = max(rec_vals) if rec_vals else 1
    fig.add_trace(go.Bar(
        name="Recommended Movie Genres (normalised)",
        x=labels,
        y=[v / rec_max for v in rec_vals],
        marker_color=p["accent"],
        opacity=0.8,
    ))
    fig.update_layout(
        **chart_layout("Your Genre Preferences vs. Recommended Movie Genres"),
        barmode="group",
        xaxis_tickangle=-30,
        yaxis_title="Relative Presence",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="font-size:0.75rem;color:{p["muted"]};">'
        f'Compares your selected genres (shown as 0 or 1) with the genre distribution '
        f'of the recommended movies (normalised). Based on real user selections and real TMDB data.</div>',
        unsafe_allow_html=True,
    )


# ─── API key error state ──────────────────────────────────────────────────────
def render_api_error(message: str) -> None:
    p = palette()
    st.markdown(
        f"""<div class="alert-error">
<b>🚨 API Key Error</b><br>{message}<br><br>
<b>To fix this:</b><br>
1. Create a <code>.env</code> file in the project folder.<br>
2. Add your key: <code>TMDB_API_KEY=your_key_here</code><br>
3. Get a free key at <a href="https://www.themoviedb.org/settings/api" style="color:{p['accent']};">
   themoviedb.org/settings/api</a><br>
4. Restart the app: <code>streamlit run app.py</code>
</div>""",
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    # Validate API key
    api_ok, api_msg = check_api_key()
    if not api_ok:
        render_api_error(api_msg)
        st.markdown(
            f'<div class="footer">This product uses the TMDB API but is not endorsed or certified by TMDB.</div>',
            unsafe_allow_html=True,
        )
        return

    render_sidebar()

    nav = st.session_state.nav

    # Detail overlay takes priority
    if st.session_state.detail_movie is not None:
        render_movie_detail()
    elif nav == "discover":
        render_discover()
    elif nav == "search":
        render_search()
    elif nav == "analytics":
        render_analytics()

    # Footer
    p = palette()
    st.markdown(
        f'<div class="footer">'
        f'🎬 <b>MovieFlix</b> &nbsp;·&nbsp; '
        f'This product uses the <a href="https://www.themoviedb.org/documentation/api" '
        f'style="color:{p["accent"]};text-decoration:none;">TMDB API</a> but is not endorsed '
        f'or certified by TMDB. &nbsp;·&nbsp; '
        f'Movie data &copy; TMDB contributors.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
