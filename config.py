"""
config.py — Application-wide constants and mappings for MovieFlix.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── TMDB ────────────────────────────────────────────────────────────────────
# Reads from .env locally, or from Streamlit Cloud secrets in production.
def _get_tmdb_key() -> str:
    # 1. Try Streamlit secrets (set via Streamlit Cloud dashboard)
    try:
        import streamlit as st
        return st.secrets.get("TMDB_API_KEY", "") or ""
    except Exception:
        pass
    # 2. Fall back to .env / environment variable
    return os.getenv("TMDB_API_KEY", "")

TMDB_API_KEY: str = _get_tmdb_key()
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
TMDB_MOVIE_PAGE = "https://www.themoviedb.org/movie"

POSTER_SIZES = {
    "small": "w185",
    "medium": "w342",
    "large": "w500",
    "original": "original",
}
BACKDROP_SIZES = {
    "medium": "w780",
    "large": "w1280",
    "original": "original",
}

# ─── TMDB genre id → name ─────────────────────────────────────────────────────
GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Sci-Fi",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}

# ─── Mood → genres + keywords ────────────────────────────────────────────────
MOOD_GENRE_MAP = {
    "😄 Happy / Fun": {
        "genres": [35, 12, 16, 10751],
        "keywords": ["comedy", "fun", "adventure", "family", "lighthearted", "feel-good"],
    },
    "😢 Emotional / Moving": {
        "genres": [18, 10749],
        "keywords": ["emotional", "drama", "tearjerker", "moving", "heartfelt", "love"],
    },
    "😨 Scared / Thrilled": {
        "genres": [27, 53, 9648],
        "keywords": ["horror", "thriller", "suspense", "scary", "terrifying", "mystery"],
    },
    "🤩 Inspired / Motivated": {
        "genres": [18, 36, 10752, 99],
        "keywords": ["inspiring", "biography", "true story", "hero", "motivation", "triumph"],
    },
    "🤯 Mind-Blown": {
        "genres": [878, 9648, 53],
        "keywords": ["mind-bending", "twist", "sci-fi", "philosophy", "existential", "mystery"],
    },
    "😎 Relaxed / Chill": {
        "genres": [35, 10749, 10402],
        "keywords": ["relaxing", "casual", "romance", "music", "feel-good", "light"],
    },
    "💥 Pumped / Excited": {
        "genres": [28, 12, 53],
        "keywords": ["action", "explosion", "fight", "chase", "adventure", "adrenaline"],
    },
    "🌙 Dark / Gritty": {
        "genres": [80, 27, 53, 18],
        "keywords": ["dark", "gritty", "neo-noir", "crime", "revenge", "violent"],
    },
}

# ─── Style ────────────────────────────────────────────────────────────────────
STYLES = [
    "Blockbuster",
    "Indie / Art-house",
    "Classic / Retro",
    "Based on True Story",
    "Franchise / Series",
    "Award-winning",
    "Cult Favorite",
    "Foreign Language",
    "Animated",
    "Documentary",
    "Short Film",
]

STYLE_KEYWORD_MAP = {
    "Blockbuster": ["blockbuster", "box office", "summer", "epic"],
    "Indie / Art-house": ["indie", "independent", "art-house", "arthouse", "low budget"],
    "Classic / Retro": ["classic", "retro", "timeless", "old hollywood", "golden age"],
    "Based on True Story": ["based on true story", "true events", "biography", "real life"],
    "Franchise / Series": ["franchise", "sequel", "series", "cinematic universe"],
    "Award-winning": ["oscar", "academy award", "cannes", "golden globe", "award"],
    "Cult Favorite": ["cult", "cult classic", "underground"],
    "Foreign Language": ["foreign", "subtitles", "international"],
    "Animated": ["animation", "animated", "cartoon", "pixar", "studio ghibli"],
    "Documentary": ["documentary", "non-fiction", "real", "true story"],
    "Short Film": ["short", "short film"],
}

# ─── Languages ────────────────────────────────────────────────────────────────
LANGUAGES = {
    "Any": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Italian": "it",
    "Portuguese": "pt",
    "Chinese (Mandarin)": "zh",
    "Arabic": "ar",
    "Russian": "ru",
}

# ─── Release period ───────────────────────────────────────────────────────────
RELEASE_PERIODS = {
    "Any Era": (None, None),
    "2020s": (2020, 2029),
    "2010s": (2010, 2019),
    "2000s": (2000, 2009),
    "1990s": (1990, 1999),
    "1980s": (1980, 1989),
    "Classic (pre-1980)": (1900, 1979),
}

# ─── Duration ────────────────────────────────────────────────────────────────
DURATION_OPTIONS = {
    "Any Length": (0, 999),
    "Short (< 90 min)": (0, 89),
    "Standard (90–120 min)": (90, 120),
    "Long (120–180 min)": (121, 180),
    "Epic (> 180 min)": (181, 999),
}

# ─── Content preference ──────────────────────────────────────────────────────
CONTENT_PREFERENCES = ["Any", "Movies only", "TV Shows only", "Both Movies & TV"]

# ─── Recommendation defaults ─────────────────────────────────────────────────
DEFAULT_CANDIDATE_COUNT = 100   # movies fetched before scoring
MAX_RECOMMENDATIONS = 20        # maximum recommendations shown
API_CACHE_TTL = 3600            # seconds to cache API responses

# ─── Colour palette (used by app.py for charts) ───────────────────────────────
PALETTE_DARK = {
    "bg": "#0e0f14",
    "surface": "#1a1b23",
    "surface2": "#22232e",
    "border": "#2e3044",
    "text": "#e8eaf6",
    "muted": "#8b8fa8",
    "accent": "#7c6af7",
    "accent2": "#e06c4a",
    "success": "#4caf7d",
    "warning": "#f0a030",
    "error": "#e05252",
    "chart_bg": "#1a1b23",
    "chart_paper": "#1a1b23",
    "chart_font": "#e8eaf6",
    "chart_grid": "#2e3044",
}

PALETTE_LIGHT = {
    "bg": "#f5f6fa",
    "surface": "#ffffff",
    "surface2": "#eef0f7",
    "border": "#d0d4e8",
    "text": "#1a1b2e",
    "muted": "#6b6e87",
    "accent": "#5b4de8",
    "accent2": "#d45a30",
    "success": "#2e8b57",
    "warning": "#c07820",
    "error": "#c0392b",
    "chart_bg": "#ffffff",
    "chart_paper": "#f5f6fa",
    "chart_font": "#1a1b2e",
    "chart_grid": "#d0d4e8",
}
