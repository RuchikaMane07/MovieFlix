"""
tmdb_client.py — All TMDB API interactions for MovieFlix.

Uses st.cache_data to avoid redundant API calls during Streamlit reruns.
Never exposes or hardcodes the API key.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import requests
import streamlit as st

from config import (
    TMDB_API_KEY,
    TMDB_BASE_URL,
    TMDB_IMAGE_BASE,
    POSTER_SIZES,
    BACKDROP_SIZES,
    API_CACHE_TTL,
    GENRE_MAP,
)

# ─── Low-level HTTP helper ────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, retries: int = 2) -> dict | None:
    """Make a GET request to the TMDB API. Returns parsed JSON or None on error."""
    if not TMDB_API_KEY:
        return None
    base_params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    if params:
        base_params.update(params)
    url = f"{TMDB_BASE_URL}/{endpoint.lstrip('/')}"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=base_params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                time.sleep(retry_after)
                continue
            if resp.status_code == 401:
                return {"_error": "invalid_api_key"}
            return None
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1)
            else:
                return None
    return None


# ─── Image URL helpers ────────────────────────────────────────────────────────

def poster_url(path: str | None, size: str = "medium") -> str | None:
    """Return the full poster URL for a TMDB poster path, or None."""
    if not path:
        return None
    sz = POSTER_SIZES.get(size, "w342")
    return f"{TMDB_IMAGE_BASE}/{sz}{path}"


def backdrop_url(path: str | None, size: str = "medium") -> str | None:
    """Return the full backdrop URL for a TMDB backdrop path, or None."""
    if not path:
        return None
    sz = BACKDROP_SIZES.get(size, "w780")
    return f"{TMDB_IMAGE_BASE}/{sz}{path}"


# ─── Cached API calls ─────────────────────────────────────────────────────────

@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_trending(media_type: str = "movie", time_window: str = "week") -> list[dict]:
    """Fetch trending movies or TV shows."""
    data = _get(f"trending/{media_type}/{time_window}")
    if not data or "results" not in data:
        return []
    return data["results"]


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_popular(media_type: str = "movie", page: int = 1) -> list[dict]:
    data = _get(f"{media_type}/popular", {"page": page})
    if not data or "results" not in data:
        return []
    return data["results"]


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_top_rated(media_type: str = "movie", page: int = 1) -> list[dict]:
    data = _get(f"{media_type}/top_rated", {"page": page})
    if not data or "results" not in data:
        return []
    return data["results"]


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def search_movies(query: str, page: int = 1) -> dict:
    """Search TMDB for movies/shows by title. Returns full response dict."""
    if not query.strip():
        return {"results": [], "total_pages": 0, "total_results": 0}
    data = _get("search/multi", {"query": query, "page": page, "include_adult": False})
    if not data:
        return {"results": [], "total_pages": 0, "total_results": 0}
    return data


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_movie_details(movie_id: int, media_type: str = "movie") -> dict | None:
    """Fetch full details for a single movie or TV show."""
    return _get(f"{media_type}/{movie_id}", {"append_to_response": "keywords,credits,videos"})


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_similar(movie_id: int, media_type: str = "movie") -> list[dict]:
    data = _get(f"{media_type}/{movie_id}/similar")
    if not data or "results" not in data:
        return []
    return data["results"][:12]


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_discover(
    media_type: str = "movie",
    genre_ids: tuple[int, ...] = (),
    language: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    min_rating: float = 0.0,
    sort_by: str = "popularity.desc",
    page: int = 1,
) -> list[dict]:
    """Discover movies/shows with optional filters."""
    params: dict[str, Any] = {
        "sort_by": sort_by,
        "vote_count.gte": 20,
        "include_adult": False,
        "page": page,
    }
    if genre_ids:
        params["with_genres"] = ",".join(str(g) for g in genre_ids)
    if language:
        params["with_original_language"] = language
    if year_from:
        if media_type == "movie":
            params["primary_release_date.gte"] = f"{year_from}-01-01"
        else:
            params["first_air_date.gte"] = f"{year_from}-01-01"
    if year_to:
        if media_type == "movie":
            params["primary_release_date.lte"] = f"{year_to}-12-31"
        else:
            params["first_air_date.lte"] = f"{year_to}-12-31"
    if min_rating > 0:
        params["vote_average.gte"] = min_rating

    data = _get(f"discover/{media_type}", params)
    if not data or "results" not in data:
        return []
    return data["results"]


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_genre_list(media_type: str = "movie") -> dict[int, str]:
    """Fetch the official TMDB genre list. Returns {id: name}."""
    data = _get(f"genre/{media_type}/list")
    if not data or "genres" not in data:
        return GENRE_MAP
    return {g["id"]: g["name"] for g in data["genres"]}


@st.cache_data(ttl=API_CACHE_TTL, show_spinner=False)
def fetch_search_suggestions(query: str) -> list[str]:
    """Return up to 8 title suggestions for autocomplete."""
    if len(query) < 2:
        return []
    data = _get("search/multi", {"query": query, "page": 1, "include_adult": False})
    if not data or "results" not in data:
        return []
    titles = []
    for r in data["results"][:8]:
        title = r.get("title") or r.get("name") or ""
        if title:
            titles.append(title)
    return titles


def check_api_key() -> tuple[bool, str]:
    """Validate the configured API key. Returns (is_valid, message)."""
    if not TMDB_API_KEY:
        return False, "No API key found. Add TMDB_API_KEY to your .env file."
    data = _get("configuration")
    if data is None:
        return False, "Could not reach TMDB API. Check your internet connection."
    if isinstance(data, dict) and data.get("_error") == "invalid_api_key":
        return False, "Invalid TMDB API key. Check your .env file."
    return True, "API key is valid."


def get_genre_names(genre_ids: list[int]) -> list[str]:
    """Convert a list of genre IDs to genre name strings."""
    return [GENRE_MAP.get(gid, "") for gid in genre_ids if gid in GENRE_MAP]


def safe_get_year(item: dict) -> int | None:
    """Extract a 4-digit release year from a TMDB item dict."""
    date = item.get("release_date") or item.get("first_air_date") or ""
    if date and len(date) >= 4:
        try:
            return int(date[:4])
        except ValueError:
            return None
    return None


def safe_get_title(item: dict) -> str:
    """Return movie title or TV show name."""
    return item.get("title") or item.get("name") or "Unknown"


def safe_get_media_type(item: dict) -> str:
    """Determine whether an item is a movie or tv show."""
    mt = item.get("media_type", "")
    if mt in ("movie", "tv"):
        return mt
    if "title" in item:
        return "movie"
    if "name" in item:
        return "tv"
    return "movie"
