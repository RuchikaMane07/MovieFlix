# 🎬 MovieFlix — AI Movie Recommendation System

A polished, interactive movie discovery platform powered by the TMDB API and a content-based AI recommendation engine.

---

## Features

| Feature | Details |
|---|---|
| 🎯 **AI Recommendations** | TF-IDF + cosine similarity + weighted scoring across mood, genre, style, language, year, rating, runtime |
| 🌙☀️ **Dark / Light Theme** | Full UI theme switcher with cinematic dark mode and clean light mode |
| 📊 **Analytics Dashboard** | 6 interactive Plotly charts — all calculated from real TMDB API data |
| 🔍 **Search** | Full TMDB search by title |
| 🎬 **Movie Details** | Posters, backdrops, full metadata, similar movies, direct TMDB links |
| 👍👎 **Feedback** | Like/dislike movies to influence subsequent recommendations |
| 📱 **Responsive** | Works on desktop, tablet, and mobile |

---

## Requirements

- **Python 3.9+**
- **macOS / Linux / Windows**
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

---

## Installation (macOS)

### 1. Get a TMDB API key

1. Go to [https://www.themoviedb.org/signup](https://www.themoviedb.org/signup) and create a free account.
2. Navigate to **Settings → API** and request an API key (choose "Developer").
3. Copy the **API Key (v3 auth)** — it is a 32-character hex string.

### 2. Clone / download the project

```bash
cd ~/Downloads
# The project folder already exists at:
cd "movie recommendation system"
```

### 3. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your API key

Create a `.env` file in the project folder:

```bash
echo "TMDB_API_KEY=your_32_char_hex_key_here" > .env
```

Replace `your_32_char_hex_key_here` with your actual TMDB API key.

### 6. Run the application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Project Structure

```
movie recommendation system/
├── app.py              # Main Streamlit application (UI, themes, navigation, analytics)
├── recommender.py      # AI recommendation engine (TF-IDF + weighted scoring)
├── tmdb_client.py      # TMDB API client (all API calls, caching, image helpers)
├── config.py           # Constants, genre maps, mood/style mappings, colour palettes
├── requirements.txt    # Python dependencies
├── .env                # Your TMDB API key (never committed to git)
├── .env.example        # Template for .env
└── README.md           # This file
```

---

## How to Use

### Discover Mode (default)
1. Open the **Preferences** panel in the sidebar.
2. Optionally enter a description, select moods, genres, and use **Advanced Filters**.
3. Click **🎬 Get Recs** to generate personalized recommendations.
4. Sort results by **Best Match / Highest Rating / Most Popular / Newest**.
5. Click **ℹ️ Details** on any movie to see full info and similar movies.
6. Use 👍/👎 to like or dislike movies — this influences future recommendations.

### Search Mode
- Use the **🔍 Search** navigation item to search TMDB by movie title.

### Analytics Mode
- Navigate to **📊 Analytics** to see interactive charts about the retrieved movie data.
- All charts update when you generate new recommendations.

### Themes
- Use the **🌙 Dark / ☀️ Light / 💻 Auto** buttons in the sidebar to switch themes.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.32` | Web UI framework |
| `requests` | HTTP calls to TMDB API |
| `python-dotenv` | Load `.env` API key |
| `pandas` | Data manipulation for analytics |
| `numpy` | Numerical operations |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `plotly` | Interactive analytics charts |
| `Pillow` | Image handling (optional) |

---

## How Analytics Are Calculated

All analytics use **real data from TMDB API responses** — no hardcoded values or dummy data.

| Chart | Data Source |
|---|---|
| **Rating Distribution** | `vote_average` field from each TMDB movie in the retrieved candidate set |
| **Genre Distribution** | `genre_ids` field → mapped to genre names using TMDB genre list |
| **Release Year Trends** | `release_date` / `first_air_date` field, parsed as integer year |
| **Popularity vs Rating** | `popularity` and `vote_average` fields plotted against each other |
| **Recommendation Comparison** | Match score (computed by `recommender.py`) vs `vote_average` (from TMDB) |
| **Summary Metrics** | Aggregated from the full candidate dataset (up to 100 movies) |

The analytics analyse the **complete candidate dataset** (up to 100 movies), not just the visible cards.

---

## How Recommendations Work

The engine combines:

1. **TF-IDF + Cosine Similarity** — when a free-text description is provided, movie overviews, genres, and keywords are vectorized and compared against the query.
2. **Genre Scoring** — proportional overlap between selected genres and the movie's genre_ids.
3. **Mood Scoring** — maps moods to genre IDs and keywords, scores based on both genre and text matches.
4. **Style Scoring** — keyword-based matching against style tags (blockbuster, indie, classic, etc.).
5. **Language Scoring** — exact match on `original_language`.
6. **Year Range Scoring** — hard filter + soft score based on release year.
7. **Rating Filter** — soft score based on `vote_average` vs minimum threshold.
8. **Feedback Adjustment** — liked movies get a +0.3 boost; disliked movies are excluded.

All component weights are **normalised by the number of active components**, so selecting only one preference works just as well as selecting many.

---

## TMDB Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.

Movie data © TMDB contributors. Images © respective rights holders.

![TMDB Logo](https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg)
