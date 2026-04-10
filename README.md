# TheMovieDetective

A hybrid movie/TV retrieval app that enriches TMDB metadata with spoiler plot summaries
(when available) to improve recall on vague, plot-driven queries.

## Setup

1. Create a `.env` file with your keys:

```
TMDB_API_KEY=your_tmdb_key
CLAUDE_API_KEY=your_claude_key
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Ingest (TMDB + optional spoilers)

```
python ingest.py
```

To enable spoiler summaries during ingestion, run the module manually and pass
`include_spoilers=True` inside `ingest.py` or call `ingest_titles` from a short script.
Spoiler summaries are cached under `data/moviespoiler/`.

## Optional: Pre-fetch spoiler summaries

```
python scripts/moviespoiler_fetch.py --allow-network
```

This script reads `data/raw/movies/*.json` (from `scripts/tmdb_fetch.py`) and stores
summaries in `data/moviespoiler/parsed/` for later corpus builds.

## Run the app

```
streamlit run app.py
```
