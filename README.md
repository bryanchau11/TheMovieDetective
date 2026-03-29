# TheMovieDetective

## Phase 1 — TMDB Corpus Collection + Cleaning

Phase 1 outputs (what you should see after it works):
- `data/raw/movies/*.json` (raw TMDB bundles)
- `data/raw/movie_ids.json` (resume list)
- `data/processed/corpus.jsonl` (cleaned corpus for embeddings later)

### Quickstart (teammate-friendly)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# create .env and paste your TMDB_API_KEY
cp .env.example .env

python scripts/tmdb_fetch.py --target-count 3000
python scripts/build_corpus.py
```

### 1) Python setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure your TMDB API key

1. Copy [.env.example](.env.example) to `.env`
2. Set `TMDB_API_KEY` inside `.env`

```env
TMDB_API_KEY=YOUR_KEY_HERE
```

### 3) Fetch raw TMDB movie JSON

This downloads movie `details` and `keywords` into `data/raw/movies/` and writes `data/raw/movie_ids.json` so you can resume.

```bash
python scripts/tmdb_fetch.py --target-count 3000
```

Optional flags:

```bash
python scripts/tmdb_fetch.py --target-count 3000 --sleep 0.05 --min-popularity 10
```

### 4) Build the cleaned text corpus

This creates `data/processed/corpus.jsonl` where each line is one movie record with a merged `document` field.

```bash
python scripts/build_corpus.py
```
