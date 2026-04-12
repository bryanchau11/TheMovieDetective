# TheMovieDetective

A hybrid movie and TV retrieval app that combines semantic search with intent-aware reranking. It can use spoiler summaries for better recall while keeping spoilers hidden from users.

## Features
- Hybrid search with semantic embeddings + reranking
- Intent-aware results (animation/anime, K-drama, Asian titles)
- Optional spoiler summaries from themoviespoiler.com, hidden from users
- Feedback loop to refine results when the top results are wrong
- Streamlit UI with quick search ideas and extracted clues

## Quickstart

### 1) Create a virtual environment
```bash
cd /Users/charlie/PycharmProjects/TheMovieDetective_clean
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3) Set environment variables
Create a `.env` file in the project root:
```
TMDB_API_KEY=your_tmdb_key
CLAUDE_API_KEY=your_claude_key
```

### 4) Ingest data
```bash
python ingest.py
```

Optional flags are inside `ingest.py` if you want spoilers or classics:
- `include_spoilers=True` to pull spoiler summaries
- `include_classics=True` to ingest older titles

### 5) Run the app
```bash
streamlit run app.py
```

## How It Works
- `ingest.py` pulls movies/TV data from TMDB and builds a local ChromaDB collection.
- `search.py` expands queries, extracts attributes, and reranks results with intent-aware logic.
- `app.py` provides the Streamlit UI and feedback loop.

### Spoilers Are Hidden
If spoiler summaries are enabled, the system uses them only for search relevance. The UI does not display spoilers to users.

## Testing
```bash
python -m pytest tests/test_search.py
```

## Example Queries
- `animated movie about Moses and the exodus`
- `k drama about lost memories and romance`
- `asian crime thriller set in a port city`
- `cartoon about toys that come to life`
- `movie where a ship hits an iceberg and there is a love story`

## Deployment (Free)
- **Streamlit Community Cloud**: connect your GitHub repo and set `CLAUDE_API_KEY` and `TMDB_API_KEY` as secrets.
- **Hugging Face Spaces**: choose a Streamlit space and add secrets in settings.

## Project Structure
```
app.py                # Streamlit UI
ingest.py             # Data ingestion + embeddings
search.py             # Query expansion + reranking
utils/                # Spoiler scraping helpers
styles/               # App CSS
tests/                # Tests for reranking
```

## Notes
- Unique title count in the UI is based on title + year + media type.
- For K-drama or Asian title intent, results are boosted using country/language metadata.

## License
Add your license here.
