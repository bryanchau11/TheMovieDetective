from __future__ import annotations

import os
import time
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from utils.moviespoiler import MovieSpoilerClient

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is missing in your environment.")

DB_PATH = "./db/movie_db"
COLLECTION_NAME = "movie_detective"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs("./db", exist_ok=True)

client = chromadb.PersistentClient(path=DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)

BASE_URL = "https://api.themoviedb.org/3"


def safe_get(url: str, params: dict | None = None, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 429:
                time.sleep(2 + attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1 + attempt)
    return None


def _extract_keywords(keywords_data: dict) -> list[str]:
    rows = keywords_data.get("keywords") or keywords_data.get("results") or []
    return [k.get("name", "") for k in rows if k.get("name")]


def build_title_document(
    item: dict,
    details: dict,
    keywords_data: dict,
    media_type: str,
    spoiler_excerpt: str = "",
    spoiler_source_url: str = "",
) -> tuple[str, dict]:
    is_movie = media_type == "movie"

    if is_movie:
        title = item.get("title", "") or details.get("title", "")
        original_title = details.get("original_title", "") or item.get("original_title", "")
        date_value = item.get("release_date", "") or details.get("release_date", "")
        media_label = "Movie"
    else:
        title = item.get("name", "") or details.get("name", "")
        original_title = details.get("original_name", "") or item.get("original_name", "")
        date_value = item.get("first_air_date", "") or details.get("first_air_date", "")
        media_label = "TV Series"

    overview = item.get("overview", "") or details.get("overview", "")
    year = date_value[:4] if date_value else "N/A"

    genres = [g.get("name", "") for g in details.get("genres", []) if g.get("name")]
    keywords = _extract_keywords(keywords_data)
    tagline = details.get("tagline", "") or ""

    collection_name = ""
    if is_movie and details.get("belongs_to_collection") and details["belongs_to_collection"].get("name"):
        collection_name = details["belongs_to_collection"]["name"]

    production_countries = [
        c.get("name", "") for c in details.get("production_countries", []) if c.get("name")
    ]
    if not production_countries:
        production_countries = [c for c in details.get("origin_country", []) if c]

    spoken_languages = [
        l.get("english_name", "") for l in details.get("spoken_languages", []) if l.get("english_name")
    ]

    poster_path = item.get("poster_path") or details.get("poster_path") or ""
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

    doc_parts = [
        f"Title: {title}",
        f"Media type: {media_label}",
        f"Original title: {original_title}" if original_title else "",
        f"Year: {year}" if year != "N/A" else "",
        f"Overview: {overview}" if overview else "",
        f"Tagline: {tagline}" if tagline else "",
        f"Genres: {', '.join(genres)}" if genres else "",
        f"Keywords: {', '.join(keywords)}" if keywords else "",
        f"Collection: {collection_name}" if collection_name else "",
        f"Countries: {', '.join(production_countries)}" if production_countries else "",
        f"Languages: {', '.join(spoken_languages)}" if spoken_languages else "",
        f"Plot summary: {spoiler_excerpt}" if spoiler_excerpt else "",
    ]

    document = ". ".join([part for part in doc_parts if part]).strip()

    metadata = {
        "title": title,
        "original_title": original_title,
        "year": year,
        "overview": overview,
        "tagline": tagline,
        "genres": ", ".join(genres),
        "keywords": ", ".join(keywords),
        "collection": collection_name,
        "poster": poster_url,
        "media_type": media_type,
        "media_label": media_label,
        "spoiler_excerpt": spoiler_excerpt,
        "spoiler_source_url": spoiler_source_url,
    }

    return document, metadata


def ingest_titles(
    total_pages: int = 150,
    reset_collection: bool = False,
    include_movies: bool = True,
    include_tv: bool = True,
    include_spoilers: bool = False,
    spoilers_cache_dir: str = "data/moviespoiler",
    spoilers_allow_network: bool = True,
    include_classics: bool = False,
    classics_start_year: int = 1920,
    classics_end_year: int = 1979,
):
    global collection

    if reset_collection:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(name=COLLECTION_NAME)

    existing_ids = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    added_count = 0
    skipped_existing = 0
    skipped_incomplete = 0

    sources = []
    if include_movies:
        sources.append(("movie", "/movie/popular", {}))
    if include_tv:
        sources.append(("tv", "/tv/popular", {}))
    if include_classics:
        sources.append((
            "movie",
            "/discover/movie",
            {
                "sort_by": "popularity.desc",
                "primary_release_date.gte": f"{classics_start_year}-01-01",
                "primary_release_date.lte": f"{classics_end_year}-12-31",
                "include_adult": False,
            },
        ))

    if not sources:
        print("Nothing to ingest. Enable include_movies and/or include_tv.")
        return

    print(f"🚀 Starting ingestion for movies/TV with {total_pages} pages each...")

    spoiler_client = None
    if include_spoilers:
        spoiler_client = MovieSpoilerClient(
            cache_dir=Path(spoilers_cache_dir),
            allow_network=spoilers_allow_network,
        )

    for media_type, endpoint, extra_params in sources:
        media_added = 0
        label = f"{media_type.upper()} {endpoint}"
        print(f"\n📦 Ingesting {label}")

        for page in range(1, total_pages + 1):
            popular_url = f"{BASE_URL}{endpoint}"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page,
            }
            params.update(extra_params)
            popular_data = safe_get(popular_url, params=params)

            if not popular_data or "results" not in popular_data:
                print(f"⚠️ [{label}] Skipping page {page} due to fetch error.")
                continue

            ids_batch = []
            docs_batch = []
            metas_batch = []
            embeds_batch = []

            for item in popular_data["results"]:
                raw_id = str(item.get("id", ""))
                if not raw_id:
                    continue

                prefixed_id = f"{media_type}_{raw_id}"
                if prefixed_id in existing_ids or (media_type == "movie" and raw_id in existing_ids):
                    skipped_existing += 1
                    continue

                details_url = f"{BASE_URL}/{media_type}/{raw_id}"
                keywords_url = f"{BASE_URL}/{media_type}/{raw_id}/keywords"

                details = safe_get(details_url, params={"api_key": TMDB_API_KEY, "language": "en-US"})
                keywords_data = safe_get(keywords_url, params={"api_key": TMDB_API_KEY})

                if not details or not keywords_data:
                    skipped_incomplete += 1
                    continue

                overview = item.get("overview", "") or details.get("overview", "")
                if not overview:
                    skipped_incomplete += 1
                    continue

                spoiler_excerpt = ""
                spoiler_source_url = ""
                if spoiler_client and media_type == "movie":
                    title = details.get("title", "") or item.get("title", "")
                    release_date = details.get("release_date", "") or item.get("release_date", "")
                    year = release_date[:4] if release_date else ""
                    spoiler = spoiler_client.get_summary(
                        title=title,
                        year=year,
                        tmdb_id=raw_id,
                    )
                    if spoiler:
                        spoiler_excerpt = spoiler.excerpt
                        spoiler_source_url = spoiler.source_url

                document, metadata = build_title_document(
                    item,
                    details,
                    keywords_data,
                    media_type,
                    spoiler_excerpt,
                    spoiler_source_url,
                )

                if not document.strip():
                    skipped_incomplete += 1
                    continue

                try:
                    embedding = model.encode(document).tolist()
                except Exception:
                    skipped_incomplete += 1
                    continue

                ids_batch.append(prefixed_id)
                docs_batch.append(document)
                metas_batch.append(metadata)
                embeds_batch.append(embedding)

                existing_ids.add(prefixed_id)

            if ids_batch:
                try:
                    collection.add(
                        ids=ids_batch,
                        documents=docs_batch,
                        metadatas=metas_batch,
                        embeddings=embeds_batch
                    )
                    added_count += len(ids_batch)
                    media_added += len(ids_batch)
                except Exception as e:
                    print(f"❌ [{media_type}] Failed to add batch on page {page}: {e}")

            if page % 10 == 0 or page == total_pages:
                print(
                    f"✅ [{label}] Page {page}/{total_pages} | "
                    f"Added: {media_added} | Existing skipped: {skipped_existing} | "
                    f"Incomplete skipped: {skipped_incomplete} | Current total: {collection.count()}"
                )

            time.sleep(0.25)

    print("\n🏁 Ingestion complete.")
    print(f"Added new titles: {added_count}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped incomplete/error: {skipped_incomplete}")
    print(f"Total titles in collection: {collection.count()}")


if __name__ == "__main__":
    # Set reset_collection=True only if you want a full rebuild.
    ingest_titles(
        total_pages=150,
        reset_collection=False,
        include_movies=True,
        include_tv=True,
        include_classics=False,
    )
