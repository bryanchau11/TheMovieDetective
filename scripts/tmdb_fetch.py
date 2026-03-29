from __future__ import annotations

"""Phase 1: download a small TMDB movie corpus (raw JSON).

What this script does:
- Uses your TMDB API key from .env (TMDB_API_KEY)
- Fetches a list of movies (default: /movie/popular)
- For each movie, saves a JSON bundle with:
    - details: /movie/{id}
    - keywords: /movie/{id}/keywords

Outputs:
- data/raw/movies/{tmdb_id}.json
- data/raw/movie_ids.json (used to resume without re-downloading)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tqdm import tqdm


TMDB_API_BASE = "https://api.themoviedb.org/3"


def tmdb_get_json(
    session: requests.Session,
    *,
    api_key: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    url = f"{TMDB_API_BASE}{path}"
    req_params: dict[str, Any] = {"api_key": api_key}
    if params:
        req_params.update(params)
    resp = session.get(url, params=req_params, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _load_existing_ids(raw_dir: Path) -> set[int]:
    # If you re-run the script, we use this file to avoid downloading duplicates.
    ids_path = raw_dir / "movie_ids.json"
    if not ids_path.exists():
        return set()
    try:
        data = json.loads(ids_path.read_text())
        if isinstance(data, list):
            return {int(x) for x in data}
    except Exception:
        return set()
    return set()


def _save_ids(raw_dir: Path, ids: list[int]) -> None:
    _safe_write_json(raw_dir / "movie_ids.json", ids)


def _fetch_list_page(
    session: requests.Session,
    *,
    api_key: str,
    endpoint: str,
    page: int,
    language: str,
    include_adult: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page": page,
        "language": language,
        "include_adult": bool(include_adult),
    }
    return tmdb_get_json(session, api_key=api_key, path=endpoint, params=params)


def _fetch_movie_bundle(
    session: requests.Session, *, api_key: str, movie_id: int, language: str
) -> dict[str, Any]:
    details = tmdb_get_json(
        session, api_key=api_key, path=f"/movie/{movie_id}", params={"language": language}
    )
    keywords = tmdb_get_json(session, api_key=api_key, path=f"/movie/{movie_id}/keywords")
    return {
        "details": details,
        "keywords": keywords,
    }

def run_fetch(
    *,
    api_key: str,
    output_dir: Path,
    target_count: int,
    start_page: int,
    list_endpoint: str,
    language: str,
    include_adult: bool,
    min_popularity: float | None,
    sleep_seconds: float,
) -> None:
    """Fetch raw TMDB JSON into output_dir/raw.

    This function is safe to re-run: it keeps track of already-downloaded movie ids
    in output_dir/raw/movie_ids.json.
    """
    raw_dir = output_dir / "raw"
    movies_dir = raw_dir / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    existing_ids = _load_existing_ids(raw_dir)
    ids: list[int] = sorted(existing_ids)

    page = start_page
    pbar = tqdm(total=target_count, initial=len(ids), desc="Movies fetched")

    try:
        while len(ids) < target_count:
            page_payload = _fetch_list_page(
                session,
                api_key=api_key,
                endpoint=list_endpoint,
                page=page,
                language=language,
                include_adult=include_adult,
            )
            results = page_payload.get("results") or []
            if not results:
                break

            for item in results:
                if len(ids) >= target_count:
                    break

                movie_id = int(item.get("id"))
                if movie_id in existing_ids:
                    continue

                popularity = item.get("popularity")
                if min_popularity is not None and isinstance(popularity, (int, float)):
                    if float(popularity) < float(min_popularity):
                        continue

                try:
                    bundle = _fetch_movie_bundle(
                        session, api_key=api_key, movie_id=movie_id, language=language
                    )
                except requests.HTTPError as e:
                    tqdm.write(f"Skipping {movie_id} due to HTTP error: {e}")
                    continue

                _safe_write_json(movies_dir / f"{movie_id}.json", bundle)
                ids.append(movie_id)
                existing_ids.add(movie_id)
                pbar.update(1)

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            _save_ids(raw_dir, ids)
            page += 1

    finally:
        pbar.close()
        _save_ids(raw_dir, ids)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a TMDB movie corpus (raw JSON) for TheMovieDetective."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=3000,
        help="How many unique movies to fetch (default: 3000)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="Starting page for pagination (default: 1)",
    )
    parser.add_argument(
        "--list-endpoint",
        type=str,
        default="/movie/popular",
        choices=["/movie/popular", "/discover/movie"],
        help="Endpoint used to enumerate movies (default: /movie/popular)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en-US",
        help="TMDB language code (default: en-US)",
    )
    parser.add_argument(
        "--include-adult",
        action="store_true",
        help="Include adult titles",
    )
    parser.add_argument(
        "--min-popularity",
        type=float,
        default=None,
        help="Optional minimum popularity threshold",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Seconds to sleep between movie detail fetches (default: 0.05)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit(
            "TMDB_API_KEY is missing. Create a .env file (see .env.example) and set TMDB_API_KEY."
        )

    run_fetch(
        api_key=api_key,
        output_dir=args.out,
        target_count=int(args.target_count),
        start_page=int(args.start_page),
        list_endpoint=str(args.list_endpoint),
        language=str(args.language),
        include_adult=bool(args.include_adult),
        min_popularity=args.min_popularity,
        sleep_seconds=float(args.sleep),
    )


if __name__ == "__main__":
    main()
