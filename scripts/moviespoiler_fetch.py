from __future__ import annotations

"""Phase 1.5: fetch spoiler plot summaries from themoviespoiler(s).com.

This script reads TMDB raw movie bundles from data/raw/movies and stores
spoiler summaries in the shared cache directory: data/moviespoiler/parsed.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from utils.moviespoiler import MovieSpoilerClient


def _iter_raw_movie_files(raw_movies_dir: Path) -> list[Path]:
    if not raw_movies_dir.exists():
        return []
    return sorted(p for p in raw_movies_dir.glob("*.json") if p.is_file())


def _load_movie_title_year(path: Path) -> tuple[int | None, str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details") or {}
    tmdb_id = details.get("id")
    title = str(details.get("title") or "").strip()
    release_date = str(details.get("release_date") or "").strip()
    year = release_date[:4] if release_date else ""
    if tmdb_id is None:
        return None, title, year
    return int(tmdb_id), title, year


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch MovieSpoiler plot summaries for TMDB raw movie bundles."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing raw/ and moviespoiler/ (default: data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of how many movies to process (default: 0 = all)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow network requests to themoviespoiler(s).com",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Seconds to sleep between requests (default: 0.4)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    raw_movies_dir = data_dir / "raw" / "movies"
    paths = _iter_raw_movie_files(raw_movies_dir)
    if not paths:
        raise SystemExit(
            f"No raw movie files found in {raw_movies_dir}. Run scripts/tmdb_fetch.py first."
        )

    client = MovieSpoilerClient(
        cache_dir=data_dir / "moviespoiler",
        allow_network=bool(args.allow_network),
        sleep_seconds=float(args.sleep),
    )

    processed = 0
    for path in paths:
        tmdb_id, title, year = _load_movie_title_year(path)
        if not tmdb_id or not title:
            continue

        client.get_summary(title=title, year=year, tmdb_id=tmdb_id)
        processed += 1
        if args.limit and processed >= args.limit:
            break

    print(f"Processed {processed} titles.")


if __name__ == "__main__":
    main()

