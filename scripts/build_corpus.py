from __future__ import annotations

"""Phase 1: turn raw TMDB JSON bundles into a cleaned text corpus (jsonl).

Input:
- data/raw/movies/{tmdb_id}.json (created by scripts/tmdb_fetch.py)

Output:
- data/processed/corpus.jsonl

Each line in corpus.jsonl is one movie (a JSON object) with a merged "document" field.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable
from utils.moviespoiler import load_spoiler_file


WS_RE = re.compile(r"\s+")


def _compact_ws(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = _compact_ws(item)
                if s:
                    out.append(s)
            elif isinstance(item, dict) and "name" in item:
                s = _compact_ws(str(item.get("name") or ""))
                if s:
                    out.append(s)
        return out
    if isinstance(value, str):
        s = _compact_ws(value)
        return [s] if s else []
    return []


def _year_from_date(date_str: Any) -> int | None:
    if not isinstance(date_str, str) or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except ValueError:
        return None


def _build_document(
    *,
    title: str,
    overview: str,
    tagline: str,
    genres: list[str],
    keywords: list[str],
    spoiler_excerpt: str,
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if tagline:
        parts.append(f"Tagline: {tagline}")
    if overview:
        parts.append(f"Overview: {overview}")
    if genres:
        parts.append("Genres: " + ", ".join(genres))
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    if spoiler_excerpt:
        parts.append(f"Plot summary: {spoiler_excerpt}")
    return _compact_ws("\n".join(parts))


def iter_raw_movie_files(raw_movies_dir: Path) -> Iterable[Path]:
    if not raw_movies_dir.exists():
        return []
    return sorted(p for p in raw_movies_dir.glob("*.json") if p.is_file())


def load_corpus_record(movie_json_path: Path) -> dict[str, Any] | None:
    """Convert one raw movie bundle into a normalized record.

    Returns None if the file is missing required fields.
    """
    payload = json.loads(movie_json_path.read_text(encoding="utf-8"))
    details = payload.get("details") or {}
    keywords_payload = payload.get("keywords") or {}

    tmdb_id = details.get("id")
    if tmdb_id is None:
        return None
    tmdb_id = int(tmdb_id)

    title = _compact_ws(str(details.get("title") or ""))
    if not title:
        return None

    release_year = _year_from_date(details.get("release_date"))
    genres = _as_list_of_str(details.get("genres"))

    keywords_list = keywords_payload.get("keywords")
    keywords = _as_list_of_str(keywords_list)

    overview = _compact_ws(str(details.get("overview") or ""))
    tagline = _compact_ws(str(details.get("tagline") or ""))

    spoiler_excerpt = ""
    spoiler_source_url = ""
    data_dir = movie_json_path.parents[2]
    spoiler_path = data_dir / "moviespoiler" / "parsed" / f"{tmdb_id}.json"
    spoiler_payload = load_spoiler_file(spoiler_path)
    if spoiler_payload:
        spoiler_excerpt = _compact_ws(str(spoiler_payload.get("excerpt") or ""))
        spoiler_source_url = _compact_ws(str(spoiler_payload.get("source_url") or ""))

    document = _build_document(
        title=title,
        overview=overview,
        tagline=tagline,
        genres=genres,
        keywords=keywords,
        spoiler_excerpt=spoiler_excerpt,
    )

    return {
        "tmdb_id": tmdb_id,
        "title": title,
        "release_year": release_year,
        "genres": genres,
        "keywords": keywords,
        "overview": overview,
        "tagline": tagline,
        "spoiler_excerpt": spoiler_excerpt,
        "spoiler_source_url": spoiler_source_url,
        "document": document,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    # JSONL = JSON Lines: one JSON object per line.
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a cleaned text corpus (jsonl) from TMDB raw movie JSON bundles."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing raw/ and processed/ (default: data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output jsonl path (default: data/processed/corpus.jsonl)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data_dir: Path = args.data_dir
    out_path: Path = args.out or (data_dir / "processed" / "corpus.jsonl")

    raw_movies_dir = data_dir / "raw" / "movies"
    paths = list(iter_raw_movie_files(raw_movies_dir))
    if not paths:
        raise SystemExit(
            f"No raw movie files found in {raw_movies_dir}. Run scripts/tmdb_fetch.py first."
        )

    records: list[dict[str, Any]] = []
    for path in paths:
        rec = load_corpus_record(path)
        if rec is not None:
            records.append(rec)

    count = write_jsonl(out_path, records)
    print(f"Wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()
