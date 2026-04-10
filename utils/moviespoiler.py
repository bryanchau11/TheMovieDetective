from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse, urljoin
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

DEFAULT_BASE_URLS = [
    "https://themoviespoilers.com",
    "https://themoviespoiler.com",
]


@dataclass
class SpoilerSummary:
    summary: str
    excerpt: str
    source_url: str


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_title(text: str) -> str:
    text = _compact_ws(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _url_map_key(title: str, year: str | int | None) -> str:
    norm_title = _normalize_title(title)
    year_text = str(year).strip() if year else ""
    return f"title:{norm_title}|{year_text}" if norm_title else ""


def _load_url_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_url_map(path: Path, url_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(url_map, ensure_ascii=True, indent=2))


def _robots_allows(url: str, user_agent: str, cache: dict[str, robotparser.RobotFileParser]) -> bool:
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            rp.read()
            cache[base] = rp
        return cache[base].can_fetch(user_agent, url)
    except Exception:
        return False


def _extract_summary_from_html(html: str, max_chars: int = 2400) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("div", id="content")
        or soup.find("div", class_=re.compile(r"(content|post|entry|article)", re.I))
    )

    paragraphs = []
    if main:
        paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

    cleaned = [p for p in (_compact_ws(p) for p in paragraphs) if len(p) > 60]
    summary = " ".join(cleaned[:6])
    summary = _compact_ws(summary)
    if max_chars and len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary


def _build_excerpt(summary: str, max_chars: int = 900) -> str:
    summary = _compact_ws(summary)
    if not max_chars or len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rsplit(" ", 1)[0] + "..."


class MovieSpoilerClient:
    def __init__(
        self,
        *,
        cache_dir: Path,
        allow_network: bool = True,
        user_agent: str = "MovieDetectiveBot/1.0",
        sleep_seconds: float = 0.4,
        timeout_seconds: int = 20,
        url_map_path: Path | None = None,
        base_urls: list[str] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.raw_dir = cache_dir / "raw"
        self.parsed_dir = cache_dir / "parsed"
        self.allow_network = allow_network
        self.user_agent = user_agent
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.base_urls = base_urls or DEFAULT_BASE_URLS
        self.url_map_path = url_map_path or (cache_dir / "url_map.json")
        self._url_map = _load_url_map(self.url_map_path)
        self._robots_cache: dict[str, robotparser.RobotFileParser] = {}

    def get_summary(
        self,
        *,
        title: str,
        year: str | int | None,
        tmdb_id: int | str,
        url_override: str | None = None,
    ) -> SpoilerSummary | None:
        tmdb_id_text = str(tmdb_id).strip()
        parsed_path = self.parsed_dir / f"{tmdb_id_text}.json"
        if parsed_path.exists():
            cached = json.loads(parsed_path.read_text(encoding="utf-8"))
            summary = str(cached.get("summary") or "").strip()
            if summary:
                return SpoilerSummary(
                    summary=summary,
                    excerpt=str(cached.get("excerpt") or ""),
                    source_url=str(cached.get("source_url") or ""),
                )

        if not self.allow_network:
            return None

        url = url_override or self._resolve_url(title=title, year=year, tmdb_id=tmdb_id_text)
        if not url:
            return None

        if not _robots_allows(url, self.user_agent, self._robots_cache):
            return None

        headers = {"User-Agent": self.user_agent}
        resp = requests.get(url, headers=headers, timeout=self.timeout_seconds)
        if resp.status_code != 200:
            return None

        html = resp.text
        summary = _extract_summary_from_html(html)
        if not summary:
            return None

        excerpt = _build_excerpt(summary)
        payload = {
            "summary": summary,
            "excerpt": excerpt,
            "source_url": url,
            "title": title,
            "year": str(year or ""),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / f"{tmdb_id_text}.html").write_text(html, encoding="utf-8")
        parsed_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        return SpoilerSummary(summary=summary, excerpt=excerpt, source_url=url)

    def _resolve_url(self, *, title: str, year: str | int | None, tmdb_id: str) -> str | None:
        tmdb_key = f"tmdb:{tmdb_id}"
        if tmdb_key in self._url_map:
            return self._url_map[tmdb_key]

        title_key = _url_map_key(title, year)
        if title_key and title_key in self._url_map:
            return self._url_map[title_key]

        query = " ".join([title, str(year or "").strip()]).strip()
        if not query:
            return None

        for base in self.base_urls:
            candidate = self._search_site(base, query)
            if candidate:
                if title_key:
                    self._url_map[title_key] = candidate
                    _save_url_map(self.url_map_path, self._url_map)
                return candidate
        return None

    def _search_site(self, base_url: str, query: str) -> str | None:
        search_urls = [
            f"{base_url}/?s={quote_plus(query)}",
            f"{base_url}/search.php?search={quote_plus(query)}",
        ]

        headers = {"User-Agent": self.user_agent}
        for url in search_urls:
            try:
                if not _robots_allows(url, self.user_agent, self._robots_cache):
                    continue
                resp = requests.get(url, headers=headers, timeout=self.timeout_seconds)
                if resp.status_code != 200:
                    continue
            except requests.RequestException:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                text = _compact_ws(a.get_text(" ", strip=True))
                if not href:
                    continue
                href = urljoin(base_url, href)
                parsed = urlparse(href)
                if "themoviespoiler" not in parsed.netloc:
                    continue
                if any(token in href.lower() for token in ["/category/", "/tag/", "wp-admin"]):
                    continue
                links.append((href, text))

            best_url = None
            best_score = 0
            query_tokens = _normalize_title(query).split()
            for href, text in links:
                link_text = _normalize_title(text)
                score = sum(1 for tok in query_tokens if tok in link_text)
                if score > best_score:
                    best_score = score
                    best_url = href

            if best_url:
                return best_url

        return None


def load_spoiler_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None

