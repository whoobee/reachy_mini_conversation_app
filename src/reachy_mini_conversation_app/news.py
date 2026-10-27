"""News headlines from configurable RSS/Atom feeds.

Sources live in a JSON file (name, url, topic, language) that the tools below can edit,
so they can be changed by voice or by hand. Feeds are fetched concurrently and cached
briefly; items are merged newest first with short plain-text summaries for speaking.
"""

from __future__ import annotations
import re
import html
import json
import time
import asyncio
import logging
import datetime as dt
import unicodedata
from typing import Any, Final
from pathlib import Path
from xml.etree import ElementTree
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime

import httpx


logger = logging.getLogger(__name__)

ATOM: Final[str] = "{http://www.w3.org/2005/Atom}"
CACHE_TTL_S: Final[float] = 600.0
MAX_ITEMS: Final[int] = 15
SUMMARY_CHARS: Final[int] = 280
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


@dataclass
class NewsSource:
    """One configured feed."""

    name: str
    url: str
    topic: str = ""  # category, e.g. "tech"; see CATEGORY_ALIASES for other words that select it
    language: str = ""
    general: bool = False  # included when no category is asked for ("read me the news")


CATEGORY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "tech": ("tehnologie", "technology", "gadgets"),
    "inginerie": ("engineering", "embedded", "electronica", "electronics", "ieee", "hardware"),
    "maker": ("robotica", "robotics", "roboti", "smart home", "casa inteligenta", "diy", "hackaday"),
    "romania": ("romanesti", "romaneasca"),
    "germania": ("germany", "deutschland", "germane"),
    "international": ("world", "lume", "externe", "mondial"),
    "regional": ("hessen", "local", "regiune", "aici"),
}

DEFAULT_SOURCES: Final[tuple[NewsSource, ...]] = (
    # General tech news
    NewsSource("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index", "tech", "en"),
    NewsSource("The Verge", "https://www.theverge.com/rss/index.xml", "tech", "en"),
    # Engineering and embedded
    NewsSource("IEEE Spectrum", "https://spectrum.ieee.org/feeds/feed.rss", "inginerie", "en"),
    NewsSource("CNX Software", "https://www.cnx-software.com/feed/", "inginerie", "en"),
    NewsSource("Embedded.com", "https://www.embedded.com/feed/", "inginerie", "en"),
    # Maker, robotics, smart home
    NewsSource("Hackaday", "https://hackaday.com/blog/feed/", "maker", "en"),
    NewsSource("The Robot Report", "https://www.therobotreport.com/feed/", "maker", "en"),
    # General news
    NewsSource("Digi24", "https://www.digi24.ro/rss", "România", "ro", general=True),
    NewsSource("G4Media", "https://www.g4media.ro/feed", "România", "ro", general=True),
    NewsSource("Mediafax", "https://www.mediafax.ro/rss", "România", "ro", general=True),
    NewsSource("Tagesschau", "https://www.tagesschau.de/index~rss2.xml", "Germania", "de", general=True),
    NewsSource("Spiegel", "https://www.spiegel.de/schlagzeilen/index.rss", "Germania", "de", general=True),
    NewsSource("Zeit Online", "https://newsfeed.zeit.de/index", "Germania", "de", general=True),
    NewsSource("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", "internațional", "en", general=True),
    NewsSource("The Guardian", "https://www.theguardian.com/world/rss", "internațional", "en", general=True),
    NewsSource("DW", "https://rss.dw.com/xml/rss-en-world", "internațional", "en", general=True),
    NewsSource("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml", "internațional", "en", general=True),
    NewsSource("Hessenschau", "https://www.hessenschau.de/index.rss", "regional", "de"),
)


def normalize(text: str) -> str:
    """Lowercase and drop diacritics, so "România" matches "romania" and "știri tech" matches "stiri tech"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def matches(query: str, source: NewsSource) -> bool:
    """Return whether ``query`` names the source, its category (or an alias of it) or its language."""
    q = normalize(query)
    words = set(re.findall(r"[a-z0-9.]+", q))
    if q == normalize(source.language):  # language codes only as the whole query ("de" is also a Romanian word)
        return True
    category = normalize(source.topic)
    candidates = [normalize(source.name), category, *CATEGORY_ALIASES.get(category, ())]
    return any(c and (q == c or (len(q) > 2 and q in c) or c in words or (" " in c and c in q)) for c in candidates)


def _text(value: str | None) -> str:
    """Strip HTML tags and entities and collapse whitespace."""
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", value or ""))).strip()


def _parse_date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        parsed = parsedate_to_datetime(value)  # RSS (RFC 822)
    except (TypeError, ValueError):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))  # Atom (ISO 8601)
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def parse_feed(xml_text: str | bytes) -> list[dict[str, Any]]:
    """Return items (title, summary, published, link) from RSS 2.0 or Atom XML."""
    root = ElementTree.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    for item in root.iter("item"):
        items.append(
            {
                "title": _text(item.findtext("title")),
                "summary": _text(item.findtext("description")),
                "published": _parse_date(
                    item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
                ),
                "link": (item.findtext("link") or "").strip(),
            }
        )
    for entry in root.iter(f"{ATOM}entry"):
        link = entry.find(f"{ATOM}link")
        items.append(
            {
                "title": _text(entry.findtext(f"{ATOM}title")),
                "summary": _text(entry.findtext(f"{ATOM}summary") or entry.findtext(f"{ATOM}content")),
                "published": _parse_date(entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated")),
                "link": link.get("href", "") if link is not None else "",
            }
        )
    return [item for item in items if item["title"]]


class NewsService:
    """Configured feeds plus the tool handlers that read and edit them."""

    def __init__(self, path: Path, http: httpx.AsyncClient | None = None, timezone: dt.tzinfo | None = None) -> None:
        """Load sources from ``path`` (seeded with defaults when missing)."""
        self._path = path
        self._http = http or httpx.AsyncClient(
            timeout=10.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Reachy Mini news)"}
        )
        self._tz = timezone or dt.datetime.now().astimezone().tzinfo
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self.sources = self._load()

    def _load(self) -> list[NewsSource]:
        try:
            return [NewsSource(**item) for item in json.loads(self._path.read_text(encoding="utf-8"))]
        except FileNotFoundError:
            sources = list(DEFAULT_SOURCES)
            self._save(sources)
            return sources
        except (OSError, ValueError, TypeError) as e:
            logger.warning("Could not read news sources from %s (%s); using defaults", self._path, e)
            return list(DEFAULT_SOURCES)

    def _save(self, sources: list[NewsSource] | None = None) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in (sources if sources is not None else self.sources)]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    async def _fetch(self, source: NewsSource) -> list[dict[str, Any]]:
        cached = self._cache.get(source.url)
        if cached and time.monotonic() - cached[0] < CACHE_TTL_S:
            return cached[1]
        response = await self._http.get(source.url)
        response.raise_for_status()
        items = parse_feed(response.content)
        self._cache[source.url] = (time.monotonic(), items)
        return items

    def _select(self, query: str) -> list[NewsSource]:
        if not query.strip():
            general = [s for s in self.sources if s.general]
            return general or list(self.sources)
        return [s for s in self.sources if matches(query, s)]

    def categories(self) -> dict[str, list[str]]:
        """Return category -> source names."""
        result: dict[str, list[str]] = {}
        for source in self.sources:
            result.setdefault(source.topic or "other", []).append(source.name)
        return result

    # ---- tool handlers ----

    async def get_news(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Latest headlines from all sources or those matching ``source`` (name, topic or language)."""
        query = str(arguments.get("source") or "")
        sources = self._select(query)
        if not sources:
            return {"error": f"no category or source matches {query!r}", "categories": self.categories()}
        try:
            count = max(1, min(MAX_ITEMS, int(arguments.get("count") or 5)))
            hours = float(arguments.get("hours") or 24)
        except (TypeError, ValueError):
            count, hours = 5, 24.0
        results = await asyncio.gather(*(self._fetch(s) for s in sources), return_exceptions=True)
        now = dt.datetime.now(dt.timezone.utc)
        merged: list[dict[str, Any]] = []
        failed = []
        for source, outcome in zip(sources, results):
            if isinstance(outcome, BaseException):
                failed.append(source.name)
                logger.warning("News feed %s failed: %s", source.name, outcome)
                continue
            for item in outcome:
                published = item["published"]
                if published is not None and (now - published).total_seconds() > hours * 3600:
                    continue
                merged.append({**item, "source": source.name, "language": source.language})
        merged.sort(key=lambda i: i["published"] or dt.datetime.min.replace(tzinfo=dt.timezone.utc), reverse=True)
        seen: set[str] = set()
        headlines = []
        per_source: dict[str, int] = {}
        limit_per_source = max(2, count // max(1, len(sources)) + 1)  # keep a mix of sources
        for item in merged:
            key = item["title"].lower()
            if key in seen or per_source.get(item["source"], 0) >= limit_per_source:
                continue
            seen.add(key)
            per_source[item["source"]] = per_source.get(item["source"], 0) + 1
            published = item["published"]
            headlines.append(
                {
                    "source": item["source"],
                    "language": item["language"],
                    "title": item["title"],
                    "summary": item["summary"][:SUMMARY_CHARS],
                    "published": published.astimezone(self._tz).strftime("%d.%m %H:%M") if published else "",
                }
            )
            if len(headlines) >= count:
                break
        result: dict[str, Any] = {"headlines": headlines}
        if failed:
            result["unavailable_sources"] = failed
        return result

    async def list_sources(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        """List configured sources grouped by category."""
        return {"categories": self.categories(), "sources": [asdict(s) for s in self.sources]}

    async def add_source(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Add (or replace by name) a feed after checking that it parses."""
        name = str(arguments.get("name") or "").strip()
        url = str(arguments.get("url") or "").strip()
        if not name or not url.startswith(("http://", "https://")):
            return {"error": "give a name and an http(s) feed url"}
        source = NewsSource(
            name,
            url,
            str(arguments.get("topic") or ""),
            str(arguments.get("language") or ""),
            general=bool(arguments.get("general", False)),
        )
        try:
            self._cache.pop(url, None)
            items = await self._fetch(source)
        except (httpx.HTTPError, ElementTree.ParseError) as e:
            return {"error": f"could not read a feed at {url}: {e}"}
        if not items:
            return {"error": f"{url} has no news items"}
        self.sources = [s for s in self.sources if s.name.lower() != name.lower()] + [source]
        self._save()
        return {"status": "source added", "source": asdict(source), "latest": items[0]["title"]}

    async def remove_source(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Remove a source by name."""
        name = str(arguments.get("name") or "").strip().lower()
        kept = [s for s in self.sources if s.name.lower() != name]
        if len(kept) == len(self.sources):
            return {"error": f"no source named {arguments.get('name')!r}", "sources": [s.name for s in self.sources]}
        self.sources = kept
        self._save()
        return {"status": "source removed", "remaining": [s.name for s in self.sources]}


TOOL_DEFINITIONS: Final[dict[str, tuple[str, dict[str, Any]]]] = {
    "get_news": (
        "Latest news headlines with short summaries from the configured RSS feeds, newest first. Without "
        "'source' it returns general news (Romania, Germany, international). Use 'source' for a category or "
        "feed: 'tech', 'inginerie' (engineering/embedded), 'maker' (robotics/smart home), 'România', "
        "'Germania', 'internațional', 'regional' (Hessen), or a source name. Summaries may be in English or "
        "German; retell them in the user's language.",
        {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Category or source name; omit for general news."},
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_ITEMS,
                    "description": "Headlines, default 5.",
                },
                "hours": {"type": "number", "description": "Only news from the last N hours, default 24."},
            },
        },
    ),
    "list_news_sources": (
        "List the configured news feeds (name, url, topic, language).",
        {"type": "object", "properties": {}},
    ),
    "add_news_source": (
        "Add an RSS/Atom news feed (or replace one with the same name). The feed is checked before saving.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short name, e.g. 'Digi24'."},
                "url": {"type": "string", "description": "Feed URL (RSS or Atom)."},
                "topic": {"type": "string", "description": "Category, e.g. 'tech', 'maker', 'sport'."},
                "general": {"type": "boolean", "description": "Include it in plain 'read me the news' requests."},
                "language": {"type": "string", "description": "Language code, e.g. 'ro', 'de', 'en'."},
            },
            "required": ["name", "url"],
        },
    ),
    "remove_news_source": (
        "Remove a configured news feed by name.",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Source name."}},
            "required": ["name"],
        },
    ),
}
