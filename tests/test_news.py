import json
import datetime as dt
from pathlib import Path
from email.utils import format_datetime

import httpx
import pytest

from reachy_mini_conversation_app.news import DEFAULT_SOURCES, NewsSource, NewsService, matches, parse_feed


def _rss(items: list[tuple[str, dt.datetime]]) -> str:
    body = "".join(
        f"<item><title>{title}</title><description>&lt;p&gt;About {title}&lt;/p&gt;</description>"
        f"<pubDate>{format_datetime(when)}</pubDate><link>https://x/{i}</link></item>"
        for i, (title, when) in enumerate(items)
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{body}</channel></rss>'


ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>a</title>
<entry><title>Atom story</title><summary>Short &amp; sweet</summary>
<updated>2026-10-27T08:00:00Z</updated><link href="https://a/1"/></entry></feed>"""


def _service(tmp_path: Path, feeds: dict[str, str], requests: list[str] | None = None) -> NewsService:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(str(request.url))
        body = feeds.get(str(request.url))
        return httpx.Response(200, text=body) if body is not None else httpx.Response(404)

    return NewsService(
        tmp_path / "news.json",
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timezone=dt.timezone.utc,
    )


def test_parse_feed_reads_rss_and_atom_as_plain_text() -> None:
    """RSS and Atom items become title/summary/published with HTML stripped."""
    now = dt.datetime(2026, 10, 27, 9, 0, tzinfo=dt.timezone.utc)
    rss = parse_feed(_rss([("Știre unu", now)]))
    assert rss[0]["title"] == "Știre unu" and rss[0]["summary"] == "About Știre unu"
    assert rss[0]["published"] == now
    atom = parse_feed(ATOM)
    assert atom[0]["title"] == "Atom story" and atom[0]["summary"] == "Short & sweet"
    assert atom[0]["link"] == "https://a/1"


def test_default_sources_are_seeded_and_saved(tmp_path: Path) -> None:
    """Without a sources file the defaults are written so they can be edited."""
    service = _service(tmp_path, {})
    names = [s.name for s in service.sources]
    assert "Hackaday" in names and "Tagesschau" in names
    assert json.loads((tmp_path / "news.json").read_text(encoding="utf-8"))[0]["name"] == names[0]


@pytest.mark.asyncio
async def test_get_news_merges_newest_first_filters_old_and_by_topic(tmp_path: Path) -> None:
    """Headlines are merged across sources newest first, old items dropped, topics filter sources."""
    now = dt.datetime.now(dt.timezone.utc)
    (tmp_path / "news.json").write_text(
        json.dumps(
            [
                {"name": "RO", "url": "https://ro/rss", "topic": "România", "language": "ro"},
                {"name": "DE", "url": "https://de/rss", "topic": "Germania", "language": "de"},
            ]
        ),
        encoding="utf-8",
    )
    feeds = {
        "https://ro/rss": _rss([("ro new", now - dt.timedelta(hours=1)), ("ro old", now - dt.timedelta(days=3))]),
        "https://de/rss": _rss([("de newest", now - dt.timedelta(minutes=10))]),
    }
    service = _service(tmp_path, feeds)

    everything = await service.get_news({})
    assert [h["title"] for h in everything["headlines"]] == ["de newest", "ro new"]
    romania = await service.get_news({"source": "românia"})
    assert [h["source"] for h in romania["headlines"]] == ["RO"]
    assert "error" in await service.get_news({"source": "sport"})


@pytest.mark.asyncio
async def test_failed_feeds_are_reported_and_results_are_cached(tmp_path: Path) -> None:
    """A broken feed does not hide the others, and repeated calls reuse the cache."""
    now = dt.datetime.now(dt.timezone.utc)
    (tmp_path / "news.json").write_text(
        json.dumps(
            [
                {"name": "OK", "url": "https://ok/rss"},
                {"name": "Down", "url": "https://down/rss"},
            ]
        ),
        encoding="utf-8",
    )
    requests: list[str] = []
    service = _service(tmp_path, {"https://ok/rss": _rss([("hello", now)])}, requests)

    first = await service.get_news({})
    await service.get_news({})

    assert first["unavailable_sources"] == ["Down"] and first["headlines"][0]["title"] == "hello"
    assert requests.count("https://ok/rss") == 1


@pytest.mark.asyncio
async def test_sources_can_be_added_after_validation_and_removed(tmp_path: Path) -> None:
    """New feeds are checked before saving; removing by name updates the file."""
    service = _service(tmp_path, {"https://sport/rss": _rss([("goal", dt.datetime.now(dt.timezone.utc))])})

    assert "error" in await service.add_source({"name": "Bad", "url": "https://missing/rss"})
    added = await service.add_source({"name": "Sport", "url": "https://sport/rss", "topic": "sport"})
    assert added["latest"] == "goal"
    saved = json.loads((tmp_path / "news.json").read_text(encoding="utf-8"))
    assert saved[-1]["name"] == "Sport"
    removed = await service.remove_source({"name": "sport"})
    assert "Sport" not in removed["remaining"]
    assert "error" in await service.remove_source({"name": "Nope"})


def test_categories_match_aliases_and_ignore_diacritics() -> None:
    """Categories are selected by name or alias in Romanian or English, with or without diacritics."""
    by_category = {s.topic: s for s in DEFAULT_SOURCES}
    assert matches("tehnologie", by_category["tech"])
    assert matches("știri de tehnologie", by_category["tech"])
    assert not matches("știri de tehnologie", by_category["Germania"])  # "de" is a Romanian word here
    assert matches("de", by_category["Germania"])  # but a bare language code still works
    assert matches("robotică", by_category["maker"]) and matches("smart home", by_category["maker"])
    assert matches("embedded", by_category["inginerie"])
    assert matches("romania", by_category["România"]) and matches("internațional", by_category["internațional"])
    assert matches("hackaday", by_category["maker"])
    assert not matches("sport", by_category["tech"])


@pytest.mark.asyncio
async def test_plain_request_reads_only_general_sources(tmp_path: Path) -> None:
    """Without a category only sources marked general are used; a category reaches the others."""
    now = dt.datetime.now(dt.timezone.utc)
    (tmp_path / "news.json").write_text(
        json.dumps(
            [
                {"name": "World", "url": "https://w/rss", "topic": "internațional", "general": True},
                {"name": "Gadgets", "url": "https://g/rss", "topic": "tech"},
            ]
        ),
        encoding="utf-8",
    )
    service = _service(
        tmp_path, {"https://w/rss": _rss([("world news", now)]), "https://g/rss": _rss([("new phone", now)])}
    )

    assert [h["source"] for h in (await service.get_news({}))["headlines"]] == ["World"]
    assert [h["source"] for h in (await service.get_news({"source": "tehnologie"}))["headlines"]] == ["Gadgets"]
    assert (await service.list_sources({}))["categories"] == {"internațional": ["World"], "tech": ["Gadgets"]}


def test_old_sources_file_without_general_flag_still_loads(tmp_path: Path) -> None:
    """Source files written before the general flag existed load with general=False."""
    (tmp_path / "news.json").write_text(
        json.dumps([{"name": "Old", "url": "https://o/rss", "topic": "x", "language": "en"}])
    )
    assert _service(tmp_path, {}).sources == [NewsSource("Old", "https://o/rss", "x", "en")]
