import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

from agno.tools import tool

_RSS_BASE = "https://news.google.com/rss"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch(url: str) -> list[dict]:
    """Fetch and parse a Google News RSS feed URL into a list of article dicts."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        xml_text = resp.read().decode("utf-8")

    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    articles = []
    for item in channel.findall("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None else "Unknown"

        try:
            dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
            pub_date = dt.strftime("%d %b %Y, %H:%M UTC")
        except Exception:
            pass

        articles.append(
            {
                "title": title,
                "source": source,
                "published": pub_date,
                "url": link,
            }
        )

    return articles


def _format(articles: list[dict], header: str, max_results: int) -> str:
    articles = articles[:max_results]
    if not articles:
        return f"No articles found for: {header}"

    lines = [f"📰 {header} — {len(articles)} article(s)\n{'─' * 52}"]
    for i, a in enumerate(articles, 1):
        lines.append(
            f"{i}. {a['title']}\n"
            f"   🏢 {a['source']}   🕒 {a['published']}\n"
            f"   🔗 {a['url']}"
        )
    return "\n\n".join(lines)


@tool
def get_top_news(max_results: int = 10) -> str:
    """
    Fetches the current top global headlines from Google News.

    Args:
        max_results : Number of articles to return (default 10, max 25).

    Returns:
        Formatted top headlines with source, publish time, and URL.
    """
    max_results = min(max(1, max_results), 25)
    url = f"{_RSS_BASE}?hl=en-US&gl=US&ceid=US:en"

    try:
        articles = _fetch(url)
        return _format(articles, "Top Global Headlines", max_results)
    except Exception as e:
        return f"Error fetching top news: {e}"


@tool
def get_region_news(region: str, max_results: int = 10) -> str:
    """
    Fetches the latest news for a specific region, city, or country
    from Google News RSS.

    Args:
        region      : Region, city, or country to search for
                      (e.g. "Mumbai", "California", "Germany").
        max_results : Number of articles to return (default 10, max 25).

    Returns:
        Formatted headlines with source, publish time, and URL.
    """
    max_results = min(max(1, max_results), 25)
    query = urllib.parse.quote(region.strip())
    url = f"{_RSS_BASE}/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    try:
        articles = _fetch(url)
        return _format(articles, f"News for '{region}'", max_results)
    except Exception as e:
        return f"Error fetching news for '{region}': {e}"


@tool
def get_topic_news(topic: str, region: str = "", max_results: int = 10) -> str:
    """
    Fetches news for a specific topic from Google News RSS,
    optionally scoped to a region.

    Args:
        topic       : Keyword or topic to search (e.g. "cricket", "AI", "earthquake").
        region      : Optional region to narrow results (e.g. "India", "Texas").
        max_results : Number of articles to return (default 10, max 25).

    Returns:
        Formatted headlines with source, publish time, and URL.
    """
    max_results = min(max(1, max_results), 25)
    search_term = f"{topic} {region}".strip() if region else topic
    query = urllib.parse.quote(search_term)
    url = f"{_RSS_BASE}/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    header = f"'{topic}' news" + (f" in {region}" if region else " (global)")

    try:
        articles = _fetch(url)
        return _format(articles, header, max_results)
    except Exception as e:
        return f"Error fetching news for '{topic}': {e}"
