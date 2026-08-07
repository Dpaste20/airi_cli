"""
AgentReachTools.py

Standalone "internet reach" tool for Airi, built in the spirit of the
Agent-Reach project (github.com/Panniantong/Agent-Reach): every platform is
a *channel* with an ordered primary -> fallback backend list, and a single
doctor-style status check reports which backend is currently active per
channel and how to fix the ones that aren't.

Unlike Agent-Reach itself, this does NOT shell out to an installed
`agent-reach` CLI and does NOT require running a remote install script.
Each channel is implemented directly against free/no-key backends where
possible, with an optional upgrade path via env vars.

Channels
--------
web      : read_webpage()          -> Jina Reader (no key)
youtube  : get_youtube_transcript()-> yt-dlp (local binary)
github   : get_github_repo()       -> gh CLI -> public REST API fallback
rss      : read_rss_feed()         -> feedparser -> manual XML fallback
search   : web_search()      -> Exa REST API (EXA_API_KEY required)
status   : reach_doctor()          -> probes every channel above

Design notes (matching Airi conventions)
-----------------------------------------
- One tool call = one channel read. No batching — each call returns before
  the next is decided, consistent with the observe-react-decide loop.
- Failures return structured error dicts/strings rather than raising,
  consistent with the rest of utils/.
- Backend selection is resolved fresh on every call (no cached "which
  backend won" state) so a channel recovers automatically once its
  primary backend comes back — mirrors `agent-reach doctor` re-probing
  live rather than trusting a stale cache.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from agno.tools import tool

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_JINA_READER_BASE = "https://r.jina.ai/"
_EXA_REST_URL = "https://api.exa.ai/search"  # requires EXA_API_KEY
_MAX_TEXT_CHARS = 6000


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _http_get(url: str, headers: Optional[dict] = None, timeout: int = 15) -> str:
    """Blocking GET returning decoded text. Raises on failure — callers catch."""
    req = urllib.request.Request(url, headers=headers or _HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_post_json(url: str, payload: dict, headers: dict, timeout: int = 15) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _truncate(text: str, limit: int = _MAX_TEXT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n...[truncated, {len(text) - limit} more characters]"


# --------------------------------------------------------------------------
# Channel: web
# --------------------------------------------------------------------------


@tool
def read_webpage(url: str) -> str:
    """
    Reads a web page as clean, readable text (not raw HTML) via Jina Reader.
    No API key required.

    Args:
        url: The page to read. Protocol is added automatically if missing.

    Returns:
        Extracted page text (truncated if very long), or an error message.
    """
    if not url or not url.strip():
        return "Error: URL cannot be empty."

    target = url.strip()
    if not target.startswith(("http://", "https://")):
        target = f"https://{target}"

    reader_url = f"{_JINA_READER_BASE}{target}"

    try:
        text = _http_get(reader_url, timeout=20)
        if not text.strip():
            return f"No readable content extracted from {target}."
        return _truncate(text)
    except urllib.error.HTTPError as e:
        return f"Error reading '{target}': HTTP {e.code} from Jina Reader."
    except Exception as e:
        return f"Error reading '{target}': {e}"


# --------------------------------------------------------------------------
# Channel: youtube
# --------------------------------------------------------------------------


def _vtt_to_text(vtt_path: Path) -> str:
    """Strips WebVTT timing/cue markup down to plain spoken text, deduped."""
    lines = vtt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)  # strip inline tags
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return " ".join(out)


@tool
def get_youtube_transcript(url: str, language: str = "en") -> str:
    """
    Extracts the transcript/subtitles of a YouTube video via yt-dlp.
    Requires yt-dlp to be installed and on PATH.

    Args:
        url: Full YouTube video URL.
        language: Subtitle language code to prefer (default "en").

    Returns:
        Plain-text transcript (truncated if long), or an error/fix message.
    """
    if not shutil.which("yt-dlp"):
        return (
            "Error: 'yt-dlp' not found on PATH. "
            "Install it with: pip install -U yt-dlp"
        )

    if not url or not url.strip():
        return "Error: URL cannot be empty."

    with tempfile.TemporaryDirectory() as tmp:
        out_template = os.path.join(tmp, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang",
            language,
            "--sub-format",
            "vtt",
            "-o",
            out_template,
            url.strip(),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            return "Error: yt-dlp timed out after 60s."
        except Exception as e:
            return f"Error running yt-dlp: {e}"

        if result.returncode != 0:
            return f"Error: yt-dlp failed:\n{result.stderr.strip()[:1000]}"

        vtt_files = list(Path(tmp).glob("*.vtt"))
        if not vtt_files:
            return (
                f"No subtitles found for '{url}' in language '{language}' "
                "(video may have no captions, or try a different language code)."
            )

        transcript = _vtt_to_text(vtt_files[0])
        if not transcript:
            return "Subtitles found but produced no readable text."
        return _truncate(transcript)


# --------------------------------------------------------------------------
# Channel: github
# --------------------------------------------------------------------------


@tool
def get_github_repo(repo: str) -> dict:
    """
    Gets basic info about a public GitHub repository.
    Prefers the 'gh' CLI (works for private repos if authenticated),
    falls back to the public REST API (public repos, rate-limited) if
    'gh' isn't installed or isn't authenticated.

    Args:
        repo: Repository in "owner/name" form, e.g. "Panniantong/Agent-Reach".

    Returns:
        dict with name, description, stars, url, default_branch, updated_at,
        and a "backend" field indicating which path served the request —
        or an "error" key.
    """
    repo = repo.strip().strip("/")
    if "/" not in repo:
        return {"error": "repo must be in 'owner/name' form."}

    if shutil.which("gh"):
        try:
            result = subprocess.run(
                [
                    "gh", "repo", "view", repo, "--json",
                    "name,description,stargazerCount,url,defaultBranchRef,updatedAt",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    "name": data.get("name"),
                    "description": data.get("description"),
                    "stars": data.get("stargazerCount"),
                    "url": data.get("url"),
                    "default_branch": (data.get("defaultBranchRef") or {}).get("name"),
                    "updated_at": data.get("updatedAt"),
                    "backend": "gh-cli",
                }
            # gh present but failed (not authenticated / repo not found) — fall through
        except Exception:
            pass  # fall through to REST fallback

    try:
        raw = _http_get(f"https://api.github.com/repos/{repo}", timeout=15)
        data = json.loads(raw)
        if "message" in data and "name" not in data:
            return {"error": f"GitHub API: {data['message']}", "backend": "rest-api"}
        return {
            "name": data.get("name"),
            "description": data.get("description"),
            "stars": data.get("stargazers_count"),
            "url": data.get("html_url"),
            "default_branch": data.get("default_branch"),
            "updated_at": data.get("updated_at"),
            "backend": "rest-api",
        }
    except Exception as e:
        return {"error": f"Both gh CLI and REST API failed: {e}"}


# --------------------------------------------------------------------------
# Channel: rss
# --------------------------------------------------------------------------


def _parse_rss_manual(xml_text: str, max_results: int) -> list[dict]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    items = channel.findall("item") if channel is not None else root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )
    articles = []
    for item in items[:max_results]:
        title = (item.findtext("title") or "").strip()
        link_el = item.find("link")
        link = (item.findtext("link") or "").strip()
        if not link and link_el is not None:
            link = link_el.get("href", "")
        pub = (item.findtext("pubDate") or item.findtext(
            "{http://www.w3.org/2005/Atom}updated"
        ) or "").strip()
        articles.append({"title": title, "url": link, "published": pub})
    return articles


@tool
def read_rss_feed(feed_url: str, max_results: int = 10) -> dict:
    """
    Reads and parses an RSS/Atom feed into recent entries.
    Uses feedparser if installed, otherwise falls back to a manual
    XML parser (RSS 2.0 and basic Atom).

    Args:
        feed_url: The RSS/Atom feed URL.
        max_results: Max number of entries to return (default 10).

    Returns:
        dict with "entries" (list of {title, url, published}) and "backend",
        or an "error" key.
    """
    max_results = max(1, min(max_results, 50))

    try:
        import feedparser  # optional dependency

        parsed = feedparser.parse(feed_url)
        if parsed.bozo and not parsed.entries:
            raise ValueError(str(parsed.bozo_exception))
        entries = [
            {
                "title": e.get("title", ""),
                "url": e.get("link", ""),
                "published": e.get("published", e.get("updated", "")),
            }
            for e in parsed.entries[:max_results]
        ]
        return {"entries": entries, "backend": "feedparser"}
    except ImportError:
        pass  # fall through to manual parser
    except Exception:
        pass  # feedparser installed but choked — try manual parser too

    try:
        xml_text = _http_get(feed_url, timeout=15)
        entries = _parse_rss_manual(xml_text, max_results)
        return {"entries": entries, "backend": "manual-xml"}
    except Exception as e:
        return {"error": f"Failed to read feed '{feed_url}': {e}"}





# --------------------------------------------------------------------------
# Channel: status / doctor
# --------------------------------------------------------------------------


@tool
def reach_doctor() -> dict:
    """
    Probes every AgentReach channel and reports which backend is currently
    active for each one, plus a fix hint for anything that isn't ready.
    Run this first if a read/search call is failing unexpectedly.

    Returns:
        dict keyed by channel name, each with "status" ("ok"/"degraded"/
        "unavailable"), "backend", and optionally "fix".
    """
    report: dict = {}

    # web — Jina Reader has no install requirement; just confirm reachability.
    try:
        _http_get(f"{_JINA_READER_BASE}https://example.com", timeout=8)
        report["web"] = {"status": "ok", "backend": "jina-reader"}
    except Exception as e:
        report["web"] = {
            "status": "degraded",
            "backend": "jina-reader",
            "fix": f"Jina Reader unreachable ({e}); check network/proxy.",
        }

    # youtube — needs yt-dlp on PATH.
    if shutil.which("yt-dlp"):
        report["youtube"] = {"status": "ok", "backend": "yt-dlp"}
    else:
        report["youtube"] = {
            "status": "unavailable",
            "backend": "none",
            "fix": "Install with: pip install -U yt-dlp",
        }

    # github — gh CLI (authenticated) preferred, else public REST fallback.
    if shutil.which("gh"):
        try:
            auth = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, text=True, timeout=10
            )
            if auth.returncode == 0:
                report["github"] = {"status": "ok", "backend": "gh-cli"}
            else:
                report["github"] = {
                    "status": "degraded",
                    "backend": "rest-api",
                    "fix": "gh installed but not authenticated — run 'gh auth login', "
                    "or continue on the rate-limited public REST fallback.",
                }
        except Exception:
            report["github"] = {"status": "degraded", "backend": "rest-api"}
    else:
        report["github"] = {
            "status": "degraded",
            "backend": "rest-api",
            "fix": "Install gh CLI for private repos and higher rate limits: "
            "https://cli.github.com",
        }

    # rss — feedparser preferred, manual XML parser always available.
    try:
        import feedparser  # noqa: F401

        report["rss"] = {"status": "ok", "backend": "feedparser"}
    except ImportError:
        report["rss"] = {
            "status": "ok",
            "backend": "manual-xml",
            "fix": "Optional: pip install feedparser for broader Atom/RSS support.",
        }

    # search — needs EXA_API_KEY; if set, confirm it actually works.
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        report["search"] = {
            "status": "unavailable",
            "backend": "none",
            "fix": "Set EXA_API_KEY. Get a free key at "
            "https://dashboard.exa.ai/api-keys.",
        }
    else:
        try:
            _http_post_json(
                _EXA_REST_URL,
                {"query": "connectivity check", "numResults": 1},
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                timeout=10,
            )
            report["search"] = {"status": "ok", "backend": "exa-rest"}
        except Exception as e:
            report["search"] = {
                "status": "degraded",
                "backend": "exa-rest",
                "fix": f"EXA_API_KEY is set but the request failed ({e}) — "
                "check the key is valid and has remaining credits.",
            }

    return report
