"""Download plain-text Wikipedia articles listed in data/seed_titles.txt.

Wikipedia text is licensed CC BY-SA 4.0. Keep the title and URL with every
chunk so answers can cite their source.

Wikimedia throttles clients that look anonymous or go too fast, so this module
identifies itself (set WIKI_CONTACT in .env), waits between requests, and backs
off when it receives HTTP 429.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
MAX_RETRIES = 6
MAX_CONSECUTIVE_FAILURES = 3


def user_agent() -> str:
    contact = os.getenv("WIKI_CONTACT", "no-contact-set")
    return f"Satyameter/0.1 (MCA dissertation project; contact: {contact})"


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def read_titles(path: Path) -> list[str]:
    titles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return titles


def _get_with_retry(session: requests.Session, params: dict) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx, honouring Retry-After."""
    delay = 5.0
    for _ in range(MAX_RETRIES):
        response = session.get(
            API_URL, params=params, headers={"User-Agent": user_agent()}, timeout=30
        )
        if response.status_code == 429 or response.status_code >= 500:
            try:
                wait = float(response.headers.get("Retry-After", delay))
            except ValueError:  # Retry-After can also be an HTTP date
                wait = delay
            wait = min(max(wait, delay), 120.0)
            print(f"  ... Wikipedia said {response.status_code}; waiting {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120.0)
            continue
        response.raise_for_status()
        return response
    raise requests.RequestException(f"gave up after {MAX_RETRIES} retries")


def fetch_article(title: str, session: requests.Session) -> dict | None:
    """Return {title, url, text} for one article, or None if it does not exist."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }
    pages = _get_with_retry(session, params).json().get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or not page.get("extract"):
        return None
    resolved = page["title"]
    return {
        "title": resolved,
        "url": "https://en.wikipedia.org/wiki/" + quote(resolved.replace(" ", "_")),
        "text": page["extract"],
    }


def download_corpus(
    titles: list[str], raw_dir: Path, limit: int | None = None, delay: float = 2.0
) -> tuple[int, list[str]]:
    """Download articles into raw_dir as JSON. Skips ones already downloaded,
    so it is safe to re-run. Stops early after repeated failures.

    Returns (number_saved_or_existing, list_of_failed_titles).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    ok = 0
    failed: list[str] = []
    consecutive = 0
    for title in titles[:limit] if limit else titles:
        target = raw_dir / f"{slugify(title)}.json"
        if target.exists():
            ok += 1
            continue
        try:
            article = fetch_article(title, session)
        except requests.RequestException as exc:
            print(f"  ! network error for '{title}': {exc}")
            failed.append(title)
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print("  ! Too many failures in a row. Wait ~10 minutes, then re-run fetch.")
                break
            continue
        consecutive = 0
        if article is None:
            print(f"  ! not found: '{title}'")
            failed.append(title)
            continue
        target.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
        ok += 1
        print(f"  saved: {article['title']}")
        time.sleep(delay)
    return ok, failed
