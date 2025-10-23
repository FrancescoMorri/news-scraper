#!/usr/bin/env python3
from __future__ import annotations
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

BASE_URL = "https://brutalist.report/topic/business"

AGE_RE = re.compile(r"\[(\d+)\s*([mhd])\]")  # e.g., [19m], [1h], [3d]

def is_external(href: str) -> bool:
    try:
        netloc = urlparse(href).netloc.lower()
    except Exception:
        return False
    return bool(netloc) and "brutalist.report" not in netloc

def parse_age_to_timedelta(age_text: str) -> timedelta | None:
    m = AGE_RE.search(age_text)
    if not m:
        return None
    val = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=val)
    if unit == "h":
        return timedelta(hours=val)
    if unit == "d":
        return timedelta(days=val)
    return None

def get_today_titles_brutalist(limit=10, today_tz="Europe/Paris"):
    params = {"limit": str(limit)}
    r = requests.get(BASE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    now = datetime.now(ZoneInfo(today_tz)).replace(tzinfo=ZoneInfo(today_tz))
    today_date = now.date()

    results = []
    seen = set()

    # Heuristic: headline anchors are external links, and the parent text includes an age tag like [19m].
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not is_external(href):
            continue
        title = a.get_text(strip=True)
        if not title:
            continue

        # Look for an age tag in the same line/container (parent text is usually "<title> [xx]")
        parent_text = a.parent.get_text(" ", strip=True) if a.parent else title
        td = parse_age_to_timedelta(parent_text)
        if td is None:
            # Some non-feed links may sneak through; skip if we don't see an age tag.
            continue

        # Compute a naive published-at by subtracting the age from now.
        published_at = now - td

        # Keep if it's "today" in the chosen timezone
        if published_at.date() != today_date:
            continue

        key = (title, href)
        if key in seen:
            continue
        seen.add(key)
        results.append({"title": title, "url": href, "age": AGE_RE.search(parent_text).group(0)})

    return results
