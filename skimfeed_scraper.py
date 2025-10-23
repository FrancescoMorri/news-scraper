#!/usr/bin/env python3
from __future__ import annotations
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, Dict, List
from dateutil import parser as dateparser

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

URL = "https://skimfeed.com/custom.php?f=l%2Cp%2C119%2C121%2C122%2C123%2C124%2C125%2C126%2C127%2C156"
TIMEZONE = "Europe/Paris"

# Try resolving full titles for "What's Hot" by fetching destination pages if truncated
RESOLVE_WHATS_HOT_FULL_TITLES = True
MAX_WHATS_HOT_RESOLVES = 10    # don't hammer sites

ELLIPSIS_RE = re.compile(r"\u2026|\.\.\.$")  # … or ...

@dataclass
class Item:
    section: str
    title: str
    url: str
    published: Optional[str] = None  # ISO-8601 if found

def fetch(url: str) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    return r

def is_external(href: str) -> bool:
    try:
        netloc = urlparse(href).netloc.lower()
    except Exception:
        return False
    return bool(netloc) and "skimfeed.com" not in netloc

def normalize_section_name(raw: str) -> str:
    # Section headers sometimes look like "Economist Business + www.economist.com"
    txt = re.sub(r"\s+", " ", raw.strip())
    if " +" in txt:
        txt = txt.split(" +", 1)[0]
    return txt

def nearest_section_heading(a: Tag) -> str:
    h = a.find_previous(["h1", "h2", "h3", "h4"])
    if h:
        return normalize_section_name(h.get_text(" ", strip=True))
    return "Uncategorized"

def prefer_full_title(section: str, a: Tag) -> str:
    visible = a.get_text(" ", strip=True)
    title_attr = (a.get("title") or "").strip()
    if section.lower().startswith("what") and title_attr and len(title_attr) > len(visible):
        return title_attr
    return visible

def try_extract_datetime(a: Tag) -> Optional[datetime]:
    """
    Try to find a timestamp near the link:
    - <time datetime="..."> or title/text inside a <time> tag
    - data-time / data-ts / datetime / title attributes on the anchor or parent
    - ISO-like date in nearby text
    """
    tz = ZoneInfo(TIMEZONE)

    # <time> next to the link
    time_tag = a.find_next("time")
    if isinstance(time_tag, Tag):
        for candidate in (time_tag.get("datetime"), time_tag.get("title"), time_tag.get_text(strip=True)):
            if candidate:
                try:
                    return dateparser.parse(candidate).astimezone(tz)
                except Exception:
                    pass

    # Attributes on anchor or its parent
    for node in (a, a.parent if isinstance(a.parent, Tag) else None):
        if not isinstance(node, Tag):
            continue
        for key in ("data-time", "data-ts", "data-timestamp", "data-pubdate", "datetime", "title"):
            val = node.get(key)
            if val:
                try:
                    return dateparser.parse(val).astimezone(tz)
                except Exception:
                    continue

    # ISO-like date in nearby text
    neighborhood = " ".join(filter(None, [
        a.parent.get_text(" ", strip=True) if isinstance(a.parent, Tag) else "",
        a.find_next(string=True) or ""
    ]))
    m = re.search(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?", neighborhood)
    if m:
        try:
            return dateparser.parse(m.group(0)).astimezone(tz)
        except Exception:
            pass

    return None

def get_today_titles_skimfeed(only_today_if_possible: bool = True) -> Dict[str, List[Item]]:
    soup = BeautifulSoup(fetch(URL).text, "lxml")
    today = datetime.now(ZoneInfo(TIMEZONE)).date()

    results: Dict[str, List[Item]] = {}
    seen = set()

    # KEY CHANGE: target anchors that are list items
    for a in soup.select("li > a[href]"):
        #print((a.get("title") or "").strip())  # Debugging line to see the anchors being processed
        #href = a["href"].strip()
        #if not is_external(href):
        #    continue

        section = nearest_section_heading(a)
        title = (a.get("title") or "").strip()
        #title = prefer_full_title(section, a)

        dt = try_extract_datetime(a)
        dt_iso = dt.isoformat() if dt else None

        #key = (section, title, href)
        key = (section, title)
        if key in seen:
            continue
        seen.add(key)

        item = Item(section=section, title=title, url="", published=dt_iso)
        results.setdefault(section, []).append(item)

    # If we found any real dates, optionally keep only "today" in your tz.
    if only_today_if_possible:
        any_dated = any(it.published for items in results.values() for it in items)
        if any_dated:
            for section, items in list(results.items()):
                filtered = [it for it in items if it.published and datetime.fromisoformat(it.published).date() == today]
                results[section] = filtered
            # drop empty sections
            results = {s: lst for s, lst in results.items() if lst}

    return results