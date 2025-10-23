from __future__ import annotations
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

URL = "https://timef.com/business/"

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}\b")
DOMAIN_LIKE_RE = re.compile(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

def is_domain_text(text: str) -> bool:
    return bool(DOMAIN_LIKE_RE.fullmatch(text.strip()))

def pick_title_anchor(container):
    """Pick the first external-link anchor that looks like a headline (not a bare domain)."""
    for a in container.find_all("a", href=True):
        href = a["href"]
        txt = a.get_text(strip=True)
        if "timef.com" in href:
            continue
        if not txt or is_domain_text(txt):
            # Skip the little domain tag like "nytimes.com"
            continue
        return a
    return None

def get_today_titles_timeF(tz="Europe/Paris"):
    today_str = datetime.now(ZoneInfo(tz)).date().isoformat()  # e.g., "2025-08-20"

    r = requests.get(URL, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    results = []
    seen = set()

    # Find the timestamp strings anywhere in the page, then climb to a container to find the title link.
    for s in soup.find_all(string=DATE_RE):
        m = DATE_RE.search(s)
        if not m:
            continue
        date_full = m.group(0)
        date_part = m.group(1)
        if date_part != today_str:
            continue

        # Walk up a few levels to find a block that contains the headline link.
        node = s.parent
        for _ in range(4):  # limited climb to stay safe
            if not node:
                break
            a = pick_title_anchor(node)
            if a:
                title = a.get_text(strip=True)
                url = a["href"]
                key = (title, url)
                if key not in seen:
                    seen.add(key)
                    results.append({"date": date_full, "title": title, "url": url})
                break
            node = node.parent

    return results