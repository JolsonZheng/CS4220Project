"""Ingestion of Steam user reviews from the public store review endpoint.

Endpoint: https://store.steampowered.com/appreviews/<appid>?json=1
No API key is required. Each page returns up to 100 reviews plus a ``cursor``
that must be passed back (URL-encoded) to fetch the next page.

Raw pages are written to ``data/raw/<appid>/page_<n>.json`` so that the parser
and the evaluation harness can run offline against committed samples.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import requests

REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"
MAX_PER_PAGE = 100
DEFAULT_RAW_DIR = Path("data/raw")


class SteamAPIError(RuntimeError):
    """Raised when the review endpoint returns an unusable response."""


def build_params(
    cursor: str = "*",
    num_per_page: int = MAX_PER_PAGE,
    language: str = "english",
    review_filter: str = "recent",
    review_type: str = "all",
    purchase_type: str = "all",
) -> dict:
    """Query parameters for one page of the appreviews endpoint."""
    if not 1 <= num_per_page <= MAX_PER_PAGE:
        raise ValueError(f"num_per_page must be in [1, {MAX_PER_PAGE}], got {num_per_page}")
    return {
        "json": 1,
        "cursor": cursor,
        "num_per_page": num_per_page,
        "language": language,
        "filter": review_filter,
        "review_type": review_type,
        "purchase_type": purchase_type,
    }


def fetch_page(
    appid: int,
    cursor: str = "*",
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    backoff_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    **param_overrides,
) -> dict:
    """Fetch one page of reviews, retrying on HTTP 429 / 5xx with exponential backoff."""
    http = session or requests.Session()
    params = build_params(cursor=cursor, **param_overrides)
    url = REVIEWS_URL.format(appid=appid)

    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        resp = http.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            if attempt < max_retries:
                sleep(backoff_s * (2 ** attempt))
                continue
            break
        if resp.status_code != 200:
            raise SteamAPIError(f"appid {appid}: HTTP {resp.status_code}")
        payload = resp.json()
        if payload.get("success") != 1:
            raise SteamAPIError(f"appid {appid}: success={payload.get('success')!r}")
        return payload

    raise SteamAPIError(f"appid {appid}: gave up after {max_retries + 1} attempts ({last_error})")


def iter_pages(
    appid: int,
    max_reviews: int = 500,
    session: Optional[requests.Session] = None,
    delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    **param_overrides,
) -> Iterator[dict]:
    """Yield successive pages until ``max_reviews`` reviews are seen or the cursor stops moving."""
    http = session or requests.Session()
    cursor = "*"
    seen_cursors = set()
    total = 0
    while total < max_reviews:
        page = fetch_page(appid, cursor=cursor, session=http, sleep=sleep, **param_overrides)
        reviews = page.get("reviews", [])
        if not reviews:
            return
        yield page
        total += len(reviews)
        next_cursor = page.get("cursor")
        if not next_cursor or next_cursor in seen_cursors:
            return
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        sleep(delay_s)


def save_pages(
    appid: int,
    pages: Iterator[dict],
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> list[Path]:
    """Write each page as JSON under ``raw_dir/<appid>/`` and return the written paths."""
    out_dir = Path(raw_dir) / str(appid)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, page in enumerate(pages):
        path = out_dir / f"page_{i:03d}.json"
        path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download Steam reviews for one app id.")
    ap.add_argument("appid", type=int)
    ap.add_argument("--max-reviews", type=int, default=500)
    ap.add_argument("--language", default="english")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = ap.parse_args(argv)

    pages = iter_pages(args.appid, max_reviews=args.max_reviews, language=args.language)
    paths = save_pages(args.appid, pages, raw_dir=args.raw_dir)
    print(f"wrote {len(paths)} page(s) to {Path(args.raw_dir) / str(args.appid)}")


if __name__ == "__main__":
    main()
