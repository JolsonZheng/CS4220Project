"""Turn raw appreviews JSON pages into a clean, analysis-ready pandas DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# Output columns, in order. Playtimes are reported by Steam in minutes; we convert to hours.
COLUMNS = [
    "appid",
    "recommendation_id",
    "language",
    "review_text",
    "voted_up",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "created_at",
    "updated_at",
    "author_num_games_owned",
    "author_num_reviews",
    "playtime_at_review_h",
    "playtime_forever_h",
    "review_length",
]


def _minutes_to_hours(value) -> float:
    try:
        return round(float(value) / 60.0, 2)
    except (TypeError, ValueError):
        return float("nan")


def review_to_row(review: dict, appid: int) -> dict:
    """Flatten one review object from the endpoint into a row dict."""
    author = review.get("author") or {}
    text = (review.get("review") or "").strip()
    return {
        "appid": int(appid),
        "recommendation_id": str(review.get("recommendationid", "")),
        "language": review.get("language"),
        "review_text": text,
        "voted_up": bool(review.get("voted_up", False)),
        "votes_up": int(review.get("votes_up", 0) or 0),
        "votes_funny": int(review.get("votes_funny", 0) or 0),
        # Steam returns this as a string such as "0.523809552192687988".
        "weighted_vote_score": float(review.get("weighted_vote_score", 0) or 0),
        "comment_count": int(review.get("comment_count", 0) or 0),
        "steam_purchase": bool(review.get("steam_purchase", False)),
        "received_for_free": bool(review.get("received_for_free", False)),
        "written_during_early_access": bool(review.get("written_during_early_access", False)),
        "created_at": pd.to_datetime(review.get("timestamp_created"), unit="s", utc=True),
        "updated_at": pd.to_datetime(review.get("timestamp_updated"), unit="s", utc=True),
        "author_num_games_owned": author.get("num_games_owned"),
        "author_num_reviews": author.get("num_reviews"),
        "playtime_at_review_h": _minutes_to_hours(author.get("playtime_at_review")),
        "playtime_forever_h": _minutes_to_hours(author.get("playtime_forever")),
        "review_length": len(text),
    }


def pages_to_dataframe(pages: Iterable[dict], appid: int, min_length: int = 1) -> pd.DataFrame:
    """Combine pages into one DataFrame, drop empty reviews and duplicate recommendation ids."""
    rows = [review_to_row(r, appid) for page in pages for r in page.get("reviews", [])]
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df = df[df["review_length"] >= min_length]
    df = df.drop_duplicates(subset="recommendation_id", keep="first")
    return df.sort_values("created_at").reset_index(drop=True)


def load_raw_dir(appid: int, raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Parse every ``page_*.json`` saved by ``fetch.save_pages`` for one app id."""
    app_dir = Path(raw_dir) / str(appid)
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(app_dir.glob("page_*.json"))]
    return pages_to_dataframe(pages, appid)


def summarize(df: pd.DataFrame) -> dict:
    """Basic descriptive stats, used for the evidence table and as a sanity check."""
    if df.empty:
        return {"n_reviews": 0}
    return {
        "n_reviews": int(len(df)),
        "pct_positive": round(float(df["voted_up"].mean()) * 100, 1),
        "median_playtime_at_review_h": float(df["playtime_at_review_h"].median()),
        "median_review_length": int(df["review_length"].median()),
        "first_review": df["created_at"].min().isoformat(),
        "last_review": df["created_at"].max().isoformat(),
    }


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse saved raw pages into a CSV.")
    ap.add_argument("appid", type=int)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    df = load_raw_dir(args.appid, args.raw_dir)
    out = args.out or Path("data/processed") / f"{args.appid}_reviews.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(json.dumps(summarize(df), indent=2))
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
