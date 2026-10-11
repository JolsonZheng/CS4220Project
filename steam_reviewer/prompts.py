"""Prompt schema for review sentiment + theme extraction, and strict validation of model output.

The model call itself is injected as ``complete: Callable[[str], str]`` (prompt in, raw text
out), so the same code works with any provider and the tests can use a fake model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

SENTIMENTS = ("positive", "negative", "mixed")

# Closed set of gameplay themes. Keeping it closed makes outputs comparable across games
# and lets the evaluation harness compute per-theme agreement with manual labels.
THEMES = (
    "gameplay",
    "story",
    "graphics",
    "audio",
    "performance",
    "bugs",
    "price_value",
    "content_length",
    "multiplayer",
    "difficulty",
    "monetization",
    "developer_support",
)

PROMPT_VERSION = "v1"

SYSTEM_INSTRUCTIONS = f"""You label Steam game reviews.
For each review, return:
- "id": the review id you were given
- "sentiment": one of {list(SENTIMENTS)}
- "praise": themes the reviewer praises, chosen only from {list(THEMES)}
- "complaints": themes the reviewer complains about, chosen only from {list(THEMES)}
- "summary": at most 20 words
Reply with a JSON array only, one object per review, no prose and no code fences."""


@dataclass
class ReviewLabel:
    id: str
    sentiment: str
    praise: list[str] = field(default_factory=list)
    complaints: list[str] = field(default_factory=list)
    summary: str = ""


class LabelValidationError(ValueError):
    pass


def build_prompt(reviews: Iterable[tuple[str, str]], max_chars: int = 1500) -> str:
    """Build one batch prompt from ``(recommendation_id, review_text)`` pairs.

    Long reviews are truncated to ``max_chars`` to bound token cost per batch.
    """
    items = [{"id": rid, "text": text[:max_chars]} for rid, text in reviews]
    return SYSTEM_INSTRUCTIONS + "\n\nReviews:\n" + json.dumps(items, ensure_ascii=False)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_labels(raw: str, expected_ids: Iterable[str]) -> list[ReviewLabel]:
    """Parse and validate a model reply. Raises LabelValidationError on any schema violation."""
    text = _FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LabelValidationError(f"not JSON: {exc}") from exc
    if not isinstance(data, list):
        raise LabelValidationError("top-level value must be a JSON array")

    expected = list(expected_ids)
    labels = []
    for obj in data:
        if not isinstance(obj, dict):
            raise LabelValidationError(f"item is not an object: {obj!r}")
        rid = str(obj.get("id", ""))
        sentiment = str(obj.get("sentiment", "")).lower()
        if sentiment not in SENTIMENTS:
            raise LabelValidationError(f"{rid}: bad sentiment {sentiment!r}")
        praise = [t for t in obj.get("praise", []) if t in THEMES]
        complaints = [t for t in obj.get("complaints", []) if t in THEMES]
        labels.append(ReviewLabel(rid, sentiment, praise, complaints, str(obj.get("summary", ""))))

    got = [l.id for l in labels]
    if sorted(got) != sorted(expected):
        missing = set(expected) - set(got)
        extra = set(got) - set(expected)
        raise LabelValidationError(f"id mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return labels


def classify_dataframe(
    df: pd.DataFrame,
    complete: Callable[[str], str],
    batch_size: int = 20,
    max_attempts: int = 2,
) -> pd.DataFrame:
    """Label every row of a parser DataFrame. Returns one row per review with the label columns.

    Batches whose reply fails validation are retried once; if they still fail, their rows get
    ``sentiment = None`` and ``label_error`` set, so failures are counted rather than hidden.
    """
    out = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        pairs = list(zip(batch["recommendation_id"], batch["review_text"]))
        ids = [p[0] for p in pairs]
        prompt = build_prompt(pairs)
        error = None
        for _ in range(max_attempts):
            try:
                labels = parse_labels(complete(prompt), ids)
                break
            except LabelValidationError as exc:
                error = str(exc)
        else:
            labels = None

        if labels is None:
            out.extend(
                {"recommendation_id": rid, "sentiment": None, "praise": [], "complaints": [],
                 "summary": "", "label_error": error}
                for rid in ids
            )
        else:
            out.extend(
                {"recommendation_id": l.id, "sentiment": l.sentiment, "praise": l.praise,
                 "complaints": l.complaints, "summary": l.summary, "label_error": None}
                for l in labels
            )
    labels_df = pd.DataFrame(out)
    labels_df["prompt_version"] = PROMPT_VERSION
    return labels_df
