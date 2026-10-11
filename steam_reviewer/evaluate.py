"""Evaluation harness: compare model labels with Steam's own thumbs-up and with manual labels.

Manual labels live in a CSV with columns ``recommendation_id,sentiment,praise,complaints``
where praise/complaints are ``;``-separated theme names (may be empty).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prompts import SENTIMENTS, THEMES


def agreement_with_thumbs(reviews: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """How often the model's polar sentiment matches the reviewer's own recommend / not-recommend.

    'mixed' predictions are reported separately and excluded from the agreement rate.
    """
    merged = reviews[["recommendation_id", "voted_up"]].merge(labels, on="recommendation_id")
    labelled = merged[merged["sentiment"].notna()]
    polar = labelled[labelled["sentiment"] != "mixed"]
    agree = (polar["sentiment"] == "positive") == polar["voted_up"]
    return {
        "n_reviews": int(len(merged)),
        "n_label_errors": int(merged["sentiment"].isna().sum()),
        "n_mixed": int((labelled["sentiment"] == "mixed").sum()),
        "polar_agreement": round(float(agree.mean()), 3) if len(polar) else None,
    }


def _split_themes(value) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {t for t in value if t in THEMES}
    if not isinstance(value, str) or not value.strip():
        return set()
    return {t.strip() for t in value.split(";") if t.strip() in THEMES}


def agreement_with_manual(labels: pd.DataFrame, manual_csv: Path) -> dict:
    """Sentiment accuracy, confusion counts and per-theme precision/recall vs. manual labels."""
    manual = pd.read_csv(manual_csv, dtype={"recommendation_id": str}).fillna("")
    merged = manual.merge(labels, on="recommendation_id", suffixes=("_true", "_pred"))
    merged = merged[merged["sentiment_pred"].notna()]
    if merged.empty:
        return {"n_compared": 0}

    confusion = {
        t: {p: int(((merged["sentiment_true"] == t) & (merged["sentiment_pred"] == p)).sum())
            for p in SENTIMENTS}
        for t in SENTIMENTS
    }
    result = {
        "n_compared": int(len(merged)),
        "sentiment_accuracy": round(float((merged["sentiment_true"] == merged["sentiment_pred"]).mean()), 3),
        "sentiment_confusion": confusion,
        "themes": {},
    }
    for col in ("praise", "complaints"):
        tp = fp = fn = 0
        for true_v, pred_v in zip(merged[f"{col}_true"], merged[f"{col}_pred"]):
            t, p = _split_themes(true_v), _split_themes(pred_v)
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)
        result["themes"][col] = {
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }
    return result
