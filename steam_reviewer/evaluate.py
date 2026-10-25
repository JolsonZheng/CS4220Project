"""Evaluation harness: compare model labels with Steam's own thumbs-up and with manual labels.

Manual labels live in a CSV with columns ``recommendation_id,sentiment,praise,complaints``
where praise/complaints are ``;``-separated theme names (may be empty).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .prompts import SENTIMENTS, THEMES

import ast


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
    """Convert theme labels into a set."""

    if isinstance(value, (list, tuple, set)):
        return {t for t in value if t in THEMES}

    if not isinstance(value, str) or not value.strip():
        return set()

    # Handle lists saved as strings in CSV files
    if value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return {t for t in parsed if t in THEMES}
        except (ValueError, SyntaxError):
            return set()

    # Handle themes separated by semicolons
    return {
        t.strip()
        for t in value.split(";")
        if t.strip() in THEMES
    }





def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "support": int(tp + fn),
    }


def _read_manual(manual_csv: Path) -> pd.DataFrame:
    manual = pd.read_csv(manual_csv, dtype={"recommendation_id": str}).fillna("")
    manual["sentiment"] = manual["sentiment"].astype(str).str.strip().str.lower()
    if "label_source" not in manual.columns:
        # Checkpoint 2 sample predates the column; its labels were AI-assisted.
        manual["label_source"] = "unspecified"
    return manual


def agreement_with_manual(labels: pd.DataFrame, manual_csv: Path) -> dict:
    """Sentiment accuracy, coverage, confusion counts, disagreements and theme
    precision/recall (overall and per theme) vs. manual labels.

    ``label_source`` in the manual CSV (``human`` / ``ai_assisted``) is reported so a
    reader can tell how much of the reference set was checked by a person.
    """
    manual = _read_manual(manual_csv)
    labels = labels.copy()
    labels["recommendation_id"] = labels["recommendation_id"].astype(str)
    merged = manual.merge(labels, on="recommendation_id", suffixes=("_true", "_pred"))
    merged = merged[merged["sentiment_pred"].notna()]
    if merged.empty:
        return {"n_compared": 0}

    confusion = {
        t: {p: int(((merged["sentiment_true"] == t) & (merged["sentiment_pred"] == p)).sum())
            for p in SENTIMENTS}
        for t in SENTIMENTS
    }
    correct = merged["sentiment_true"] == merged["sentiment_pred"]
    polar_pred = merged["sentiment_pred"].isin(["positive", "negative"])
    wrong = merged[~correct]

    result = {
        "n_compared": int(len(merged)),
        "label_sources": {k: int(v) for k, v in merged["label_source"].value_counts().items()},
        "true_sentiment_counts": {s: int((merged["sentiment_true"] == s).sum()) for s in SENTIMENTS},
        "sentiment_accuracy": round(float(correct.mean()), 3),
        "sentiment_coverage": round(float(polar_pred.mean()), 3),
        "polar_accuracy": round(float(correct[polar_pred].mean()), 3) if polar_pred.any() else None,
        "sentiment_confusion": confusion,
        "n_disagreements": int(len(wrong)),
        "disagreements": [
            {"recommendation_id": rid, "true": t, "pred": p}
            for rid, t, p in zip(wrong["recommendation_id"], wrong["sentiment_true"],
                                 wrong["sentiment_pred"])
        ],
        "themes": {},
        "per_theme": {},
    }

    per_theme = {theme: [0, 0, 0] for theme in THEMES}
    for col in ("praise", "complaints"):
        tp = fp = fn = 0
        for true_v, pred_v in zip(merged[f"{col}_true"], merged[f"{col}_pred"]):
            t, p = _split_themes(true_v), _split_themes(pred_v)
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)
            for theme in t & p:
                per_theme[theme][0] += 1
            for theme in p - t:
                per_theme[theme][1] += 1
            for theme in t - p:
                per_theme[theme][2] += 1
        result["themes"][col] = _prf(tp, fp, fn)

    # Theme detection regardless of polarity: manual praise|complaints vs predicted mentions.
    if "themes" in merged.columns:
        tp = fp = fn = 0
        for true_p, true_c, pred_v in zip(merged["praise_true"], merged["complaints_true"],
                                          merged["themes"]):
            t = _split_themes(true_p) | _split_themes(true_c)
            p = _split_themes(pred_v)
            tp += len(t & p)
            fp += len(p - t)
            fn += len(t - p)
        result["themes"]["mentions"] = _prf(tp, fp, fn)

    result["per_theme"] = {
        theme: _prf(*counts) for theme, counts in per_theme.items() if any(counts)
    }
    return result


def compare_methods(method_labels: dict[str, pd.DataFrame], manual_csv: Path) -> dict:
    """Run ``agreement_with_manual`` for several labelers on the same manual set.

    Only reviews labelled by every method are compared, so the numbers are paired.
    """
    if not method_labels:
        return {"methods": {}, "n_common": 0}
    common = None
    for labels in method_labels.values():
        ids = set(labels.loc[labels["sentiment"].notna(), "recommendation_id"].astype(str))
        common = ids if common is None else common & ids
    results = {}
    for name, labels in method_labels.items():
        subset = labels[labels["recommendation_id"].astype(str).isin(common)]
        results[name] = agreement_with_manual(subset, manual_csv)
    summary = {
        name: {
            "sentiment_accuracy": r.get("sentiment_accuracy"),
            "sentiment_coverage": r.get("sentiment_coverage"),
            "praise_recall": r.get("themes", {}).get("praise", {}).get("recall"),
            "complaints_recall": r.get("themes", {}).get("complaints", {}).get("recall"),
        }
        for name, r in results.items()
    }
    return {"n_common": len(common), "summary": summary, "methods": results}
