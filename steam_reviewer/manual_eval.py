"""Manual-label evaluation workflow.

Modes
-----
sample    random sample (Checkpoint 2 behaviour) -> data/manual/<appid>_manual.csv
balanced  stratified sample for the Checkpoint 3 reference set: balanced across the
          reviewer's thumbs-up/down and across short/long reviews, excluding reviews
          already in an earlier manual file -> data/manual/<appid>_manual_v2.csv
evaluate  compare one saved labels CSV with a manual file -> <appid>_evaluation.json
compare   relabel the manual reviews with keyword v1 and v2 (and optionally a model
          labels CSV) and compare them on the same reviews -> <appid>_comparison.json

The manual CSV has columns
``recommendation_id,review_text,voted_up,sentiment,praise,complaints,label_source,notes``.
``label_source`` is ``human`` when a team member read the review and set/confirmed the label,
``ai_assisted`` when the label came from a tool and has not been checked yet.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .analyze import label_dataframe
from .evaluate import agreement_with_manual, compare_methods
from .parser import load_raw_dir

MANUAL_DIR = Path("data/manual")
RESULTS_DIR = Path("data/results")
RAW_DIR = Path("data/raw")

VALID_SENTIMENTS = {"positive", "negative", "mixed"}
LABEL_SOURCES = {"human", "ai_assisted"}
MANUAL_COLUMNS = [
    "recommendation_id", "review_text", "voted_up",
    "sentiment", "praise", "complaints", "label_source", "notes",
]


def _load_reviews(appid, raw_dirs):
    frames = [load_raw_dir(appid, Path(d)) for d in raw_dirs]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    reviews = pd.concat(frames, ignore_index=True)
    return reviews.drop_duplicates(subset="recommendation_id", keep="first").reset_index(drop=True)


def _blank_labels(sample):
    sample = sample[["recommendation_id", "review_text", "voted_up"]].copy()
    for col in ("sentiment", "praise", "complaints", "label_source", "notes"):
        sample[col] = ""
    return sample


def create_sample(appid, count=30, raw_dir=RAW_DIR, manual_dir=MANUAL_DIR):
    reviews = load_raw_dir(appid, Path(raw_dir))

    if reviews.empty:
        print("No reviews found.")
        return None

    # Pick a random sample that can be reproduced
    count = min(count, len(reviews))
    sample = reviews.sample(n=count, random_state=42)

    output = Path(manual_dir) / f"{appid}_manual.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Leave these columns blank for manual labeling
    _blank_labels(sample).to_csv(output, index=False)

    print(f"Created {output} with {count} reviews.")
    return output


def balanced_sample(reviews, count=50, exclude_ids=(), min_length=1, seed=42):
    """Stratified sample: half thumbs-up / half thumbs-down, each half split evenly
    between reviews shorter and longer than the median length.

    When a stratum has too few reviews (TF2 recent reviews are ~92% positive), the
    shortfall is filled from the other strata and reported by the caller.
    """
    pool = reviews[~reviews["recommendation_id"].astype(str).isin({str(i) for i in exclude_ids})]
    pool = pool[pool["review_length"] >= min_length]
    if pool.empty:
        return pool.iloc[0:0]

    median_len = pool["review_length"].median()
    pool = pool.assign(_long=pool["review_length"] > median_len)

    strata = [
        pool[(pool["voted_up"] == up) & (pool["_long"] == long_)]
        for up in (False, True)
        for long_ in (False, True)
    ]
    target = count // 4
    picks = []
    for i, stratum in enumerate(strata):
        n = min(len(stratum), target + (1 if i < count % 4 else 0))
        picks.append(stratum.sample(n=n, random_state=seed))
    chosen = pd.concat(picks)

    shortfall = min(count, len(pool)) - len(chosen)
    if shortfall > 0:
        rest = pool.drop(index=chosen.index)
        # Prefer the minority class (thumbs-down) when filling the shortfall.
        rest = rest.sort_values("voted_up", kind="stable")
        negatives = rest[~rest["voted_up"]]
        fill = negatives.head(shortfall)
        if len(fill) < shortfall:
            fill = pd.concat([
                fill,
                rest[rest["voted_up"]].sample(n=shortfall - len(fill), random_state=seed),
            ])
        chosen = pd.concat([chosen, fill])

    return chosen.drop(columns="_long").sample(frac=1, random_state=seed)


def create_balanced_sample(appid, count=50, raw_dirs=(RAW_DIR,), manual_dir=MANUAL_DIR,
                           exclude_files=(), min_length=1, seed=42):
    reviews = _load_reviews(appid, raw_dirs)
    if reviews.empty:
        print("No reviews found.")
        return None

    exclude = set()
    for path in exclude_files:
        path = Path(path)
        if path.exists():
            exclude |= set(pd.read_csv(path, dtype={"recommendation_id": str})["recommendation_id"])

    sample = balanced_sample(reviews, count, exclude, min_length, seed)
    output = Path(manual_dir) / f"{appid}_manual_v2.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    _blank_labels(sample).to_csv(output, index=False)

    n_down = int((~sample["voted_up"].astype(bool)).sum())
    print(f"Created {output} with {len(sample)} reviews "
          f"({n_down} thumbs-down, {len(sample) - n_down} thumbs-up; "
          f"{len(exclude)} earlier reviews excluded).")
    if n_down < len(sample) // 2:
        print("Too few thumbs-down reviews for a 50/50 split; fetch more with "
              "`python -m steam_reviewer.fetch APPID --review-type negative --raw-dir data/raw_negative` "
              "and pass --raw-dir data/raw --raw-dir data/raw_negative.")
    return output


def check_manual(manual):
    """Return a list of problems that block evaluation (empty list means OK)."""
    problems = []
    bad = manual[~manual["sentiment"].astype(str).str.strip().str.lower().isin(VALID_SENTIMENTS)]
    if len(bad):
        problems.append(f"{len(bad)} review(s) without a valid sentiment "
                        f"(first: {bad['recommendation_id'].iloc[0]})")
    if manual["recommendation_id"].duplicated().any():
        problems.append("duplicate recommendation_id in manual file")
    if "label_source" in manual.columns:
        sources = set(manual["label_source"].astype(str).str.strip()) - {""}
        unknown = sources - LABEL_SOURCES
        if unknown:
            problems.append(f"unknown label_source value(s): {sorted(unknown)}")
    return problems


def _manual_path(appid, manual_dir, manual_file):
    return Path(manual_file) if manual_file else Path(manual_dir) / f"{appid}_manual.csv"


def evaluate_sample(appid, manual_dir=MANUAL_DIR, results_dir=RESULTS_DIR,
                    manual_file=None, labels_file=None, output=None):
    manual_path = _manual_path(appid, manual_dir, manual_file)
    labels_path = Path(labels_file) if labels_file else Path(results_dir) / f"{appid}_labels.csv"

    if not manual_path.exists() or not labels_path.exists():
        print("Missing manual labels or predictions.")
        return None

    manual = pd.read_csv(manual_path, dtype=str).fillna("")
    problems = check_manual(manual)
    if problems:
        print("Finish labeling every review before evaluating: " + "; ".join(problems))
        return None

    # Saved theme lists ("['gameplay']") are parsed by evaluate._split_themes.
    labels = pd.read_csv(labels_path, dtype={"recommendation_id": str})

    result = agreement_with_manual(labels, manual_path)
    result["manual_file"] = manual_path.as_posix()
    result["labels_file"] = labels_path.as_posix()

    output = Path(output) if output else Path(results_dir) / f"{appid}_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved evaluation to {output}")
    return result


def compare_on_manual(appid, manual_dir=MANUAL_DIR, results_dir=RESULTS_DIR,
                      manual_file=None, model_labels=None, output=None):
    """Label the manual-set reviews with every keyword method (and an optional model
    labels CSV) and compare them pairwise on the same reviews."""
    manual_path = _manual_path(appid, manual_dir, manual_file)
    if not manual_path.exists():
        print(f"Missing {manual_path}.")
        return None
    manual = pd.read_csv(manual_path, dtype=str).fillna("")
    problems = check_manual(manual)
    if problems:
        print("Finish labeling every review before comparing: " + "; ".join(problems))
        return None

    reviews = manual[["recommendation_id", "review_text"]]
    methods = {
        "keyword_baseline_v1": label_dataframe(reviews, method="v1"),
        "keyword_baseline_v2": label_dataframe(reviews, method="v2"),
    }
    if model_labels:
        model = pd.read_csv(model_labels, dtype={"recommendation_id": str})
        name = "model"
        if "model" in model.columns and model["model"].notna().any():
            name = f"model:{model['model'].dropna().iloc[0]}"
        methods[name] = model

    result = compare_methods(methods, manual_path)
    result["manual_file"] = manual_path.as_posix()

    output = Path(output) if output else Path(results_dir) / f"{appid}_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved comparison to {output}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int)
    parser.add_argument(
        "--mode",
        choices=["sample", "balanced", "evaluate", "compare"],
        required=True
    )
    parser.add_argument("--count", type=int, default=None,
                        help="sample size (default 30 for sample, 50 for balanced)")
    parser.add_argument("--raw-dir", type=Path, action="append", default=None,
                        help="raw page directory; repeat to combine (default data/raw)")
    parser.add_argument("--manual-file", type=Path, default=None,
                        help="manual CSV (default data/manual/<appid>_manual.csv)")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="manual CSV whose reviews must not be re-sampled (balanced mode)")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--model-labels", type=Path, default=None,
                        help="labels CSV written by steam_reviewer.model_label (compare mode)")
    parser.add_argument("--out", type=Path, default=None)

    args = parser.parse_args(argv)
    raw_dirs = args.raw_dir or [RAW_DIR]

    if args.mode == "sample":
        create_sample(args.appid, args.count or 30, raw_dir=raw_dirs[0])
    elif args.mode == "balanced":
        create_balanced_sample(args.appid, args.count or 50, raw_dirs=raw_dirs,
                               exclude_files=args.exclude, min_length=args.min_length)
    elif args.mode == "evaluate":
        evaluate_sample(args.appid, manual_file=args.manual_file,
                        labels_file=args.labels_file, output=args.out)
    else:
        compare_on_manual(args.appid, manual_file=args.manual_file,
                          model_labels=args.model_labels, output=args.out)


if __name__ == "__main__":
    main()
