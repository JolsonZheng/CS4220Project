import argparse
import json
from pathlib import Path

import pandas as pd

from .parser import load_raw_dir
from .evaluate import agreement_with_manual


def create_sample(appid, count=30):
    reviews = load_raw_dir(appid)

    if reviews.empty:
        print("No reviews found.")
        return

    # Pick a random sample that can be reproduced
    count = min(count, len(reviews))
    sample = reviews.sample(n=count, random_state=42)

    output = Path("data/manual") / f"{appid}_manual.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    sample = sample[
        ["recommendation_id", "review_text", "voted_up"]
    ].copy()

    # Leave these columns blank for manual labeling
    sample["sentiment"] = ""
    sample["praise"] = ""
    sample["complaints"] = ""

    sample.to_csv(output, index=False)

    print(f"Created {output} with {count} reviews.")


def evaluate_sample(appid):
    manual_path = Path("data/manual") / f"{appid}_manual.csv"
    labels_path = Path("data/results") / f"{appid}_labels.csv"

    if not manual_path.exists() or not labels_path.exists():
        print("Missing manual labels or predictions.")
        return

    manual = pd.read_csv(manual_path, dtype=str).fillna("")

    # Check that every review has a valid manual sentiment
    valid = {"positive", "negative", "mixed"}
    if not manual["sentiment"].isin(valid).all():
        print("Finish labeling every review before evaluating.")
        return

    labels = pd.read_csv(
        labels_path,
        dtype={"recommendation_id": str}
    )

    # Convert saved theme lists into the expected format
    import ast

    for col in ("praise", "complaints"):
        def parse_themes(value):
            if pd.isna(value):
                return []
            if isinstance(value, list):
                return value
            try:
                result = ast.literal_eval(value)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []

        labels[col] = labels[col].apply(parse_themes)

    result = agreement_with_manual(labels, manual_path)

    output = Path("data/results") / f"{appid}_evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved evaluation to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("appid", type=int)
    parser.add_argument(
        "--mode",
        choices=["sample", "evaluate"],
        required=True
    )
    parser.add_argument("--count", type=int, default=30)

    args = parser.parse_args()

    if args.mode == "sample":
        create_sample(args.appid, args.count)
    else:
        evaluate_sample(args.appid)


if __name__ == "__main__":
    main()
