"""Optional model-based labeler, for comparison against the keyword baselines.

Uses the provider-agnostic prompt/validation code in ``prompts.py`` and any
OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, a local Ollama or LM Studio
server, an institutional proxy). Nothing here is needed for the offline baseline.

Configuration comes from the environment so no key is ever committed:

    MODEL_API_KEY    bearer token (may be empty for a local server)
    MODEL_BASE_URL   default https://api.openai.com/v1
    MODEL_NAME       default gpt-4o-mini

Usage (labels only the manual-set reviews, to keep the number of calls small):

    python -m steam_reviewer.model_label 440 --manual-file data/manual/440_manual_v2.csv
    python -m steam_reviewer.manual_eval 440 --mode compare \
        --manual-file data/manual/440_manual_v2.csv --model-labels data/results/440_model_labels.csv

Every request (status, latency, retries) is appended to a JSON run log next to the labels so
cost / rate-limit assumptions can be checked against a committed artifact.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

from .parser import load_raw_dir
from .prompts import PROMPT_VERSION, LabelValidationError, classify_dataframe

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class ModelAPIError(RuntimeError):
    """The model endpoint failed after all retries."""


class ChatCompletion:
    """``complete(prompt) -> str`` backed by an OpenAI-compatible chat endpoint.

    Retries on HTTP 429 / 5xx with exponential backoff and records every call in ``log``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        session: Optional[requests.Session] = None,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        temperature: float = 0.0,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.http = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.temperature = temperature
        self.log: list[dict] = []

    def __call__(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        last = None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            resp = self.http.post(self.url, headers=headers, json=body, timeout=120)
            entry = {
                "at": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
                "status": resp.status_code,
                "latency_s": round(time.monotonic() - started, 3),
                "prompt_chars": len(prompt),
            }
            if resp.status_code == 429 or resp.status_code >= 500:
                self.log.append(entry)
                last = f"HTTP {resp.status_code}"
                if attempt < self.max_retries:
                    self.sleep(self.backoff_s * (2 ** attempt))
                    continue
                break
            if resp.status_code != 200:
                self.log.append(entry)
                raise ModelAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()
            usage = payload.get("usage") or {}
            entry["prompt_tokens"] = usage.get("prompt_tokens")
            entry["completion_tokens"] = usage.get("completion_tokens")
            self.log.append(entry)
            try:
                return payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelAPIError(f"unexpected response shape: {exc}") from exc
        raise ModelAPIError(f"gave up after {self.max_retries + 1} attempts ({last})")


def tolerant(complete: Callable[[str], str]) -> Callable[[str], str]:
    """Turn endpoint failures into validation failures, so ``classify_dataframe`` records the
    batch as a counted ``label_error`` instead of aborting the whole run."""

    def wrapped(prompt: str) -> str:
        try:
            return complete(prompt)
        except ModelAPIError as exc:
            raise LabelValidationError(f"api: {exc}") from exc

    return wrapped


def label_with_model(
    reviews: pd.DataFrame,
    complete: Callable[[str], str],
    model_name: str,
    batch_size: int = 10,
) -> pd.DataFrame:
    labels = classify_dataframe(
        reviews[["recommendation_id", "review_text"]], tolerant(complete), batch_size=batch_size
    )
    labels["recommendation_id"] = labels["recommendation_id"].astype(str)
    labels["label_method"] = f"model_{PROMPT_VERSION}"
    labels["model"] = model_name
    return labels


def run_summary(labels: pd.DataFrame, log: list[dict], model_name: str) -> dict:
    statuses: dict[str, int] = {}
    for entry in log:
        statuses[str(entry["status"])] = statuses.get(str(entry["status"]), 0) + 1
    ok = [e for e in log if e["status"] == 200]
    return {
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "n_reviews": int(len(labels)),
        "n_label_errors": int(labels["sentiment"].isna().sum()),
        "n_requests": len(log),
        "status_counts": statuses,
        "prompt_tokens": sum(e.get("prompt_tokens") or 0 for e in ok),
        "completion_tokens": sum(e.get("completion_tokens") or 0 for e in ok),
        "requests": log,
    }


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Label reviews with an LLM for comparison.")
    ap.add_argument("appid", type=int)
    ap.add_argument("--manual-file", type=Path, default=None,
                    help="label only the reviews in this manual CSV (recommended)")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.manual_file:
        reviews = pd.read_csv(args.manual_file, dtype={"recommendation_id": str}).fillna("")
    else:
        reviews = load_raw_dir(args.appid, args.raw_dir)
    if reviews.empty:
        ap.error("no reviews to label")

    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    complete = ChatCompletion(
        model=model_name,
        base_url=os.environ.get("MODEL_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("MODEL_API_KEY", ""),
    )
    labels = label_with_model(reviews, complete, model_name, args.batch_size)

    out = args.out or Path("data/results") / f"{args.appid}_model_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    log_path = out.with_name(out.stem + "_runlog.json")
    summary = run_summary(labels, complete.log, model_name)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=2))
    print(f"wrote {out} and {log_path}")


if __name__ == "__main__":
    main()
