# CS4220Project
Steam Game Reviewer: ingest Steam user reviews, structure them with pandas, label sentiment and
gameplay themes with a model, and evaluate those labels against Steam's own recommend flag and
manual labels.

## Layout

- `steam_reviewer/fetch.py` - pages through `store.steampowered.com/appreviews/<appid>` (cursor
  pagination, retry/backoff on 429/5xx) and saves raw JSON to `data/raw/<appid>/page_NNN.json`.
- `steam_reviewer/parser.py` - flattens raw pages into a DataFrame (dedup by recommendation id,
  empty reviews dropped, UTC timestamps, playtime in hours) and writes `data/processed/<appid>_reviews.csv`.
- `steam_reviewer/prompts.py` - closed sentiment/theme schema, batch prompt builder, strict JSON
  validation of model replies; the model is injected as `complete(prompt) -> str`.
- `steam_reviewer/evaluate.py` - agreement with the reviewer's thumbs-up and with a manual-label CSV
  (`recommendation_id,sentiment,praise,complaints`, themes `;`-separated).
- `tests/` - unit tests using a synthetic fixture and a fake HTTP session / fake model (no network).

## Usage

    pip install -r requirements.txt
    python -m steam_reviewer.fetch 440 --max-reviews 500
    python -m steam_reviewer.parser 440
    pytest

## Checkpoint 3: label quality

- `steam_reviewer/analyze.py` - two keyword baselines. `v1` is the Checkpoint 2 labeler, kept
  unchanged for comparison. `v2` (default) scores sentiment per clause (split on punctuation and
  "but/however/though"), flips a term negated within three words ("cannot enjoy", "no lag"),
  reads ratings such as `10/10` and `<3`, and uses a broader vocabulary built from the
  Checkpoint 2 false negatives. Themes take the polarity of the clause they appear in, so a
  review that is mixed overall still yields praise and complaints; every theme found is also
  kept in a `themes` (mentions) column.
- `steam_reviewer/cli.py --method v1|v2` - the report now includes sentiment coverage, theme
  coverage and theme mentions.
- `steam_reviewer/manual_eval.py`
  - `--mode balanced` draws a 50-review set balanced across thumbs-up/down and short/long
    reviews, excluding the Checkpoint 2 sample (`--exclude data/manual/440_manual.csv`), into
    `data/manual/<appid>_manual_v2.csv`. Fill `sentiment`, `praise`, `complaints` (themes
    `;`-separated) and set `label_source` to `human` for every row you read and labelled.
  - `--mode evaluate` reports accuracy, coverage (share of non-mixed predictions), polar
    accuracy, confusion matrix, the list of disagreements, theme precision/recall for praise,
    complaints and mentions, and per-theme precision/recall.
  - `--mode compare` relabels the manual reviews with v1 and v2 (and, with `--model-labels`,
    an LLM) and compares them on the same reviews -> `data/results/<appid>_comparison.json`.
- `steam_reviewer/model_label.py` - optional LLM labeler over any OpenAI-compatible endpoint
  (`MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL_NAME`), with retry/backoff and a run log of every
  request (status, latency, tokens).

Checkpoint 3 workflow:

    python -m steam_reviewer.fetch 440 --max-reviews 300 --review-type negative --raw-dir data/raw_negative
    python -m steam_reviewer.cli 440                       # v2 labels + report
    python -m steam_reviewer.manual_eval 440 --mode balanced --raw-dir data/raw --raw-dir data/raw_negative \
        --exclude data/manual/440_manual.csv --min-length 10
    # label data/manual/440_manual_v2.csv by hand, label_source=human
    python -m steam_reviewer.manual_eval 440 --mode compare --manual-file data/manual/440_manual_v2.csv
    # optional: MODEL_API_KEY=... python -m steam_reviewer.model_label 440 --manual-file data/manual/440_manual_v2.csv
    #           then rerun compare with --model-labels data/results/440_model_labels.csv
    pytest
