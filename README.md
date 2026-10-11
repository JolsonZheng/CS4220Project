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
