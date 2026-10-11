import json
from pathlib import Path

from steam_reviewer.parser import COLUMNS, load_raw_dir, pages_to_dataframe, summarize

FIXTURE = Path(__file__).parent / "fixtures" / "sample_page.json"


def _page():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_columns_and_dedup_and_empty_dropped():
    df = pages_to_dataframe([_page()], appid=440)
    assert list(df.columns) == COLUMNS
    # 4 raw reviews: one duplicate id, one whitespace-only -> 2 rows
    assert sorted(df["recommendation_id"]) == ["1001", "1002"]


def test_types_and_conversions():
    df = pages_to_dataframe([_page()], appid=440)
    row = df.set_index("recommendation_id").loc["1001"]
    assert row["appid"] == 440
    assert bool(row["voted_up"])
    assert abs(row["weighted_vote_score"] - 0.612345) < 1e-9
    assert row["playtime_at_review_h"] == 10.0
    assert str(row["created_at"].tz) == "UTC"


def test_sorted_by_created_at():
    df = pages_to_dataframe([_page()], appid=440)
    assert df["created_at"].is_monotonic_increasing


def test_empty_pages():
    df = pages_to_dataframe([{"success": 1, "reviews": []}], appid=1)
    assert df.empty and list(df.columns) == COLUMNS
    assert summarize(df) == {"n_reviews": 0}


def test_load_raw_dir(tmp_path):
    app_dir = tmp_path / "440"
    app_dir.mkdir()
    (app_dir / "page_000.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    df = load_raw_dir(440, raw_dir=tmp_path)
    assert len(df) == 2
    s = summarize(df)
    assert s["n_reviews"] == 2 and s["pct_positive"] == 50.0
