import json

import pandas as pd
import pytest

from steam_reviewer.evaluate import agreement_with_manual, agreement_with_thumbs
from steam_reviewer.prompts import LabelValidationError, build_prompt, classify_dataframe, parse_labels


def test_build_prompt_contains_ids_and_truncates():
    prompt = build_prompt([("1", "x" * 3000)], max_chars=100)
    items = json.loads(prompt.split("Reviews:\n", 1)[1])
    assert items == [{"id": "1", "text": "x" * 100}]


def test_parse_labels_accepts_fenced_json_and_filters_unknown_themes():
    raw = '```json\n[{"id": "1", "sentiment": "Positive", "praise": ["story", "vibes"], "complaints": []}]\n```'
    labels = parse_labels(raw, ["1"])
    assert labels[0].sentiment == "positive"
    assert labels[0].praise == ["story"]


@pytest.mark.parametrize("raw", [
    "not json",
    '{"id": "1"}',
    '[{"id": "1", "sentiment": "angry"}]',
    '[{"id": "2", "sentiment": "positive"}]',
])
def test_parse_labels_rejects_bad_output(raw):
    with pytest.raises(LabelValidationError):
        parse_labels(raw, ["1"])


def _reviews():
    return pd.DataFrame({
        "recommendation_id": ["1", "2", "3"],
        "review_text": ["great story", "crashes constantly", "ok but short"],
        "voted_up": [True, False, True],
    })


def _fake_model(prompt):
    items = json.loads(prompt.split("Reviews:\n", 1)[1])
    out = []
    for it in items:
        if "crash" in it["text"]:
            out.append({"id": it["id"], "sentiment": "negative", "praise": [], "complaints": ["bugs"]})
        elif "short" in it["text"]:
            out.append({"id": it["id"], "sentiment": "mixed", "praise": [], "complaints": ["content_length"]})
        else:
            out.append({"id": it["id"], "sentiment": "positive", "praise": ["story"], "complaints": []})
    return json.dumps(out)


def test_classify_dataframe_and_thumbs_agreement():
    labels = classify_dataframe(_reviews(), _fake_model, batch_size=2)
    assert list(labels["recommendation_id"]) == ["1", "2", "3"]
    m = agreement_with_thumbs(_reviews(), labels)
    assert m == {"n_reviews": 3, "n_label_errors": 0, "n_mixed": 1, "polar_agreement": 1.0}


def test_classify_dataframe_records_failures():
    labels = classify_dataframe(_reviews(), lambda p: "garbage", batch_size=3)
    assert labels["sentiment"].isna().all()
    assert labels["label_error"].notna().all()


def test_agreement_with_manual(tmp_path):
    manual = tmp_path / "manual.csv"
    manual.write_text(
        "recommendation_id,sentiment,praise,complaints\n"
        "1,positive,story,\n"
        "2,negative,,bugs;performance\n"
        "3,negative,,content_length\n"
    )
    labels = classify_dataframe(_reviews(), _fake_model)
    m = agreement_with_manual(labels, manual)
    assert m["n_compared"] == 3
    assert m["sentiment_accuracy"] == round(2 / 3, 3)
    assert m["sentiment_confusion"]["negative"]["mixed"] == 1
    assert m["themes"]["complaints"] == {"precision": 1.0, "recall": round(2 / 3, 3), "support": 3}
