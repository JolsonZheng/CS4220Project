"""Regression tests for the manual-label workflow and the CLI edge cases."""

import json
from pathlib import Path

import pandas as pd
import pytest

from steam_reviewer import cli, manual_eval
from steam_reviewer.analyze import label_dataframe
from steam_reviewer.evaluate import agreement_with_manual, compare_methods

FIXTURE = Path(__file__).parent / "fixtures" / "sample_page.json"


def _write_manual(path, rows, with_source=True):
    df = pd.DataFrame(rows)
    if with_source and "label_source" not in df.columns:
        df["label_source"] = "human"
    df.to_csv(path, index=False)
    return path


MANUAL_ROWS = [
    {"recommendation_id": "1", "review_text": "Great maps but the servers are dead",
     "voted_up": True, "sentiment": "mixed", "praise": "gameplay", "complaints": "multiplayer"},
    {"recommendation_id": "2", "review_text": "awesome", "voted_up": True,
     "sentiment": "positive", "praise": "", "complaints": ""},
    {"recommendation_id": "3", "review_text": "crashes every match", "voted_up": False,
     "sentiment": "negative", "praise": "", "complaints": "bugs;multiplayer"},
]


# --- evaluate.agreement_with_manual -------------------------------------------------

def test_agreement_reports_coverage_disagreements_and_sources(tmp_path):
    manual = _write_manual(tmp_path / "m.csv", MANUAL_ROWS)
    labels = label_dataframe(pd.DataFrame(MANUAL_ROWS)[["recommendation_id", "review_text"]])
    r = agreement_with_manual(labels, manual)
    assert r["n_compared"] == 3
    assert r["label_sources"] == {"human": 3}
    assert r["sentiment_accuracy"] == 1.0
    assert r["sentiment_coverage"] == round(2 / 3, 3)
    assert r["n_disagreements"] == 0
    assert r["themes"]["praise"] == {"precision": 1.0, "recall": 1.0, "support": 1}
    # "crashes every match" -> bugs + multiplayer complaints
    assert r["themes"]["complaints"]["recall"] == 1.0
    assert r["themes"]["mentions"]["recall"] == 1.0
    assert r["per_theme"]["multiplayer"]["support"] == 2


def test_agreement_lists_disagreements(tmp_path):
    rows = [dict(MANUAL_ROWS[1], sentiment="negative")]
    manual = _write_manual(tmp_path / "m.csv", rows)
    labels = label_dataframe(pd.DataFrame(rows)[["recommendation_id", "review_text"]])
    r = agreement_with_manual(labels, manual)
    assert r["disagreements"] == [{"recommendation_id": "2", "true": "negative", "pred": "positive"}]
    assert r["polar_accuracy"] == 0.0


def test_agreement_without_label_source_column(tmp_path):
    manual = _write_manual(tmp_path / "m.csv", MANUAL_ROWS, with_source=False)
    labels = label_dataframe(pd.DataFrame(MANUAL_ROWS)[["recommendation_id", "review_text"]])
    assert agreement_with_manual(labels, manual)["label_sources"] == {"unspecified": 3}


def test_agreement_with_no_overlap(tmp_path):
    manual = _write_manual(tmp_path / "m.csv", MANUAL_ROWS)
    labels = pd.DataFrame([{"recommendation_id": "999", "sentiment": "positive",
                            "praise": [], "complaints": []}])
    assert agreement_with_manual(labels, manual) == {"n_compared": 0}


def test_compare_methods_uses_common_reviews(tmp_path):
    manual = _write_manual(tmp_path / "m.csv", MANUAL_ROWS)
    reviews = pd.DataFrame(MANUAL_ROWS)[["recommendation_id", "review_text"]]
    v1 = label_dataframe(reviews, method="v1")
    model = pd.DataFrame([
        {"recommendation_id": "1", "sentiment": "mixed", "praise": "['gameplay']",
         "complaints": "['multiplayer']"},
        {"recommendation_id": "2", "sentiment": None, "praise": "[]", "complaints": "[]"},
        {"recommendation_id": "3", "sentiment": "negative", "praise": "[]",
         "complaints": "['bugs']"},
    ])
    r = compare_methods({"v1": v1, "model": model}, manual)
    assert r["n_common"] == 2
    assert r["methods"]["model"]["n_compared"] == 2
    assert r["summary"]["model"]["sentiment_accuracy"] == 1.0


# --- manual_eval --------------------------------------------------------------------

def _reviews(n_up, n_down):
    rows = []
    for i in range(n_up + n_down):
        text = "x" * (5 if i % 2 else 50)
        rows.append({"recommendation_id": str(i), "review_text": text,
                     "voted_up": i < n_up, "review_length": len(text)})
    return pd.DataFrame(rows)


def test_balanced_sample_splits_classes_and_lengths():
    sample = manual_eval.balanced_sample(_reviews(40, 40), count=20)
    assert len(sample) == 20
    assert sample["voted_up"].sum() == 10
    assert (sample["review_length"] > 5).sum() == 10
    assert sample["recommendation_id"].is_unique


def test_balanced_sample_fills_shortfall_and_excludes():
    reviews = _reviews(60, 4)
    sample = manual_eval.balanced_sample(reviews, count=20, exclude_ids={"60"})
    assert len(sample) == 20
    down = sample[~sample["voted_up"]]
    assert set(down["recommendation_id"]) == {"61", "62", "63"}


def test_balanced_sample_is_reproducible():
    reviews = _reviews(30, 30)
    a = manual_eval.balanced_sample(reviews, count=12, seed=7)
    b = manual_eval.balanced_sample(reviews, count=12, seed=7)
    assert list(a["recommendation_id"]) == list(b["recommendation_id"])


def test_create_balanced_sample_writes_blank_label_columns(tmp_path):
    raw = tmp_path / "raw" / "440"
    raw.mkdir(parents=True)
    (raw / "page_000.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    out = manual_eval.create_balanced_sample(440, count=4, raw_dirs=[tmp_path / "raw"],
                                             manual_dir=tmp_path / "manual")
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == manual_eval.MANUAL_COLUMNS
    assert len(df) == 2  # fixture has two usable reviews


def test_check_manual_flags_problems():
    df = pd.DataFrame([
        {"recommendation_id": "1", "sentiment": "positive", "label_source": "human"},
        {"recommendation_id": "1", "sentiment": "", "label_source": "guess"},
    ])
    problems = manual_eval.check_manual(df)
    assert len(problems) == 3


def test_evaluate_sample_writes_json(tmp_path):
    manual = _write_manual(tmp_path / "440_manual.csv", MANUAL_ROWS)
    labels = label_dataframe(pd.DataFrame(MANUAL_ROWS)[["recommendation_id", "review_text"]])
    labels_path = tmp_path / "440_labels.csv"
    labels.to_csv(labels_path, index=False)  # lists are saved as "['gameplay']"
    out = tmp_path / "eval.json"
    result = manual_eval.evaluate_sample(440, manual_file=manual, labels_file=labels_path,
                                         output=out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["n_compared"] == 3 == result["n_compared"]
    assert saved["themes"]["praise"]["recall"] == 1.0


def test_evaluate_sample_refuses_unlabelled(tmp_path, capsys):
    rows = [dict(MANUAL_ROWS[0], sentiment="")]
    manual = _write_manual(tmp_path / "440_manual.csv", rows)
    labels_path = tmp_path / "440_labels.csv"
    label_dataframe(pd.DataFrame(rows)[["recommendation_id", "review_text"]]).to_csv(
        labels_path, index=False)
    assert manual_eval.evaluate_sample(440, manual_file=manual, labels_file=labels_path) is None
    assert "Finish labeling" in capsys.readouterr().out


def test_evaluate_sample_missing_files(tmp_path, capsys):
    assert manual_eval.evaluate_sample(440, manual_dir=tmp_path, results_dir=tmp_path) is None
    assert "Missing" in capsys.readouterr().out


def test_compare_on_manual_v1_vs_v2(tmp_path):
    manual = _write_manual(tmp_path / "440_manual.csv", MANUAL_ROWS)
    out = tmp_path / "cmp.json"
    result = manual_eval.compare_on_manual(440, manual_file=manual, output=out)
    assert set(result["summary"]) == {"keyword_baseline_v1", "keyword_baseline_v2"}
    assert result["summary"]["keyword_baseline_v2"]["sentiment_accuracy"] >= \
        result["summary"]["keyword_baseline_v1"]["sentiment_accuracy"]
    assert out.exists()


# --- cli ------------------------------------------------------------------------------

def test_cli_writes_labels_and_report(tmp_path):
    raw = tmp_path / "raw" / "440"
    raw.mkdir(parents=True)
    (raw / "page_000.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    cli.main(["440", "--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out")])
    report = json.loads((tmp_path / "out" / "440_report.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(tmp_path / "out" / "440_labels.csv", dtype=str)
    assert report["label_method"] == "keyword_baseline_v2"
    assert report["insights"]["reviews_analyzed"] == 2
    assert "note" in report
    assert set(labels["label_method"]) == {"keyword_baseline_v2"}


def test_cli_v1_method(tmp_path):
    raw = tmp_path / "raw" / "440"
    raw.mkdir(parents=True)
    (raw / "page_000.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    cli.main(["440", "--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out"),
              "--method", "v1"])
    labels = pd.read_csv(tmp_path / "out" / "440_labels.csv", dtype=str)
    assert set(labels["label_method"]) == {"keyword_baseline_v1"}


def test_cli_errors_without_saved_reviews(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["440", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
