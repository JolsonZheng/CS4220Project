import pandas as pd
import pytest

from steam_reviewer.analyze import label_dataframe, aggregate_insights


def test_keyword_baseline_and_disagreement():
    reviews = pd.DataFrame([
        {
            'recommendation_id': '1',
            'review_text': 'Great combat and beautiful graphics',
            'voted_up': True
        },
        {
            'recommendation_id': '2',
            'review_text': 'Terrible bugs and lag',
            'voted_up': True
        },
        {
            'recommendation_id': '3',
            'review_text': 'Story exists',
            'voted_up': False
        },
    ])

    labels = label_dataframe(reviews)
    report = aggregate_insights(reviews, labels)

    assert report['reviews_analyzed'] == 3
    assert report['top_praise']['gameplay'] == 1
    assert report['top_complaints']['bugs'] == 1
    assert report['disagreement_ids'] == ['2']
    assert report['sentiment_counts'] == {
        'positive': 1,
        'negative': 1,
        'mixed': 1
    }


def test_no_substring_false_positive():
    reviews = pd.DataFrame([
        {
            'recommendation_id': 'x',
            'review_text': 'The hardship is memorable',
            'voted_up': True
        }
    ])

    assert label_dataframe(reviews).iloc[0]['sentiment'] == 'mixed'


def test_duplicate_ids_rejected():
    reviews = pd.DataFrame([
        {
            'recommendation_id': 'x',
            'review_text': 'Great gameplay',
            'voted_up': True
        }
    ] * 2)

    with pytest.raises(ValueError, match='duplicate'):
        aggregate_insights(reviews, label_dataframe(reviews))
