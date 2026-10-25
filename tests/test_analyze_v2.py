"""Regression tests for keyword_baseline_v2 (clause-level sentiment, negation,
broader vocabulary, theme mentions). Review texts are modelled on the false
negatives found in the Checkpoint 2 30-review sample."""

import pandas as pd
import pytest

from steam_reviewer.analyze import (
    aggregate_insights,
    clause_score,
    label_dataframe,
    label_review,
    split_clauses,
)


@pytest.mark.parametrize('text', [
    'AWESOME SAUCE',
    'cool',
    'goated nothing can stop it',
    'it is a good game would recomand 111/10',
    'A important game to me<3',
    'its the best free to play game u can get.',
    'Joyita 10/10',
])
def test_short_praise_is_positive(text):
    assert label_review('1', text)['sentiment'] == 'positive'


@pytest.mark.parametrize('text', [
    'I cannot enjoy this game due to joining late in a match, most of the time.',
    'not fun anymore',
    'worst update ever',
    'dead game full of bots',
    '2/10',
])
def test_negative_and_negated_praise(text):
    assert label_review('1', text)['sentiment'] == 'negative'


def test_negated_negative_counts_as_positive():
    assert clause_score('not bad at all') > 0
    assert label_review('1', 'no lag and no crashes')['sentiment'] == 'positive'


def test_v1_behaviour_is_unchanged():
    row = label_review('1', 'AWESOME SAUCE', method='v1')
    assert row['sentiment'] == 'mixed'
    assert row['label_method'] == 'keyword_baseline_v1'


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        label_review('1', 'great', method='v9')


def test_clause_split_on_but():
    assert split_clauses('great maps but the servers are dead') == [
        'great maps', 'the servers are dead'
    ]


def test_mixed_review_keeps_clause_level_themes():
    row = label_review('1', 'Great maps but the servers are dead')
    assert row['sentiment'] == 'mixed'
    assert row['praise'] == ['gameplay']
    assert row['complaints'] == ['multiplayer']
    assert row['themes'] == ['gameplay', 'multiplayer']


def test_neutral_clause_follows_overall_polarity():
    row = label_review('1', 'Love it. The servers.')
    assert row['sentiment'] == 'positive'
    assert row['praise'] == ['multiplayer']


def test_mentions_kept_when_no_polarity():
    row = label_review('1', 'Story exists')
    assert row['sentiment'] == 'mixed'
    assert row['praise'] == [] and row['complaints'] == []
    assert row['themes'] == ['story']


def test_empty_and_missing_text():
    assert label_review('1', '')['sentiment'] == 'mixed'
    assert label_review('1', None)['sentiment'] == 'mixed'
    assert label_review('1', float('nan'))['themes'] == []


def test_curly_apostrophe_negation():
    assert label_review('1', 'I don’t enjoy it')['sentiment'] == 'negative'


def test_aggregate_reports_coverage_and_mentions():
    reviews = pd.DataFrame([
        {'recommendation_id': '1', 'review_text': 'Great maps but the servers are dead',
         'voted_up': True},
        {'recommendation_id': '2', 'review_text': 'awesome', 'voted_up': True},
        {'recommendation_id': '3', 'review_text': 'hm', 'voted_up': False},
    ])
    report = aggregate_insights(reviews, label_dataframe(reviews))
    assert report['sentiment_coverage'] == round(1 / 3, 3)
    assert report['theme_coverage'] == round(1 / 3, 3)
    assert report['theme_mentions'] == {'gameplay': 1, 'multiplayer': 1}
    assert report['top_praise'] == {'gameplay': 1}
    assert report['top_complaints'] == {'multiplayer': 1}
    assert report['n_disagreements'] == 0


@pytest.mark.parametrize(('text', 'theme'), [
    ('Love the game but not the fandom', 'multiplayer'),
    ('Do NOT install if you like a casual gaming experience', 'multiplayer'),
    ('tf3 when?', 'developer_support'),
    ('when heavy update', 'developer_support'),
])
def test_implicit_complaint_phrasing(text, theme):
    row = label_review('1', text)
    assert theme in row['complaints']


def test_explicit_positive_topic_is_not_complaint():
    row = label_review('1', 'I love the community and the servers')
    assert 'multiplayer' in row['praise']
    assert 'multiplayer' not in row['complaints']
