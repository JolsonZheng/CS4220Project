from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .prompts import THEMES

# Keywords used to identify review themes
KEYWORDS = {
    'gameplay': ('gameplay', 'combat', 'controls', 'mechanics'),
    'story': ('story', 'plot', 'writing', 'narrative'),
    'graphics': ('graphics', 'visuals', 'art style', 'textures'),
    'audio': ('music', 'soundtrack', 'audio', 'sound design'),
    'performance': ('fps', 'framerate', 'lag', 'stutter', 'optimization'),
    'bugs': ('bug', 'bugs', 'glitch', 'crash', 'broken'),
    'price_value': ('price', 'expensive', 'worth', 'value'),
    'content_length': ('content', 'short', 'hours', 'replayability'),
    'multiplayer': ('multiplayer', 'servers', 'matchmaking', 'co-op'),
    'difficulty': ('difficulty', 'hard', 'easy', 'challenging'),
    'monetization': ('microtransaction', 'pay to win', 'battle pass', 'dlc'),
    'developer_support': ('developer', 'updates', 'patches', 'support'),
}

POSITIVE = (
    'good', 'great', 'fun', 'love', 'excellent',
    'amazing', 'enjoy', 'smooth', 'beautiful'
)

NEGATIVE = (
    'bad', 'awful', 'hate', 'boring', 'terrible',
    'broken', 'buggy', 'crash', 'lag', 'expensive'
)


def _contains(text: str, term: str) -> bool:
    return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text) is not None


def label_review(review_id: str, text: str) -> dict:
    """Find the sentiment and themes of a review."""
    lowered = str(text).lower()
    positive = sum(_contains(lowered, word) for word in POSITIVE)
    negative = sum(_contains(lowered, word) for word in NEGATIVE)

    sentiment = (
        'positive' if positive > negative
        else 'negative' if negative > positive
        else 'mixed'
    )

    themes = [
        theme for theme in THEMES
        if any(_contains(lowered, term) for term in KEYWORDS[theme])
    ]

    # Sort themes into positive and negative categories
    return {
        'recommendation_id': str(review_id),
        'sentiment': sentiment,
        'praise': themes if sentiment == 'positive' else [],
        'complaints': themes if sentiment == 'negative' else [],
        'summary': '',
        'label_error': None,
        'label_method': 'keyword_baseline_v1',
    }


def label_dataframe(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = [
        label_review(str(r.recommendation_id), r.review_text)
        for r in reviews.itertuples(index=False)
    ]
    return pd.DataFrame(
        rows,
        columns=[
            'recommendation_id', 'sentiment', 'praise',
            'complaints', 'summary', 'label_error', 'label_method'
        ]
    )


def aggregate_insights(reviews: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Count common themes and compare results with Steam ratings."""
    from .evaluate import agreement_with_thumbs

    if (
        reviews['recommendation_id'].astype(str).duplicated().any()
        or labels['recommendation_id'].astype(str).duplicated().any()
    ):
        raise ValueError('duplicate recommendation_id in reviews or labels')

    praise, complaints = Counter(), Counter()

    for row in labels.itertuples(index=False):
        praise.update(set(row.praise))
        complaints.update(set(row.complaints))

    merged = reviews[['recommendation_id', 'voted_up']].merge(
        labels[['recommendation_id', 'sentiment']],
        on='recommendation_id'
    )

    disagreements = merged[
        (merged.sentiment.isin(['positive', 'negative'])) &
        ((merged.sentiment == 'positive') != merged.voted_up)
    ]

    return {
        'reviews_analyzed': int(len(merged)),
        'sentiment_counts': {
            key: int((merged.sentiment == key).sum())
            for key in ('positive', 'negative', 'mixed')
        },
        'top_praise': dict(praise.most_common()),
        'top_complaints': dict(complaints.most_common()),
        'agreement_with_steam': agreement_with_thumbs(reviews, labels),
        'disagreement_ids': disagreements.recommendation_id.astype(str).tolist()[:25],
    }
