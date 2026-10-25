from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .prompts import THEMES

# ---------------------------------------------------------------------------
# keyword_baseline_v1 (Checkpoint 2) - kept unchanged so v1 and v2 can be compared
# on the same evaluation set.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# keyword_baseline_v2 (Checkpoint 3)
#  - vocabulary broadened with the false negatives seen in the 30-review sample
#    (short slang praise such as "goated", "awesome", "cool", "best", "10/10", "<3")
#  - negation within three words flips a term ("cannot enjoy", "not fun")
#  - sentiment and theme polarity are decided per clause, so a review that is
#    "mixed" overall still contributes praise/complaint themes
#  - every theme found is kept in a separate ``themes`` column (mentions), so
#    coverage of theme detection is measurable independently of polarity
# ---------------------------------------------------------------------------

KEYWORDS_V2 = {
    'gameplay': KEYWORDS['gameplay'] + (
        'mechanic', 'movement', 'gunplay', 'classes', 'class', 'maps', 'map',
        'weapons', 'weapon', 'game modes', 'gamemodes',
    ),
    'story': KEYWORDS['story'] + ('lore', 'characters', 'character', 'comics'),
    'graphics': KEYWORDS['graphics'] + ('graphic', 'art', 'animations', 'visual'),
    'audio': KEYWORDS['audio'] + ('sound', 'sounds', 'voice acting', 'voice lines'),
    'performance': KEYWORDS['performance'] + (
        'performance', 'frames', 'laggy', 'lagging', 'stuttering', 'optimized', 'ping',
    ),
    'bugs': KEYWORDS['bugs'] + ('glitches', 'glitchy', 'crashes', 'crashing', 'crashed', 'buggy'),
    'price_value': KEYWORDS['price_value'] + (
        'free', 'free to play', 'f2p', 'money', 'overpriced', 'cheap', 'refund',
    ),
    'content_length': KEYWORDS['content_length'] + ('length', 'replay', 'replayable'),
    'multiplayer': KEYWORDS['multiplayer'] + (
        'server', 'players', 'player', 'online', 'queue', 'community', 'friends',
        'bots', 'bot', 'cheaters', 'cheater', 'hackers', 'match', 'casual', 'teammates',
        'fandom',
    ),
    'difficulty': KEYWORDS['difficulty'] + ('learning curve', 'skill', 'difficult', 'sweaty'),
    'monetization': KEYWORDS['monetization'] + (
        'microtransactions', 'crates', 'keys', 'cosmetics', 'hats', 'p2w',
        'loot boxes', 'lootboxes', 'trading',
    ),
    'developer_support': KEYWORDS['developer_support'] + (
        'valve', 'update', 'devs', 'dev', 'abandoned', 'tf3', 'patch',
    ),
}

POSITIVE_V2 = POSITIVE + (
    'awesome', 'cool', 'best', 'goated', 'goat', 'peak', 'masterpiece', 'classic',
    'recommend', 'recommended', 'recomend', 'recomand', 'fantastic', 'perfect',
    'favorite', 'favourite', 'addictive', 'nice', 'solid', 'wonderful', 'incredible',
    'legendary', 'enjoyable', 'enjoyed', 'loved', 'loving', 'brilliant', 'underrated',
)

NEGATIVE_V2 = NEGATIVE + (
    'worst', 'waste', 'unplayable', 'dead', 'dull', 'dreary', 'useless', 'toxic',
    'cheaters', 'hackers', 'sucks', 'trash', 'garbage', 'disappointing', 'disappointed',
    'overpriced', 'laggy', 'crashes', 'crashing', 'annoying', 'frustrating', 'refund',
    'abandoned', 'unbalanced', 'tired', 'bots',
)

NEGATORS = (
    'not', 'no', 'never', 'dont', "don't", 'doesnt', "doesn't", 'isnt', "isn't",
    'wasnt', "wasn't", 'cant', "can't", 'cannot', 'wont', "won't", 'aint', "ain't",
    'hardly', 'nothing', 'didnt', "didn't",
)

POSITIVE_SYMBOLS = ('<3', '❤', '\U0001F44D')  # <3, heart, thumbs-up
NEGATIVE_SYMBOLS = ('\U0001F44E',)  # thumbs-down

_CLAUSE_SPLIT = re.compile(r'[.!?;:\n]+|,|\bbut\b|\bhowever\b|\bthough\b')
_RATING = re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*10(?!\d)')
_WORD = re.compile(r"[a-z0-9']+")

LABEL_COLUMNS = [
    'recommendation_id', 'sentiment', 'praise', 'complaints',
    'summary', 'label_error', 'label_method', 'themes',
]

METHODS = ('v1', 'v2')


def _contains(text: str, term: str) -> bool:
    return re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', text) is not None


def _normalize(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ''
    return str(text).lower().replace('’', "'").replace('‘', "'")


def _negated(clause: str, start: int) -> bool:
    """True if one of the three words before ``start`` is a negator."""
    previous = _WORD.findall(clause[:start])[-3:]
    return any(word in NEGATORS for word in previous)


def _term_polarity(clause: str, terms, polarity: int) -> int:
    score = 0
    for term in terms:
        for match in re.finditer(r'(?<!\w)' + re.escape(term) + r'(?!\w)', clause):
            score += -polarity if _negated(clause, match.start()) else polarity
    return score


# These patterns catch a few complaints that contain no conventional negative
# adjective. They are deliberately narrow to avoid treating every question or
# negation as a negative review.
_DISCOURAGED_INSTALL = re.compile(r'\b(?:do\s+not|don\'t|dont|never)\s+(?:install|download|buy|play)\b')
_NEGATED_TOPIC = re.compile(r'\b(?:not|hate)\s+(?:the\s+|this\s+)?(?:fandom|community|servers|matchmaking)\b')
_UPDATE_REQUEST = re.compile(
    r'\b(?:when\s+(?:is|are|will|do\s+we\s+get)?\s*(?:the\s+)?(?:(?:next|heavy|major|new)\s+)?(?:update|patch|tf3)'
    r'|(?:update|patch|tf3)\s+when)\b'
)


def clause_score(clause: str) -> int:
    """Signed sentiment of one clause: >0 positive, <0 negative, 0 neutral."""
    score = _term_polarity(clause, POSITIVE_V2, 1) + _term_polarity(clause, NEGATIVE_V2, -1)
    # An explicit warning or rejection of a named topic can convey a complaint
    # without words like 'bad' or 'terrible'.
    if _DISCOURAGED_INSTALL.search(clause):
        score -= 2
    if _NEGATED_TOPIC.search(clause):
        score -= 1
    if _UPDATE_REQUEST.search(clause):
        score -= 1
    score += sum(clause.count(symbol) for symbol in POSITIVE_SYMBOLS)
    score -= sum(clause.count(symbol) for symbol in NEGATIVE_SYMBOLS)
    for match in _RATING.finditer(clause):
        rating = float(match.group(1))
        if rating >= 7:
            score += 1
        elif rating <= 4:
            score -= 1
    return score


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def _themes_in(text: str, keywords: dict) -> list[str]:
    return [
        theme for theme in THEMES
        if any(_contains(text, term) for term in keywords[theme])
    ]


def label_review_v1(review_id: str, text: str) -> dict:
    """Checkpoint 2 baseline: whole-review word counts, themes follow overall sentiment."""
    lowered = str(text).lower()
    positive = sum(_contains(lowered, word) for word in POSITIVE)
    negative = sum(_contains(lowered, word) for word in NEGATIVE)

    sentiment = (
        'positive' if positive > negative
        else 'negative' if negative > positive
        else 'mixed'
    )

    themes = _themes_in(lowered, KEYWORDS)

    # Sort themes into positive and negative categories
    return {
        'recommendation_id': str(review_id),
        'sentiment': sentiment,
        'praise': themes if sentiment == 'positive' else [],
        'complaints': themes if sentiment == 'negative' else [],
        'summary': '',
        'label_error': None,
        'label_method': 'keyword_baseline_v1',
        'themes': themes,
    }


def label_review_v2(review_id: str, text: str) -> dict:
    """Clause-level baseline with negation, broader vocabulary and theme mentions."""
    lowered = _normalize(text)
    clauses = split_clauses(lowered)
    scores = [clause_score(c) for c in clauses]

    positive = sum(s for s in scores if s > 0)
    negative = -sum(s for s in scores if s < 0)
    sentiment = (
        'positive' if positive > negative
        else 'negative' if negative > positive
        else 'mixed'
    )

    praise: list[str] = []
    complaints: list[str] = []
    mentions: list[str] = []
    for clause, score in zip(clauses, scores):
        found = _themes_in(clause, KEYWORDS_V2)
        for theme in found:
            if theme not in mentions:
                mentions.append(theme)
        # Neutral clause ("the servers") takes the review's overall polarity;
        # in a mixed review it stays a mention only.
        polarity = score if score else (
            1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0
        )
        target = praise if polarity > 0 else complaints if polarity < 0 else None
        if target is not None:
            for theme in found:
                if theme not in target:
                    target.append(theme)

    order = {t: i for i, t in enumerate(THEMES)}
    return {
        'recommendation_id': str(review_id),
        'sentiment': sentiment,
        'praise': sorted(praise, key=order.get),
        'complaints': sorted(complaints, key=order.get),
        'summary': '',
        'label_error': None,
        'label_method': 'keyword_baseline_v2',
        'themes': sorted(mentions, key=order.get),
    }


def label_review(review_id: str, text: str, method: str = 'v2') -> dict:
    """Find the sentiment and themes of a review with the chosen keyword baseline."""
    if method == 'v1':
        return label_review_v1(review_id, text)
    if method == 'v2':
        return label_review_v2(review_id, text)
    raise ValueError(f'unknown labeling method {method!r}; expected one of {METHODS}')


def label_dataframe(reviews: pd.DataFrame, method: str = 'v2') -> pd.DataFrame:
    rows = [
        label_review(str(r.recommendation_id), r.review_text, method)
        for r in reviews.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def aggregate_insights(reviews: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Count common themes and compare results with Steam ratings."""
    from .evaluate import agreement_with_thumbs

    if (
        reviews['recommendation_id'].astype(str).duplicated().any()
        or labels['recommendation_id'].astype(str).duplicated().any()
    ):
        raise ValueError('duplicate recommendation_id in reviews or labels')

    praise, complaints, mentions = Counter(), Counter(), Counter()

    for row in labels.itertuples(index=False):
        praise.update(set(row.praise))
        complaints.update(set(row.complaints))
        row_themes = getattr(row, 'themes', None)
        if isinstance(row_themes, (list, tuple, set)):
            mentions.update(set(row_themes))

    merged = reviews[['recommendation_id', 'voted_up']].merge(
        labels[['recommendation_id', 'sentiment']],
        on='recommendation_id'
    )

    disagreements = merged[
        (merged.sentiment.isin(['positive', 'negative'])) &
        ((merged.sentiment == 'positive') != merged.voted_up)
    ]

    n = int(len(merged))
    polar = int(merged.sentiment.isin(['positive', 'negative']).sum())
    theme_lists = labels['themes'] if 'themes' in labels.columns else []
    with_theme = int(sum(1 for t in theme_lists if isinstance(t, (list, tuple, set)) and t))

    return {
        'reviews_analyzed': n,
        'sentiment_counts': {
            key: int((merged.sentiment == key).sum())
            for key in ('positive', 'negative', 'mixed')
        },
        'sentiment_coverage': round(polar / n, 3) if n else None,
        'theme_coverage': round(with_theme / len(labels), 3) if len(labels) else None,
        'top_praise': dict(praise.most_common()),
        'top_complaints': dict(complaints.most_common()),
        'theme_mentions': dict(mentions.most_common()),
        'agreement_with_steam': agreement_with_thumbs(reviews, labels),
        'n_disagreements': int(len(disagreements)),
        'disagreement_ids': disagreements.recommendation_id.astype(str).tolist()[:25],
    }
