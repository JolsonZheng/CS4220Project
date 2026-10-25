import argparse
import json
from pathlib import Path

from .analyze import METHODS, aggregate_insights, label_dataframe
from .parser import load_raw_dir, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Analyze saved Steam reviews without an API key'
    )
    parser.add_argument('appid', type=int, help='Steam game app ID')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--out-dir', type=Path, default=Path('data/results'))
    parser.add_argument(
        '--method', choices=METHODS, default='v2',
        help='keyword baseline version (v1 = Checkpoint 2, v2 = clause-level with negation)'
    )

    args = parser.parse_args(argv)

    reviews = load_raw_dir(args.appid, args.raw_dir)

    if reviews.empty:
        parser.error(
            'No saved reviews found. Run python -m steam_reviewer.fetch APPID first.'
        )

    labels = label_dataframe(reviews, method=args.method)

    report = {
        'game_id': args.appid,
        'label_method': f'keyword_baseline_{args.method}',
        'review_statistics': summarize(reviews),
        'insights': aggregate_insights(reviews, labels),
        # Written by the code (not added by hand) so the committed report matches the CLI.
        'note': 'Keyword baseline; theme polarity is approximate, not validated AI classification.',
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.out_dir / f'{args.appid}_labels.csv'
    report_path = args.out_dir / f'{args.appid}_report.json'

    labels.to_csv(labels_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding='utf-8'
    )

    print(json.dumps(report, indent=2))
    print(f'\nSaved: {labels_path} and {report_path}')


if __name__ == '__main__':
    main()
