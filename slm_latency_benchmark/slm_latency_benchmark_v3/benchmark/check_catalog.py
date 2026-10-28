#!/usr/bin/env python3
"""Verify remote generation endpoints without making billable inference calls."""
import argparse
import json
from pathlib import Path
import requests

try:
    from benchmark.catalog import matched_providers
except ModuleNotFoundError:
    from catalog import matched_providers

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Check current OpenRouter endpoint availability at no token cost.')
    parser.add_argument('--model-ids', default='1,2,3,4', help='Comma-separated 1-based model IDs.')
    args = parser.parse_args()
    models = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
    ids = [int(i.strip()) for i in args.model_ids.split(',')]
    failed = []
    for i in ids:
        model = models[i - 1]
        slug = model['openrouter']
        try:
            eps = matched_providers(slug, model.get('provider', ''))
            print(f'{i}. {model["name"]} | {slug} | {len(eps)} eligible endpoint(s)')
            for ep in eps:
                price = ep.get('pricing') or {}
                print('   Provider:', ep.get('provider_name') or '(unknown)',
                      '| Route:', ep.get('tag') or '(unspecified)',
                      '| Input USD/token:', price.get('prompt', '?'),
                      '| Output USD/token:', price.get('completion', '?'))
            if not eps:
                failed.append(model['name'])
        except (requests.RequestException, ValueError, KeyError) as exc:
            failed.append(model['name'])
            print(f'{i}. {model["name"]}: endpoint catalog check failed: {exc}')
    if failed:
        print('\nUnavailable or unverified:', ', '.join(failed))
        print('Do not assume a listed free model can actually accept generation requests.')
        print('Use --skip-unavailable in run.py to omit unavailable models from BOTH environments.')
        raise SystemExit(1)
    print('\nAll selected models have at least one advertised eligible endpoint.')


if __name__ == '__main__':
    main()
