"""Read-only OpenRouter catalog checks. These requests do not generate tokens."""
from urllib.parse import quote
import requests

BASE = 'https://openrouter.ai/api/v1/models'


def get_model_endpoints(model_id, session=None, timeout=20):
    """Return the advertised active endpoints or raise on network/catalog errors."""
    http = session or requests
    url = f'{BASE}/{quote(model_id, safe="/")}/endpoints'
    response = http.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.json().get('data') or {}
    endpoints = data.get('endpoints')
    if not isinstance(endpoints, list):
        raise ValueError(f'Missing endpoints list for {model_id}')
    return endpoints


def matched_providers(model_id, requested='', session=None):
    eps = get_model_endpoints(model_id, session=session)
    if requested:
        eps = [p for p in eps if requested.lower() in {
            str(p.get('provider_name', '')).lower(),
            str(p.get('tag', '')).lower(),
            str(p.get('provider_slug', '')).lower()
        }]
    return eps
