from __future__ import annotations

import json
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30):
    request_headers = {
        "User-Agent": "RAKSHA-AI/5.1 (disaster-data-ingestion; contact-configured-by-operator)",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
            return json.loads(payload.decode("utf-8")), dict(response.headers.items())
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc
