"""Shared HTTP helpers: rate limiting, retry/backoff, and error logging.

Used by fetch_upcoming_fixtures.py (football-data.org, needs throttling) and
fetch_historical_results.py (football-data.co.uk, no auth/rate limit, but
still benefits from the same retry/backoff and logging behavior).
"""
import time
from datetime import datetime, timezone
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # fixes CERTIFICATE_VERIFY_FAILED behind this machine's TLS-inspecting proxy

import requests

ERROR_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "collection_errors.log"


def log_error(context: str, reason: str) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {context} | {reason}\n")


class RateLimiter:
    """Ensures at least `min_interval` seconds pass between calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def get_json(
    session: requests.Session,
    url: str,
    rate_limiter: RateLimiter,
    context: str,
    params: dict | None = None,
    headers: dict | None = None,
    max_retries: int = 5,
    timeout: int = 20,
):
    """GET a URL and return parsed JSON, or None on unrecoverable failure.

    Retries on network errors, 429 (respecting Retry-After), and 5xx with
    exponential backoff. Logs every failure (transient or final) via
    log_error and never raises - callers can rely on a None return to skip
    a record without crashing.
    """
    for attempt in range(1, max_retries + 1):
        rate_limiter.wait()
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as e:
            log_error(context, f"network error on attempt {attempt}/{max_retries}: {e}")
            time.sleep(min(2 ** attempt, 30))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as e:
                log_error(context, f"invalid JSON in response: {e}")
                return None

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after else min(2 ** attempt * 5, 60)
            log_error(context, f"rate limited (429), waiting {wait_s}s (attempt {attempt}/{max_retries})")
            time.sleep(wait_s)
            continue

        if resp.status_code in (500, 502, 503, 504):
            log_error(context, f"server error {resp.status_code} (attempt {attempt}/{max_retries})")
            time.sleep(min(2 ** attempt, 30))
            continue

        # Non-retryable client error (401, 403, 404, etc.)
        log_error(context, f"HTTP {resp.status_code}: {resp.text[:300]}")
        return None

    log_error(context, f"gave up after {max_retries} attempts")
    return None


def get_csv_bytes(
    session: requests.Session,
    url: str,
    context: str,
    max_retries: int = 3,
    timeout: int = 20,
) -> bytes | None:
    """GET a URL and return the raw response bytes (for CSV downloads), or None on failure.

    football-data.co.uk has no auth and no documented rate limit, so this
    skips RateLimiter, but still retries transient network/server errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
        except requests.RequestException as e:
            log_error(context, f"network error on attempt {attempt}/{max_retries}: {e}")
            time.sleep(min(2 ** attempt, 15))
            continue

        if resp.status_code == 200:
            return resp.content

        if resp.status_code in (500, 502, 503, 504):
            log_error(context, f"server error {resp.status_code} (attempt {attempt}/{max_retries})")
            time.sleep(min(2 ** attempt, 15))
            continue

        # 404 is expected for season codes that don't exist yet/anymore - not retryable.
        log_error(context, f"HTTP {resp.status_code}")
        return None

    log_error(context, f"gave up after {max_retries} attempts")
    return None
