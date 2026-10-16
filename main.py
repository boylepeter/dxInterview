#!/usr/bin/env python3
"""Fetch merged pull requests from the GitHub API and write them to a CSV.

Usage:
    python fetch_merged_prs.py OWNER/REPO [--limit 25] [--output merged_prs.csv]

Example:
    python fetch_merged_prs.py psf/requests --limit 10

Authentication (optional but strongly recommended):
    export GITHUB_TOKEN=ghp_yourtoken      # macOS/Linux
    $env:GITHUB_TOKEN = "ghp_yourtoken"    # Windows PowerShell
Unauthenticated requests are limited to 60/hour; authenticated, 5,000/hour.
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime

import requests

API_BASE = "https://api.github.com"
TIMEOUT = (5, 15)  # (connect, read) seconds. requests has NO default timeout.
MAX_ATTEMPTS = 4
MAX_RATE_LIMIT_WAIT = 300  # never sleep longer than 5 min on a rate limit

logger = logging.getLogger("merged_prs")

CSV_COLUMNS = [
    "pr_number", "pr_title", "pr_url",
    "author_login", "author_id", "author_name", "author_email",
    "author_company", "author_location", "author_public_repos",
    "author_followers", "author_account_created", "author_profile_url",
    "merger_login", "merger_id", "merger_name", "merger_email",
    "merger_company", "merger_location", "merger_public_repos",
    "merger_followers", "merger_account_created", "merger_profile_url",
    "additions", "deletions",
    "created_at", "merged_at",
    "time_to_merge_seconds", "time_to_merge_human",
]


# --------------------------------------------------------------------------
# Errors: our own exception so callers don't need to know we use `requests`
# --------------------------------------------------------------------------
class GitHubApiError(Exception):
    """Raised when we can't get a usable response from GitHub."""


# --------------------------------------------------------------------------
# HTTP layer: one place that handles timeouts, retries, and rate limits
# --------------------------------------------------------------------------
def build_session() -> requests.Session:
    """A Session reuses TCP connections and lets us set headers once."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "merged-prs-exercise",  # GitHub rejects requests without one
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    else:
        logger.warning("No GITHUB_TOKEN set; limited to 60 requests/hour.")
    return session


def _seconds_until_reset(response: requests.Response) -> float | None:
    """If GitHub says we're rate limited, work out how long to wait."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    if response.headers.get("X-RateLimit-Remaining") == "0":
        reset = response.headers.get("X-RateLimit-Reset")
        if reset and reset.isdigit():
            return max(0.0, int(reset) - time.time()) + 1
    return None


def get(session: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    """GET with timeout, retry on transient failures, and rate-limit awareness.

    Returns the Response for any status we should NOT retry (200, 404, ...);
    the caller decides what a 404 means.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        last_attempt = attempt == MAX_ATTEMPTS
        wait = None

        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if last_attempt:
                raise GitHubApiError(f"Network failure calling {url}: {exc}") from exc
            logger.warning("Attempt %d: network error (%s); retrying", attempt, exc)
        else:
            status = response.status_code
            # GitHub signals rate limiting with 403 or 429 (not just 429)
            if status in (403, 429):
                wait = _seconds_until_reset(response)
                if wait is None:
                    return response  # a real 403 (permissions), not a rate limit
                if wait > MAX_RATE_LIMIT_WAIT or last_attempt:
                    raise GitHubApiError(
                        f"Rate limited; would need to wait {wait:.0f}s. "
                        "Set GITHUB_TOKEN or try again later."
                    )
                logger.warning("Rate limited; sleeping %.0fs", wait)
            elif status in (500, 502, 503, 504):
                if last_attempt:
                    raise GitHubApiError(f"HTTP {status} from {url} after {attempt} attempts")
                logger.warning("Attempt %d: HTTP %d; retrying", attempt, status)
            else:
                return response

        # Exponential backoff with jitter, unless the server told us exactly how long
        delay = wait if wait is not None else 0.5 * 2 ** (attempt - 1)
        time.sleep(delay + random.uniform(0, 0.25))

    raise GitHubApiError("unreachable")  # loop always returns or raises; keeps linters happy


def get_json(session: requests.Session, url: str, params: dict | None = None):
    """GET and return parsed JSON, raising GitHubApiError on any failure."""
    response = get(session, url, params)
    if response.status_code != 200:
        raise GitHubApiError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:  # body wasn't valid JSON
        raise GitHubApiError(f"Invalid JSON from {url}") from exc


# --------------------------------------------------------------------------
# GitHub-specific logic
# --------------------------------------------------------------------------
def iter_merged_prs(session: requests.Session, repo: str, limit: int):
    """Yield full details of merged PRs, newest-updated first, up to `limit`.

    A generator (note `yield`): it produces one PR at a time instead of
    building a giant list, so we can stop early once we have enough.

    Why two calls per PR? The *list* endpoint omits additions/deletions and
    merged_by. Only the single-PR "detail" endpoint includes them.
    """
    url = f"{API_BASE}/repos/{repo}/pulls"
    params = {"state": "closed", "sort": "updated", "direction": "desc", "per_page": 100}
    found = 0

    while url and found < limit:
        response = get(session, url, params)
        if response.status_code == 404:
            raise GitHubApiError(f"Repo '{repo}' not found (or it's private and you lack access).")
        if response.status_code != 200:
            raise GitHubApiError(f"HTTP {response.status_code}: {response.text[:200]}")

        for summary in response.json():
            # "closed" includes PRs closed WITHOUT merging; those have merged_at = None
            if not summary.get("merged_at"):
                continue
            yield get_json(session, summary["url"])
            found += 1
            if found >= limit:
                return

        # Pagination: GitHub gives the next page's full URL in the Link header.
        # requests parses it into response.links. Params are baked into that URL.
        url = response.links.get("next", {}).get("url")
        params = None


def get_user(session: requests.Session, login: str | None, cache: dict) -> dict:
    """Fetch user details, caching so each person is looked up only once."""
    if not login:
        return {}
    if login in cache:
        return cache[login]
    response = get(session, f"{API_BASE}/users/{login}")
    if response.status_code == 200:
        cache[login] = response.json()
    else:
        # Deleted accounts ("ghost") or odd cases: keep going with what we know
        logger.warning("Could not fetch user '%s' (HTTP %d)", login, response.status_code)
        cache[login] = {"login": login}
    return cache[login]


def parse_timestamp(value: str) -> datetime:
    """GitHub sends '2024-01-15T10:30:00Z'. Python < 3.11 can't parse the 'Z'."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def user_columns(prefix: str, user: dict) -> dict:
    """Flatten a user object into CSV columns like 'author_login'."""
    return {
        f"{prefix}_login": user.get("login"),
        f"{prefix}_id": user.get("id"),
        f"{prefix}_name": user.get("name"),
        f"{prefix}_email": user.get("email"),  # often None: users can hide it
        f"{prefix}_company": user.get("company"),
        f"{prefix}_location": user.get("location"),
        f"{prefix}_public_repos": user.get("public_repos"),
        f"{prefix}_followers": user.get("followers"),
        f"{prefix}_account_created": user.get("created_at"),
        f"{prefix}_profile_url": user.get("html_url"),
    }


def build_row(pr: dict, session: requests.Session, user_cache: dict) -> dict:
    author = get_user(session, (pr.get("user") or {}).get("login"), user_cache)
    # merged_by can be missing; the exercise says it equals the author, so fall back to that
    merger_login = (pr.get("merged_by") or {}).get("login")
    merger = get_user(session, merger_login, user_cache) if merger_login else author

    created = parse_timestamp(pr["created_at"])
    merged = parse_timestamp(pr["merged_at"])
    delta = merged - created  # subtracting datetimes gives a timedelta

    row = {
        "pr_number": pr["number"],
        "pr_title": pr["title"],
        "pr_url": pr["html_url"],
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "created_at": pr["created_at"],
        "merged_at": pr["merged_at"],
        "time_to_merge_seconds": int(delta.total_seconds()),
        "time_to_merge_human": str(delta),  # e.g. "2 days, 3:04:05"
    }
    row.update(user_columns("author", author))
    row.update(user_columns("merger", merger))
    return row


def write_csv(path: str, rows: list[dict]) -> None:
    # newline="" prevents blank lines between rows on Windows (csv module quirk)
    # encoding="utf-8" because names and titles contain non-ASCII characters
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged GitHub PRs to CSV.")
    parser.add_argument("repo", help="Repository as OWNER/REPO, e.g. psf/requests")
    parser.add_argument("--limit", type=int, default=25, help="Max merged PRs to fetch (default 25)")
    parser.add_argument("--output", default="merged_prs.csv", help="CSV path (default merged_prs.csv)")
    args = parser.parse_args()
    if args.repo.count("/") != 1:
        parser.error("repo must look like OWNER/REPO")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    session = build_session()
    user_cache: dict = {}
    rows = []

    try:
        for pr in iter_merged_prs(session, args.repo, args.limit):
            rows.append(build_row(pr, session, user_cache))
            logger.info("Processed PR #%s (%d/%d)", pr["number"], len(rows), args.limit)
    except GitHubApiError as exc:
        logger.error("%s", exc)
        if rows:  # don't throw away work already done
            logger.info("Writing the %d rows collected before the failure.", len(rows))
            write_csv(args.output, rows)
        return 1
    finally:
        session.close()

    if not rows:
        logger.warning("No merged PRs found in %s.", args.repo)
        return 0

    write_csv(args.output, rows)
    logger.info("Wrote %d rows to %s", len(rows), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())