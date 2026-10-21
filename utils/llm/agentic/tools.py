"""Search and read web evidence through Tavily using the Python standard library.

Copied from AIRO's fri-forecaster package; see THIRD_PARTY_NOTICES.md.
The functions return JSON-compatible results, including an error field when a
request fails. The pi extension records these results as forecast evidence.
"""

import json
import os
import re
import urllib.error
import urllib.request
from datetime import date, timedelta

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"

# One read_page reply is this many characters of the page. pi_tools.ts
# caps a tool reply at 8000 characters, so a window has to fit under that with
# its envelope; the model pages through a longer document with `offset`.
PAGE_CHARS = 7000

# URL GUARD (2026-10-16, AIRO's anchor-guard experiment). OFF unless the
# environment sets FORECASTER_BLOCK_URL_PATTERNS: regexes separated by newlines
# or commas (so no commas inside a pattern), searched case-insensitively
# against each URL. pi_tools.ts hands its
# environment to every tool process, so one variable reaches both tools.
# read_page refuses a matching URL without calling Tavily, with the ordinary
# failure shape and a neutral reason; web_search drops matching results, and
# results titled as the AIRO paper (BLOCK_TITLES), and blanks Tavily's
# `answer` when it dropped anything, because that summary was written from
# the unfiltered results. ForecastBench shares these tools, so nothing here
# changes when the variable is unset.
BLOCK_ENV = "FORECASTER_BLOCK_URL_PATTERNS"
BLOCK_TITLES = ("Automated Artificial Intelligence Risk Outlook",)


def blocked_patterns():
    """Compile the caller's optional URL exclusion patterns."""
    raw = os.environ.get(BLOCK_ENV) or ""
    return [re.compile(p.strip(), re.I) for p in re.split(r"[\n,]", raw) if p.strip()]


def is_blocked(url, patterns):
    """Return whether a URL matches an exclusion pattern."""
    return bool(url) and any(p.search(url) for p in patterns)


def search_body(query, max_results=5, recent_days=None, today=None):
    """Build a Tavily search request without credentials.

    With `recent_days`, the
    search runs on Tavily's NEWS index bounded to [today - recent_days, today]
    by exact dates -- probed 2026-09-02: start_date/end_date filter precisely
    there, and not at all on the general index, whose only bound is the coarse
    time_range bucket. Without it, the general index, unbounded.
    """
    body = {
        "query": query,
        "max_results": max(1, min(int(max_results or 5), 10)),
        # "advanced" (2026-09-02 evening, the source-quality pass): Tavily
        # re-ranks and extracts the relevant passage per result instead of
        # returning the page head; two credits a search instead of one.
        "search_depth": "advanced",
        "include_answer": True,
    }
    if recent_days:
        today = today or date.today()
        days = max(1, min(int(recent_days), 3650))
        body.update(
            {
                "topic": "news",
                "include_answer": False,
                "start_date": (today - timedelta(days=days)).isoformat(),
                "end_date": today.isoformat(),
            }
        )
    return body


def web_search(query, max_results=5, recent_days=None, cause=None):
    """Search the web via Tavily.

    Returns {query, answer?, results:[{title,url,
    content, published?}]}. `recent_days` bounds the search to news published in
    the last N days (see search_body); the date each result was published comes
    back with it.

    `cause` (2026-09-11) is the question group the search is for. It is echoed
    on the result and never sent to Tavily: the forecasting loop's coverage
    rule in pi_tools.ts reads it, with `recent_days`,
    off the recorded evidence to decide whether every group has been checked.

    Needs TAVILY_API_KEY; without it (or on any network error) returns a result
    carrying an `error` string and no results, so the loop keeps going ungrounded.
    """
    tag = {"cause": cause} if cause is not None else {}
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return {
            "query": query,
            **tag,
            "results": [],
            "error": "web search unavailable (no TAVILY_API_KEY set)",
        }
    spec = search_body(query, max_results, recent_days)
    body = json.dumps(dict(spec, api_key=key)).encode()
    req = urllib.request.Request(
        TAVILY_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        return {"query": query, **tag, "results": [], "error": f"web search failed: {e}"}
    out = {"query": query, **tag, "answer": data.get("answer"), "results": []}
    guard = blocked_patterns()
    if recent_days:
        out["window"] = {"start": spec["start_date"], "end": spec["end_date"]}
    for r in data.get("results", []):
        if guard and (
            is_blocked(r.get("url"), guard)
            or any(t in (r.get("title") or "") for t in BLOCK_TITLES)
        ):
            out["answer"] = None
            continue
        item = {
            "title": r.get("title"),
            "url": r.get("url"),
            # Tavily already returns trimmed snippets; cap defensively
            "content": (r.get("content") or "")[:1500],
        }
        if r.get("published_date"):
            item["published"] = r["published_date"]
        out["results"].append(item)
    return out


def read_page(url, offset=0):
    """Read one window of web-page text through Tavily extract.

    Return PAGE_CHARS
    characters from `offset`. Returns {url, offset, total_chars, text,
    next_offset?}; `next_offset` is present only when there is more.

    Needs TAVILY_API_KEY (the same key as web_search); without it, or when the
    page cannot be fetched, returns a result carrying an `error` string and no
    text, so the loop keeps going.
    """
    if is_blocked(url, blocked_patterns()):
        return {"url": url, "text": "", "error": "page read failed: page unavailable"}
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return {"url": url, "text": "", "error": "page reading unavailable (no TAVILY_API_KEY set)"}
    body = json.dumps({"urls": [url], "extract_depth": "basic"}).encode()
    req = urllib.request.Request(
        TAVILY_EXTRACT_URL,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        return {"url": url, "text": "", "error": f"page read failed: {e}"}
    results = data.get("results") or []
    if not results:
        failed = data.get("failed_results") or []
        why = (
            failed[0].get("error") if failed and isinstance(failed[0], dict) else None
        ) or "no content returned"
        return {"url": url, "text": "", "error": f"page read failed: {why}"}
    text = results[0].get("raw_content") or ""
    try:
        start = max(0, int(offset or 0))
    except (TypeError, ValueError):
        start = 0
    out = {
        "url": url,
        "offset": start,
        "total_chars": len(text),
        "text": text[start : start + PAGE_CHARS],
    }
    if start + PAGE_CHARS < len(text):
        out["next_offset"] = start + PAGE_CHARS
    return out


WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current facts, recent events, base rates, expert "
        "estimates and reporting relevant to a question. Use it to ground your "
        "forecast in evidence past your training cutoff, and use it repeatedly: "
        "one query per thing you need to know, refined by what earlier results "
        "said. Set recent_days to search only news published in the last N days "
        "(counted back from today's date, which your prompt states); each such "
        "result carries its publication date. Returns titled result snippets with "
        "URLs; call read_page on a URL when the snippet is not enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {
                "type": "integer",
                "description": "how many results to return (default 5, max 10)",
            },
            "recent_days": {
                "type": "integer",
                "description": "only news published within this many days "
                "before today; omit to search the whole web",
            },
            # A forecasting run narrows this to its question groups (an enum)
            # and can require it with tool_parameters.
            "cause": {
                "type": "string",
                "description": "the question group this search is for, by the key "
                "the prompt uses for it",
            },
        },
        "required": ["query"],
    },
    "fn": web_search,
}

READ_PAGE_TOOL = {
    "name": "read_page",
    "description": (
        "Read the text of one web page (a search result, a report, a paper, a "
        "dataset page) when a snippet is not enough to rely on. Returns up to "
        f"{PAGE_CHARS} characters from `offset`; if the reply carries `next_offset`, "
        "call again with it to keep reading."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "the page to read"},
            "offset": {
                "type": "integer",
                "description": "character offset to start from (default 0)",
            },
        },
        "required": ["url"],
    },
    "fn": read_page,
}

# the default grounding toolset for a forecasting run
FORECAST_TOOLS = [WEB_SEARCH_TOOL, READ_PAGE_TOOL]
