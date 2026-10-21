"""Check research filtering, page windows, and retry boundaries without HTTP."""

import io
import json
from unittest.mock import Mock

import pytest

from utils.llm.agentic import tool_cli, tools


def response(data):
    """Encode a Tavily-shaped response for urllib's context manager."""
    return io.BytesIO(json.dumps(data).encode())


@pytest.mark.parametrize("url", ["https://blocked.example/report", "https://BLOCKED.example/"])
def test_blocked_page_never_reaches_tavily(url, monkeypatch):
    """Apply URL exclusions before issuing any research request."""
    monkeypatch.setenv("FORECASTER_BLOCK_URL_PATTERNS", r"blocked\.example")
    network = Mock(side_effect=AssertionError("blocked URL must not be fetched"))
    monkeypatch.setattr(tools.urllib.request, "urlopen", network)
    result = tools.read_page(url)
    assert result["text"] == ""
    assert "page unavailable" in result["error"]
    network.assert_not_called()


@pytest.mark.parametrize("patterns", [None, "", r"blocked\.example"])
def test_search_filter_removes_derived_answer_only_when_results_are_removed(patterns, monkeypatch):
    """Avoid exposing a summary derived from excluded search results."""
    data = {
        "answer": "Summary of both sources.",
        "results": [
            {"title": "Blocked source", "url": "https://blocked.example/report", "content": "x"},
            {"title": "Allowed source", "url": "https://allowed.example/report", "content": "y"},
        ],
    }
    if patterns is not None:
        monkeypatch.setenv("FORECASTER_BLOCK_URL_PATTERNS", patterns)
    monkeypatch.setattr(tools.urllib.request, "urlopen", lambda *args, **kwargs: response(data))
    result = tools.web_search("base rate")
    if patterns:
        assert result["answer"] is None
        assert [row["url"] for row in result["results"]] == ["https://allowed.example/report"]
    else:
        assert result["answer"] == data["answer"]
        assert len(result["results"]) == 2


def test_read_page_returns_contiguous_windows(monkeypatch):
    """Let callers page through extracted text without gaps or duplication."""
    text = "x" * 7000 + "tail"
    monkeypatch.setattr(
        tools.urllib.request,
        "urlopen",
        lambda *args, **kwargs: response({"results": [{"raw_content": text}]}),
    )
    first = tools.read_page("https://allowed.example/report")
    second = tools.read_page("https://allowed.example/report", offset=first["next_offset"])
    assert first["text"] + second["text"] == text
    assert first["total_chars"] == len(text)
    assert "next_offset" not in second


def test_missing_research_key_returns_error_without_http(monkeypatch):
    """Report unavailable research rather than making an unauthenticated request."""
    monkeypatch.delenv("TAVILY_API_KEY")
    network = Mock(side_effect=AssertionError("missing key must stop HTTP"))
    monkeypatch.setattr(tools.urllib.request, "urlopen", network)
    assert "TAVILY_API_KEY" in tools.web_search("base rate")["error"]
    assert "TAVILY_API_KEY" in tools.read_page("https://example.org")["error"]
    network.assert_not_called()


def test_cli_retries_rate_limits_but_not_other_failures():
    """Retry a temporary quota response without repeatedly fetching a missing page."""
    replies = iter([{"error": "HTTP Error 429: Too Many Requests"}, {"results": []}])
    waits = []
    assert tool_cli.call(lambda **kwargs: next(replies), {}, sleep=waits.append) == {"results": []}
    assert len(waits) == 1
    assert waits[0] > 0
    waits.clear()
    missing = {"error": "page read failed: 404"}
    assert tool_cli.call(lambda **kwargs: missing, {}, sleep=waits.append) == missing
    assert waits == []
