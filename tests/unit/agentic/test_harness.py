"""Check the process boundary using recorded outputs from a fake pi."""

import json
import os
import sys
import textwrap

import pytest

from utils.llm.agentic import harness

SCHEMA = {
    "type": "object",
    "properties": {"k": {"type": "string"}, "probability": {"type": "number"}},
}
DESCRIPTIONS = {
    "web_search": "Search.",
    "read_page": "Read.",
    "submit_cells": "Submit some cells.",
    "submit_forecast": "Submit the forecast.",
}

FAKE_PI = textwrap.dedent("""\
    #!{python}
    import json, os, sys
    from pathlib import Path

    if "--version" in sys.argv:
        print({version!r})
        raise SystemExit(0)
    out = Path(os.environ["FORECASTER_OUT"])
    (out / "stdin.txt").write_text(sys.stdin.read())
    system = Path(sys.argv[sys.argv.index("--system-prompt") + 1])
    (out / "seen-system.txt").write_text(system.read_text())
    (out / "env.json").write_text(json.dumps({{
        name: os.environ.get(name) for name in (
            "PI_CODING_AGENT_DIR", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY", "FORECASTER_BLOCK_URL_PATTERNS"
        )
    }}))
    if {mode!r} == "broken":
        raise SystemExit(1)
    if {mode!r} == "provider-error":
        print(json.dumps({{"type": "message_end", "message": {{
            "role": "assistant", "stopReason": "error", "errorMessage": "rate limited",
            "usage": {{"input": 3}}
        }}}}))
        raise SystemExit(0)
    (out / "pieces.jsonl").write_text(json.dumps({{"forecasts": [
        {{"k": "a", "probability": 0.1}}, {{"k": "b", "probability": 0.2}},
        {{"k": "unknown-piece", "probability": 0.9}}
    ]}}) + "\\n")
    (out / "final.json").write_text(json.dumps({{"answer": {{
        "forecasts": [{{"k": "b", "probability": 0.7}},
                      {{"k": "unknown-final", "probability": 0.8}}],
        "rationale": "because", "key_sources": ["https://example.org"]
    }}}}))
    for event in [
        {{"type": "tool_execution_start", "toolCallId": "one", "toolName": "web_search",
          "args": {{"query": "base rate"}}}},
        {{"type": "tool_execution_end", "toolCallId": "one", "toolName": "web_search",
          "result": {{"details": {{"results": [1, 2]}}}}}},
        {{"type": "tool_execution_start", "toolCallId": "two", "toolName": "read_page",
          "args": {{"url": "https://example.org"}}}},
        {{"type": "tool_execution_end", "toolCallId": "two", "toolName": "read_page",
          "result": {{"details": {{"text": "", "error": "page read failed"}}}}}},
        {{"type": "tool_execution_end", "toolName": "submit_forecast", "result": {{}}}},
        {{"type": "message_end", "message": {{"role": "assistant", "usage": {{
            "input": 10, "output": 5, "cacheRead": 2, "cost": {{"total": 0.01}}
        }}}}}},
        {{"type": "message_end", "message": {{"role": "assistant", "usage": {{
            "input": 1, "output": 1, "cost": {{"total": 0.02}}
        }}}}}}
    ]:
        print(json.dumps(event))
    """)


def fake_pi(tmp_path, *, version="0.99.2", mode="success", name="pi"):
    """Write a deterministic executable with a real version-check boundary."""
    path = tmp_path / name
    path.write_text(FAKE_PI.format(python=sys.executable, version=version, mode=mode))
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def run_args(tmp_path):
    """Provide a small forecasting request with known output keys."""
    return {
        "model": "anthropic/test-model",
        "system": "System instructions.",
        "prompt": "Forecast these two cells.",
        "workdir": str(tmp_path / "work"),
        "cell_schema": SCHEMA,
        "key_fields": ["k"],
        "allowed_keys": ["a", "b"],
        "tool_descriptions": DESCRIPTIONS,
    }


def test_merges_allowed_cells_and_collects_evidence_and_usage(tmp_path, run_args):
    """Keep valid pieces, prefer final cells, and join events by tool call ID."""
    result = harness.run(pi=fake_pi(tmp_path), **run_args)

    assert {key: value["probability"] for key, value in result["cells"].items()} == {
        "a": 0.1,
        "b": 0.7,
    }
    assert result["answer"]["rationale"] == "because"
    assert result["evidence"] == [
        {
            "tool": "web_search",
            "args": {"query": "base rate"},
            "result": {"results": [1, 2]},
            "error": None,
        },
        {
            "tool": "read_page",
            "args": {"url": "https://example.org"},
            "result": {"text": "", "error": "page read failed"},
            "error": "page read failed",
        },
    ]
    usage = result["usage"]
    assert [usage[key] for key in ("turns", "input_tokens", "output_tokens", "cached_tokens")] == [
        2,
        11,
        6,
        2,
    ]
    assert usage["cost_usd"] == pytest.approx(0.03)
    assert (usage["research_ok"], usage["research_failed"]) == (1, 1)
    attempt = tmp_path / "work" / "attempt-1"
    assert (attempt / "stdin.txt").read_text() == run_args["prompt"]
    assert (attempt / "seen-system.txt").read_text() == run_args["system"]


def test_failed_process_cannot_reuse_previous_answers(tmp_path, run_args):
    """Use a fresh attempt directory even after an earlier successful call."""
    harness.run(pi=fake_pi(tmp_path), **run_args)
    result = harness.run(pi=fake_pi(tmp_path, mode="broken"), **run_args)
    assert result["rc"] == 1
    assert result["workdir"].endswith("attempt-2")
    assert result["answer"] == {}
    assert result["cells"] == {}


def test_provider_error_is_reported_and_retried(tmp_path, run_args, monkeypatch):
    """Retry provider errors even when pi exits with a successful status."""
    monkeypatch.setattr(harness, "RETRY_BACKOFF", 0)
    result = harness.run(pi=fake_pi(tmp_path, mode="provider-error"), retries=2, **run_args)
    assert result["rc"] == 0
    assert result["usage"]["api_error"] == "rate limited"
    assert result["cells"] == {}
    assert sorted(path.name for path in (tmp_path / "work").iterdir()) == [
        "attempt-1",
        "attempt-2",
        "attempt-3",
    ]


def test_runtime_selection_is_explicit_then_call_env_then_process(tmp_path, run_args, monkeypatch):
    """Resolve the runtime at call time and honor per-call overrides."""
    process_pi = fake_pi(tmp_path, name="process-pi")
    call_pi = fake_pi(tmp_path, name="call-pi")
    explicit_pi = fake_pi(tmp_path, name="explicit-pi")
    monkeypatch.setenv("FORECASTER_PI", process_pi)
    assert harness.resolve_pi() == process_pi
    assert harness.resolve_pi(env={"FORECASTER_PI": call_pi}) == call_pi
    assert harness.resolve_pi(explicit_pi, env={"FORECASTER_PI": call_pi}) == explicit_pi
    monkeypatch.setenv("FORECASTER_PI", call_pi)
    assert harness.resolve_pi() == call_pi
    monkeypatch.delenv("FORECASTER_PI")
    with pytest.raises(RuntimeError, match="FORECASTER_PI"):
        harness.run(**run_args)
    assert not (tmp_path / "work").exists()


def test_wrong_runtime_version_stops_before_forecasting(tmp_path, run_args):
    """Reject an incompatible pi before a model process receives the request."""
    with pytest.raises(RuntimeError, match="0.99.2"):
        harness.run(pi=fake_pi(tmp_path, version="0.98.0"), **run_args)
    assert not (tmp_path / "work").exists()


def test_environment_overrides_are_private_and_oauth_tokens_are_removed(
    tmp_path, run_args, monkeypatch
):
    """Pass call-specific keys and URL filters without changing process state."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-bearer")
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "test-oauth")
    monkeypatch.setenv("FORECASTER_BLOCK_URL_PATTERNS", "original")
    before = dict(os.environ)
    result = harness.run(
        pi=fake_pi(tmp_path),
        env={"ANTHROPIC_API_KEY": "call-key", "FORECASTER_BLOCK_URL_PATTERNS": "blocked"},
        **run_args,
    )
    assert dict(os.environ) == before
    seen = json.loads((tmp_path / "work" / "attempt-1" / "env.json").read_text())
    assert seen["ANTHROPIC_API_KEY"] == "call-key"
    assert seen["FORECASTER_BLOCK_URL_PATTERNS"] == "blocked"
    assert seen["ANTHROPIC_AUTH_TOKEN"] is None
    assert seen["ANTHROPIC_OAUTH_TOKEN"] is None
    assert result["rc"] == 0


def test_missing_key_and_saved_oauth_login_stop_before_forecasting(tmp_path, run_args):
    """Require API billing even when an agent directory contains a login."""
    pi = fake_pi(tmp_path)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        harness.run(pi=pi, env={"ANTHROPIC_API_KEY": ""}, **run_args)
    agent = tmp_path / "logged-in-agent"
    agent.mkdir()
    (agent / "auth.json").write_text(
        json.dumps({"anthropic": {"type": "oauth", "access": "test-token"}})
    )
    with pytest.raises(RuntimeError, match="login"):
        harness.run(pi=pi, env={"PI_CODING_AGENT_DIR": str(agent)}, **run_args)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("agent_value", ["agent", "~/agent"])
def test_agent_directory_checked_and_used_is_same_under_other_cwd(
    agent_value, tmp_path, run_args, monkeypatch
):
    """Resolve the billing directory before checking it and changing the child cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    expected = tmp_path / "home" / "agent" if agent_value.startswith("~") else tmp_path / "agent"
    child_cwd = tmp_path / "elsewhere"
    child_cwd.mkdir()
    pi = fake_pi(tmp_path)
    call_options = {
        "pi": pi,
        "cwd": str(child_cwd),
        "env": {"PI_CODING_AGENT_DIR": agent_value},
        **run_args,
    }
    result = harness.run(**call_options)
    assert result["rc"] == 0
    seen = json.loads((tmp_path / "work" / "attempt-1" / "env.json").read_text())
    assert seen["PI_CODING_AGENT_DIR"] == str(expected)
    (expected / "auth.json").write_text(
        json.dumps({"anthropic": {"type": "oauth", "access": "test-token"}})
    )
    with pytest.raises(RuntimeError, match="login"):
        harness.run(**call_options)
    assert sorted(path.name for path in (tmp_path / "work").iterdir()) == ["attempt-1"]
