"""Keep agentic tests independent of local credentials and pi state."""

import pytest


@pytest.fixture(autouse=True)
def isolated_agentic_environment(monkeypatch, tmp_path):
    """Use dummy credentials and a fresh agent directory for every test."""
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPINFRA_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")
    for name in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_OAUTH_TOKEN",
        "FORECASTER_BLOCK_URL_PATTERNS",
        "FORECASTER_ENV_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "agent"))
    monkeypatch.setenv("PI_OFFLINE", "1")
    monkeypatch.setenv("PI_TELEMETRY", "0")
