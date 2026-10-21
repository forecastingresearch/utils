"""Environment key names used by the pi providers supported by this harness."""

# Credentials are inherited from the caller, with optional per-call overrides.
# The application may explicitly load a .env file before calling the harness.
KEY_FOR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}
