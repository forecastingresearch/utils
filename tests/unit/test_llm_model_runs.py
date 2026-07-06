"""Unit tests for shared LLM model-run declarations."""

import ast
import inspect
import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.llm.lab_registry import LABS
from utils.llm.provider_registry import PROVIDERS

FILENAME_SAFE_NAME_PATTERN = re.compile(r"^k-[a-z0-9~-]+$")
AA_MODEL_RUN_KEY_SUFFIX_PATTERN = re.compile(r"^aa-run-variant-[0-9]{2}$")
MODEL_RUN_KEY_SUFFIX_PATTERN = re.compile(r"^run-variant-[0-9]{2}$")
ROUTE_OWNED_PROVIDER_OPTION_KEYS = frozenset({"contents", "input", "messages", "model"})
OUTPUT_TOKEN_LIMIT_OPTION_KEYS = (
    "max_tokens",
    "max_output_tokens",
    "maxOutputTokens",
)

# This ledger intentionally duplicates the registry declarations so every new
# model run requires an explicit historical-key test update.
HISTORICAL_MODEL_RUN_KEYS = (
    "claude-2.1-run-variant-01",
    "claude-3-5-sonnet-20240620-run-variant-01",
    "claude-3-5-sonnet-20241022-run-variant-01",
    "claude-3-7-sonnet-20250219-run-variant-01",
    "claude-3-haiku-20240307-run-variant-01",
    "claude-3-opus-20240229-run-variant-01",
    "claude-haiku-4-5-20251001-run-variant-01",
    "claude-haiku-4-5-20251001-run-variant-02",
    "claude-opus-4-1-20250805-run-variant-01",
    "claude-opus-4-20250514-run-variant-01",
    "claude-opus-4-5-20251101-run-variant-01",
    "claude-opus-4-6-run-variant-01",
    "claude-opus-4-6-run-variant-02",
    "claude-opus-4-7-aa-run-variant-01",
    "claude-opus-4-7-aa-run-variant-02",
    "claude-opus-4-7-run-variant-01",
    "claude-opus-4-7-run-variant-02",
    "claude-opus-4-7-run-variant-03",
    "claude-opus-4-7-run-variant-04",
    "claude-opus-4-8-run-variant-01",
    "claude-opus-4-8-run-variant-02",
    "claude-opus-4-8-run-variant-03",
    "claude-opus-4-8-run-variant-04",
    "claude-sonnet-4-20250514-run-variant-01",
    "claude-sonnet-4-5-20250929-run-variant-01",
    "claude-sonnet-4-5-20250929-run-variant-02",
    "claude-sonnet-4-6-run-variant-01",
    "claude-sonnet-4-6-run-variant-02",
    "claude-sonnet-4-6-run-variant-03",
    "claude-sonnet-5-run-variant-01",
    "deepseek-r1-run-variant-01",
    "deepseek-v3-run-variant-01",
    "deepseek-v3.1-run-variant-01",
    "deepseek-v4-pro-run-variant-01",
    "gemini-1.5-flash-run-variant-01",
    "gemini-1.5-pro-run-variant-01",
    "gemini-2.0-flash-lite-001-run-variant-01",
    "gemini-2.5-flash-preview-04-17-run-variant-01",
    "gemini-2.5-flash-run-variant-01",
    "gemini-2.5-pro-exp-03-25-run-variant-01",
    "gemini-2.5-pro-preview-03-25-run-variant-01",
    "gemini-2.5-pro-run-variant-01",
    "gemini-2.5-pro-run-variant-02",
    "gemini-3-flash-preview-run-variant-01",
    "gemini-3-pro-preview-run-variant-01",
    "gemini-3.1-flash-lite-preview-run-variant-01",
    "gemini-3.1-flash-lite-run-variant-01",
    "gemini-3.1-pro-preview-run-variant-01",
    "gemini-3.1-pro-preview-run-variant-02",
    "gemini-3.1-pro-preview-run-variant-03",
    "gemini-3.5-flash-run-variant-01",
    "gemini-3.5-flash-run-variant-02",
    "gemma-4-31b-it-run-variant-01",
    "gemma-4-31b-it-run-variant-02",
    "glm-4.5-air-fp8-run-variant-01",
    "glm-4.6-run-variant-01",
    "glm-4.7-run-variant-01",
    "glm-5-run-variant-01",
    "glm-5.1-run-variant-01",
    "glm-5.2-run-variant-01",
    "glm-5.2-run-variant-02",
    "gpt-3.5-turbo-0125-run-variant-01",
    "gpt-4-0613-run-variant-01",
    "gpt-4-turbo-2024-04-09-run-variant-01",
    "gpt-4.1-2025-04-14-run-variant-01",
    "gpt-4.5-preview-2025-02-27-run-variant-01",
    "gpt-4o-2024-05-13-run-variant-01",
    "gpt-4o-2024-11-20-run-variant-01",
    "gpt-4o-2024-11-20-run-variant-02",
    "gpt-4o-mini-2024-07-18-run-variant-01",
    "gpt-5-2025-08-07-run-variant-01",
    "gpt-5-mini-2025-08-07-run-variant-01",
    "gpt-5-mini-2025-08-07-run-variant-02",
    "gpt-5-nano-2025-08-07-run-variant-01",
    "gpt-5.1-2025-11-13-run-variant-01",
    "gpt-5.2-2025-12-11-run-variant-01",
    "gpt-5.4-2026-03-05-run-variant-01",
    "gpt-5.4-2026-03-05-run-variant-02",
    "gpt-5.4-2026-03-05-run-variant-03",
    "gpt-5.4-mini-2026-03-17-run-variant-01",
    "gpt-5.4-nano-2026-03-17-run-variant-01",
    "gpt-5.5-2026-04-23-run-variant-01",
    "gpt-5.5-2026-04-23-run-variant-02",
    "gpt-5.5-2026-04-23-run-variant-03",
    "gpt-5.5-2026-04-23-run-variant-04",
    "grok-4-0709-run-variant-01",
    "grok-4-1-fast-non-reasoning-run-variant-01",
    "grok-4-1-fast-reasoning-run-variant-01",
    "grok-4-fast-non-reasoning-run-variant-01",
    "grok-4-fast-reasoning-run-variant-01",
    "grok-4.20-0309-non-reasoning-run-variant-01",
    "grok-4.20-0309-reasoning-run-variant-01",
    "grok-4.20-0309-reasoning-run-variant-02",
    "grok-4.20-beta-0309-non-reasoning-run-variant-01",
    "grok-4.20-beta-0309-reasoning-run-variant-01",
    "grok-4.3-run-variant-01",
    "grok-4.3-run-variant-02",
    "grok-beta-run-variant-01",
    "kimi-k2-instruct-0905-run-variant-01",
    "kimi-k2-instruct-run-variant-01",
    "kimi-k2-thinking-run-variant-01",
    "kimi-k2.5-moonshot-ai-run-variant-01",
    "kimi-k2.5-moonshot-ai-run-variant-02",
    "kimi-k2.5-run-variant-01",
    "kimi-k2.6-moonshot-ai-run-variant-01",
    "kimi-k2.6-moonshot-ai-run-variant-02",
    "kimi-k2.6-run-variant-01",
    "kimi-k2.6-run-variant-02",
    "llama-2-70b-chat-hf-run-variant-01",
    "llama-3-70b-chat-hf-run-variant-01",
    "llama-3-8b-chat-hf-run-variant-01",
    "llama-3.2-3b-instruct-turbo-run-variant-01",
    "llama-3.3-70b-instruct-turbo-run-variant-01",
    "llama-4-maverick-17b-128e-instruct-fp8-run-variant-01",
    "llama-4-scout-17b-16e-instruct-run-variant-01",
    "magistral-medium-2506-run-variant-01",
    "meta-llama-3.1-405b-instruct-turbo-run-variant-01",
    "minimax-m2.5-run-variant-01",
    "minimax-m2.7-run-variant-01",
    "minimax-m2.7-run-variant-02",
    "minimax-m3-run-variant-01",
    "mistral-large-2407-run-variant-01",
    "mistral-large-2411-run-variant-01",
    "mistral-large-latest-run-variant-01",
    "mixtral-8x22b-instruct-v0.1-run-variant-01",
    "mixtral-8x7b-instruct-v0.1-run-variant-01",
    "o3-2025-04-16-run-variant-01",
    "o3-mini-2025-01-31-run-variant-01",
    "o4-mini-2025-04-16-run-variant-01",
    "qwen1.5-110b-chat-run-variant-01",
    "qwen2.5-72b-instruct-turbo-run-variant-01",
    "qwen3-235b-a22b-fp8-tput-run-variant-01",
    "qwen3-235b-a22b-thinking-2507-run-variant-01",
    "qwq-32b-preview-run-variant-01",
)


def test_model_keys_are_unique_and_file_safe():
    """Keep base model keys unique and safe for downstream identifiers."""
    from utils.llm import model_registry

    model_keys = [model.model_key for model in model_registry.MODELS]
    filename_safe_names = [model.filename_safe_name for model in model_registry.MODELS]

    assert len(model_keys) == len(set(model_keys))
    assert len(filename_safe_names) == len(set(filename_safe_names))
    assert all(FILENAME_SAFE_NAME_PATTERN.fullmatch(name) for name in filename_safe_names)


def test_model_key_and_provider_model_id_are_distinct_for_together_models():
    """Keep canonical model keys separate from routed provider model IDs."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["deepseek-v3.1"]

    assert model.model_key == "deepseek-v3.1"
    assert model.provider_model_id == "deepseek-ai/DeepSeek-V3.1"
    assert model.lab == LABS["DeepSeek"]
    assert model.provider == PROVIDERS["Together"]
    assert model.release_date == date(2025, 8, 21)
    assert model.active is False


def test_forecastbench_origin_main_models_are_in_canonical_registry():
    """Include recent ForecastBench origin/main models in the shared registry."""
    from utils.llm import model_registry

    expected_models = {
        "deepseek-v4-pro": (
            "deepseek-ai/DeepSeek-V4-Pro",
            LABS["DeepSeek"],
            PROVIDERS["Together"],
            date(2026, 4, 24),
        ),
        "gemini-3.5-flash": (
            "gemini-3.5-flash",
            LABS["Google DeepMind"],
            PROVIDERS["Google"],
            date(2026, 5, 19),
        ),
        "glm-5.2": (
            "zai-org/GLM-5.2",
            LABS["Z.ai"],
            PROVIDERS["Together"],
            date(2026, 6, 13),
        ),
        "minimax-m3": (
            "MiniMaxAI/MiniMax-M3",
            LABS["MiniMax"],
            PROVIDERS["Together"],
            date(2026, 6, 1),
        ),
    }

    for model_key, (
        provider_model_id,
        lab,
        provider,
        release_date,
    ) in expected_models.items():
        model = model_registry.MODELS_BY_KEY[model_key]
        assert model.provider_model_id == provider_model_id
        assert model.lab == lab
        assert model.provider == provider
        assert model.release_date == release_date

    minimax_m3 = model_registry.MODELS_BY_KEY["minimax-m3"]
    assert minimax_m3.models_dev_reference == model_registry.ModelsDevReference(
        provider_id="minimax",
        model_id="MiniMax-M3",
    )
    assert minimax_m3.models_dev_metadata is not None


def test_model_release_date_resolves_from_models_dev_metadata():
    """Resolve model release dates from configured Models.dev metadata."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["gpt-4o-2024-11-20"]

    assert model.models_dev_reference == model_registry.ModelsDevReference(
        provider_id="openai",
        model_id="gpt-4o-2024-11-20",
    )
    assert model.models_dev_metadata is not None
    assert model.release_date == model.models_dev_metadata.release_date
    assert model.release_date == date(2024, 11, 20)


def test_claude_sonnet_5_uses_models_dev_metadata_and_supported_options():
    """Keep Claude Sonnet 5 linked to Models.dev and avoid unsupported temperature options."""
    from utils.llm import model_registry, model_runs

    model = model_registry.MODELS_BY_KEY["claude-sonnet-5"]
    run = model_runs.MODEL_RUNS_BY_KEY["claude-sonnet-5-run-variant-01"]

    assert model.provider_model_id == "claude-sonnet-5"
    assert model.lab == LABS["Anthropic"]
    assert model.provider == PROVIDERS["Anthropic"]
    assert model.models_dev_reference == model_registry.ModelsDevReference(
        provider_id="anthropic",
        model_id="claude-sonnet-5",
    )
    assert model.release_date == date(2026, 6, 30)
    assert model.models_dev_metadata.raw["limit"] == {
        "context": 1000000,
        "output": 128000,
    }
    assert model.models_dev_metadata.raw["reasoning"] is True
    assert model.models_dev_metadata.raw["temperature"] is False
    assert model.models_dev_metadata.raw["tool_call"] is True
    assert run.model is model
    assert run.slug == "claude-sonnet-5-adaptive-thinking-16000"
    assert run.options == {
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
    }


def test_model_api_provider_route_is_independent_from_models_dev_provider():
    """Keep API routing separate from the Models.dev metadata provider."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["glm-5.2"]

    assert model.provider == PROVIDERS["Together"]
    assert model.provider_model_id == "zai-org/GLM-5.2"
    assert model.models_dev_reference is None
    assert model.release_date == date(2026, 6, 13)


def test_models_without_models_dev_metadata_use_manual_release_dates():
    """Use manual release dates only when Models.dev metadata is unavailable."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["gpt-4-0613"]

    assert model.models_dev_metadata is None
    assert model.manual_release_date == date(2023, 6, 13)
    assert model.release_date == date(2023, 6, 13)


def test_manual_release_dates_override_models_dev_metadata():
    """Allow explicit corrections when Models.dev metadata is not the desired date."""
    from utils.llm import model_registry

    model = model_registry.openai_model(
        model_key="manual-override-model",
        models_dev_reference=model_registry.ModelsDevReference(
            provider_id="openai",
            model_id="gpt-4o-2024-05-13",
        ),
        manual_release_date=date(2026, 1, 1),
    )

    assert model.models_dev_metadata.release_date == date(2024, 5, 13)
    assert model.manual_release_date == date(2026, 1, 1)
    assert model.release_date == date(2026, 1, 1)


def test_provider_specific_model_helpers_default_lab_provider_and_provider_model_id():
    """Use helper constructors to avoid repeated provider and lab boilerplate."""
    from utils.llm import model_registry

    model = model_registry.openai_model(
        model_key="gpt-test",
        models_dev_reference=model_registry.ModelsDevReference(
            provider_id="openai",
            model_id="gpt-4o-2024-05-13",
        ),
    )

    assert model.model_key == "gpt-test"
    assert model.provider_model_id == "gpt-test"
    assert model.lab == LABS["OpenAI"]
    assert model.provider == PROVIDERS["OpenAI"]
    assert model.models_dev_reference == model_registry.ModelsDevReference(
        provider_id="openai",
        model_id="gpt-4o-2024-05-13",
    )
    assert model.active is True


def test_together_model_helper_keeps_lab_and_route_explicit():
    """Keep Together creator lab and provider route explicit while reducing noise."""
    from utils.llm import model_registry

    model = model_registry.together_model(
        model_key="glm-5.1",
        provider_model_id="zai-org/GLM-5.1",
        lab_key="Z.ai",
        models_dev_reference=model_registry.ModelsDevReference(
            provider_id="zai",
            model_id="glm-5.1",
        ),
    )

    assert model.lab == LABS["Z.ai"]
    assert model.provider == PROVIDERS["Together"]
    assert model.provider_model_id == "zai-org/GLM-5.1"
    assert model.release_date == date(2026, 4, 7)


def test_model_without_models_dev_or_manual_release_date_fails_on_initialization():
    """Fail on construction when a model lacks a release date source."""
    from utils.llm import model_registry

    with pytest.raises(ValueError, match="missing-date-model"):
        model_registry.Model(
            model_key="missing-date-model",
            provider_model_id="missing-date-model",
            lab=LABS["OpenAI"],
            provider=PROVIDERS["OpenAI"],
        )


def test_model_rejects_missing_models_dev_reference_on_initialization():
    """Reject model declarations whose Models.dev reference is not in the snapshot."""
    from utils.llm import model_registry

    with pytest.raises(ValueError, match="bad-reference-model"):
        model_registry.openai_model(
            model_key="bad-reference-model",
            models_dev_reference=model_registry.ModelsDevReference(
                provider_id="openai",
                model_id="missing-model",
            ),
        )


def test_openai_dated_model_uses_dated_provider_model_id():
    """Use fixed dated OpenAI provider model IDs instead of moving aliases."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["gpt-4o-mini-2024-07-18"]

    assert model.provider_model_id == "gpt-4o-mini-2024-07-18"


def test_model_registry_models_are_grouped_by_provider():
    """Build the shared model registry from provider-specific groups."""
    from utils.llm import model_registry

    assert model_registry.MODELS == [
        *model_registry.OPENAI_MODELS,
        *model_registry.TOGETHER_MODELS,
        *model_registry.MOONSHOT_AI_MODELS,
        *model_registry.ANTHROPIC_MODELS,
        *model_registry.XAI_MODELS,
        *model_registry.GOOGLE_MODELS,
    ]
    assert {model.provider.name for model in model_registry.OPENAI_MODELS} == {"OpenAI"}
    assert {model.provider.name for model in model_registry.TOGETHER_MODELS} == {"Together"}
    assert {model.provider.name for model in model_registry.MOONSHOT_AI_MODELS} == {"Moonshot AI"}
    assert {model.provider.name for model in model_registry.ANTHROPIC_MODELS} == {"Anthropic"}
    assert {model.provider.name for model in model_registry.XAI_MODELS} == {"xAI"}
    assert {model.provider.name for model in model_registry.GOOGLE_MODELS} == {"Google"}


def test_model_registry_provider_groups_are_sorted_by_release_date():
    """Keep provider-specific model groups sorted by release date."""
    from utils.llm import model_registry

    provider_groups = [
        model_registry.OPENAI_MODELS,
        model_registry.TOGETHER_MODELS,
        model_registry.MOONSHOT_AI_MODELS,
        model_registry.ANTHROPIC_MODELS,
        model_registry.XAI_MODELS,
        model_registry.GOOGLE_MODELS,
    ]

    for models in provider_groups:
        release_order = [(model.release_date, model.model_key) for model in models]
        assert release_order == sorted(release_order)


def test_create_models_list_rejects_duplicate_model_keys():
    """Reject duplicate model keys when creating the full model registry list."""
    from utils.llm import model_registry

    model = model_registry.MODELS_BY_KEY["gpt-4-0613"]

    with pytest.raises(ValueError, match="Duplicate LLM model_key: gpt-4-0613"):
        model_registry.create_models_list([model, model])


def test_model_filename_safe_names_encode_model_keys_without_collisions():
    """Encode model keys into unique filename-safe names without lossy replacement."""
    from utils.llm import model_registry

    models = [
        model_registry.provider_model(
            model_key=model_key,
            lab_key="OpenAI",
            provider_key="OpenAI",
            manual_release_date=date(2026, 1, 1),
        )
        for model_key in [
            "//*",
            "opus/4.3",
            "opus*4.3",
            "opus_4.3",
            "opus 4.3",
            "Opus-4.3",
            ".opus-4.3",
            "opus-4.3.",
            "opus~2f4.3",
        ]
    ]
    filename_safe_names = [model.filename_safe_name for model in models]

    assert len(filename_safe_names) == len(set(filename_safe_names))
    assert all(FILENAME_SAFE_NAME_PATTERN.fullmatch(name) for name in filename_safe_names)
    assert models[1].filename_safe_name != models[2].filename_safe_name
    assert models[1].filename_safe_name == "k-opus~2f4~2e3"
    assert models[2].filename_safe_name == "k-opus~2a4~2e3"


def test_model_runs_match_canonical_model_registry_entries():
    """Keep shared model runs aligned with canonical model registry entries."""
    from utils.llm import model_registry, model_runs

    run = model_runs.MODEL_RUNS_BY_SLUG["deepseek-v3.1"]

    assert run.model == model_registry.MODELS_BY_KEY["deepseek-v3.1"]


def test_o3_mini_model_run_is_not_artificial_analysis_backed():
    """Keep o3-mini selectable without treating it as an AA-backed run."""
    from utils.llm import model_runs

    run = model_runs.MODEL_RUNS_BY_SLUG["o3-mini-2025-01-31"]

    assert run.model_run_key.startswith("o3-mini-2025-01-31-run-variant-")
    assert MODEL_RUN_KEY_SUFFIX_PATTERN.fullmatch(
        run.model_run_key.removeprefix(f"{run.model_key}-")
    )
    assert run.slug == "o3-mini-2025-01-31"
    assert run.artificial_analysis_id is None
    assert run.display_name == run.model_key


def test_model_run_constructor_requires_explicit_model_run_key():
    """Do not allow ModelRun keys to be generated implicitly."""
    from utils.llm import model_registry, model_runs

    with pytest.raises(TypeError):
        model_runs.ModelRun(
            model=model_registry.MODELS_BY_KEY["gpt-5.5-2026-04-23"],
        )


def test_model_run_helper_rejects_unknown_model_key_with_value_error():
    """Report declaration typos as registry construction errors."""
    from utils.llm import model_runs

    with pytest.raises(ValueError, match="Unknown model_key typo-model"):
        model_runs._model_run(
            model_run_key="typo-model-run-variant-01",
            slug="typo-model",
            model_key="typo-model",
        )


def test_model_run_key_prefix_error_names_expected_model_key_without_delimiter():
    """Report the expected model key without folding in the key delimiter."""
    from utils.llm import model_registry, model_runs

    with pytest.raises(ValueError) as excinfo:
        model_runs.ModelRun(
            model_run_key="gpt-5.5-run-variant-01",
            slug="gpt-5.5",
            model=model_registry.MODELS_BY_KEY["gpt-5.5-2026-04-23"],
        )

    message = str(excinfo.value)
    assert "Expected model_key: gpt-5.5-2026-04-23" in message
    assert "Expected prefix: gpt-5.5-2026-04-23-" not in message


def test_model_run_declarations_use_literal_model_run_keys():
    """Keep shared model-run keys and slugs handwritten at declaration sites."""
    from utils.llm import model_runs

    source = Path(model_runs.__file__).read_text()
    tree = ast.parse(source)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_model_run"
    ]

    assert calls
    for call in calls:
        keyword = next(
            (keyword for keyword in call.keywords if keyword.arg == "model_run_key"),
            None,
        )
        slug_keyword = next((keyword for keyword in call.keywords if keyword.arg == "slug"), None)
        assert keyword is not None
        assert isinstance(keyword.value, ast.Constant)
        assert isinstance(keyword.value.value, str)
        assert slug_keyword is not None
        assert isinstance(slug_keyword.value, ast.Constant)
        assert isinstance(slug_keyword.value.value, str)


def test_artificial_analysis_model_runs_are_declared_in_dedicated_module():
    """Keep AA-backed model runs separate from the main registry list."""
    from utils.llm import artificial_analysis_model_runs, model_runs

    declaration_keys = [
        declaration["model_run_key"]
        for declaration in artificial_analysis_model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS
    ]
    declaration_slugs = [
        declaration["slug"]
        for declaration in artificial_analysis_model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS
    ]
    registry_keys = [run.model_run_key for run in model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUNS]
    registry_slugs = [run.slug for run in model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUNS]
    aa_keys_in_model_runs = [
        run.model_run_key for run in model_runs.MODEL_RUNS if run.artificial_analysis_id is not None
    ]
    aa_slugs_in_model_runs = [
        run.slug for run in model_runs.MODEL_RUNS if run.artificial_analysis_id is not None
    ]

    assert registry_keys == declaration_keys
    assert registry_slugs == declaration_slugs
    assert aa_keys_in_model_runs == declaration_keys
    assert aa_slugs_in_model_runs == declaration_slugs
    assert "o3-mini-2025-01-31" not in declaration_slugs
    assert all(isinstance(key, str) for key in declaration_keys)
    assert all(isinstance(slug, str) for slug in declaration_slugs)
    assert all(
        key.startswith(f"{declaration['model_key']}-aa-run-variant-")
        for key, declaration in zip(
            declaration_keys,
            artificial_analysis_model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS,
        )
    )
    assert all(
        declaration["artificial_analysis_id"]
        for declaration in artificial_analysis_model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUN_DECLARATIONS
    )


def test_artificial_analysis_opus_runs_use_model_display_names_and_token_caps():
    """Keep AA metadata separate from display names and runtime options."""
    from utils.llm import model_runs

    non_reasoning = model_runs.MODEL_RUNS_BY_SLUG["claude-opus-4-7-high-16384"]
    adaptive = model_runs.MODEL_RUNS_BY_SLUG["claude-opus-4-7-adaptive-thinking-max-128000"]

    assert non_reasoning.artificial_analysis_id == "2fa8e143-77a8-4d05-bfa8-d3b54634c00f"
    assert non_reasoning.display_name == non_reasoning.model_key
    assert non_reasoning.options == {
        "max_tokens": 16384,
        "output_config": {"effort": "high"},
    }
    assert adaptive.artificial_analysis_id == "e9a09db3-8fd6-41dd-ba2f-20e0a2bff7f2"
    assert adaptive.display_name == adaptive.model_key
    assert adaptive.options == {
        "max_tokens": 128000,
        "output_config": {"effort": "max"},
        "thinking": {"type": "adaptive"},
    }


def test_artificial_analysis_non_reasoning_runs_default_to_16384_max_tokens():
    """Keep AA-backed non-reasoning runs at the documented default token cap."""
    from utils.llm import model_runs

    non_reasoning_runs = [
        run for run in model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUNS if "thinking" not in run.options
    ]
    invalid_caps = {
        run.model_run_key: run.options.get("max_tokens")
        for run in non_reasoning_runs
        if run.options.get("max_tokens") != 16384
    }

    assert non_reasoning_runs
    assert invalid_caps == {}


def test_anthropic_1024_token_runs_use_temperature_only_when_models_dev_supports_it():
    """Keep deterministic sampling on supported 1024-token Anthropic runs."""
    from utils.llm import model_runs

    matching_runs = [
        run
        for run in model_runs.MODEL_RUNS
        if run.provider == PROVIDERS["Anthropic"]
        and run.options.get("max_tokens") == 1024
        and (
            run.model.models_dev_metadata is None
            or run.model.models_dev_metadata.raw.get("temperature") is True
        )
    ]
    unsupported_temperature_runs = [
        run
        for run in model_runs.MODEL_RUNS
        if run.provider == PROVIDERS["Anthropic"]
        and run.options.get("max_tokens") == 1024
        and run.model.models_dev_metadata is not None
        and run.model.models_dev_metadata.raw.get("temperature") is not True
    ]
    invalid_temperatures = {
        run.slug: run.options.get("temperature")
        for run in matching_runs
        if run.options.get("temperature") != 0
    }
    invalid_unsupported_temperatures = {
        run.slug: run.options.get("temperature")
        for run in unsupported_temperature_runs
        if "temperature" in run.options
    }

    assert matching_runs
    assert unsupported_temperature_runs
    assert invalid_temperatures == {}
    assert invalid_unsupported_temperatures == {}


def test_minimax_variants_declare_provider_controls_on_existing_run_keys():
    """Keep MiniMax provider controls on the intended stable run keys."""
    from utils.llm import model_runs

    m2_7_original = model_runs.MODEL_RUNS_BY_SLUG["minimax-m2.7"]
    m2_7_capped = model_runs.MODEL_RUNS_BY_SLUG["minimax-m2.7-12000"]
    m3_adaptive = model_runs.MODEL_RUNS_BY_SLUG["minimax-m3-adaptive-thinking-12000"]

    assert m2_7_original.model_run_key == "minimax-m2.7-run-variant-01"
    assert m2_7_original.options == {"temperature": 0}
    assert m2_7_capped.model_run_key == "minimax-m2.7-run-variant-02"
    assert m2_7_capped.options == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 12000,
    }
    assert m3_adaptive.model_run_key == "minimax-m3-run-variant-01"
    assert m3_adaptive.options == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "chat_template_kwargs": {"thinking_mode": "adaptive"},
        "max_tokens": 12000,
    }


def test_glm_5_2_declares_capped_variant():
    """Keep GLM 5.2 output cap on a separate stable run key."""
    from utils.llm import model_runs

    original = model_runs.MODEL_RUNS_BY_KEY["glm-5.2-run-variant-01"]
    capped = model_runs.MODEL_RUNS_BY_KEY["glm-5.2-run-variant-02"]

    assert original.slug == "glm-5.2"
    assert original.options == {"temperature": 0}
    assert capped.slug == "glm-5.2-12000"
    assert capped.model_key == "glm-5.2"
    assert capped.options == {
        "temperature": 0,
        "max_tokens": 12000,
    }


def test_gemini_3_5_flash_web_search_run_uses_high_thinking():
    """Keep Gemini 3.5 Flash web-search options on the intended stable run key."""
    from utils.llm import model_runs

    run = model_runs.MODEL_RUNS_BY_KEY["gemini-3.5-flash-run-variant-02"]

    assert run.slug == "gemini-3.5-flash-high-web-search"
    assert run.options == {
        "thinking_config": {"thinking_level": "high"},
        "tools": [{"googleSearch": {}}],
    }


def test_gemini_3_1_pro_preview_declares_high_thinking_variant():
    """Keep Gemini 3.1 Pro Preview high thinking on a separate stable run key."""
    from utils.llm import model_runs

    original = model_runs.MODEL_RUNS_BY_KEY["gemini-3.1-pro-preview-run-variant-01"]
    high_thinking = model_runs.MODEL_RUNS_BY_KEY["gemini-3.1-pro-preview-run-variant-02"]
    high_web_search = model_runs.MODEL_RUNS_BY_KEY["gemini-3.1-pro-preview-run-variant-03"]

    assert original.slug == "gemini-3.1-pro-preview"
    assert original.options == {
        "candidate_count": 1,
        "temperature": 0,
        "automatic_function_calling": {"disable": True},
    }
    assert high_thinking.slug == "gemini-3.1-pro-preview-high"
    assert high_thinking.model_key == "gemini-3.1-pro-preview"
    assert high_thinking.options == {
        "thinking_config": {"thinking_level": "high"},
    }
    assert high_web_search.slug == "gemini-3.1-pro-preview-high-web-search"
    assert high_web_search.model_key == "gemini-3.1-pro-preview"
    assert high_web_search.options == {
        "thinking_config": {"thinking_level": "high"},
        "tools": [{"googleSearch": {}}],
    }


def test_kimi_k2_5_and_k2_6_are_callable_via_together_and_moonshot_ai():
    """Kimi K2.5/K2.6 keep their historical Together route and gain a direct Moonshot route."""
    from utils.llm import model_registry

    together_k25 = model_registry.MODELS_BY_KEY["kimi-k2.5"]
    together_k26 = model_registry.MODELS_BY_KEY["kimi-k2.6"]
    moonshot_k25 = model_registry.MODELS_BY_KEY["kimi-k2.5-moonshot-ai"]
    moonshot_k26 = model_registry.MODELS_BY_KEY["kimi-k2.6-moonshot-ai"]

    # Both routes are the same Moonshot-lab model reached through different APIs.
    for model in (together_k25, together_k26, moonshot_k25, moonshot_k26):
        assert model.lab == LABS["Moonshot"]

    # Historical Together route sends Together's routed model IDs.
    assert together_k25.provider == PROVIDERS["Together"]
    assert together_k25.provider_model_id == "moonshotai/Kimi-K2.5"
    assert together_k26.provider == PROVIDERS["Together"]
    assert together_k26.provider_model_id == "moonshotai/Kimi-K2.6"

    # Direct Moonshot route sends Moonshot's own model IDs, not the suffixed keys.
    assert moonshot_k25.provider == PROVIDERS["Moonshot AI"]
    assert moonshot_k25.provider_model_id == "kimi-k2.5"
    assert moonshot_k26.provider == PROVIDERS["Moonshot AI"]
    assert moonshot_k26.provider_model_id == "kimi-k2.6"


def test_moonshot_ai_kimi_runs_enable_thinking_without_temperature():
    """Direct Moonshot Kimi runs enable thinking, omit temperature (API rejects it), cap at 131072."""
    from utils.llm import model_runs

    k25_default = model_runs.MODEL_RUNS_BY_SLUG["kimi-k2.5-moonshot-ai-thinking"]
    k25_capped = model_runs.MODEL_RUNS_BY_SLUG["kimi-k2.5-moonshot-ai-thinking-128k"]
    k26_default = model_runs.MODEL_RUNS_BY_SLUG["kimi-k2.6-moonshot-ai-thinking"]
    k26_capped = model_runs.MODEL_RUNS_BY_SLUG["kimi-k2.6-moonshot-ai-thinking-128k"]

    thinking = {"thinking": {"type": "enabled"}}
    for run in (k25_default, k25_capped, k26_default, k26_capped):
        assert run.provider == PROVIDERS["Moonshot AI"]
        assert "temperature" not in run.options
        assert run.options["extra_body"] == thinking

    assert k25_default.model_run_key == "kimi-k2.5-moonshot-ai-run-variant-01"
    assert k25_default.options == {"extra_body": thinking}
    assert k25_capped.model_run_key == "kimi-k2.5-moonshot-ai-run-variant-02"
    assert k25_capped.options == {"extra_body": thinking, "max_tokens": 131072}
    assert k26_default.model_run_key == "kimi-k2.6-moonshot-ai-run-variant-01"
    assert k26_default.options == {"extra_body": thinking}
    assert k26_capped.model_run_key == "kimi-k2.6-moonshot-ai-run-variant-02"
    assert k26_capped.options == {"extra_body": thinking, "max_tokens": 131072}


# TODO: Add all-run coverage for AA-backed reasoning max-token caps once
# expected caps are recorded per reasoning configuration.
def test_artificial_analysis_model_runs_require_snapshot_ids():
    """Reject AA-backed model runs that reference missing snapshot IDs."""
    from utils.llm import model_registry, model_runs

    with pytest.raises(ValueError, match="Artificial Analysis"):
        model_runs.ModelRun(
            model_run_key="o3-mini-2025-01-31-aa-run-variant-01",
            slug="o3-mini-2025-01-31",
            model=model_registry.MODELS_BY_KEY["o3-mini-2025-01-31"],
            artificial_analysis_id="missing-aa-model",
        )


def test_model_run_exposes_stable_key_and_mutable_slug():
    """Use stable keys for identity and descriptive slugs for human lookup."""
    from utils.llm import model_registry, model_runs

    run = model_runs.ModelRun(
        model_run_key="gemini-3.1-pro-preview-run-variant-01",
        slug="gemini-3.1-pro-preview",
        model=model_registry.MODELS_BY_KEY["gemini-3.1-pro-preview"],
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    )

    assert run.model_run_key == "gemini-3.1-pro-preview-run-variant-01"
    assert run.slug == "gemini-3.1-pro-preview"
    assert run.filename_safe_name == "k-gemini-3~2e1-pro-preview-run-variant-01"


def test_model_run_filename_safe_names_encode_keys_without_collisions():
    """Encode model-run keys into unique filename-safe names without lossy replacement."""
    from utils.llm import model_registry, model_runs

    runs = [
        model_runs.ModelRun(
            model_run_key=f"{model.model_key}-run-variant-01",
            slug=f"{model.model_key}-slug",
            model=model,
        )
        for model in [
            model_registry.provider_model(
                model_key=model_key,
                lab_key="OpenAI",
                provider_key="OpenAI",
                manual_release_date=date(2026, 1, 1),
            )
            for model_key in [
                "//*",
                "opus/4.3",
                "opus*4.3",
                "opus_4.3",
                "opus 4.3",
                "Opus-4.3",
                ".opus-4.3",
                "opus-4.3.",
                "opus~2f4.3",
            ]
        ]
    ]
    filename_safe_names = [run.filename_safe_name for run in runs]

    assert len(filename_safe_names) == len(set(filename_safe_names))
    assert all(FILENAME_SAFE_NAME_PATTERN.fullmatch(name) for name in filename_safe_names)
    assert runs[1].filename_safe_name != runs[2].filename_safe_name
    assert runs[1].filename_safe_name == "k-opus~2f4~2e3-run-variant-01"
    assert runs[2].filename_safe_name == "k-opus~2a4~2e3-run-variant-01"


def test_model_run_routes_provider_model_id_to_get_response():
    """Route model-run calls through the provider model ID and merged options."""
    from utils.llm import model_registry, model_runs

    run = model_runs.ModelRun(
        model_run_key="deepseek-v3.1-run-variant-01",
        slug="deepseek-v3.1",
        model=model_registry.MODELS_BY_KEY["deepseek-v3.1"],
        options={"temperature": 0},
    )

    with patch("utils.llm.model_registry.get_response", return_value="forecast") as get_response:
        response = run.get_response("prompt", max_tokens=10000)

    assert response == "forecast"
    get_response.assert_called_once_with(
        provider=PROVIDERS["Together"],
        model_id="deepseek-ai/DeepSeek-V3.1",
        prompt="prompt",
        options={"temperature": 0, "max_tokens": 10000},
    )


def test_model_run_get_response_isolates_nested_options_from_provider_mutation():
    """Provider handling should not mutate registry-owned nested option values."""
    from utils.llm import model_registry, model_runs

    run = model_runs.ModelRun(
        model_run_key="gpt-5.5-2026-04-23-run-variant-02",
        slug="gpt-5.5-2026-04-23-high",
        model=model_registry.MODELS_BY_KEY["gpt-5.5-2026-04-23"],
        options={
            "reasoning": {"effort": "high"},
            "tools": [{"type": "web_search"}],
        },
    )

    def mutate_options(*, provider, model_id, prompt, options):
        del provider, model_id, prompt
        options["reasoning"]["effort"] = "low"
        options["tools"].append({"type": "x_search"})
        return "forecast"

    with patch("utils.llm.model_registry.get_response", side_effect=mutate_options):
        response = run.get_response("prompt", max_output_tokens=10000)

    assert response == "forecast"
    assert run.options == {
        "reasoning": {"effort": "high"},
        "tools": [{"type": "web_search"}],
    }


def test_shared_model_run_keys_and_slugs_are_unique_and_file_safe():
    """Keep shared model-run keys and slugs unique and safe for filenames."""
    from utils.llm import model_runs

    keys = [run.model_run_key for run in model_runs.MODEL_RUNS]
    slugs = [run.slug for run in model_runs.MODEL_RUNS]
    filename_safe_names = [run.filename_safe_name for run in model_runs.MODEL_RUNS]

    assert len(keys) == len(set(keys))
    assert len(slugs) == len(set(slugs))
    assert all(run.model_run_key.startswith(f"{run.model_key}-") for run in model_runs.MODEL_RUNS)
    assert all(
        (
            AA_MODEL_RUN_KEY_SUFFIX_PATTERN.fullmatch(
                run.model_run_key.removeprefix(f"{run.model_key}-")
            )
            if run.artificial_analysis_id is not None
            else MODEL_RUN_KEY_SUFFIX_PATTERN.fullmatch(
                run.model_run_key.removeprefix(f"{run.model_key}-")
            )
        )
        for run in model_runs.MODEL_RUNS
    )
    assert len(filename_safe_names) == len(set(filename_safe_names))
    assert all(FILENAME_SAFE_NAME_PATTERN.fullmatch(name) for name in filename_safe_names)


def test_explicit_model_run_groups_are_grouped_by_provider_and_sorted_by_key():
    """Keep explicit provider model-run groups alphabetized by stable key."""
    from utils.llm import model_runs

    provider_groups = (
        ("ANTHROPIC_MODEL_RUNS", PROVIDERS["Anthropic"]),
        ("GOOGLE_MODEL_RUNS", PROVIDERS["Google"]),
        ("MOONSHOT_AI_MODEL_RUNS", PROVIDERS["Moonshot AI"]),
        ("OAI_MODEL_RUNS", PROVIDERS["OpenAI"]),
        ("TOGETHER_MODEL_RUNS", PROVIDERS["Together"]),
        ("XAI_MODEL_RUNS", PROVIDERS["xAI"]),
    )
    grouped_runs = []
    for group_name, provider in provider_groups:
        runs = getattr(model_runs, group_name)
        keys = [run.model_run_key for run in runs]

        assert runs
        assert all(run.provider == provider for run in runs)
        assert keys == sorted(keys)
        grouped_runs.extend(runs)

    assert not any(run.artificial_analysis_id is not None for run in grouped_runs)
    assert model_runs.MODEL_RUNS == [
        *grouped_runs,
        *model_runs.ARTIFICIAL_ANALYSIS_MODEL_RUNS,
    ]


def test_model_run_keys_are_recorded_in_historical_key_ledger():
    """Require every model-run key to be listed in the historical key ledger."""
    from utils.llm import model_runs

    declared_keys = {run.model_run_key for run in model_runs.MODEL_RUNS}

    assert list(HISTORICAL_MODEL_RUN_KEYS) == sorted(HISTORICAL_MODEL_RUN_KEYS)
    assert len(HISTORICAL_MODEL_RUN_KEYS) == len(set(HISTORICAL_MODEL_RUN_KEYS))
    assert declared_keys == set(HISTORICAL_MODEL_RUN_KEYS)


def _keyword_parameter_names(callable_obj):
    return frozenset(
        name
        for name, parameter in inspect.signature(callable_obj).parameters.items()
        if parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def _google_generate_content_config_names() -> frozenset[str]:
    from google.genai import types

    names = set()
    for field_name, field in types.GenerateContentConfig.model_fields.items():
        names.add(field_name)
        if field.alias:
            names.add(field.alias)
    return frozenset(names)


def _declared_option_names_by_provider() -> dict[str, frozenset[str]]:
    from anthropic import Anthropic
    from openai import OpenAI
    from together import Together

    openai_names = _keyword_parameter_names(OpenAI(api_key="test").responses.create)
    moonshot_ai_names = _keyword_parameter_names(OpenAI(api_key="test").chat.completions.create)
    return {
        "Anthropic": _keyword_parameter_names(Anthropic(api_key="test").messages.stream),
        "Google": _google_generate_content_config_names(),
        "Moonshot AI": moonshot_ai_names,
        "OpenAI": openai_names,
        "Together": _keyword_parameter_names(Together(api_key="test").chat.completions.create),
        "xAI": openai_names,
    }


def test_declared_model_run_options_are_provider_parameters():
    """Catch misspelled top-level provider options in declared model runs."""
    from utils.llm import model_runs

    option_names_by_provider = _declared_option_names_by_provider()
    invalid_options = {
        run.model_run_key: sorted(invalid_keys)
        for run in model_runs.MODEL_RUNS
        if (
            invalid_keys := set(run.options)
            - option_names_by_provider[run.provider.name]
            - ROUTE_OWNED_PROVIDER_OPTION_KEYS
        )
    }

    assert invalid_options == {}


def _models_dev_raw_metadata(run) -> dict | None:
    metadata = run.model.models_dev_metadata
    if metadata is None:
        return None
    return metadata.raw


def _declared_reasoning_options(options: dict) -> set[str]:
    reasoning_options = set(options) & {"reasoning", "thinking"}
    output_config = options.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        reasoning_options.add("output_config.effort")
    return reasoning_options


def test_declared_temperature_options_match_models_dev_support():
    """Models.dev should confirm temperature support before runs set temperature."""
    from utils.llm import model_runs

    invalid_temperature_runs = {
        run.model_run_key: _models_dev_raw_metadata(run).get("temperature")
        for run in model_runs.MODEL_RUNS
        if "temperature" in run.options
        and _models_dev_raw_metadata(run) is not None
        and _models_dev_raw_metadata(run).get("temperature") is not True
    }

    assert invalid_temperature_runs == {}


def test_declared_output_token_caps_do_not_exceed_models_dev_limits():
    """Declared output-token caps should fit within Models.dev output limits."""
    from utils.llm import model_runs

    invalid_token_caps = {}
    for run in model_runs.MODEL_RUNS:
        raw_metadata = _models_dev_raw_metadata(run)
        if raw_metadata is None:
            continue
        output_limit = (raw_metadata.get("limit") or {}).get("output")
        for option_name in OUTPUT_TOKEN_LIMIT_OPTION_KEYS:
            option_value = run.options.get(option_name)
            if option_value is None:
                continue
            if not isinstance(output_limit, int) or option_value > output_limit:
                invalid_token_caps[run.model_run_key] = {
                    "option": option_name,
                    "option_value": option_value,
                    "models_dev_output_limit": output_limit,
                }

    assert invalid_token_caps == {}


def test_declared_reasoning_options_match_models_dev_support():
    """Models.dev should confirm reasoning support before runs set reasoning controls."""
    from utils.llm import model_runs

    invalid_reasoning_runs = {
        run.model_run_key: sorted(_declared_reasoning_options(run.options))
        for run in model_runs.MODEL_RUNS
        if _declared_reasoning_options(run.options)
        if _models_dev_raw_metadata(run) is not None
        and _models_dev_raw_metadata(run).get("reasoning") is not True
    }

    assert invalid_reasoning_runs == {}


def test_active_model_runs_exclude_runs_for_inactive_models():
    """Keep inactive provider routes in history while excluding them from live runs."""
    from utils.llm import model_runs

    all_slugs = {run.slug for run in model_runs.MODEL_RUNS}
    active_slugs = {run.slug for run in model_runs.ACTIVE_MODEL_RUNS}

    assert "deepseek-v3.1" in all_slugs
    assert "deepseek-v3.1" not in active_slugs
    assert all(run.model.active for run in model_runs.ACTIVE_MODEL_RUNS)
    assert model_runs.ACTIVE_MODEL_RUNS_BY_KEY == {
        run.model_run_key: run for run in model_runs.ACTIVE_MODEL_RUNS
    }


def test_mistral_models_are_inactive():
    """Keep Mistral-created Together routes out of active live benchmark runs."""
    from utils.llm import model_registry, model_runs

    mistral_model_keys = {
        model.model_key for model in model_registry.MODELS if model.lab == LABS["Mistral AI"]
    }
    active_mistral_model_keys = {
        model.model_key
        for model in model_registry.MODELS
        if model.lab == LABS["Mistral AI"] and model.active
    }
    active_run_model_keys = {run.model.model_key for run in model_runs.ACTIVE_MODEL_RUNS}

    assert mistral_model_keys == {
        "magistral-medium-2506",
        "mistral-large-2407",
        "mistral-large-2411",
        "mistral-large-latest",
        "mixtral-8x22b-instruct-v0.1",
        "mixtral-8x7b-instruct-v0.1",
    }
    assert active_mistral_model_keys == set()
    assert mistral_model_keys.isdisjoint(active_run_model_keys)


def test_create_model_runs_list_rejects_duplicate_model_run_keys_slugs_and_fingerprints():
    """Reject duplicate keys, slugs, and model-plus-options fingerprints."""
    from utils.llm import model_runs

    run = model_runs.MODEL_RUNS_BY_SLUG["gpt-5.5-2026-04-23"]
    duplicate_slug = model_runs.ModelRun(
        model_run_key="gpt-5.5-2026-04-23-run-variant-99",
        slug=run.slug,
        model=run.model,
        options={"reasoning": {"effort": "medium"}},
    )
    duplicate_fingerprint = model_runs.ModelRun(
        model_run_key="gpt-5.5-2026-04-23-run-variant-99",
        slug="gpt-5.5-2026-04-23-copy",
        model=run.model,
        options=run.options,
    )

    with pytest.raises(ValueError, match="Duplicate LLM model_run_key"):
        model_runs.create_model_runs_list([run, run])
    with pytest.raises(ValueError, match="Duplicate LLM model-run slug"):
        model_runs.create_model_runs_list([run, duplicate_slug])
    with pytest.raises(ValueError, match="Duplicate LLM model-run fingerprint"):
        model_runs.create_model_runs_list([run, duplicate_fingerprint])


def test_declared_model_run_options_do_not_share_mutable_objects():
    """Avoid sharing mutable nested option objects across declared runs."""
    from utils.llm import model_runs

    preview_run = model_runs.MODEL_RUNS_BY_SLUG["gemini-3-flash-preview"]
    lite_run = model_runs.MODEL_RUNS_BY_SLUG["gemini-3.1-flash-lite-preview"]

    assert preview_run.options is not lite_run.options
    assert (
        preview_run.options["automatic_function_calling"]
        is not lite_run.options["automatic_function_calling"]
    )


def test_release_dates_exist_for_all_shared_models():
    """Expose release dates for every shared canonical model."""
    from utils.llm import model_registry

    release_dates = model_registry.model_release_dates_by_key()

    assert release_dates["gpt-5.5-2026-04-23"] == date(2026, 4, 23)
    assert release_dates["deepseek-r1"] == date(2025, 1, 20)
    assert release_dates["deepseek-v3"] == date(2024, 12, 26)
    assert release_dates["deepseek-v3.1"] == date(2025, 8, 21)
    assert release_dates["gemini-3.1-flash-lite"] == date(2026, 5, 7)
    for model in model_registry.MODELS:
        assert release_dates[model.model_key] == model.release_date


def test_historical_forecastbench_llm_release_dates_are_available():
    """Keep historical ForecastBench LLM release dates in the model registry."""
    from utils.llm import model_registry

    historical_model_keys = {
        "claude-2.1",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-20250514",
        "deepseek-r1",
        "deepseek-v3",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-lite-001",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.5-pro-preview-03-25",
        "gemini-3-pro-preview",
        "gemma-4-31b-it",
        "glm-4.5-air-fp8",
        "glm-4.6",
        "glm-4.7",
        "glm-5",
        "gpt-3.5-turbo-0125",
        "gpt-4-0613",
        "gpt-4-turbo-2024-04-09",
        "gpt-4.1-2025-04-14",
        "gpt-4.5-preview-2025-02-27",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-11-20",
        "gpt-5-2025-08-07",
        "gpt-5.1-2025-11-13",
        "grok-4-0709",
        "grok-4-fast-non-reasoning",
        "grok-4-fast-reasoning",
        "grok-4.20-beta-0309-non-reasoning",
        "grok-4.20-beta-0309-reasoning",
        "grok-beta",
        "kimi-k2-instruct",
        "kimi-k2-instruct-0905",
        "kimi-k2-thinking",
        "llama-2-70b-chat-hf",
        "llama-3-70b-chat-hf",
        "llama-3-8b-chat-hf",
        "llama-3.2-3b-instruct-turbo",
        "llama-3.3-70b-instruct-turbo",
        "llama-4-maverick-17b-128e-instruct-fp8",
        "llama-4-scout-17b-16e-instruct",
        "magistral-medium-2506",
        "meta-llama-3.1-405b-instruct-turbo",
        "mistral-large-2407",
        "mistral-large-2411",
        "mistral-large-latest",
        "mixtral-8x22b-instruct-v0.1",
        "mixtral-8x7b-instruct-v0.1",
        "o3-2025-04-16",
        "o3-mini-2025-01-31",
        "o4-mini-2025-04-16",
        "qwen1.5-110b-chat",
        "qwen2.5-72b-instruct-turbo",
        "qwen3-235b-a22b-fp8-tput",
        "qwen3-235b-a22b-thinking-2507",
        "qwq-32b-preview",
    }
    release_dates = model_registry.model_release_dates_by_key()

    assert historical_model_keys <= set(model_registry.MODELS_BY_KEY)
    assert release_dates["gpt-4-0613"] == date(2023, 6, 13)
    assert release_dates["claude-2.1"] == date(2023, 11, 21)
    assert release_dates["kimi-k2-instruct"] == date(2025, 7, 12)
    assert release_dates["qwen3-235b-a22b-thinking-2507"] == date(2025, 7, 25)
    assert all(key in release_dates for key in historical_model_keys)
    assert not any(key.startswith("unusedgrok") for key in model_registry.MODELS_BY_KEY)
    assert "Always 0" not in release_dates
    assert "Naive Forecaster" not in release_dates


def test_select_model_runs_preserves_order_and_rejects_unknown_keys():
    """Select shared model runs in requested order and fail on unknown keys."""
    from utils.llm import model_runs

    first, second = model_runs.ACTIVE_MODEL_RUNS[:2]
    selected = model_runs.select_model_runs([first.model_run_key, second.model_run_key])

    assert [run.model_run_key for run in selected] == [
        first.model_run_key,
        second.model_run_key,
    ]
    with pytest.raises(KeyError, match="missing-model"):
        model_runs.select_model_runs(["missing-model"])


def test_select_model_runs_rejects_inactive_runs_by_default():
    """Keep benchmark-facing model-run selection limited to active runs."""
    from utils.llm import model_runs

    inactive_run = model_runs.MODEL_RUNS_BY_SLUG["deepseek-v3.1"]

    assert inactive_run.model.active is False
    with pytest.raises(KeyError, match="Inactive LLM model_run_key") as excinfo:
        model_runs.select_model_runs([inactive_run.model_run_key])
    assert excinfo.value.__cause__ is None
    assert model_runs.select_model_runs(
        [inactive_run.model_run_key],
        active_only=False,
    ) == [inactive_run]


def test_integration_model_run_default_selection_uses_latest_active_run_per_provider():
    """Keep default live model-run integration coverage broad but bounded."""
    from tests.integration.llm import test_model_runs
    from utils.llm import model_runs

    default_run_keys = test_model_runs.DEFAULT_SMOKE_MODEL_RUN_KEYS
    default_runs = model_runs.select_model_runs(default_run_keys)
    active_providers = {run.provider.name for run in model_runs.ACTIVE_MODEL_RUNS}

    expected_latest_by_provider = {}
    for run in model_runs.ACTIVE_MODEL_RUNS:
        provider_name = run.provider.name
        previous = expected_latest_by_provider.get(provider_name)
        if previous is None or (
            run.release_date,
            run.model_key,
            run.model_run_key,
        ) > (
            previous.release_date,
            previous.model_key,
            previous.model_run_key,
        ):
            expected_latest_by_provider[provider_name] = run

    assert len(default_runs) == len(active_providers)
    assert {run.provider.name for run in default_runs} == active_providers
    assert default_run_keys == tuple(
        expected_latest_by_provider[provider_name].model_run_key
        for provider_name in sorted(expected_latest_by_provider)
    )
    assert any(
        run.provider.name == "Anthropic" and "thinking" in run.options for run in default_runs
    )
    assert any("tools" in run.options for run in default_runs)
    assert any(run.provider.name == "Together" for run in default_runs)


@pytest.mark.parametrize("raw_keys", [",", "   "])
def test_integration_model_run_selection_falls_back_for_empty_env_selection(
    monkeypatch,
    raw_keys,
):
    """Use the default smoke selection when LLM_MODEL_RUN_KEYS yields no keys."""
    from tests.integration.llm import test_model_runs

    monkeypatch.setenv("LLM_MODEL_RUN_KEYS", raw_keys)

    assert (
        test_model_runs._selected_model_run_keys() == test_model_runs.DEFAULT_SMOKE_MODEL_RUN_KEYS
    )


def test_get_model_run_by_slug_is_convenience_lookup_for_human_slugs():
    """Look up model runs by slug while preserving key lookup as canonical."""
    from utils.llm import model_runs

    run = model_runs.MODEL_RUNS_BY_SLUG["gpt-5.5-2026-04-23-high"]

    assert model_runs.get_model_run(run.model_run_key) is run
    assert model_runs.get_model_run_by_slug("gpt-5.5-2026-04-23-high") is run
    with pytest.raises(KeyError, match="missing-slug"):
        model_runs.get_model_run_by_slug("missing-slug")
