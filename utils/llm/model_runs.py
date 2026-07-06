"""Shared LLM model-run registry.

`model_run_key` is handwritten at each declaration site and is the stable immutable identifier
used by benchmark files.

`slug` is handwritten at each declaration site as a human-readable convenience identifier. Slugs
are unique, but may be renamed in the future.
"""

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import model_registry
from ._identifiers import filename_safe_name, validate_registry_key
from .artificial_analysis_model_runs import create_artificial_analysis_model_runs
from .lab_registry import Lab
from .metadata.artificial_analysis import load_artificial_analysis_snapshot
from .provider_registry import Provider

logger = logging.getLogger(__name__)
AA_MODEL_RUN_KEY_SUFFIX_PATTERN = re.compile(r"^aa-run-variant-[0-9]{2}$")
MODEL_RUN_KEY_SUFFIX_PATTERN = re.compile(r"^run-variant-[0-9]{2}$")


@dataclass(frozen=True, slots=True)
class ModelRun:
    """Concrete LLM run with provider options."""

    # Immutable benchmark identifier for this exact model-plus-options run.
    # This is the key for durable references, persisted results, and filenames.
    model_run_key: str
    # Human-readable convenience identifier for lookup and display. Slugs are unique,
    # but they are not durable and may be renamed as conventions change.
    slug: str
    # Canonical base model registry entry that supplies provider routing and metadata.
    model: model_registry.Model
    # Provider call options that distinguish this run from other runs of the same model.
    options: dict[str, Any] = field(default_factory=dict)
    # Stable Artificial Analysis snapshot ID when this run is backed by AA metadata.
    # None means the run is declared only in this registry.
    artificial_analysis_id: str | None = None

    def __post_init__(self) -> None:
        """Validate model-run metadata."""
        _validate_model_run_key(
            self.model_run_key,
            model_key=self.model.model_key,
            artificial_analysis_id=self.artificial_analysis_id,
        )
        validate_registry_key(self.slug, field_name="ModelRun slug")

        if self.artificial_analysis_id is None:
            return

        try:
            load_artificial_analysis_snapshot().get_model(self.artificial_analysis_id)
        except KeyError as exc:
            raise ValueError(
                "Artificial Analysis model runs must reference a valid "
                f"artificial_analysis_id: {self.artificial_analysis_id}"
            ) from exc

    @property
    def display_name(self) -> str:
        """Return the display name for leaderboards and reports."""
        return self.model_key

    @property
    def filename_safe_name(self) -> str:
        """Return a filename-safe model-run name."""
        return filename_safe_name(self.model_run_key, field_name="ModelRun model_run_key")

    @property
    def model_key(self) -> str:
        """Return the canonical base model key."""
        return self.model.model_key

    @property
    def provider_model_id(self) -> str:
        """Return the provider API model identifier."""
        return self.model.provider_model_id

    @property
    def lab(self) -> Lab:
        """Return the model-making lab."""
        return self.model.lab

    @property
    def provider(self) -> Provider:
        """Return the API provider route."""
        return self.model.provider

    @property
    def release_date(self) -> date:
        """Return the underlying model release date."""
        return self.model.release_date

    def __repr__(self) -> str:
        """Return a concise model-run representation."""
        if self.options:
            return f"<ModelRun {self.model_run_key} " f"({self.provider_model_id}) {self.options}>"
        return f"<ModelRun {self.model_run_key}>"

    def get_response(self, prompt: str, **kwargs: Any) -> str:
        """Request a response from the configured provider and model."""
        merged_options = deepcopy(self.options)
        merged_options.update(deepcopy(kwargs))
        logger.info(
            "Requesting LLM response provider=%s provider_model_id=%s options=%s",
            self.provider.name,
            self.provider_model_id,
            merged_options,
        )
        return model_registry.get_response(
            provider=self.provider,
            model_id=self.provider_model_id,
            prompt=prompt,
            options=merged_options,
        )


def _model_run_options_fingerprint(model_key: str, options: dict[str, Any]) -> str:
    """Return a stable fingerprint for a model key plus provider options."""
    validate_registry_key(model_key, field_name="ModelRun model_key")
    try:
        payload = json.dumps(
            {"model_key": model_key, "options": options},
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise TypeError("ModelRun options must be JSON-serializable for fingerprinting") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_model_run_key(
    model_run_key: str,
    *,
    model_key: str,
    artificial_analysis_id: str | None,
) -> None:
    """Reject invalid model-run keys."""
    validate_registry_key(model_run_key, field_name="ModelRun model_run_key")
    expected_prefix = f"{model_key}-"
    if not model_run_key.startswith(expected_prefix):
        raise ValueError(
            "ModelRun model_run_key must be the model_key followed by '-run-variant-XX' "
            "or '-aa-run-variant-XX' for Artificial Analysis-backed runs. "
            f"Expected model_key: {model_key}"
        )
    suffix = model_run_key[len(expected_prefix) :]
    expected_pattern = (
        AA_MODEL_RUN_KEY_SUFFIX_PATTERN
        if artificial_analysis_id is not None
        else MODEL_RUN_KEY_SUFFIX_PATTERN
    )
    if expected_pattern.fullmatch(suffix) is None:
        expected_suffix = (
            "-aa-run-variant-XX" if artificial_analysis_id is not None else "-run-variant-XX"
        )
        raise ValueError(
            "ModelRun model_run_key must be the model_key followed by "
            f"'{expected_suffix}'. "
            f"Invalid suffix: {suffix}"
        )


def _model_run(
    *,
    model_run_key: str,
    slug: str,
    model_key: str,
    options: dict[str, Any] | None = None,
    artificial_analysis_id: str | None = None,
) -> ModelRun:
    """Create a model run from a canonical model key."""
    try:
        model = model_registry.MODELS_BY_KEY[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model_key {model_key}") from exc

    return ModelRun(
        model_run_key=model_run_key,
        slug=slug,
        model=model,
        options=deepcopy(options) if options is not None else {},
        artificial_analysis_id=artificial_analysis_id,
    )


def _validate_unique_model_runs(runs: Sequence[ModelRun]) -> None:
    """Reject duplicate model-run keys, slugs, and semantic fingerprints."""
    seen_model_run_keys = set()
    seen_slugs = set()
    seen_fingerprints = set()
    for run in runs:
        if run.model_run_key in seen_model_run_keys:
            raise ValueError(f"Duplicate LLM model_run_key: {run.model_run_key}")
        seen_model_run_keys.add(run.model_run_key)
        if run.slug in seen_slugs:
            raise ValueError(f"Duplicate LLM model-run slug: {run.slug}")
        seen_slugs.add(run.slug)
        fingerprint = _model_run_options_fingerprint(run.model_key, run.options)
        if fingerprint in seen_fingerprints:
            raise ValueError(
                "Duplicate LLM model-run fingerprint for model_key/options: "
                f"{run.model_key} {run.options}"
            )
        seen_fingerprints.add(fingerprint)


def create_model_runs_list(runs: Sequence[ModelRun]) -> list[ModelRun]:
    """Create a validated model-run registry list."""
    _validate_unique_model_runs(runs)
    return list(runs)


ARTIFICIAL_ANALYSIS_MODEL_RUNS = create_artificial_analysis_model_runs(_model_run)


ANTHROPIC_MODEL_RUNS: list[ModelRun] = [
    _model_run(
        model_run_key="claude-2.1-run-variant-01",
        slug="claude-2.1",
        model_key="claude-2.1",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-3-5-sonnet-20240620-run-variant-01",
        slug="claude-3-5-sonnet-20240620",
        model_key="claude-3-5-sonnet-20240620",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-3-5-sonnet-20241022-run-variant-01",
        slug="claude-3-5-sonnet-20241022",
        model_key="claude-3-5-sonnet-20241022",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-3-7-sonnet-20250219-run-variant-01",
        slug="claude-3-7-sonnet-20250219",
        model_key="claude-3-7-sonnet-20250219",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-3-haiku-20240307-run-variant-01",
        slug="claude-3-haiku-20240307",
        model_key="claude-3-haiku-20240307",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-3-opus-20240229-run-variant-01",
        slug="claude-3-opus-20240229",
        model_key="claude-3-opus-20240229",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-haiku-4-5-20251001-run-variant-01",
        slug="claude-haiku-4-5-20251001-1024",
        model_key="claude-haiku-4-5-20251001",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-haiku-4-5-20251001-run-variant-02",
        slug="claude-haiku-4-5-20251001-4096",
        model_key="claude-haiku-4-5-20251001",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-opus-4-1-20250805-run-variant-01",
        slug="claude-opus-4-1-20250805",
        model_key="claude-opus-4-1-20250805",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-opus-4-20250514-run-variant-01",
        slug="claude-opus-4-20250514",
        model_key="claude-opus-4-20250514",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-opus-4-5-20251101-run-variant-01",
        slug="claude-opus-4-5-20251101",
        model_key="claude-opus-4-5-20251101",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-opus-4-6-run-variant-01",
        slug="claude-opus-4-6-1024",
        model_key="claude-opus-4-6",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-opus-4-6-run-variant-02",
        slug="claude-opus-4-6-4096",
        model_key="claude-opus-4-6",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-opus-4-7-run-variant-01",
        slug="claude-opus-4-7-1024",
        model_key="claude-opus-4-7",
        options={"max_tokens": 1024},
    ),
    _model_run(
        model_run_key="claude-opus-4-7-run-variant-02",
        slug="claude-opus-4-7-4096",
        model_key="claude-opus-4-7",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-opus-4-7-run-variant-03",
        slug="claude-opus-4-7-adaptive-thinking-high-24000",
        model_key="claude-opus-4-7",
        options={
            "max_tokens": 24000,
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
        },
    ),
    _model_run(
        model_run_key="claude-opus-4-7-run-variant-04",
        slug="claude-opus-4-7-adaptive-thinking-high-web-search-64000",
        model_key="claude-opus-4-7",
        options={
            "max_tokens": 64000,
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
            "tools": [
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 5,
                }
            ],
        },
    ),
    _model_run(
        model_run_key="claude-opus-4-8-run-variant-01",
        slug="claude-opus-4-8-1024",
        model_key="claude-opus-4-8",
        options={"max_tokens": 1024},
    ),
    _model_run(
        model_run_key="claude-opus-4-8-run-variant-02",
        slug="claude-opus-4-8-4096",
        model_key="claude-opus-4-8",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-opus-4-8-run-variant-03",
        slug="claude-opus-4-8-adaptive-thinking-high-24000",
        model_key="claude-opus-4-8",
        options={
            "max_tokens": 24000,
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
        },
    ),
    _model_run(
        model_run_key="claude-opus-4-8-run-variant-04",
        slug="claude-opus-4-8-adaptive-thinking-max-web-search-128000",
        model_key="claude-opus-4-8",
        options={
            "max_tokens": 128000,
            "output_config": {"effort": "max"},
            "thinking": {"type": "adaptive"},
            "tools": [
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                }
            ],
        },
    ),
    _model_run(
        model_run_key="claude-sonnet-4-20250514-run-variant-01",
        slug="claude-sonnet-4-20250514",
        model_key="claude-sonnet-4-20250514",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-sonnet-4-5-20250929-run-variant-01",
        slug="claude-sonnet-4-5-20250929-1024",
        model_key="claude-sonnet-4-5-20250929",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-sonnet-4-5-20250929-run-variant-02",
        slug="claude-sonnet-4-5-20250929-4096",
        model_key="claude-sonnet-4-5-20250929",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-sonnet-4-6-run-variant-01",
        slug="claude-sonnet-4-6-1024",
        model_key="claude-sonnet-4-6",
        options={"max_tokens": 1024, "temperature": 0},
    ),
    _model_run(
        model_run_key="claude-sonnet-4-6-run-variant-02",
        slug="claude-sonnet-4-6-4096",
        model_key="claude-sonnet-4-6",
        options={"max_tokens": 4096},
    ),
    _model_run(
        model_run_key="claude-sonnet-4-6-run-variant-03",
        slug="claude-sonnet-4-6-adaptive-thinking-16000",
        model_key="claude-sonnet-4-6",
        options={
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
        },
    ),
    _model_run(
        model_run_key="claude-sonnet-5-run-variant-01",
        slug="claude-sonnet-5-adaptive-thinking-16000",
        model_key="claude-sonnet-5",
        options={
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
        },
    ),
]


GOOGLE_MODEL_RUNS: list[ModelRun] = [
    _model_run(
        model_run_key="gemini-1.5-flash-run-variant-01",
        slug="gemini-1.5-flash",
        model_key="gemini-1.5-flash",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-1.5-pro-run-variant-01",
        slug="gemini-1.5-pro",
        model_key="gemini-1.5-pro",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.0-flash-lite-001-run-variant-01",
        slug="gemini-2.0-flash-lite-001",
        model_key="gemini-2.0-flash-lite-001",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-flash-preview-04-17-run-variant-01",
        slug="gemini-2.5-flash-preview-04-17",
        model_key="gemini-2.5-flash-preview-04-17",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-flash-run-variant-01",
        slug="gemini-2.5-flash",
        model_key="gemini-2.5-flash",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-pro-exp-03-25-run-variant-01",
        slug="gemini-2.5-pro-exp-03-25",
        model_key="gemini-2.5-pro-exp-03-25",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-pro-preview-03-25-run-variant-01",
        slug="gemini-2.5-pro-preview-03-25",
        model_key="gemini-2.5-pro-preview-03-25",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-pro-run-variant-01",
        slug="gemini-2.5-pro",
        model_key="gemini-2.5-pro",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-2.5-pro-run-variant-02",
        slug="gemini-2.5-pro-web-search",
        model_key="gemini-2.5-pro",
        options={
            "temperature": 0,
            "tools": [{"googleSearch": {}}],
        },
    ),
    _model_run(
        model_run_key="gemini-3-flash-preview-run-variant-01",
        slug="gemini-3-flash-preview",
        model_key="gemini-3-flash-preview",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3-pro-preview-run-variant-01",
        slug="gemini-3-pro-preview",
        model_key="gemini-3-pro-preview",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3.1-flash-lite-preview-run-variant-01",
        slug="gemini-3.1-flash-lite-preview",
        model_key="gemini-3.1-flash-lite-preview",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3.1-flash-lite-run-variant-01",
        slug="gemini-3.1-flash-lite",
        model_key="gemini-3.1-flash-lite",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3.1-pro-preview-run-variant-01",
        slug="gemini-3.1-pro-preview",
        model_key="gemini-3.1-pro-preview",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3.1-pro-preview-run-variant-02",
        slug="gemini-3.1-pro-preview-high",
        model_key="gemini-3.1-pro-preview",
        options={
            "thinking_config": {"thinking_level": "high"},
        },
    ),
    _model_run(
        model_run_key="gemini-3.1-pro-preview-run-variant-03",
        slug="gemini-3.1-pro-preview-high-web-search",
        model_key="gemini-3.1-pro-preview",
        options={
            "thinking_config": {"thinking_level": "high"},
            "tools": [{"googleSearch": {}}],
        },
    ),
    _model_run(
        model_run_key="gemini-3.5-flash-run-variant-01",
        slug="gemini-3.5-flash",
        model_key="gemini-3.5-flash",
        options={
            "candidate_count": 1,
            "temperature": 0,
            "automatic_function_calling": {"disable": True},
        },
    ),
    _model_run(
        model_run_key="gemini-3.5-flash-run-variant-02",
        slug="gemini-3.5-flash-high-web-search",
        model_key="gemini-3.5-flash",
        options={
            "thinking_config": {"thinking_level": "high"},
            "tools": [{"googleSearch": {}}],
        },
    ),
]


MOONSHOT_AI_MODEL_RUNS: list[ModelRun] = [
    # The Moonshot API fixes temperature internally for the kimi-k2.x models and
    # errors on any explicit temperature, so these runs omit it. Thinking mode is
    # enabled through extra_body because the OpenAI SDK has no thinking keyword.
    _model_run(
        model_run_key="kimi-k2.5-moonshot-ai-run-variant-01",
        slug="kimi-k2.5-moonshot-ai-thinking",
        model_key="kimi-k2.5-moonshot-ai",
        options={"extra_body": {"thinking": {"type": "enabled"}}},
    ),
    _model_run(
        model_run_key="kimi-k2.5-moonshot-ai-run-variant-02",
        slug="kimi-k2.5-moonshot-ai-thinking-128k",
        model_key="kimi-k2.5-moonshot-ai",
        options={"extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 131072},
    ),
    _model_run(
        model_run_key="kimi-k2.6-moonshot-ai-run-variant-01",
        slug="kimi-k2.6-moonshot-ai-thinking",
        model_key="kimi-k2.6-moonshot-ai",
        options={"extra_body": {"thinking": {"type": "enabled"}}},
    ),
    _model_run(
        model_run_key="kimi-k2.6-moonshot-ai-run-variant-02",
        slug="kimi-k2.6-moonshot-ai-thinking-128k",
        model_key="kimi-k2.6-moonshot-ai",
        options={"extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 131072},
    ),
]


OAI_MODEL_RUNS: list[ModelRun] = [
    _model_run(
        model_run_key="gpt-3.5-turbo-0125-run-variant-01",
        slug="gpt-3.5-turbo-0125",
        model_key="gpt-3.5-turbo-0125",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4-0613-run-variant-01",
        slug="gpt-4-0613",
        model_key="gpt-4-0613",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4-turbo-2024-04-09-run-variant-01",
        slug="gpt-4-turbo-2024-04-09",
        model_key="gpt-4-turbo-2024-04-09",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4.1-2025-04-14-run-variant-01",
        slug="gpt-4.1-2025-04-14",
        model_key="gpt-4.1-2025-04-14",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4.5-preview-2025-02-27-run-variant-01",
        slug="gpt-4.5-preview-2025-02-27",
        model_key="gpt-4.5-preview-2025-02-27",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4o-2024-05-13-run-variant-01",
        slug="gpt-4o-2024-05-13",
        model_key="gpt-4o-2024-05-13",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4o-2024-11-20-run-variant-01",
        slug="gpt-4o-2024-11-20",
        model_key="gpt-4o-2024-11-20",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-4o-2024-11-20-run-variant-02",
        slug="gpt-4o-2024-11-20-web-search",
        model_key="gpt-4o-2024-11-20",
        options={
            "temperature": 0,
            "tools": [{"type": "web_search_preview"}],
        },
    ),
    _model_run(
        model_run_key="gpt-4o-mini-2024-07-18-run-variant-01",
        slug="gpt-4o-mini-2024-07-18",
        model_key="gpt-4o-mini-2024-07-18",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gpt-5-2025-08-07-run-variant-01",
        slug="gpt-5-2025-08-07",
        model_key="gpt-5-2025-08-07",
    ),
    _model_run(
        model_run_key="gpt-5-mini-2025-08-07-run-variant-01",
        slug="gpt-5-mini-2025-08-07",
        model_key="gpt-5-mini-2025-08-07",
    ),
    _model_run(
        model_run_key="gpt-5-mini-2025-08-07-run-variant-02",
        slug="gpt-5-mini-2025-08-07-1024",
        model_key="gpt-5-mini-2025-08-07",
        options={"max_output_tokens": 1024},
    ),
    _model_run(
        model_run_key="gpt-5-nano-2025-08-07-run-variant-01",
        slug="gpt-5-nano-2025-08-07",
        model_key="gpt-5-nano-2025-08-07",
    ),
    _model_run(
        model_run_key="gpt-5.1-2025-11-13-run-variant-01",
        slug="gpt-5.1-2025-11-13",
        model_key="gpt-5.1-2025-11-13",
    ),
    _model_run(
        model_run_key="gpt-5.2-2025-12-11-run-variant-01",
        slug="gpt-5.2-2025-12-11",
        model_key="gpt-5.2-2025-12-11",
    ),
    _model_run(
        model_run_key="gpt-5.4-2026-03-05-run-variant-01",
        slug="gpt-5.4-2026-03-05",
        model_key="gpt-5.4-2026-03-05",
    ),
    _model_run(
        model_run_key="gpt-5.4-2026-03-05-run-variant-02",
        slug="gpt-5.4-2026-03-05-high",
        model_key="gpt-5.4-2026-03-05",
        options={"reasoning": {"effort": "high"}},
    ),
    _model_run(
        model_run_key="gpt-5.4-2026-03-05-run-variant-03",
        slug="gpt-5.4-2026-03-05-high-web-search",
        model_key="gpt-5.4-2026-03-05",
        options={
            "reasoning": {"effort": "high"},
            "tools": [{"type": "web_search"}],
        },
    ),
    _model_run(
        model_run_key="gpt-5.4-mini-2026-03-17-run-variant-01",
        slug="gpt-5.4-mini-2026-03-17",
        model_key="gpt-5.4-mini-2026-03-17",
    ),
    _model_run(
        model_run_key="gpt-5.4-nano-2026-03-17-run-variant-01",
        slug="gpt-5.4-nano-2026-03-17",
        model_key="gpt-5.4-nano-2026-03-17",
    ),
    _model_run(
        model_run_key="gpt-5.5-2026-04-23-run-variant-01",
        slug="gpt-5.5-2026-04-23",
        model_key="gpt-5.5-2026-04-23",
    ),
    _model_run(
        model_run_key="gpt-5.5-2026-04-23-run-variant-02",
        slug="gpt-5.5-2026-04-23-medium",
        model_key="gpt-5.5-2026-04-23",
        options={"reasoning": {"effort": "medium"}},
    ),
    _model_run(
        model_run_key="gpt-5.5-2026-04-23-run-variant-03",
        slug="gpt-5.5-2026-04-23-high",
        model_key="gpt-5.5-2026-04-23",
        options={"reasoning": {"effort": "high"}},
    ),
    _model_run(
        model_run_key="gpt-5.5-2026-04-23-run-variant-04",
        slug="gpt-5.5-2026-04-23-high-web-search",
        model_key="gpt-5.5-2026-04-23",
        options={
            "reasoning": {"effort": "high"},
            "tools": [{"type": "web_search"}],
        },
    ),
    _model_run(
        model_run_key="o3-2025-04-16-run-variant-01",
        slug="o3-2025-04-16",
        model_key="o3-2025-04-16",
    ),
    _model_run(
        model_run_key="o3-mini-2025-01-31-run-variant-01",
        slug="o3-mini-2025-01-31",
        model_key="o3-mini-2025-01-31",
    ),
    _model_run(
        model_run_key="o4-mini-2025-04-16-run-variant-01",
        slug="o4-mini-2025-04-16",
        model_key="o4-mini-2025-04-16",
    ),
]


TOGETHER_MODEL_RUNS: list[ModelRun] = [
    _model_run(
        model_run_key="deepseek-r1-run-variant-01",
        slug="deepseek-r1",
        model_key="deepseek-r1",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="deepseek-v3-run-variant-01",
        slug="deepseek-v3",
        model_key="deepseek-v3",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="deepseek-v3.1-run-variant-01",
        slug="deepseek-v3.1",
        model_key="deepseek-v3.1",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="deepseek-v4-pro-run-variant-01",
        slug="deepseek-v4-pro",
        model_key="deepseek-v4-pro",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gemma-4-31b-it-run-variant-01",
        slug="gemma-4-31b-it",
        model_key="gemma-4-31b-it",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="gemma-4-31b-it-run-variant-02",
        slug="gemma-4-31b-it-16000",
        model_key="gemma-4-31b-it",
        options={"max_tokens": 16000},
    ),
    _model_run(
        model_run_key="glm-4.5-air-fp8-run-variant-01",
        slug="glm-4.5-air-fp8",
        model_key="glm-4.5-air-fp8",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-4.6-run-variant-01",
        slug="glm-4.6",
        model_key="glm-4.6",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-4.7-run-variant-01",
        slug="glm-4.7",
        model_key="glm-4.7",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-5-run-variant-01",
        slug="glm-5",
        model_key="glm-5",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-5.1-run-variant-01",
        slug="glm-5.1",
        model_key="glm-5.1",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-5.2-run-variant-01",
        slug="glm-5.2",
        model_key="glm-5.2",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="glm-5.2-run-variant-02",
        slug="glm-5.2-12000",
        model_key="glm-5.2",
        options={"temperature": 0, "max_tokens": 12000},
    ),
    _model_run(
        model_run_key="kimi-k2-instruct-0905-run-variant-01",
        slug="kimi-k2-instruct-0905",
        model_key="kimi-k2-instruct-0905",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="kimi-k2-instruct-run-variant-01",
        slug="kimi-k2-instruct",
        model_key="kimi-k2-instruct",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="kimi-k2-thinking-run-variant-01",
        slug="kimi-k2-thinking",
        model_key="kimi-k2-thinking",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="kimi-k2.5-run-variant-01",
        slug="kimi-k2.5",
        model_key="kimi-k2.5",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="kimi-k2.6-run-variant-01",
        slug="kimi-k2.6",
        model_key="kimi-k2.6",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="kimi-k2.6-run-variant-02",
        slug="kimi-k2.6-16000",
        model_key="kimi-k2.6",
        options={"max_tokens": 16000},
    ),
    _model_run(
        model_run_key="llama-2-70b-chat-hf-run-variant-01",
        slug="llama-2-70b-chat-hf",
        model_key="llama-2-70b-chat-hf",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-3-70b-chat-hf-run-variant-01",
        slug="llama-3-70b-chat-hf",
        model_key="llama-3-70b-chat-hf",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-3-8b-chat-hf-run-variant-01",
        slug="llama-3-8b-chat-hf",
        model_key="llama-3-8b-chat-hf",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-3.2-3b-instruct-turbo-run-variant-01",
        slug="llama-3.2-3b-instruct-turbo",
        model_key="llama-3.2-3b-instruct-turbo",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-3.3-70b-instruct-turbo-run-variant-01",
        slug="llama-3.3-70b-instruct-turbo",
        model_key="llama-3.3-70b-instruct-turbo",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-4-maverick-17b-128e-instruct-fp8-run-variant-01",
        slug="llama-4-maverick-17b-128e-instruct-fp8",
        model_key="llama-4-maverick-17b-128e-instruct-fp8",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="llama-4-scout-17b-16e-instruct-run-variant-01",
        slug="llama-4-scout-17b-16e-instruct",
        model_key="llama-4-scout-17b-16e-instruct",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="magistral-medium-2506-run-variant-01",
        slug="magistral-medium-2506",
        model_key="magistral-medium-2506",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="meta-llama-3.1-405b-instruct-turbo-run-variant-01",
        slug="meta-llama-3.1-405b-instruct-turbo",
        model_key="meta-llama-3.1-405b-instruct-turbo",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="minimax-m2.5-run-variant-01",
        slug="minimax-m2.5",
        model_key="minimax-m2.5",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="minimax-m2.7-run-variant-01",
        slug="minimax-m2.7",
        model_key="minimax-m2.7",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="minimax-m2.7-run-variant-02",
        slug="minimax-m2.7-12000",
        model_key="minimax-m2.7",
        # MiniMax options from:
        # https://github.com/MiniMax-AI/MiniMax-M2.7#inference-parameters
        options={"temperature": 1.0, "top_p": 0.95, "top_k": 40, "max_tokens": 12000},
    ),
    _model_run(
        model_run_key="minimax-m3-run-variant-01",
        slug="minimax-m3-adaptive-thinking-12000",
        model_key="minimax-m3",
        # MiniMax options from:
        # https://github.com/MiniMax-AI/MiniMax-M3#inference-parameters
        options={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "chat_template_kwargs": {"thinking_mode": "adaptive"},
            "max_tokens": 12000,
        },
    ),
    _model_run(
        model_run_key="mistral-large-2407-run-variant-01",
        slug="mistral-large-2407",
        model_key="mistral-large-2407",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="mistral-large-2411-run-variant-01",
        slug="mistral-large-2411",
        model_key="mistral-large-2411",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="mistral-large-latest-run-variant-01",
        slug="mistral-large-latest",
        model_key="mistral-large-latest",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="mixtral-8x22b-instruct-v0.1-run-variant-01",
        slug="mixtral-8x22b-instruct-v0.1",
        model_key="mixtral-8x22b-instruct-v0.1",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="mixtral-8x7b-instruct-v0.1-run-variant-01",
        slug="mixtral-8x7b-instruct-v0.1",
        model_key="mixtral-8x7b-instruct-v0.1",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="qwen1.5-110b-chat-run-variant-01",
        slug="qwen1.5-110b-chat",
        model_key="qwen1.5-110b-chat",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="qwen2.5-72b-instruct-turbo-run-variant-01",
        slug="qwen2.5-72b-instruct-turbo",
        model_key="qwen2.5-72b-instruct-turbo",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="qwen3-235b-a22b-fp8-tput-run-variant-01",
        slug="qwen3-235b-a22b-fp8-tput",
        model_key="qwen3-235b-a22b-fp8-tput",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="qwen3-235b-a22b-thinking-2507-run-variant-01",
        slug="qwen3-235b-a22b-thinking-2507",
        model_key="qwen3-235b-a22b-thinking-2507",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="qwq-32b-preview-run-variant-01",
        slug="qwq-32b-preview",
        model_key="qwq-32b-preview",
        options={"temperature": 0},
    ),
]


XAI_MODEL_RUNS: list[ModelRun] = [
    _model_run(
        model_run_key="grok-4-0709-run-variant-01",
        slug="grok-4-0709",
        model_key="grok-4-0709",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4-1-fast-non-reasoning-run-variant-01",
        slug="grok-4-1-fast-non-reasoning",
        model_key="grok-4-1-fast-non-reasoning",
    ),
    _model_run(
        model_run_key="grok-4-1-fast-reasoning-run-variant-01",
        slug="grok-4-1-fast-reasoning",
        model_key="grok-4-1-fast-reasoning",
    ),
    _model_run(
        model_run_key="grok-4-fast-non-reasoning-run-variant-01",
        slug="grok-4-fast-non-reasoning",
        model_key="grok-4-fast-non-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4-fast-reasoning-run-variant-01",
        slug="grok-4-fast-reasoning",
        model_key="grok-4-fast-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.20-0309-non-reasoning-run-variant-01",
        slug="grok-4.20-0309-non-reasoning",
        model_key="grok-4.20-0309-non-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.20-0309-reasoning-run-variant-01",
        slug="grok-4.20-0309-reasoning",
        model_key="grok-4.20-0309-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.20-0309-reasoning-run-variant-02",
        slug="grok-4.20-0309-reasoning-web-search-x-search",
        model_key="grok-4.20-0309-reasoning",
        options={
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
        },
    ),
    _model_run(
        model_run_key="grok-4.20-beta-0309-non-reasoning-run-variant-01",
        slug="grok-4.20-beta-0309-non-reasoning",
        model_key="grok-4.20-beta-0309-non-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.20-beta-0309-reasoning-run-variant-01",
        slug="grok-4.20-beta-0309-reasoning",
        model_key="grok-4.20-beta-0309-reasoning",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.3-run-variant-01",
        slug="grok-4.3",
        model_key="grok-4.3",
        options={"temperature": 0},
    ),
    _model_run(
        model_run_key="grok-4.3-run-variant-02",
        slug="grok-4.3-high-web-x-search",
        model_key="grok-4.3",
        options={
            "reasoning": {"effort": "high"},
            "tools": [{"type": "web_search"}, {"type": "x_search"}],
        },
    ),
    _model_run(
        model_run_key="grok-beta-run-variant-01",
        slug="grok-beta",
        model_key="grok-beta",
        options={"temperature": 0},
    ),
]


MODEL_RUNS: list[ModelRun] = create_model_runs_list(
    [
        *ANTHROPIC_MODEL_RUNS,
        *GOOGLE_MODEL_RUNS,
        *MOONSHOT_AI_MODEL_RUNS,
        *OAI_MODEL_RUNS,
        *TOGETHER_MODEL_RUNS,
        *XAI_MODEL_RUNS,
        # AA declarations are benchmark-selectable runs, not metadata-only
        # records, so declaring them in the AA module adds them here.
        *ARTIFICIAL_ANALYSIS_MODEL_RUNS,
    ]
)
MODEL_RUNS_BY_KEY: dict[str, ModelRun] = {run.model_run_key: run for run in MODEL_RUNS}
MODEL_RUNS_BY_SLUG: dict[str, ModelRun] = {run.slug: run for run in MODEL_RUNS}

# MODEL_RUNS is historical. ACTIVE_MODEL_RUNS is the current live-callable
# subset for benchmarks and integration sweeps.
ACTIVE_MODEL_RUNS: list[ModelRun] = [run for run in MODEL_RUNS if run.model.active]
ACTIVE_MODEL_RUNS_BY_KEY: dict[str, ModelRun] = {
    run.model_run_key: run for run in ACTIVE_MODEL_RUNS
}
ACTIVE_MODEL_RUNS_BY_SLUG: dict[str, ModelRun] = {run.slug: run for run in ACTIVE_MODEL_RUNS}


def get_model_run(model_run_key: str) -> ModelRun:
    """Return a shared model run by immutable key. Prefer this for durable references."""
    try:
        return MODEL_RUNS_BY_KEY[model_run_key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_RUNS_BY_KEY))
        raise KeyError(
            f"Unknown LLM model_run_key {model_run_key}. Available: {available}"
        ) from exc


def get_model_run_by_slug(slug: str) -> ModelRun:
    """Return a shared model run by human-readable slug. Convenience lookup only."""
    try:
        return MODEL_RUNS_BY_SLUG[slug]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_RUNS_BY_SLUG))
        raise KeyError(f"Unknown LLM model-run slug {slug}. Available: {available}") from exc


def _get_selectable_model_run(model_run_key: str, *, active_only: bool) -> ModelRun:
    """Return a model run from the requested selection scope."""
    if not active_only:
        return get_model_run(model_run_key)

    try:
        return ACTIVE_MODEL_RUNS_BY_KEY[model_run_key]
    except KeyError as exc:
        if model_run_key in MODEL_RUNS_BY_KEY:
            raise KeyError(
                f"Inactive LLM model_run_key {model_run_key}. "
                "Pass active_only=False to select_model_runs() for historical runs."
            ) from None
        available = ", ".join(sorted(ACTIVE_MODEL_RUNS_BY_KEY))
        raise KeyError(
            f"Unknown active LLM model_run_key {model_run_key}. Available: {available}"
        ) from exc


def select_model_runs(
    model_run_keys: Sequence[str],
    *,
    active_only: bool = True,
) -> list[ModelRun]:
    """Return active model runs in the requested order unless historical runs are allowed."""
    return [
        _get_selectable_model_run(model_run_key, active_only=active_only)
        for model_run_key in model_run_keys
    ]
