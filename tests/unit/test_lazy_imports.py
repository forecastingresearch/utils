# ABOUTME: Tests that importing individual utils submodules doesn't eagerly load unrelated ones.
# ABOUTME: Ensures consumers like the leaderboard can use utils.gcp without pulling in LLM deps.

"""Verify that the utils package lazily loads heavy submodules."""

import subprocess
import sys


def test_import_utils_does_not_load_llm_model_runs_or_providers():
    """Importing utils must not trigger import of heavy LLM registries."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import utils; "
            "unexpected_modules = ["
            "module_name for module_name in ("
            "'utils.llm', "
            "'utils.llm.model_runs', "
            "'utils.llm.model_registry', "
            "'utils.llm.providers', "
            "'utils.llm.metadata.models_dev', "
            "'utils.llm.metadata.artificial_analysis'"
            ") if module_name in sys.modules"
            "]; "
            "assert not unexpected_modules, unexpected_modules",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Importing utils eagerly pulled in LLM modules:\n{result.stderr}"


def test_import_llm_does_not_load_model_runs_providers_or_snapshots():
    """Importing utils.llm must not trigger heavy model-run/provider imports."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import utils.llm; "
            "unexpected_modules = ["
            "module_name for module_name in ("
            "'utils.llm.model_runs', "
            "'utils.llm.model_registry', "
            "'utils.llm.providers', "
            "'utils.llm.metadata.models_dev', "
            "'utils.llm.metadata.artificial_analysis'"
            ") if module_name in sys.modules"
            "]; "
            "assert not unexpected_modules, unexpected_modules",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Importing utils.llm eagerly pulled in LLM modules:\n{result.stderr}"


def test_import_gcp_does_not_load_llm():
    """Importing utils.gcp must not trigger import of utils.llm or its providers."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from utils import gcp; import sys; "
            "assert 'utils.llm' not in sys.modules, "
            "'utils.llm was eagerly imported when only gcp was requested'",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Importing utils.gcp eagerly pulled in utils.llm:\n{result.stderr}"


def test_import_llm_metadata_does_not_load_model_providers():
    """Importing lightweight LLM metadata must not trigger provider SDK imports."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import utils.llm.lab_registry; import utils.llm.provider_registry; import sys; "
            "assert 'utils.llm.model_registry' not in sys.modules, "
            "'utils.llm.model_registry was eagerly imported'; "
            "assert 'utils.llm.providers' not in sys.modules, "
            "'utils.llm.providers was eagerly imported'",
        ],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Importing LLM metadata eagerly pulled in provider modules:\n{result.stderr}"
