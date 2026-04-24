# ABOUTME: Tests that importing individual utils submodules doesn't eagerly load unrelated ones.
# ABOUTME: Ensures consumers like the leaderboard can use utils.gcp without pulling in LLM deps.

"""Verify that the utils package lazily loads heavy submodules."""

import subprocess
import sys


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
