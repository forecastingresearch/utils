"""Check the installed agentic wheel with pi and a localhost model, without API calls.

Run from outside the checkout, with PYTHONPATH unset, using the wheel's Python
environment. FORECASTER_PI must point to a separate pinned pi installation.
"""

import importlib.util
import json
import os
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

from utils.llm.agentic import harness


def main() -> None:
    """Exercise bundled TypeScript and Python tools from the installed wheel."""
    checkout = Path(__file__).resolve().parents[1]
    package_dir = Path(harness.__file__).resolve().parent
    assert not Path.cwd().resolve().is_relative_to(checkout), "Run outside the checkout."
    assert not os.environ.get("PYTHONPATH"), "Unset PYTHONPATH for the installed-wheel check."
    assert package_dir.is_relative_to(Path(sys.prefix).resolve()), "Import the installed wheel."
    pi = Path(os.environ["FORECASTER_PI"]).resolve()
    assert not pi.is_relative_to(package_dir), "Install pi outside the Python package."
    resources = files("utils.llm.agentic")
    package = json.loads(resources.joinpath("package.json").read_text())
    lock = json.loads(resources.joinpath("package-lock.json").read_text())
    dependency = "@earendil-works/pi-coding-agent"
    expected_version = package["dependencies"][dependency]
    assert lock["packages"][f"node_modules/{dependency}"]["version"] == expected_version
    assert harness.pi_version(str(pi)) == expected_version, "The installed pi version differs."

    # Load only the stdlib localhost fixture, keeping the checkout off sys.path.
    fixture_path = checkout / "tests/unit/agentic/scripted_model.py"
    spec = importlib.util.spec_from_file_location("wheel_scripted_model", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture.ScriptedModel(
        [
            ("read_page", {"url": "https://example.invalid/blocked"}),
            (
                "submit_forecast",
                {"forecasts": [{"id": "q1", "probability": 0.4}], "rationale": "Local fixture."},
            ),
            ("submit_debrief", {"checked": False}),
            ("submit_debrief", {"checked": True}),
            "done",
        ]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="fri-wheel-smoke-") as directory:
            root = Path(directory)
            agent = root / "agent"
            agent.mkdir()
            (agent / "models.json").write_text(
                json.dumps(
                    {
                        "providers": {
                            "fake": {
                                "baseUrl": model.url,
                                "api": "openai-completions",
                                "apiKey": "local-test-key",
                                "models": [{"id": "scripted"}],
                            }
                        }
                    }
                )
            )
            validator = root / "validator.py"
            validator.write_text(
                "def check(value):\n"
                "    return [] if value.get('checked') is True else ['Check the forecast.']\n"
            )
            result = harness.run(
                model="fake/scripted",
                system="Return the scripted forecast.",
                prompt="Forecast q1.",
                workdir=str(root / "calls"),
                cell_schema={
                    "type": "object",
                    "required": ["id", "probability"],
                    "properties": {
                        "id": {"type": "string"},
                        "probability": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                key_fields=["id"],
                allowed_keys=["q1"],
                cells_tool=False,
                research_tools=("read_page",),
                tool_descriptions={"read_page": "Read a page.", "submit_forecast": "Submit."},
                min_research=0,
                thinking="off",
                pi=str(pi),
                timeout=60,
                env={
                    "PI_CODING_AGENT_DIR": str(agent),
                    "PI_OFFLINE": "1",
                    "PI_TELEMETRY": "0",
                    "FORECASTER_BLOCK_URL_PATTERNS": ".*",
                    "TAVILY_API_KEY": "",
                },
                debrief={
                    "tool": {
                        "name": "submit_debrief",
                        "description": "Confirm the forecast was checked.",
                        "parameters": {
                            "type": "object",
                            "required": ["checked"],
                            "properties": {"checked": {"type": "boolean"}},
                        },
                    },
                    "prompt": "Check the forecast and submit the debrief.",
                    "retries": 1,
                    "validator": {"file": str(validator), "function": "check"},
                },
            )
            stderr = (Path(result["workdir"]) / "stderr.txt").read_text()
            assert result["rc"] == 0, stderr
            assert result["cells"] == {"q1": {"id": "q1", "probability": 0.4}}, result
            assert result["usage"]["research_failed"] == 1, result
            assert result["evidence"][0]["error"] == "page read failed: page unavailable", result
            assert result["debrief"]["complete"] is True, result
            assert result["debrief"]["attempts"] == 2, result
            assert result["debrief"]["error"] is None, result
    finally:
        model.close()
    print("Installed wheel passed: pi extension, research tool CLI, and debrief validator.")


if __name__ == "__main__":
    main()
