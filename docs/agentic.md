# Agentic forecasting

`utils.llm.agentic.run` runs AIRO's forecasting harness with web research and
structured submissions. Callers supply the prompt and forecast schema. Existing
`ModelRun.get_response()` calls remain one-shot; importing this module starts no work.

This API does not register model runs or change AIRO, ForecastBench, or DWH callers.
Before publishing benchmark results, give the full agent configuration a distinct
identity. An existing one-shot `model_run_key` does not identify an agentic run.

## Runtime

Install `fri-utils` in your Python environment. Agentic calls also require
Node.js 22.19 or newer and pi 0.99.2. Install the packaged, locked runtime outside
the Python package:

```bash
export FORECASTER_RUNTIME="$HOME/.cache/fri-utils/pi-0.99.2"
python - <<'PY'
import os
from importlib.resources import files
from pathlib import Path

runtime = Path(os.environ["FORECASTER_RUNTIME"])
runtime.mkdir(parents=True, exist_ok=True)
resources = files("utils.llm.agentic")
for name in ("package.json", "package-lock.json"):
    (runtime / name).write_bytes(resources.joinpath(name).read_bytes())
PY
npm ci --prefix "$FORECASTER_RUNTIME"
export FORECASTER_PI="$FORECASTER_RUNTIME/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
node "$FORECASTER_PI" --version
```

The harness checks pi's version before each run. The `pi=` argument takes precedence
over `env["FORECASTER_PI"]`, then the process's `FORECASTER_PI`. Set an absolute path;
the harness does not select a pi installation from `PATH`.

## Keys

Set `FORECASTER_MODEL` to the exact `provider/model-id` supported by pi and your
API account. This is a pi identifier, not a `model_run_key` or slug. Provide that
provider's key and `TAVILY_API_KEY` when using `web_search` or `read_page`.

| Provider prefix | Environment variable |
| --- | --- |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GEMINI_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `deepinfra` | `DEEPINFRA_API_KEY` |

The harness reads the process environment plus explicit `env={...}` overrides.
It does not search for `.env` files. For local use, put the selected model and keys
in a git-ignored `.env`, then call `load_dotenv(".env", override=False)` as below.
Existing environment values take precedence over that file. Deployment jobs should
inject secrets under the same environment names.

`configure_api_keys(from_gcp=True)` configures Python clients in memory. It does not
pass keys to pi. For example, inject GCP secret `API_KEY_OPENAI` as `OPENAI_API_KEY`
in the job environment. Do not put key values in command-line arguments, logs, or source control.

Calls use API billing. The harness removes Anthropic bearer/OAuth token variables
and rejects a `PI_CODING_AGENT_DIR` containing an OAuth login. Use a dedicated directory
without a `pi /login`; omit the variable to use the harness's temporary agent directory.

## Call

This example makes paid provider and Tavily calls. It forecasts one question, sets
a turn limit and a three-minute timeout, and disables retries. Select
`FORECASTER_MODEL` and provide keys before running it.

```python
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from utils.llm.agentic import run

load_dotenv(".env", override=False)

result = run(
    model=os.environ["FORECASTER_MODEL"],
    system=(
        "You are a calibrated forecaster. Research the question with web_search "
        "and read_page, then call submit_forecast with your probability, rationale, "
        "and key_sources."
    ),
    prompt=(
        f"Today is {datetime.now(timezone.utc).date()}. "
        "Question q1: Will NASA report a higher global annual temperature anomaly "
        "for 2027 than for 2026? Resolve using NASA's published annual values "
        "available on 2028-03-01. Give the probability that the answer is Yes."
    ),
    workdir="agentic-run",
    cell_schema={
        "type": "object",
        "required": ["question_id", "probability"],
        "properties": {
            "question_id": {"type": "string", "enum": ["q1"]},
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    },
    key_fields=["question_id"],
    allowed_keys=["q1"],
    tool_descriptions={
        "web_search": "Search the web for relevant evidence.",
        "read_page": "Read a source page returned by web search.",
        "submit_forecast": "Submit forecasts, rationale, and key_sources once, at the end.",
    },
    cells_tool=False,
    research_tools=("web_search", "read_page"),
    min_research=1,
    thinking="low",
    max_turns=6,
    timeout=180,
    retries=0,
)

missing = {"q1"} - result["cells"].keys()
if result["rc"] != 0 or result["usage"]["api_error"] or missing:
    raise RuntimeError(f"Forecast failed or incomplete; inspect {result['workdir']}")
if not result["answer"] or result["usage"]["research_ok"] < 1:
    raise RuntimeError(f"No researched final answer; inspect {result['workdir']}")
print(result["cells"]["q1"])
print(result["answer"])
print(result["usage"])
```

`cell_schema` can also describe numeric values or quantiles. `key_fields` identifies
each cell; multiple field values join with `|` in `allowed_keys`. Provide descriptions
for each enabled research tool, `submit_forecast`, and `submit_cells` if enabled.
The final submission contains `forecasts`, `rationale`, and optional `key_sources`.

## Results and limits

Each attempt writes a fresh `attempt-N` directory under `workdir`. It contains the
prompt, system prompt, tool specification, event trace, and stderr. Submissions create
`pieces.jsonl` and/or `final.json`; an optional debrief creates `debrief.json`.
The result includes merged `cells`, the final `answer`, `evidence`, `usage`, `rc`,
`forced`, `trace`, and `workdir`. Final cells override earlier pieces. Check expected
keys and errors before consuming results: pi can exit with code zero after an API error.

`max_turns` requests final submission at the limit and permits up to two more final
turns. `forced` records a submission forced by this limit. `timeout` bounds each
attempt; neither setting is a dollar cap. On macOS and Linux, a timeout also stops
the research subprocesses. Other platforms stop pi only. Retries are opt-in and can spend again.
Returned usage covers only the final attempt and excludes Tavily charges. Earlier
attempt traces remain on disk. Review all attempts and provider billing before a
large run.

Run the offline tests without secrets:

```bash
pytest tests/unit/agentic
FRI_REQUIRE_PI=1 FORECASTER_PI="$FORECASTER_PI" pytest tests/unit/agentic
```

The second command requires the installed runtime and exercises it without paid
provider calls. Before rollout, review one bounded live result for valid cells,
evidence, complete artifacts, latency, and actual cost. Consumer migrations and a
ForecastBench biweekly run need separate validation.
