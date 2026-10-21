"""Check the real pi extension with scripted localhost responses and no paid calls."""

import json
import os
import tempfile
import textwrap
import unittest

from utils.llm.agentic import harness

from .scripted_model import REFUSAL, ScriptedModel

SCHEMA = {
    "type": "object",
    "required": ["k", "probability"],
    "properties": {"k": {"type": "string"}, "probability": {"type": "number"}},
}
DESC = {"web_search": "s", "read_page": "r", "submit_cells": "c", "submit_forecast": "f"}

# Research without the network: web_search answers, read_page fails. A
# debrief validator runs as the real tool_cli runs it.
FAKE_TOOL_CLI = textwrap.dedent("""\
    import json, sys
    tool = sys.argv[1]
    if tool == "validate":
        from utils.llm.agentic.tool_cli import validate
        print(json.dumps(validate(**json.load(sys.stdin))))
    else:
        print(json.dumps({"results": [{"url": "u", "title": "t"}]} if tool == "web_search"
                         else {"error": "page read failed: 404"}))
    """)
VALIDATOR = "def check(value, need):\n    return [f'{k}: missing' for k in need if k not in (value or {})]\n"


# Configured runtimes must work; required CI runs must never silently skip.
NEEDS_PI = unittest.skipUnless(
    bool(os.environ.get("FORECASTER_PI")) or os.environ.get("FRI_REQUIRE_PI") == "1",
    "set FORECASTER_PI to run the local scripted-model tests",
)


class PiCase(unittest.TestCase):
    """Run the pinned pi against a scripted model and fake research tools."""

    def setUp(self):
        """Create temporary files and replace the research CLI."""
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        cli = os.path.join(d, "tool_cli.py")
        open(cli, "w").write(FAKE_TOOL_CLI)
        self.real_cli, harness.TOOL_CLI = harness.TOOL_CLI, [cli]

    def tearDown(self):
        """Restore the real research CLI and remove temporary files."""
        harness.TOOL_CLI = self.real_cli
        self.tmp.cleanup()

    def run_pi(self, script, **kw):
        """Execute one scripted model conversation in an isolated agent directory."""
        model = ScriptedModel(script)
        self.addCleanup(model.close)
        agent = os.path.join(self.tmp.name, "agent")
        os.makedirs(agent, exist_ok=True)
        json.dump(
            {
                "providers": {
                    "fake": {
                        "baseUrl": model.url,
                        "api": "openai-completions",
                        "apiKey": "x",
                        "models": [{"id": "scripted"}],
                    }
                }
            },
            open(os.path.join(agent, "models.json"), "w"),
        )
        args = dict(
            model="fake/scripted",
            system="S",
            prompt="P",
            workdir=os.path.join(self.tmp.name, "w"),
            cell_schema=SCHEMA,
            key_fields=["k"],
            allowed_keys=["a", "b"],
            tool_descriptions=DESC,
            min_research=2,
            thinking="off",
            timeout=120,
            env={"PI_CODING_AGENT_DIR": agent, "PI_OFFLINE": "1", "PI_TELEMETRY": "0"},
        )
        args["env"].update(kw.pop("env", {}))
        args.update(kw)
        return harness.run(**args), model


@NEEDS_PI
class PiToolsTest(PiCase):
    """Check research and submission behavior through the real pi extension."""

    def test_research_floor_pieces_and_final(self):
        """Verify research floor pieces and final."""
        res, model = self.run_pi(
            [
                (
                    "submit_forecast",
                    {"forecasts": [{"k": "a", "probability": 0.1}], "rationale": "early"},
                ),  # 1
                ("read_page", {"url": "u"}),  # 2
                ("web_search", {"query": "q1"}),  # 3
                ("web_search", {"query": "q2"}),  # 4
                (
                    "submit_cells",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.2},
                            {"k": "zzz", "probability": 0.9},
                        ]
                    },
                ),  # 5
                (
                    "submit_forecast",
                    {"forecasts": [{"k": "zzz", "probability": 0.5}], "rationale": "r"},
                ),  # 6
                (
                    "submit_forecast",
                    {
                        "forecasts": [{"k": "b", "probability": 0.3}],
                        "rationale": "r",
                        "key_sources": ["s"],
                    },
                ),  # 7
                ("submit_forecast", {"forecasts": [], "rationale": "again"}),  # 8
                "done",  # 9
            ]
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(len(model.requests), 9)

        # 1: submitting before any research is refused
        self.assertIn(
            "submit_forecast is not offered yet: 2 more successful research call(s)",
            model.tool_result(2),
        )
        # 2: a failed read does not count toward the floor
        self.assertIn("page read failed", model.tool_result(3))
        # 5: only wanted keys are accepted from a piece
        piece = json.loads(model.tool_result(6))
        self.assertEqual(
            (piece["accepted"], piece["received"], piece["missing_first"]), (1, 1, ["b"])
        )
        # 6: unknown and missing keys get one warning, and nothing is written
        warn = json.loads(model.tool_result(7))
        self.assertEqual((warn["unknown_keys"], warn["missing_keys"]), (["zzz"], ["b"]))
        # 7: accepted; 8: a second final answer is refused
        self.assertTrue(json.loads(model.tool_result(8).split(" Done")[0])["ok"])
        self.assertIn("already called", model.tool_result(9))

        final = json.load(open(os.path.join(res["workdir"], "final.json")))
        self.assertEqual((final["research"], final["research_failed"]), (2, 1))
        self.assertEqual(final["answer"]["key_sources"], ["s"])
        self.assertEqual(
            [
                json.loads(line)["accepted"]
                for line in open(os.path.join(res["workdir"], "pieces.jsonl"))
            ],
            [1],
        )
        self.assertEqual(
            {k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.2, "b": 0.3}
        )
        self.assertEqual((res["usage"]["research_ok"], res["usage"]["research_failed"]), (2, 1))

    def test_no_research_offers_only_submit_forecast(self):
        """Verify no research offers only submit forecast."""
        res, model = self.run_pi(
            [
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.4},
                            {"k": "b", "probability": 0.6},
                        ],
                        "rationale": "r",
                    },
                ),
                "done",
            ],
            research_tools=(),
            min_research=0,
            cells_tool=False,
            tool_descriptions={"submit_forecast": "f"},
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        offered = {t["function"]["name"] for t in model.requests[0]["tools"]}
        self.assertEqual(
            offered, {"submit_forecast"}
        )  # no research tools, and none of pi's built-ins
        self.assertEqual(
            {k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.4, "b": 0.6}
        )
        self.assertEqual(
            model.last_user(1), "P"
        )  # the prompt, from stdin, as the first user message

    def test_numeric_forecast_schema_is_preserved(self):
        """Accept caller-defined numeric forecasts instead of requiring probabilities."""
        schema = {
            "type": "object",
            "required": ["date", "estimate"],
            "properties": {"date": {"type": "string"}, "estimate": {"type": "number"}},
        }
        result, model = self.run_pi(
            [
                (
                    "submit_forecast",
                    {"forecasts": [{"date": "2030-01-01", "estimate": 42.5}], "rationale": "r"},
                ),
                "done",
            ],
            cell_schema=schema,
            key_fields=["date"],
            allowed_keys=["2030-01-01"],
            research_tools=(),
            min_research=0,
            cells_tool=False,
            tool_descriptions={"submit_forecast": "Submit the numeric forecast."},
        )
        self.assertEqual(result["rc"], 0)
        self.assertEqual(result["cells"], {"2030-01-01": {"date": "2030-01-01", "estimate": 42.5}})
        sent = model.requests[0]["tools"][0]["function"]["parameters"]
        self.assertEqual(sent["properties"]["forecasts"]["items"]["required"], ["date", "estimate"])

    def test_a_prompt_past_the_argv_limit_reaches_the_model(self):
        """Verify a prompt past the argv limit reaches the model."""
        prompt = "Forecast.\n" + "".join(
            f"--- question {i} ---\nWill it happen?\n" for i in range(6000)
        )
        self.assertGreater(len(prompt.encode()), 131072)
        res, model = self.run_pi(
            [
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.4},
                            {"k": "b", "probability": 0.6},
                        ],
                        "rationale": "r",
                    },
                ),
                "done",
            ],
            prompt=prompt,
            research_tools=(),
            min_research=0,
            cells_tool=False,
            tool_descriptions={"submit_forecast": "f"},
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.last_user(1), prompt.strip())  # verbatim but for pi's trim
        self.assertEqual(open(os.path.join(res["workdir"], "prompt.txt")).read(), prompt)

    def test_a_system_prompt_past_the_argv_limit_reaches_the_model(self):
        """Verify a system prompt past the argv limit reaches the model."""
        system = "The survey.\n" + "".join(
            f"=== G{i:04d} ===\nWhat will it be?\n" for i in range(6000)
        )
        self.assertGreater(len(system.encode()), 131072)
        res, model = self.run_pi(
            [
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.4},
                            {"k": "b", "probability": 0.6},
                        ],
                        "rationale": "r",
                    },
                ),
                "done",
            ],
            system=system,
            research_tools=(),
            min_research=0,
            cells_tool=False,
            tool_descriptions={"submit_forecast": "f"},
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertTrue(
            model.text(1, "system", "developer").startswith(system)
        )  # pi appends its <cwd> section
        self.assertEqual(model.last_user(1), "P")


SEARCH_WITH_CAUSE = {
    "type": "object",
    "required": ["query", "cause"],
    "properties": {
        "query": {"type": "string"},
        "recent_days": {"type": "integer"},
        "cause": {"type": "string", "enum": ["ai", "bio"]},
    },
}


@NEEDS_PI
class LoopOptionsTest(PiCase):
    """Check the optional coverage, turn-limit, and debrief behavior."""

    def test_withheld_until_every_call_and_group_counts(self):
        """Verify withheld until every call and group counts."""
        res, model = self.run_pi(
            [
                ("read_page", {"url": "u"}),  # fails, still counts
                ("web_search", {"query": "q", "cause": "ai", "recent_days": 7}),
                ("web_search", {"query": "q", "cause": "bio"}),  # no recent_days: not coverage
                ("web_search", {"query": "q", "cause": "bio", "recent_days": 30}),
                (
                    "submit_forecast",
                    {"forecasts": [{"k": "a", "probability": 0.1}], "rationale": "r"},
                ),
                "done",
            ],
            min_research=4,
            floor_counts="returned",
            coverage=["ai", "bio"],
            withhold_final=True,
            warn_missing_keys=False,
            tool_parameters={"web_search": SEARCH_WITH_CAUSE},
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        for n in (1, 2, 3, 4):  # 4 returned (one failed) and both groups covered after call 4
            self.assertNotIn("submit_forecast", model.tools(n))
            self.assertNotIn("submit_cells", model.tools(n))
        self.assertTrue({"submit_forecast", "submit_cells"} <= model.tools(5))
        sent = next(t for t in model.requests[0]["tools"] if t["function"]["name"] == "web_search")
        self.assertEqual(
            sent["function"]["parameters"]["properties"]["cause"]["enum"], ["ai", "bio"]
        )
        final = json.load(open(os.path.join(res["workdir"], "final.json")))
        self.assertEqual(
            (final["research"], final["research_failed"], final["forced"]), (3, 1, False)
        )
        self.assertEqual(
            {k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.1}
        )  # no warning for b

    def test_turn_limit_offers_only_submission(self):
        """Verify turn limit offers only submission."""
        res, model = self.run_pi(
            [
                ("web_search", {"query": "q1"}),
                ("web_search", {"query": "q2"}),
                (
                    "submit_forecast",
                    {"forecasts": [{"k": "a", "probability": 0.2}], "rationale": "forced"},
                ),
                "done",
            ],
            min_research=5,
            withhold_final=True,
            warn_missing_keys=False,
            max_turns=3,
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.tools(3), {"submit_forecast"})
        self.assertTrue(
            model.last_user(3).startswith("Now call submit_forecast with your final answer.")
        )
        self.assertTrue(res["forced"])

    def test_turn_limit_ends_a_call_that_never_submits(self):
        """Verify turn limit ends a call that never submits."""
        res, model = self.run_pi(
            ["thinking", "still thinking", "no", "no", "no", "no"],
            min_research=1,
            withhold_final=True,
            max_turns=3,
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(len(model.requests), 5)  # the limit is max_turns + 2
        self.assertIn("Keep researching with the tools", model.last_user(2))
        self.assertEqual(model.tools(5), {"submit_forecast"})
        self.assertFalse(os.path.exists(os.path.join(res["workdir"], "final.json")))

    def test_debrief_merges_and_retries(self):
        """Verify debrief merges and retries."""
        validator = os.path.join(self.tmp.name, "validator.py")
        open(validator, "w").write(VALIDATOR)
        debrief = {
            "tool": {
                "name": "submit_debrief",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                },
            },
            "prompt": "Debrief now.",
            "retries": 1,
            "validator": {"file": validator, "function": "check", "kwargs": {"need": ["a", "b"]}},
        }
        res, model = self.run_pi(
            [
                ("web_search", {"query": "q1"}),
                ("web_search", {"query": "q2"}),
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.1},
                            {"k": "b", "probability": 0.2},
                        ],
                        "rationale": "r",
                    },
                ),
                ("submit_debrief", {"a": "why a"}),
                ("submit_debrief", {"b": "why b"}),
                "done",
            ],
            debrief=debrief,
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertNotIn("submit_debrief", model.tools(1))
        self.assertEqual(model.tools(4), {"submit_debrief"})
        self.assertEqual(model.last_user(4), "Debrief now.")
        self.assertIn("b: missing", model.tool_result(5))
        self.assertIn("fixing: b: missing", model.last_user(5))
        self.assertEqual(
            res["debrief"],
            {
                "result": {"a": "why a", "b": "why b"},
                "complete": True,
                "attempts": 2,
                "problems": [],
                "error": None,
            },
        )

    def test_debrief_asks_again_after_a_refusal(self):
        # A refused turn ends pi's run unless something asks again.
        """Verify debrief asks again after a refusal."""
        debrief = {
            "tool": {
                "name": "submit_debrief",
                "description": "d",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
            "prompt": "Debrief now.",
            "retries": 1,
        }
        res, model = self.run_pi(
            [
                ("web_search", {"query": "q1"}),
                ("web_search", {"query": "q2"}),
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.1},
                            {"k": "b", "probability": 0.2},
                        ],
                        "rationale": "r",
                    },
                ),
                REFUSAL,
                ("submit_debrief", {"a": "why a"}),
                "done",
            ],
            debrief=debrief,
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.last_user(5), "Now call submit_debrief with the debrief.")
        self.assertEqual(model.tools(5), {"submit_debrief"})
        self.assertEqual(
            res["debrief"],
            {
                "result": {"a": "why a"},
                "complete": True,
                "attempts": 2,
                "problems": [],
                "error": None,
            },
        )

    def test_validator_exceptions_are_retried_and_never_count_as_valid(self):
        """Recover from a validator exception or stop incomplete after bounded retries."""
        validator = os.path.join(self.tmp.name, "fallible_validator.py")
        with open(validator, "w") as handle:
            handle.write(
                "from pathlib import Path\n"
                "def check(value, marker, recovers):\n"
                "    marker = Path(marker)\n"
                "    first = not marker.exists()\n"
                "    marker.touch()\n"
                "    if first or not recovers:\n"
                "        raise RuntimeError('validator unavailable')\n"
                "    return []\n"
            )
        for recovers in (True, False):
            with self.subTest(recovers=recovers):
                debrief = {
                    "tool": {
                        "name": "submit_debrief",
                        "description": "Explain the forecast.",
                        "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
                    },
                    "prompt": "Debrief now.",
                    "retries": 1,
                    "validator": {
                        "file": validator,
                        "function": "check",
                        "kwargs": {
                            "marker": os.path.join(self.tmp.name, f"validated-{recovers}"),
                            "recovers": recovers,
                        },
                    },
                }
                result, model = self.run_pi(
                    [
                        (
                            "submit_forecast",
                            {"forecasts": [{"k": "a", "probability": 0.2}], "rationale": "r"},
                        ),
                        ("submit_debrief", {"a": "why a"}),
                        ("submit_debrief", {"a": "why a"}),
                        "done",
                    ],
                    min_research=0,
                    warn_missing_keys=False,
                    debrief=debrief,
                )
                self.assertEqual(result["rc"], 0)
                rejected = json.loads(model.tool_result(3))
                self.assertFalse(rejected["accepted"])
                self.assertIn("validator failed:", rejected["problems"][0])
                self.assertEqual(result["debrief"]["attempts"], 2)
                self.assertEqual(result["debrief"]["complete"], recovers)
                self.assertEqual(len(model.requests), 4)
                if recovers:
                    self.assertIsNone(result["debrief"]["error"])
                    self.assertEqual(result["debrief"]["problems"], [])
                else:
                    self.assertIn("validator failed:", result["debrief"]["error"])
                    self.assertEqual(result["debrief"]["problems"], [result["debrief"]["error"]])


@NEEDS_PI
class UrlGuardPassthroughTest(PiCase):
    """Check per-call URL exclusions through the real Python tool process."""

    def test_env_reaches_read_page(self):
        """Verify env reaches read page."""
        harness.TOOL_CLI = self.real_cli  # the real CLI; with no Tavily key nothing is fetched
        res, model = self.run_pi(
            [
                ("read_page", {"url": "https://example.com/paper.pdf"}),  # 1: blocked by the guard
                (
                    "read_page",
                    {"url": "https://example.org/other"},
                ),  # 2: passes it, stops at the key
                (
                    "submit_forecast",
                    {
                        "forecasts": [
                            {"k": "a", "probability": 0.5},
                            {"k": "b", "probability": 0.5},
                        ],
                        "rationale": "r",
                    },
                ),
                "done",
            ],
            min_research=0,
            env={"FORECASTER_BLOCK_URL_PATTERNS": "example\\.com", "TAVILY_API_KEY": ""},
        )
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertIn("page read failed: page unavailable", model.tool_result(2))
        self.assertIn("no TAVILY_API_KEY set", model.tool_result(3))
        self.assertEqual((res["usage"]["research_ok"], res["usage"]["research_failed"]), (0, 2))
