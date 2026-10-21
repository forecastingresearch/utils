#!/usr/bin/env python3
"""Run research tools as a Python module for the pi extension.

``python -m utils.llm.agentic.tool_cli web_search '{"query": ...}'`` prints
the tool result as JSON. Keys come from the environment the harness passes down.

`tool_cli.py validate` (arguments on stdin: {file, function, kwargs, value})
runs a caller's debrief validator, `function(value, **kwargs)` from the
Python file `file`, and prints {"problems": [...]}.
"""

import importlib.util
import json
import random
import sys
import time

from . import tools

FN = {"web_search": tools.web_search, "read_page": tools.read_page}
# Tavily answers 429 when many agents search at once (2026-10-06 practice
# run: 32 concurrent calls, ~1,700 failures). Back off and retry here, not in
# tools.py, so the direct tool functions retain their single-request behavior.
BACKOFF = [2, 5, 10, 20, 40]


def rate_limited(res):
    """Return whether a tool result reports an HTTP 429 error."""
    return isinstance(res, dict) and "429" in str(res.get("error") or "")


def call(fn, args, sleep=time.sleep):
    """Retry rate-limited tool calls with bounded exponential backoff."""
    res = fn(**args)
    for wait in BACKOFF:
        if not rate_limited(res):
            break
        sleep(wait * random.uniform(0.75, 1.25))
        res = fn(**args)
    return res


def validate(file, function, value, kwargs=None):
    """Run a caller-supplied debrief validator and return its problems."""
    spec = importlib.util.spec_from_file_location("forecaster_validator", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "problems": [str(p) for p in (getattr(module, function)(value, **(kwargs or {})) or [])]
    }


def main():
    """Dispatch one tool or validator call and print its JSON result."""
    if sys.argv[1] == "validate":
        print(json.dumps(validate(**json.load(sys.stdin))))
        return
    tool, args = sys.argv[1], json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")
    if tool not in FN:
        sys.exit(f"unknown tool {tool!r}")
    print(json.dumps(call(FN[tool], args)))


if __name__ == "__main__":
    main()
