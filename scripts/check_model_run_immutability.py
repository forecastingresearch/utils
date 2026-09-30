"""Verify that published LLM model runs never change.

`model_run_key` is the immutable benchmark identity of one exact model plus
option set. Downstream systems (benchmark result files, the data warehouse's
`dim_model_run`) key on it, so once a run has been merged to `main` its
identity must never be edited or deleted: a new option set is a NEW run with
the next variant number, never an edit of an existing one.

This script enforces that in CI by comparing two dumps of the registry, one
from the pull request's base branch and one from the PR head:

    python scripts/check_model_run_immutability.py dump  > base.ndjson   # on base
    python scripts/check_model_run_immutability.py dump  > pr.ndjson     # on PR
    python scripts/check_model_run_immutability.py check base.ndjson pr.ndjson

`check` fails when a run present on the base is missing from the PR, or is
present with a different `model_key` or `options`. Adding runs is always
allowed. Everything else is deliberately NOT compared: `slug` and `active`
are documented as changeable (AGENTS.md), and release dates plus the
`models_dev_*` metadata are resolved from a refreshable snapshot, so none of
them is a stable part of a run's identity.

`dump` imports `utils.llm.model_runs` from whichever interpreter runs it and
uses only the long-stable public attributes (`MODEL_RUNS`, `model_run_key`,
`model_key`, `options`), so the PR's copy of this script can dump the BASE
checkout too: run it by file path with a Python whose site-packages holds the
base version of the package (see .github/workflows/model-run-immutability.yml).
Running by path keeps the current directory off `sys.path`, so `utils` resolves
to the installed package, not to a source tree that happens to be the cwd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# The identity of a run. Everything else on a ModelRun/Model may change.
IMMUTABLE_FIELDS: tuple[str, ...] = ("model_run_key", "model_key", "options")


def canonical_json(value: Any) -> str:
    """Serialize `value` deterministically so equal options compare equal as text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def fingerprint(record: dict[str, Any]) -> str:
    """Return a SHA256 over the immutable fields of one dumped record."""
    payload = canonical_json({field: record.get(field) for field in IMMUTABLE_FIELDS})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registry_records() -> list[dict[str, Any]]:
    """Return one record per registered model run, sorted by `model_run_key`."""
    from utils.llm import model_runs  # imported lazily: `check` needs no registry

    records = [
        {
            "model_run_key": run.model_run_key,
            "model_key": run.model_key,
            "options": run.options,
        }
        for run in model_runs.MODEL_RUNS
    ]
    for record in records:
        record["fingerprint"] = fingerprint(record)
    return sorted(records, key=lambda record: record["model_run_key"])


def dump_ndjson(records: Iterable[dict[str, Any]]) -> str:
    """Render records as newline-delimited canonical JSON, one run per line."""
    return "".join(f"{canonical_json(record)}\n" for record in records)


def load_ndjson(path: Path) -> dict[str, dict[str, Any]]:
    """Load a dump into a mapping keyed by `model_run_key`.

    Rejects an empty dump: an import failure upstream must not masquerade as
    "no runs on the base branch, nothing to protect".
    """
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["model_run_key"]
        if key in records:
            raise ValueError(f"{path}:{line_number}: duplicate model_run_key {key}")
        records[key] = record
    if not records:
        raise ValueError(f"{path}: dump contains no model runs")
    return records


@dataclass(frozen=True)
class Violation:
    """One published run that the candidate registry removed or altered."""

    model_run_key: str
    kind: str  # "removed" | "changed"
    detail: str

    def __str__(self) -> str:
        """Render as one report line."""
        return f"[{self.kind}] {self.model_run_key}: {self.detail}"


def find_violations(
    base: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[Violation]:
    """Return every base run that is missing from, or altered in, the candidate.

    Runs only present in the candidate are additions and never a violation.
    """
    violations: list[Violation] = []
    for key in sorted(base):
        if key not in candidate:
            violations.append(Violation(key, "removed", "present on base, missing from PR"))
            continue
        for field in IMMUTABLE_FIELDS:
            before = canonical_json(base[key].get(field))
            after = canonical_json(candidate[key].get(field))
            if before != after:
                violations.append(
                    Violation(key, "changed", f"{field} was {before} and is now {after}")
                )
    return violations


def check(base_path: Path, candidate_path: Path) -> int:
    """Compare two dumps, print a report, and return the process exit code."""
    base = load_ndjson(base_path)
    candidate = load_ndjson(candidate_path)
    violations = find_violations(base, candidate)
    added = sorted(set(candidate) - set(base))

    print(f"Base runs verified: {len(base)}")
    print(f"Runs added in PR:   {len(added)}")
    for key in added:
        print(f"  + {key}")

    if not violations:
        print("OK: every published model run is unchanged.")
        return 0

    print("")
    print(f"FAIL: {len(violations)} published model run(s) were removed or changed.")
    print("model_run_key is immutable once merged: declare a new run with the next")
    print("variant number instead of editing or deleting an existing one.")
    for violation in violations:
        print(f"  {violation}")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to `dump` or `check`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser(
        "dump", help="write the registry's immutable fields as NDJSON"
    )
    dump_parser.add_argument(
        "--output", type=Path, default=None, help="write to this path instead of stdout"
    )

    check_parser = subparsers.add_parser("check", help="compare a base dump against a PR dump")
    check_parser.add_argument("base", type=Path, help="dump taken on the PR's base branch")
    check_parser.add_argument("candidate", type=Path, help="dump taken on the PR head")

    args = parser.parse_args(argv)
    if args.command == "dump":
        text = dump_ndjson(registry_records())
        if args.output is None:
            sys.stdout.write(text)
        else:
            args.output.write_text(text)
        return 0
    return check(args.base, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
