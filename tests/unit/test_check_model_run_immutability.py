"""Tests for the CI model-run immutability check."""

import json
from pathlib import Path

from scripts import check_model_run_immutability as immutability
from utils.llm import model_runs


def _dump(records: list[dict]) -> dict[str, dict]:
    return {record["model_run_key"]: record for record in records}


BASE = [
    {
        "model_run_key": "a-run-variant-01",
        "model_key": "a",
        "options": {"max_tokens": 1024, "temperature": 0},
    },
    {"model_run_key": "b-run-variant-01", "model_key": "b", "options": {}},
]


# --- find_violations -----------------------------------------------------------


def test_identical_registries_have_no_violations() -> None:
    """A PR that touches nothing passes."""
    assert immutability.find_violations(_dump(BASE), _dump(BASE)) == []


def test_adding_a_run_is_allowed() -> None:
    """New variants are the sanctioned way to change options."""
    candidate = BASE + [
        {"model_run_key": "a-run-variant-02", "model_key": "a", "options": {"max_tokens": 2048}}
    ]
    assert immutability.find_violations(_dump(BASE), _dump(candidate)) == []


def test_removing_a_published_run_is_a_violation() -> None:
    """Deleting a run breaks every downstream reference to its key."""
    violations = immutability.find_violations(_dump(BASE), _dump(BASE[:1]))

    assert [(v.model_run_key, v.kind) for v in violations] == [("b-run-variant-01", "removed")]


def test_changing_options_is_a_violation_and_names_the_field() -> None:
    """Editing options in place is exactly what the check exists to catch."""
    candidate = [dict(BASE[0], options={"max_tokens": 1024, "temperature": 0.5}), BASE[1]]

    violations = immutability.find_violations(_dump(BASE), _dump(candidate))

    assert len(violations) == 1
    assert violations[0].kind == "changed"
    assert violations[0].model_run_key == "a-run-variant-01"
    assert "options" in violations[0].detail
    assert '"temperature":0.5' in violations[0].detail


def test_changing_model_key_is_a_violation() -> None:
    """Re-pointing a run at another base model changes its identity."""
    candidate = [dict(BASE[0], model_key="a-v2"), BASE[1]]

    violations = immutability.find_violations(_dump(BASE), _dump(candidate))

    assert [(v.model_run_key, v.kind) for v in violations] == [("a-run-variant-01", "changed")]
    assert "model_key" in violations[0].detail


def test_option_key_order_does_not_matter() -> None:
    """Only the option VALUES are identity; dict ordering is a serialization detail."""
    reordered = [dict(BASE[0], options={"temperature": 0, "max_tokens": 1024}), BASE[1]]

    assert immutability.find_violations(_dump(BASE), _dump(reordered)) == []


# --- dump / load ---------------------------------------------------------------


def test_registry_dump_covers_every_run_sorted_with_only_immutable_fields() -> None:
    """The dump is the full registry, ordered for stable diffs, nothing mutable in it."""
    records = immutability.registry_records()

    assert [r["model_run_key"] for r in records] == sorted(
        run.model_run_key for run in model_runs.MODEL_RUNS
    )
    assert len(records) == len(model_runs.MODEL_RUNS)
    assert set(records[0]) == set(immutability.IMMUTABLE_FIELDS) | {"fingerprint"}


def test_dump_round_trips_through_load(tmp_path: Path) -> None:
    """What `dump` writes, `load_ndjson` reads back unchanged."""
    records = immutability.registry_records()
    path = tmp_path / "dump.ndjson"
    path.write_text(immutability.dump_ndjson(records))

    loaded = immutability.load_ndjson(path)

    assert loaded == {r["model_run_key"]: r for r in records}
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_load_rejects_empty_dump(tmp_path: Path) -> None:
    """An empty base dump must fail loudly, not pass as 'nothing to protect'."""
    path = tmp_path / "empty.ndjson"
    path.write_text("\n")

    try:
        immutability.load_ndjson(path)
    except ValueError as exc:
        assert "no model runs" in str(exc)
    else:
        raise AssertionError("expected ValueError for an empty dump")


def test_fingerprint_ignores_mutable_fields_and_option_order() -> None:
    """Same identity, same fingerprint, whatever else rides along on the record."""
    base = {"model_run_key": "k", "model_key": "m", "options": {"a": 1, "b": 2}}
    same = {"model_run_key": "k", "model_key": "m", "options": {"b": 2, "a": 1}, "slug": "x"}
    different = {"model_run_key": "k", "model_key": "m", "options": {"a": 1, "b": 3}}

    assert immutability.fingerprint(base) == immutability.fingerprint(same)
    assert immutability.fingerprint(base) != immutability.fingerprint(different)


# --- CLI -----------------------------------------------------------------------


def test_cli_check_exit_codes(tmp_path: Path, capsys) -> None:
    """`check` returns 0 when unchanged and 1 with a readable report on a violation."""
    base_path = tmp_path / "base.ndjson"
    base_path.write_text(immutability.dump_ndjson(BASE))
    ok_path = tmp_path / "ok.ndjson"
    ok_path.write_text(
        immutability.dump_ndjson(BASE + [dict(BASE[0], model_run_key="a-run-variant-02")])
    )
    bad_path = tmp_path / "bad.ndjson"
    bad_path.write_text(immutability.dump_ndjson([dict(BASE[0], options={}), BASE[1]]))

    assert immutability.main(["check", str(base_path), str(ok_path)]) == 0
    assert "Runs added in PR:   1" in capsys.readouterr().out

    assert immutability.main(["check", str(base_path), str(bad_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "[changed] a-run-variant-01" in out


def test_cli_dump_writes_registry(tmp_path: Path) -> None:
    """`dump --output` writes the same document `registry_records` describes."""
    out = tmp_path / "registry.ndjson"

    assert immutability.main(["dump", "--output", str(out)]) == 0
    assert out.read_text() == immutability.dump_ndjson(immutability.registry_records())
