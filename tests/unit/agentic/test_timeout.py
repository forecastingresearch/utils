"""Check that a timed-out forecast stops its local tool subprocesses."""

import os
import signal
import sys
import textwrap
import time

import pytest

from utils.llm.agentic import harness


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup requires POSIX.")
def test_timeout_stops_tool_child_before_returning(tmp_path):
    """A tool child cannot write results after its forecast has timed out."""
    child = tmp_path / "tool.py"
    child.write_text(textwrap.dedent("""\
            import os
            import time
            from pathlib import Path

            root = Path(os.environ["FRI_TEST_CHILD_DIR"])
            (root / "child.pid").write_text(str(os.getpid()))
            while not (root / "release").exists():
                time.sleep(0.01)
            (root / "late-result").write_text("The tool survived its forecast.")
            """))
    pi = tmp_path / "pi"
    pi.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
            import os
            import subprocess
            import sys
            import time
            from pathlib import Path

            if "--version" in sys.argv:
                print("0.99.2")
                raise SystemExit(0)
            root = Path(os.environ["FRI_TEST_CHILD_DIR"])
            (root / "pi.pid").write_text(str(os.getpid()))
            subprocess.Popen([sys.executable, str(root / "tool.py")])
            while True:
                time.sleep(0.1)
            """))
    pi.chmod(0o755)
    child_pid = None
    try:
        result = harness.run(
            model="fake/scripted",
            system="Forecast.",
            prompt="Forecast q1.",
            workdir=str(tmp_path / "calls"),
            cell_schema={"type": "object"},
            key_fields=["id"],
            allowed_keys=["q1"],
            tool_descriptions={"submit_forecast": "Submit."},
            research_tools=(),
            cells_tool=False,
            pi=str(pi),
            timeout=1,
            env={"FRI_TEST_CHILD_DIR": str(tmp_path)},
        )
        child_pid = int((tmp_path / "child.pid").read_text())
        assert result["rc"] == "timeout"
        assert result["cells"] == {}
        pi_pid = int((tmp_path / "pi.pid").read_text())
        with pytest.raises(ChildProcessError):
            os.waitpid(pi_pid, os.WNOHANG)

        (tmp_path / "release").touch()
        marker = tmp_path / "late-result"
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not marker.exists():
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        assert not marker.exists(), "The tool child continued after the forecast timed out."
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
