import subprocess
import sys

import pytest

from globerode import __version__
from globerode.cli import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"globerode {__version__}"


def test_no_args_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: globerode" in capsys.readouterr().out


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "globerode.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.startswith("globerode ")
