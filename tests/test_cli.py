"""Tests for the command line interface."""
from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from typer.testing import CliRunner

from explocal.cli import app


def test_cli_train_predict(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "default.yml"
    data_src = root / "examples" / "data" / "example_sequences.csv"
    data_dst = tmp_path / "examples" / "data" / "example_sequences.csv"
    data_dst.parent.mkdir(parents=True, exist_ok=True)
    copyfile(data_src, data_dst)
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        result = runner.invoke(app, ["train", "--config", str(cfg_path)])
        assert result.exit_code == 0
        result = runner.invoke(
            app, ["predict", "MKTIIALSYIFCLVFADYKDDDDK", "--model-path", "model.pkl"]
        )
        assert result.exit_code == 0
