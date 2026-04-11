from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from genesis_sensors import __version__
from genesis_sensors import cli


def test_cli_help_lists_available_scenes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "genesis-sensors" in help_text
    assert "drone" in help_text
    assert "perception" in help_text
    assert "franka" in help_text
    assert "go2" in help_text
    assert "synthetic" in help_text
    assert "Examples:" in help_text
    assert "--steps" in help_text
    assert "--summary-every" in help_text
    assert "--list-scenes" in help_text
    assert "--dry-run" in help_text
    assert "--summary-format" in help_text


def test_cli_version_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_surfaces_runtime_dependency_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error() -> dict[str, object]:
        raise ImportError("missing genesis")

    monkeypatch.setattr(cli, "_get_scene_builders", _raise_import_error)

    with pytest.raises(SystemExit, match="Install torch"):
        cli.main(["drone"])


@pytest.mark.parametrize(
    "argv",
    [["drone", "--steps", "0"], ["drone", "--dt", "0"], ["drone", "--summary-every", "-1"]],
)
def test_cli_rejects_non_positive_numeric_args(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2
    error_text = capsys.readouterr().err
    if "--summary-every" in argv:
        assert "expected a non-negative" in error_text
    else:
        assert "expected a positive" in error_text


def test_cli_lists_built_in_scenes_without_runtime(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-scenes"])

    output = capsys.readouterr().out
    assert "synthetic" in output
    assert "headless" in output
    assert "navigation" in output


def test_cli_runs_selected_builder(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class _DummyRig:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.step_times: list[float] = []

        def reset(self) -> None:
            self.reset_calls += 1

        def step(self, sim_time: float) -> dict[str, float]:
            self.step_times.append(sim_time)
            return {"imu": sim_time, "gnss": sim_time}

    class _DummyScene:
        def __init__(self) -> None:
            self.step_calls = 0

        def step(self) -> None:
            self.step_calls += 1

    demo = SimpleNamespace(name="synthetic", scene=_DummyScene(), rig=_DummyRig(), controller=None)
    captured_kwargs: dict[str, object] = {}

    def _builder(**kwargs: object) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return demo

    monkeypatch.setattr(
        cli,
        "_get_scene_builders",
        lambda: {
            "drone": _builder,
            "perception": _builder,
            "franka": _builder,
            "go2": _builder,
            "synthetic": _builder,
        },
    )

    cli.main(["synthetic", "--steps", "3", "--dt", "0.02", "--summary-every", "1", "--gpu", "--vis"])

    assert captured_kwargs["dt"] == pytest.approx(0.02)
    assert captured_kwargs["show_viewer"] is True
    assert captured_kwargs["use_gpu"] is True
    assert demo.rig.reset_calls == 1
    assert demo.scene.step_calls == 3
    assert demo.rig.step_times == [0.0, 0.02, 0.04]
    assert "[synthetic] step=000 sensors=imu, gnss" in capsys.readouterr().out


def test_cli_builtin_synthetic_demo_runs_headless(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["synthetic", "--steps", "2", "--summary-every", "1"])

    output = capsys.readouterr().out
    assert "[synthetic] step=000" in output
    assert "imu" in output


def test_cli_builtin_synthetic_dry_run_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["synthetic", "--dry-run", "--summary-format", "json"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["scene"] == "synthetic"
    assert summary["rig"]["profile"] == "synthetic_multimodal"
    assert summary["rig"]["metadata"]["dt"] == pytest.approx(0.05)
    assert summary["rig"]["sensor_count"] >= 10
