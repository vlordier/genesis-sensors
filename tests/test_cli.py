from __future__ import annotations

import json
from pathlib import Path
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
    assert "--list-phases" in help_text
    assert "--list-profiles" in help_text
    assert "--describe-scene" in help_text
    assert "--search" in help_text
    assert "--dry-run" in help_text
    assert "--summary-format" in help_text
    assert "--profile" in help_text
    assert "--headless-only" in help_text
    assert "--write-summary" in help_text


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


def test_cli_lists_filtered_scenes_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-scenes", "--profile", "synthetic_multimodal", "--headless-only", "--summary-format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["name"] == "synthetic"
    assert payload[0]["requires_runtime"] is False


def test_cli_lists_synthetic_phases_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-phases", "--summary-format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["phase"] == "takeoff"
    assert payload[-1]["phase"] == "signal_recovery"
    assert payload[0]["duration"] == pytest.approx(0.2)


def test_cli_lists_profile_catalog_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-profiles", "--summary-format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert any(item["profile"] == "synthetic_multimodal" for item in payload)
    assert all(item["scene_count"] >= 1 for item in payload)


def test_cli_describes_single_scene_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--describe-scene", "synthetic", "--summary-format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "synthetic"
    assert payload["runtime_mode"] == "headless"
    assert payload["profile"] == "synthetic_multimodal"


def test_cli_search_filters_scene_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-scenes", "--search", "syn", "--summary-format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["name"] == "synthetic"


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


def test_cli_can_write_dry_run_summary_to_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "nested" / "synthetic.json"

    cli.main(["synthetic", "--dry-run", "--summary-format", "json", "--write-summary", str(summary_path)])

    summary = json.loads(capsys.readouterr().out)
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
