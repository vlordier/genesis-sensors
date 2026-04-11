from __future__ import annotations

from types import SimpleNamespace

import pytest

from genesis_sensors import __version__
from genesis_sensors import cli


def test_cli_help_lists_available_scenes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "drone" in help_text
    assert "perception" in help_text
    assert "franka" in help_text
    assert "go2" in help_text


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


def test_cli_runs_selected_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyRig:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.step_times: list[float] = []

        def reset(self) -> None:
            self.reset_calls += 1

        def step(self, sim_time: float) -> None:
            self.step_times.append(sim_time)

    class _DummyScene:
        def __init__(self) -> None:
            self.step_calls = 0

        def step(self) -> None:
            self.step_calls += 1

    demo = SimpleNamespace(scene=_DummyScene(), rig=_DummyRig(), controller=None)

    def _builder(**_: object) -> SimpleNamespace:
        return demo

    monkeypatch.setattr(
        cli,
        "_get_scene_builders",
        lambda: {"drone": _builder, "perception": _builder, "franka": _builder, "go2": _builder},
    )

    cli.main(["drone", "--steps", "3", "--dt", "0.02"])

    assert demo.rig.reset_calls == 1
    assert demo.scene.step_calls == 3
    assert demo.rig.step_times == [0.0, 0.02, 0.04]
