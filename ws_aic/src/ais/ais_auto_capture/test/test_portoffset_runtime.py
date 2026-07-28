"""PortOffset collector 외부 프로세스 실행 명령 회귀 테스트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

AUTO_CAPTURE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_CAPTURE_DIR))

from portoffset_randomization import runtime


def test_start_gazebo_disables_distrobox_tty(monkeypatch) -> None:
    """Distrobox가 호스트 터미널 줄바꿈 설정을 변경하지 못하게 한다."""
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    args = argparse.Namespace(
        distrobox="aic_eval",
        headless=False,
        rootless_distrobox=True,
    )

    result = runtime.start_gazebo(
        args,
        Path("/tmp/engine_config.yaml"),
        None,
        "test-run",
    )

    assert result is sentinel
    assert captured["command"][:3] == ["distrobox", "enter", "--no-tty"]
    command = " ".join(captured["command"])
    assert "launch_rviz:=false" in command
    assert "gazebo_gui:=false" not in command
    kwargs = captured["kwargs"]
    assert kwargs["stderr"] is runtime.subprocess.STDOUT
    assert kwargs["start_new_session"] is True
