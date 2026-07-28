"""PortOffset collector의 trial별 rosbag 회귀 테스트."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

AUTO_CAPTURE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_CAPTURE_DIR))

from portoffset_randomization.cli import build_parser
from portoffset_randomization.lifecycle import OwnedProcessGroup
from portoffset_randomization.runtime import (
    _forward_process_output,
    rosbag_output_dir,
    validate_rosbag,
)
import collect_portoffset_randomization_data as collector


def _write_metadata(path: Path, message_count: int) -> None:
    """테스트용 rosbag2 metadata를 기록한다."""
    path.write_text(
        "rosbag2_bagfile_information:\n"
        f"  message_count: {message_count}\n",
        encoding="utf-8",
    )


def test_rosbag_cli_is_opt_in_and_uses_per_trial_path(tmp_path: Path) -> None:
    """rosbag은 opt-in이며 version/run/trial별 경로를 격리한다."""
    parser = build_parser()
    default_args = parser.parse_args([])
    assert default_args.record_rosbag is False
    assert default_args.sync_wait_timeout_s == 1.0

    args = parser.parse_args(
        [
            "--record-rosbag",
            "true",
            "--rosbag-output-dir",
            str(tmp_path),
            "--dataset-version",
            "0726-001",
        ]
    )
    output = rosbag_output_dir(
        args,
        run_id="run-123",
        index=2,
        task_id="portoffset_sc_0002_rail0",
    )
    assert args.record_rosbag is True
    assert output == (
        tmp_path
        / "0726-001"
        / "run-123"
        / "trial_0002_portoffset_sc_0002_rail0"
    )


def test_validate_rosbag_accepts_finalized_nonempty_mcap(tmp_path: Path) -> None:
    """메시지가 있고 양끝 magic이 있는 MCAP만 완료로 인정한다."""
    _write_metadata(tmp_path / "metadata.yaml", message_count=42)
    magic = b"\x89MCAP0\r\n"
    (tmp_path / "trial_0.mcap").write_bytes(magic + b"payload" + magic)

    valid, detail = validate_rosbag(tmp_path)

    assert valid
    assert "messages=42" in detail


def test_validate_rosbag_rejects_unfinalized_mcap(tmp_path: Path) -> None:
    """Footer magic이 없는 중단 파일을 다음 trial 진행 전에 거부한다."""
    _write_metadata(tmp_path / "metadata.yaml", message_count=42)
    magic = b"\x89MCAP0\r\n"
    (tmp_path / "trial_0.mcap").write_bytes(magic + b"partial")

    valid, detail = validate_rosbag(tmp_path)

    assert not valid
    assert "not finalized" in detail


def test_validate_rosbag_rejects_zero_messages(tmp_path: Path) -> None:
    """형식이 완결돼도 실제 메시지가 없으면 성공으로 표시하지 않는다."""
    _write_metadata(tmp_path / "metadata.yaml", message_count=0)
    magic = b"\x89MCAP0\r\n"
    (tmp_path / "trial_0.mcap").write_bytes(magic + b"payload" + magic)

    valid, detail = validate_rosbag(tmp_path)

    assert not valid
    assert detail == "message_count is zero"


def test_trial_finalizes_rosbag_before_stopping_simulator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """trial 종료 시 recorder 완료 검증이 Gazebo teardown보다 선행한다."""
    args = build_parser().parse_args(
        [
            "--record-rosbag",
            "true",
            "--rosbag-output-dir",
            str(tmp_path),
            "--policy-start-wait-s",
            "0",
            "--post-summary-wait-s",
            "0",
            "--between-trial-wait-s",
            "0",
        ]
    )
    ctx = collector.RunContext(
        args=args,
        run_id="run-order",
        run_dir=tmp_path / "runtime",
        stop_file=tmp_path / "stop",
        registry_path=tmp_path / "registry.json",
    )
    order: list[str] = []
    config_path = tmp_path / "engine.yaml"
    scenario_path = tmp_path / "scenario.json"
    simulator_group = OwnedProcessGroup(None, 10, "simulator", ctx.run_id, str(config_path))

    monkeypatch.setattr(
        collector,
        "_prepare_trial",
        lambda *_: (
            "portoffset_sc_0000_rail0",
            {},
            config_path,
            scenario_path,
            None,
            {},
        ),
    )
    monkeypatch.setattr(collector, "known_episode_summaries", lambda: set())
    monkeypatch.setattr(
        collector,
        "start_gazebo",
        lambda *_: order.append("start_gazebo") or SimpleNamespace(),
    )
    monkeypatch.setattr(collector, "_persist_groups", lambda *_: None)
    monkeypatch.setattr(
        collector,
        "_register_inner_simulator_groups",
        lambda *_: [simulator_group],
    )
    monkeypatch.setattr(
        collector,
        "start_rosbag",
        lambda *_, output_dir, **__: order.append("start_rosbag")
        or SimpleNamespace(proc=SimpleNamespace(), output_dir=output_dir),
    )
    monkeypatch.setattr(
        collector,
        "wait_for_rosbag_start",
        lambda *_: order.append("rosbag_ready") or True,
    )
    monkeypatch.setattr(
        collector,
        "start_policy",
        lambda *_, **__: order.append("start_policy") or SimpleNamespace(),
    )
    monkeypatch.setattr(collector, "wait_for_trial_summary", lambda *_: True)
    monkeypatch.setattr(
        collector,
        "stop_policy",
        lambda *_: order.append("stop_policy") or True,
    )
    monkeypatch.setattr(
        collector,
        "stop_rosbag",
        lambda *_: order.append("stop_rosbag") or True,
    )
    monkeypatch.setattr(
        collector,
        "terminate_owned_group",
        lambda *_args, **_kwargs: order.append("stop_simulator") or True,
    )

    next_pgid = iter(range(20, 30))

    def register_group(proc, *, kind, run_id, marker):
        return OwnedProcessGroup(None, next(next_pgid), kind, run_id, marker)

    monkeypatch.setattr(collector, "register_owned_group", register_group)

    collector._run_trial(ctx, 0, object())

    assert order.index("start_gazebo") < order.index("start_rosbag")
    assert order.index("rosbag_ready") < order.index("start_policy")
    assert order.index("stop_policy") < order.index("stop_rosbag")
    assert order.index("stop_rosbag") < order.index("stop_simulator")


def test_policy_color_log_is_preserved_but_callback_receives_plain_text(
    monkeypatch,
    capsys,
) -> None:
    """policy의 ANSI 경고는 화면에 유지하고 lifecycle callback에는 제거해 전달한다."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    received: list[str] = []
    proc = SimpleNamespace(stdout=["\033[1m\033[33mwarning\033[0m\n"])

    _forward_process_output(proc, received.append)

    assert received == ["warning"]
    assert "\033[33mwarning\033[0m" in capsys.readouterr().out
