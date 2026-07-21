"""PortOffset collector의 PGID 소유권 및 crash recovery 회귀 테스트."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

AUTO_CAPTURE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_CAPTURE_DIR))

from portoffset_randomization import lifecycle


def _lifecycle_args() -> argparse.Namespace:
    """테스트가 빠르게 signal 단계를 진행하도록 짧은 유예시간을 만든다."""
    return argparse.Namespace(
        sim_sigint_grace_s=0.2,
        sim_cleanup_grace_s=0.2,
        sim_sigkill_grace_s=0.2,
    )


def _start_sleep(*, run_id: str | None = None) -> subprocess.Popen:
    """선택적으로 run marker를 가진 격리된 테스트 프로세스 그룹을 시작한다."""
    env = os.environ.copy()
    if run_id is not None:
        env[lifecycle.RUN_MARKER_ENV] = run_id
    return subprocess.Popen(["sleep", "60"], env=env, start_new_session=True)


def _start_marked_sleep(marker: str, run_id: str) -> subprocess.Popen:
    """run marker 환경과 config marker 명령행을 모두 가진 테스트 그룹을 시작한다."""
    env = os.environ.copy()
    env[lifecycle.RUN_MARKER_ENV] = run_id
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", marker],
        env=env,
        start_new_session=True,
    )


def _force_stop(proc: subprocess.Popen) -> None:
    """테스트 실패 여부와 무관하게 임시 프로세스 그룹을 회수한다."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=2.0)


def _redirect_runtime(monkeypatch, tmp_path: Path) -> None:
    """stale cleanup 테스트가 실제 runtime 디렉터리를 건드리지 않게 격리한다."""
    monkeypatch.setattr(lifecycle, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        lifecycle,
        "LEGACY_POLICY_STOP_FILE",
        tmp_path / "legacy_policy_stop",
    )


def test_terminate_owned_group_preserves_other_pgid() -> None:
    """소유 PGID 종료가 같은 이름의 다른 프로세스 그룹에 영향을 주지 않는다."""
    owned = _start_sleep()
    unrelated = _start_sleep()
    try:
        group = lifecycle.register_owned_group(
            owned,
            kind="simulator",
            run_id="owned-test",
            marker="owned-test",
        )
        assert lifecycle.terminate_owned_group(
            group,
            _lifecycle_args(),
            graceful_ros_shutdown=True,
        )
        assert not lifecycle.process_group_members(group.pgid)
        assert unrelated.poll() is None
    finally:
        _force_stop(owned)
        _force_stop(unrelated)


def test_cleanup_preserves_marker_mismatch(monkeypatch, tmp_path: Path) -> None:
    """registry PGID가 재사용됐거나 marker가 다르면 해당 그룹을 종료하지 않는다."""
    _redirect_runtime(monkeypatch, tmp_path)
    unrelated = _start_sleep()
    registry = tmp_path / "stale-run" / lifecycle.REGISTRY_FILENAME
    registry.parent.mkdir(parents=True)
    registry.write_text(
        '{"run_id":"wrong-run","stop_file":"","groups":['
        f'{{"kind":"simulator","pgid":{unrelated.pid},"marker":"wrong"}}]}}',
        encoding="utf-8",
    )
    try:
        assert lifecycle.cleanup_stale_processes(_lifecycle_args())
        assert unrelated.poll() is None
    finally:
        _force_stop(unrelated)


def test_cleanup_recovers_registered_group(monkeypatch, tmp_path: Path) -> None:
    """run marker가 일치하는 crash 잔여 PGID를 registry에서 찾아 종료한다."""
    _redirect_runtime(monkeypatch, tmp_path)
    run_id = "registered-run"
    proc = _start_sleep(run_id=run_id)
    group = lifecycle.register_owned_group(
        proc,
        kind="policy",
        run_id=run_id,
        marker="unused-marker",
    )
    registry = tmp_path / run_id / lifecycle.REGISTRY_FILENAME
    lifecycle.write_group_registry(
        registry,
        run_id=run_id,
        stop_file=tmp_path / run_id / "policy_stop",
        groups=[group],
    )
    try:
        assert lifecycle.cleanup_stale_processes(_lifecycle_args())
        assert not lifecycle.process_group_members(group.pgid)
        assert not registry.exists()
    finally:
        _force_stop(proc)


def test_discovers_and_terminates_inner_marker_pgid() -> None:
    """Distrobox 내부처럼 분리된 PGID를 config marker로 발견해 종료한다."""
    run_id = "inner-run"
    marker = "/tmp/inner-run/engine_config.yaml"
    proc = _start_marked_sleep(marker, run_id)
    try:
        assert proc.pid in lifecycle.process_groups_with_cmdline_marker(marker)
        group = lifecycle.register_owned_pgid(
            proc.pid,
            kind="simulator",
            run_id=run_id,
            marker=marker,
        )
        assert group.proc is None
        assert lifecycle.terminate_owned_group(
            group,
            _lifecycle_args(),
            graceful_ros_shutdown=True,
        )
        assert not lifecycle.process_group_members(proc.pid)
    finally:
        _force_stop(proc)
