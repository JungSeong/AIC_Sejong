"""PortOffsetCollect timestamp gating의 회귀 테스트."""

from types import SimpleNamespace

from data_gen_node.port_offset_dataset import (
    _observation_sync_metadata,
    _tf_sync_metadata,
    _wait_for_synchronized_observation,
)


def _stamp(nanoseconds: int):
    """nanosecond 정수로 가짜 ROS time message를 만든다."""
    return SimpleNamespace(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )


def _message(nanoseconds: int):
    """header timestamp를 가진 가짜 ROS message를 만든다."""
    return SimpleNamespace(header=SimpleNamespace(stamp=_stamp(nanoseconds)))


def _observation(left: int, center: int, right: int, controller: int):
    """세 camera와 controller timestamp를 가진 observation을 만든다."""
    return SimpleNamespace(
        left_image=_message(left),
        center_image=_message(center),
        right_image=_message(right),
        controller_state=_message(controller),
    )


def _policy(tolerance_ns: int = 30_000_000):
    """timestamp helper 호출에 필요한 최소 policy 대역을 만든다."""
    return SimpleNamespace(
        collect_sync_tolerance_ns=tolerance_ns,
        _image_msg_for_camera=lambda obs, name: getattr(obs, f"{name}_image"),
    )


def test_observation_within_tolerance_is_accepted() -> None:
    """모든 source가 허용 오차 안이면 center 시각 기준으로 승인한다."""
    policy = _policy()
    valid, metadata = _observation_sync_metadata(
        policy,
        _observation(
            1_000_000_000,
            1_010_000_000,
            1_020_000_000,
            1_025_000_000,
        ),
    )

    assert valid is True
    assert metadata["capture_stamp_ns"] == 1_010_000_000
    assert metadata["skew_ns"] == {
        "camera": 20_000_000,
        "controller": 15_000_000,
    }


def test_camera_skew_over_tolerance_is_rejected() -> None:
    """camera timestamp 범위가 허용 오차를 넘으면 sample을 거부한다."""
    valid, metadata = _observation_sync_metadata(
        _policy(),
        _observation(
            1_000_000_000,
            1_010_000_000,
            1_050_000_000,
            1_010_000_000,
        ),
    )

    assert valid is False
    assert metadata["rejection_reason"] == "camera_timestamp_skew"


def test_controller_skew_over_tolerance_is_rejected() -> None:
    """controller timestamp가 center camera에서 멀면 sample을 거부한다."""
    valid, metadata = _observation_sync_metadata(
        _policy(),
        _observation(
            1_000_000_000,
            1_010_000_000,
            1_020_000_000,
            1_050_000_000,
        ),
    )

    assert valid is False
    assert metadata["rejection_reason"] == "controller_timestamp_skew"


def test_static_tf_is_allowed_but_stale_dynamic_tf_is_rejected() -> None:
    """static TF의 0 stamp는 허용하고 오래된 dynamic TF는 거부한다."""
    policy = _policy()
    timestamps = {
        "capture_stamp_ns": 1_010_000_000,
        "skew_ns": {},
        "sync_valid": True,
    }
    valid, metadata = _tf_sync_metadata(
        policy,
        timestamps,
        {
            "port": _message(0),
            "plug": _message(1_050_000_000),
        },
    )

    assert valid is False
    assert metadata["tf"]["port"]["is_static"] is True
    assert metadata["rejection_reason"] == "tf_timestamp_skew"


def test_nonzero_port_snapshot_is_treated_as_trial_static() -> None:
    """trial 시작에 저장한 port TF는 과거 stamp여도 정적 source로 승인한다."""
    timestamps = {
        "capture_stamp_ns": 2_000_000_000,
        "skew_ns": {},
        "sync_valid": True,
    }

    valid, metadata = _tf_sync_metadata(
        _policy(),
        timestamps,
        {
            "port": _message(1_000_000_000),
            "plug": _message(2_010_000_000),
        },
        static_sources={"port"},
    )

    assert valid is True
    assert metadata["tf"]["port"]["is_static_snapshot"] is True
    assert metadata["tf"]["port"]["skew_ns"] == 0


def test_wait_selects_next_synchronized_observation() -> None:
    """첫 observation이 어긋나면 제한 시간 안의 다음 유효 observation을 선택한다."""
    policy = _policy()
    policy.collect_sync_wait_timeout_sec = 0.1
    policy.collect_sync_poll_sec = 0.001
    policy._observation_sync_metadata = lambda obs: _observation_sync_metadata(
        policy,
        obs,
    )
    policy._collect_log_text = lambda message, _color: message
    policy.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    policy.sleep_for = lambda _seconds: None
    observations = iter(
        [
            _observation(1_000_000_000, 1_010_000_000, 1_100_000_000, 1_010_000_000),
            _observation(2_000_000_000, 2_010_000_000, 2_020_000_000, 2_015_000_000),
        ]
    )

    observation, metadata = _wait_for_synchronized_observation(
        policy,
        lambda: next(observations),
    )

    assert observation is not None
    assert metadata["capture_stamp_ns"] == 2_010_000_000
    assert metadata["sync_valid"] is True
    assert "observation" in metadata["wait_ns"]
