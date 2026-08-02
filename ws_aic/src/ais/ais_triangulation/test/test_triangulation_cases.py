"""Randomized triangulation case 생성과 PRED 좌표 경로의 회귀 검사."""

from __future__ import annotations

import argparse
import ast
import math
import sys
import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml


TRIANGULATION_ROOT = Path(__file__).resolve().parents[1]
AIS_ROOT = TRIANGULATION_ROOT.parent
sys.path.insert(0, str(TRIANGULATION_ROOT))
sys.path.insert(0, str(AIS_ROOT / "ais_auto_capture"))

from portoffset_randomization.constants import (  # noqa: E402
    BASE_ROBOT_HOME,
    CLI_DEFAULTS,
    LIMITS,
)
from run_triangulation_cases import (  # noqa: E402
    ANSI_RESET,
    BOLD_GREEN,
    BOLD_MAGENTA,
    _bold_log,
    default_output_path,
    generate_cases,
    parse_args,
    simulator_command,
)
from evaluate_triangulation_euclidean import (  # noqa: E402
    lookup_synced_transform,
    prediction_xyz_in_base,
    timestamp_from_msg,
    transform_xyz,
)


def _scenario_args(**overrides) -> argparse.Namespace:
    values = {
        "port_types": CLI_DEFAULTS["port_types"],
        "port_order": CLI_DEFAULTS["port_order"],
        "time_limit_s": CLI_DEFAULTS["time_limit_s"],
        "robot_joint_noise_deg": CLI_DEFAULTS["robot_joint_noise_deg"],
        "cable_rpy_noise_deg": CLI_DEFAULTS["cable_rpy_noise_deg"],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _active_target_rail(trial: dict) -> tuple[int, dict]:
    task = next(iter(trial["tasks"].values()))
    prefix = "nic_rail_" if task["port_type"] == "sfp" else "sc_rail_"
    active = [
        (int(name.removeprefix(prefix)), value)
        for name, value in trial["scene"]["task_board"].items()
        if name.startswith(prefix)
        and isinstance(value, dict)
        and value.get("entity_present")
    ]
    assert len(active) == 1
    return active[0]


def _assert_in_range(value: float, bounds: tuple[float, float]) -> None:
    assert bounds[0] - 1e-12 <= value <= bounds[1] + 1e-12


def test_seed_is_deterministic_and_num_cases_is_not_fixed() -> None:
    args = _scenario_args()
    three = generate_cases(seed=71, num_cases=3, scenario_args=args)
    seven = generate_cases(seed=71, num_cases=7, scenario_args=args)
    repeated = generate_cases(seed=71, num_cases=3, scenario_args=args)

    assert len(three["trials"]) == 3
    assert len(seven["trials"]) == 7
    assert three == repeated
    assert list(three["trials"].items()) == list(seven["trials"].items())[:3]
    assert three["robot"] == seven["robot"]


def test_generated_cases_use_portoffset_ranges() -> None:
    config = generate_cases(
        seed=30,
        num_cases=100,
        scenario_args=_scenario_args(),
    )
    assert len(config["trials"]) == 100

    joint_noise = math.radians(CLI_DEFAULTS["robot_joint_noise_deg"])
    for joint, value in config["robot"]["home_joint_positions"].items():
        _assert_in_range(
            value,
            (
                BASE_ROBOT_HOME[joint] - joint_noise,
                BASE_ROBOT_HOME[joint] + joint_noise,
            ),
        )

    for trial in config["trials"].values():
        task = next(iter(trial["tasks"].values()))
        port_type = task["port_type"]
        prefix = "sc" if port_type == "sc" else "sfp"
        board = trial["scene"]["task_board"]["pose"]
        _assert_in_range(board["x"], LIMITS[f"{prefix}_board_x"])
        _assert_in_range(board["y"], LIMITS[f"{prefix}_board_y"])
        _assert_in_range(board["yaw"], LIMITS[f"{prefix}_board_yaw"])
        assert (board["z"], board["roll"], board["pitch"]) == (1.14, 0.0, 0.0)

        rail_index, rail = _active_target_rail(trial)
        rail_pose = rail["entity_pose"]
        if port_type == "sfp":
            _assert_in_range(rail_pose["translation"], LIMITS["nic_translation"])
            _assert_in_range(rail_pose["yaw"], LIMITS["nic_yaw"])
            assert task["target_module_name"] == f"nic_card_mount_{rail_index}"
        else:
            _assert_in_range(rail_pose["translation"], LIMITS["sc_translation"])
            assert rail_pose["yaw"] == 0.0
            assert task["target_module_name"] == f"sc_port_{rail_index}"

        cable = trial["scene"]["cables"][task["cable_name"]]["pose"]
        gripper_noise = LIMITS["gripper_offset_noise"]
        for axis in ("x", "y", "z"):
            base = LIMITS[f"{prefix}_gripper_offset_{axis}"]
            _assert_in_range(
                cable["gripper_offset"][axis],
                (base + gripper_noise[0], base + gripper_noise[1]),
            )
        rpy_noise = math.radians(CLI_DEFAULTS["cable_rpy_noise_deg"])
        for axis in ("roll", "pitch", "yaw"):
            base = LIMITS[f"cable_{axis}"]
            _assert_in_range(cable[axis], (base - rpy_noise, base + rpy_noise))


def test_cli_alias_output_name_and_simulator_command() -> None:
    args = parse_args(["--seed", "9", "--num_cases", "4", "--generate-only"])
    assert args.seed == 9
    assert args.num_cases == 4
    assert (
        default_output_path(date(2026, 7, 29)).name
        == "20260729_triangulation_cases.yaml"
    )

    cases_path = Path("/tmp/20260729_triangulation_cases.yaml")
    command = simulator_command(args, cases_path)
    assert command[:5] == ["distrobox", "enter", "aic_eval", "--", "/entrypoint.sh"]
    assert "ground_truth:=true" in command
    assert "start_aic_engine:=true" in command
    assert "gazebo_gui:=false" in command
    assert "launch_rviz:=false" in command
    assert "spawn_task_board:=false" in command
    assert "spawn_cable:=false" in command
    assert f"aic_engine_config_file:={cases_path}" in command


def test_simulator_parameters_are_forwarded_to_entrypoint() -> None:
    args = parse_args(
        [
            "--launch_rviz",
            "true",
            "--gazebo_gui",
            "false",
            "--ground_truth",
            "false",
            "--spawn_task_board",
            "true",
            "--sim-arg",
            "shutdown_on_aic_engine_exit=true",
            "--generate-only",
        ]
    )
    command = simulator_command(args, Path("/tmp/cases.yaml"))
    assert "launch_rviz:=true" in command
    assert "gazebo_gui:=false" in command
    assert "ground_truth:=false" in command
    assert "spawn_task_board:=true" in command
    assert "shutdown_on_aic_engine_exit:=true" in command


def test_runner_highlight_logs_are_bold_and_colored(capsys) -> None:
    _bold_log(BOLD_MAGENTA, "[generated] yaml")
    _bold_log(BOLD_GREEN, "[simulator start] command")
    output = capsys.readouterr().out
    assert f"\033[1;35m[generated] yaml{ANSI_RESET}" in output
    assert f"\033[1;32m[simulator start] command{ANSI_RESET}" in output


def _function_node(path: Path, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_published_and_opencv_debug_pred_use_same_cached_3d_value() -> None:
    """CSV 입력 topic과 triangulation debug PRED가 같은 base 3D 값을 써야 한다."""
    policy_path = (
        AIS_ROOT / "ais_policy" / "final_policy" / "final_policy" / "FinalPolicy.py"
    )
    function = _function_node(policy_path, "_cache_detected_port")
    publish_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_publish_triangulated_port_xyz"
    )
    debug_call = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_save_triangulation_debug_images"
    )
    predicted_keyword = next(
        keyword for keyword in debug_call.keywords if keyword.arg == "predicted_port"
    )
    expected = "self._cached_port_base"
    assert ast.unparse(publish_call.args[0]) == expected
    assert ast.unparse(predicted_keyword.value) == expected
    capture_stamp_keyword = next(
        keyword for keyword in publish_call.keywords if keyword.arg == "capture_stamp"
    )
    assert ast.unparse(capture_stamp_keyword.value) == "self._capture_stamp_from_obs(obs)"

    debug_path = AIS_ROOT / "ais_policy" / "final_policy" / "final_policy" / "debug.py"
    debug_function = _function_node(debug_path, "_save_triangulation_debug_images")
    projected_values = [
        ast.unparse(node.args[0])
        for node in ast.walk(debug_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "project_3d_to_pixel"
    ]
    assert "predicted_port" in projected_values


def test_rviz_debug_image_preserves_source_timestamp_and_frame() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import Image
    from final_policy.debug import FinalPolicyDebugMixin

    source = Image()
    source.header.stamp.sec = 123
    source.header.stamp.nanosec = 456_789
    source.header.frame_id = "center_camera/optical"
    debug_image = np.zeros((4, 5, 3), dtype=np.uint8)
    published = FinalPolicyDebugMixin._bgr_debug_image_message(debug_image, source)

    assert published.header.stamp.sec == source.header.stamp.sec
    assert published.header.stamp.nanosec == source.header.stamp.nanosec
    assert published.header.frame_id == source.header.frame_id
    assert published.encoding == "bgr8"
    assert (published.height, published.width, published.step) == (4, 5, 15)
    assert len(published.data) == debug_image.size


def test_rviz_debug_image_is_republished_with_original_header() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import Image
    from final_policy.debug import FinalPolicyDebugMixin

    class Publisher:
        def __init__(self) -> None:
            self.messages = []

        @staticmethod
        def get_subscription_count() -> int:
            return 1

        def publish(self, message) -> None:
            self.messages.append(message)

    message = Image()
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456_789
    detection_publisher = Publisher()
    debug = FinalPolicyDebugMixin()
    debug._detection_debug_image_pubs = {"left": detection_publisher}
    debug._triangulation_debug_image_pubs = {}
    debug._latest_detection_debug_images = {"left": message}
    debug._latest_triangulation_debug_images = {}

    debug._republish_debug_images()

    assert detection_publisher.messages == [message]
    assert detection_publisher.messages[0].header.stamp.sec == 123
    assert detection_publisher.messages[0].header.stamp.nanosec == 456_789


def test_yolo_detection_overlay_callback_receives_source_image() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import Image
    from final_policy.vision import VisionPortEstimator
    import torch

    class FakeBox:
        cls = torch.tensor([0.0])
        conf = torch.tensor([0.95])
        xyxy = torch.tensor([[1.0, 2.0, 8.0, 9.0]])

    class FakeResult:
        boxes = [FakeBox()]
        keypoints = None

    predict_kwargs = {}

    def fake_model(_image, **kwargs):
        predict_kwargs.update(kwargs)
        return [FakeResult()]

    callbacks = []
    source = Image()
    source.header.stamp.sec = 123
    source.header.stamp.nanosec = 456_000_000
    source.header.frame_id = "left_camera/optical"

    estimator = VisionPortEstimator.__new__(VisionPortEstimator)
    estimator._model = fake_model
    estimator._conf_thresh = 0.8
    estimator._yolo_device = "cpu"
    estimator._logger = None
    estimator._debug_save_enabled = False
    estimator._debug_save_dir = None
    estimator._debug_task_label = "m4_sfp1"
    estimator._debug_call_count = 0
    estimator._detection_debug_callback = (
        lambda camera, image, image_msg: callbacks.append(
            (camera, image.copy(), image_msg)
        )
    )

    detections = estimator._detect(
        np.zeros((12, 12, 3), dtype=np.uint8),
        cam_name="left",
        target_class_id=0,
        source_image_msg=source,
        sync_span_ms=12.5,
    )

    assert len(detections) == 1
    assert predict_kwargs["device"] == "cpu"
    assert len(callbacks) == 1
    camera, overlay, callback_source = callbacks[0]
    assert camera == "left"
    assert callback_source is source
    assert np.count_nonzero(overlay) > 0


def test_triangulation_marker_array_uses_camera_stamp_and_base_link() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from builtin_interfaces.msg import Time
    from visualization_msgs.msg import Marker
    from final_policy.debug import FinalPolicyDebugMixin

    stamp = Time(sec=77, nanosec=123_000_000)
    markers = FinalPolicyDebugMixin._triangulation_marker_array(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.11, 0.18, 0.31]),
        stamp,
        "sfp_port_1",
    ).markers

    assert [marker.type for marker in markers] == [
        Marker.SPHERE,
        Marker.SPHERE,
        Marker.LINE_LIST,
    ]
    assert all(marker.header.frame_id == "base_link" for marker in markers)
    assert all(marker.header.stamp == stamp for marker in markers)
    assert markers[0].pose.position.x == pytest.approx(0.1)
    assert markers[1].pose.position.y == pytest.approx(0.18)
    assert len(markers[2].points) == 2


def test_cached_triangulation_keeps_its_source_observation() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from final_policy.vision import VisionPortEstimator

    estimator = VisionPortEstimator.__new__(VisionPortEstimator)
    estimator._cache_lock = threading.Lock()
    estimator._cache_max_age_sec = 1.0
    source_observation = object()
    position = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    estimator._cache = {
        "target_class_id": 7,
        "candidates": [{"pos": position}],
        "observation": source_observation,
        "updated_at": time.time(),
        "request_id": 11,
    }

    cached_position, cached_observation = estimator.cached_estimate_with_observation(7)
    assert np.array_equal(cached_position, position)
    assert cached_observation is source_observation


def test_evaluator_uses_prediction_header_timestamp() -> None:
    stamp = type("Stamp", (), {"sec": 42, "nanosec": 125_000_000})()
    header = type("Header", (), {"stamp": stamp})()
    message = type("Message", (), {"header": header})()
    assert timestamp_from_msg(message) == 42.125


def test_camera_observation_timing_uses_pairwise_span() -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import Image
    from final_policy.vision import observation_camera_timing

    def image_at(nanosec: int) -> Image:
        image = Image()
        image.header.stamp.sec = 10
        image.header.stamp.nanosec = nanosec
        return image

    observation = type(
        "Observation",
        (),
        {
            "left_image": image_at(10_000_000),
            "center_image": image_at(20_000_000),
            "right_image": image_at(35_000_000),
        },
    )()
    reference_ns, stamps_ns, sync_span_ms = observation_camera_timing(observation)
    assert reference_ns == 10_020_000_000
    assert stamps_ns["left"] == 10_010_000_000
    assert sync_span_ms == 25.0


def test_triangulation_rejects_camera_span_above_threshold(monkeypatch) -> None:
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import Image
    from final_policy.config import FinalPolicyConfig
    from final_policy.vision import VisionPortEstimator

    def image_at(nanosec: int) -> Image:
        image = Image()
        image.header.stamp.sec = 5
        image.header.stamp.nanosec = nanosec
        return image

    observation = type(
        "Observation",
        (),
        {
            "left_image": image_at(0),
            "center_image": image_at(10_000_000),
            "right_image": image_at(40_000_000),
        },
    )()
    warnings = []
    logger = type("Logger", (), {"warn": lambda self, message: warnings.append(message)})()
    estimator = VisionPortEstimator.__new__(VisionPortEstimator)
    estimator._logger = logger
    monkeypatch.setattr(FinalPolicyConfig, "TRIANGULATION_SYNC_THRESHOLD_MS", 30.0)

    assert estimator._estimate_all_sync(observation, target_class_id=0) == []
    assert any("outside threshold" in message for message in warnings)


def test_prediction_frame_is_required_and_identity_frame_is_preserved() -> None:
    point = type("Point", (), {"x": 0.1, "y": 0.2, "z": 0.3})()
    stamp = type("Stamp", (), {"sec": 12, "nanosec": 1})()
    header = type("Header", (), {"stamp": stamp, "frame_id": "base_link"})()
    message = type("PointStamped", (), {"header": header, "point": point})()
    xyz, source_frame, mode, sync_delta_ms = prediction_xyz_in_base(
        message,
        "point",
        buffer=None,
        node=None,
        base_frame="base_link",
        timeout_s=0.0,
    )
    assert xyz == (0.1, 0.2, 0.3)
    assert source_frame == "base_link"
    assert mode == "identity"
    assert sync_delta_ms == 0.0

    message.header.frame_id = ""
    with pytest.raises(ValueError, match="header.frame_id"):
        prediction_xyz_in_base(
            message,
            "point",
            buffer=None,
            node=None,
            base_frame="base_link",
            timeout_s=0.0,
        )


def test_transform_xyz_applies_rotation_and_translation() -> None:
    translation = type("Vector", (), {"x": 1.0, "y": 2.0, "z": 3.0})()
    half_sqrt = math.sqrt(0.5)
    rotation = type(
        "Quaternion",
        (),
        {"x": 0.0, "y": 0.0, "z": half_sqrt, "w": half_sqrt},
    )()
    transform = type(
        "Transform",
        (),
        {"translation": translation, "rotation": rotation},
    )()
    transformed = transform_xyz((1.0, 0.0, 0.0), transform)
    assert transformed == pytest.approx((1.0, 3.0, 3.0))


def test_tf_fallback_must_stay_inside_sync_threshold() -> None:
    from builtin_interfaces.msg import Time as TimeMsg

    requested_stamp = TimeMsg(sec=10, nanosec=0)
    latest_stamp = TimeMsg(sec=10, nanosec=20_000_000)
    header = type("Header", (), {"stamp": latest_stamp})()
    transform = type("Transform", (), {})()
    latest_message = type(
        "TransformStamped",
        (),
        {"header": header, "transform": transform},
    )()

    class LatestOnlyBuffer:
        def lookup_transform(self, _destination, _source, query_time, timeout=None):
            if query_time.nanoseconds != 0:
                raise RuntimeError("exact transform unavailable")
            return latest_message

    accepted = lookup_synced_transform(
        LatestOnlyBuffer(),
        node=None,
        destination_frame="base_link",
        source_frame="world",
        timeout_s=0.0,
        stamp=requested_stamp,
        sync_threshold_ms=30.0,
    )
    assert accepted.mode == "nearby_latest"
    assert accepted.sync_delta_ms == pytest.approx(20.0)

    with pytest.raises(RuntimeError, match="exceeds threshold"):
        lookup_synced_transform(
            LatestOnlyBuffer(),
            node=None,
            destination_frame="base_link",
            source_frame="world",
            timeout_s=0.0,
            stamp=requested_stamp,
            sync_threshold_ms=10.0,
        )


def test_best_camera_pair_is_selected_by_reprojection_rms(monkeypatch) -> None:
    """마지막 pair가 아니라 전체 camera 재투영 RMS가 가장 작은 pair를 선택한다."""
    final_policy_root = AIS_ROOT / "ais_policy" / "final_policy"
    sys.path.insert(0, str(final_policy_root))

    from sensor_msgs.msg import CameraInfo, Image
    import final_policy.vision as vision_module
    from final_policy.vision import VisionPortEstimator

    def image_at() -> Image:
        image = Image()
        image.header.stamp.sec = 1
        return image

    camera_info = CameraInfo()
    camera_info.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    observation = type(
        "Observation",
        (),
        {
            "left_image": image_at(),
            "center_image": image_at(),
            "right_image": image_at(),
            "left_camera_info": camera_info,
            "center_camera_info": camera_info,
            "right_camera_info": camera_info,
        },
    )()
    detections = {
        "left": [
            {
                "class_id": 0,
                "point_name": "port",
                "port_index": 0,
                "u": 10.0,
                "v": 10.0,
                "conf": 0.9,
            }
        ],
        "center": [
            {
                "class_id": 0,
                "point_name": "port",
                "port_index": 0,
                "u": 20.0,
                "v": 20.0,
                "conf": 0.9,
            }
        ],
        "right": [
            {
                "class_id": 0,
                "point_name": "port",
                "port_index": 0,
                "u": 30.0,
                "v": 30.0,
                "conf": 0.9,
            }
        ],
    }
    pair_points = {
        (10.0, 20.0): np.array([-0.380, 0.22, 0.10]),
        (20.0, 30.0): np.array([-0.378, 0.22, 0.10]),
        (10.0, 30.0): np.array([-0.376, 0.22, 0.10]),
    }
    reprojection_error = {
        -0.380: 4.0,
        -0.378: 1.0,
        -0.376: 8.0,
    }
    camera_uv = {
        0: (10.0, 10.0),
        1: (20.0, 20.0),
        2: (30.0, 30.0),
    }

    estimator = VisionPortEstimator.__new__(VisionPortEstimator)
    estimator._logger = None
    estimator._loaded = True
    estimator._ensure_loaded = lambda: None
    estimator._image_from_msg = lambda _image: np.zeros((2, 2, 3))
    estimator._detect = (
        lambda _image, *, cam_name, **_kwargs: detections[cam_name]
    )

    def camera_matrix(_observation, camera_name):
        matrix = np.eye(4)
        matrix[0, 3] = {"left": 0, "center": 1, "right": 2}[camera_name]
        return matrix

    estimator._base_to_camera_optical_matrix = camera_matrix
    estimator._triangulate = (
        lambda u_a, _v_a, _k_a, _t_a, u_b, _v_b, _k_b, _t_b:
        pair_points[(u_a, u_b)].copy()
    )

    def fake_project(point, _k_matrix, camera_matrix):
        camera_index = int(camera_matrix[0, 3])
        u, v = camera_uv[camera_index]
        error = reprojection_error[round(float(point[0]), 3)]
        return u + error, v

    monkeypatch.setattr(vision_module, "project_3d_to_pixel", fake_project)

    candidates = estimator._estimate_all_sync(observation, target_class_id=0)

    assert len(candidates) == 1
    assert candidates[0]["camera_pair"] == ("center", "right")
    assert candidates[0]["reprojection_error_px"] == pytest.approx(1.0)
    assert np.array_equal(candidates[0]["pos"], pair_points[(20.0, 30.0)])


def test_generated_yaml_is_engine_readable(tmp_path: Path) -> None:
    from run_triangulation_cases import write_cases

    config = generate_cases(seed=5, num_cases=2, scenario_args=_scenario_args())
    output = tmp_path / "cases.yaml"
    write_cases(output, config, seed=5, num_cases=2)
    loaded = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert loaded == config
