from __future__ import annotations
"""PortOffsetCollect의 ROS 2/TF와 실행 상태 초기화."""

import json
import math
import os
import signal
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Header
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose, Transform, TransformStamped, Vector3, Wrench
from data_gen_node.lib.cheatcode import CheatCodePlanner
from data_gen_node.port_offset_config import (
    DAMPING_DEFAULT,
    SFP_PLUG_REFERENCE_OFFSET_IN_CABLE_TIP_FRAME,
    STIFFNESS_DEFAULT,
    TF_RECONSTRUCTION_ANGLE_TOLERANCE_RAD,
    TF_RECONSTRUCTION_POSITION_TOLERANCE_M,
    TOOL0_TO_OPTICAL,
    TOOL0_TO_TCP_Z,
    COLLECT_BASE_Z_OFFSET_DEFAULT,
)
from data_gen_node.port_offset_geometry import (
    _matrix_to_rpy_xyz,
    _matrix_from_pose,
    _matrix_from_translation_quat,
    _quat_to_matrix_xyzw,
)
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException

_LOG_COLORS = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}

def init_runtime(self, parent_node):
    """GT 기반 수집에 필요한 planner, camera 좌표계, 데이터셋, 종료 처리를 초기화한다."""
    
    # 1. 상태 및 제어 변수 초기화
    self._task: Optional[Task] = None
    # 나선형 진동(Oscillation) 방지를 위해 적분 한계를 낮추고(안전 범위), 부족한 힘은 Stiffness로 보상
    self._max_integrator_windup = 0.15 
    
    # 2. 제어 주기 및 경로 설정
    fps = int(os.environ.get("AIC_COLLECT_FPS", "0"))
    self.step_sleep_sec = 1.0 / (fps if fps > 0 else 20.0)
    self.capture_root = Path(os.environ.get("AIC_CAPTURE_DIR", "/tmp/aic_episodes"))
    
    # 3. 플래너 및 환경 설정
    self._planner = CheatCodePlanner(
        i_gain=float(os.environ.get("AIC_CAPTURE_CHEATCODE_I_GAIN", "0.15")),
        max_integrator_windup=self._max_integrator_windup,
    )
    self.approach_steps = int(os.environ.get("AIC_CAPTURE_CHEATCODE_APPROACH_STEPS", "100"))
    self.collect_steps = int(os.environ.get("AIC_COLLECT_STEPS", "160"))
    self.collect_turns = float(os.environ.get("AIC_COLLECT_TURNS", "2.0"))
    self.collect_start_radius = float(os.environ.get("AIC_COLLECT_START_RADIUS", "0.020"))
    self.collect_end_radius = float(os.environ.get("AIC_COLLECT_END_RADIUS", "0.0"))
    self.collect_rotate_angle = float(os.environ.get("AIC_COLLECT_ROTATE_ANGLE", "0.0"))
    self.collect_pattern = os.environ.get("AIC_COLLECT_PATTERN", "spiral").strip().lower()
    if self.collect_pattern not in {"gaussian", "spiral"}:
        self.get_logger().warn(
            f"[PortOffsetCollect] Invalid AIC_COLLECT_PATTERN={self.collect_pattern}; using spiral"
        )
        self.collect_pattern = "spiral"
    self.collect_gaussian_sigma = float(os.environ.get("AIC_COLLECT_GAUSSIAN_SIGMA", "0.006"))
    self.collect_gaussian_max_radius = float(os.environ.get("AIC_COLLECT_GAUSSIAN_MAX_RADIUS", str(self.collect_start_radius)))
    self.collect_sync_tolerance_ns = int(
        max(0.0, float(os.environ.get("AIC_COLLECT_SYNC_TOLERANCE_MS", "30.0")))
        * 1_000_000
    )
    self.collect_sync_wait_timeout_sec = max(
        0.0,
        float(os.environ.get("AIC_COLLECT_SYNC_WAIT_TIMEOUT_SEC", "1.0")),
    )
    self.collect_sync_poll_sec = max(
        0.001,
        float(os.environ.get("AIC_COLLECT_SYNC_POLL_SEC", "0.01")),
    )
    self.collect_color_log = (
        os.environ.get("AIC_COLLECT_COLOR_LOG", "true").lower()
        not in {"0", "false", "no"}
        and not os.environ.get("NO_COLOR")
    )
    self._tf_quality_buffer = Buffer(
        cache_time=Duration(seconds=10.0),
        node=parent_node,
    )
    self._tf_quality_lock = threading.Lock()
    self._tf_quality_dynamic_stamps = {}
    self._tf_quality_static_edges = set()
    self._tf_quality_last_prune_ns = 0
    self._tf_quality_callback_group = ReentrantCallbackGroup()
    dynamic_qos = QoSProfile(
        depth=100,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    static_qos = QoSProfile(
        depth=100,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    self._tf_quality_subscriptions = [
        parent_node.create_subscription(
            TFMessage,
            topic,
            lambda message, source_topic=topic: _record_tf_quality_message(
                self, message, source_topic, is_static=False
            ),
            dynamic_qos,
            callback_group=self._tf_quality_callback_group,
        )
        for topic in ("/tf", "/scoring/tf")
    ]
    self._tf_quality_subscriptions.append(
        parent_node.create_subscription(
            TFMessage,
            "/tf_static",
            lambda message: _record_tf_quality_message(
                self, message, "/tf_static", is_static=True
            ),
            static_qos,
            callback_group=self._tf_quality_callback_group,
        )
    )
    seed_text = os.environ.get("AIC_COLLECT_RANDOM_SEED", "").strip()
    seed = int(seed_text) if seed_text else None
    self._collect_rng = np.random.default_rng(seed)

    # 4. Ground Truth 수집 기준과 camera 좌표계 설정
    self.collect_base_z_offset_m = float(
        os.environ.get(
            "AIC_PORT_COLLECT_BASE_Z_OFFSET_M",
            str(COLLECT_BASE_Z_OFFSET_DEFAULT),
        )
    )
    self._tool0_to_tcp_z = float(os.environ.get("AIC_TOOL0_TO_TCP_Z", str(TOOL0_TO_TCP_Z)))
    self._t_tool0_tcp = np.eye(4, dtype=float)
    self._t_tool0_tcp[2, 3] = self._tool0_to_tcp_z
    self._t_tool0_to_optical = {
        name: _matrix_from_translation_quat(translation, quat)
        for name, (translation, quat) in TOOL0_TO_OPTICAL.items()
    }
    self._vision_offset_record_enabled = os.environ.get("AIC_VISION_OFFSET_RECORD", "1").lower() not in {"0", "false", "no"}
    
    self._vision_offset_dataset_dir = Path(
        os.environ.get(
            "AIC_VISION_OFFSET_DATASET_DIR",
            str(self.capture_root / "vision_offset_dataset"),
        )
    )
    self._debug_image_dir = self._vision_offset_dataset_dir / "debug" / "image"
    self._vision_offset_images_dir = self._vision_offset_dataset_dir / "images"
    self._vision_offset_samples_path = self._vision_offset_dataset_dir / "samples.jsonl"
    if self._vision_offset_record_enabled:
        self._vision_offset_images_dir.mkdir(parents=True, exist_ok=True)
    self._scenario_params_file = Path(os.environ.get("AIC_SCENARIO_PARAMS_FILE", "/tmp/aic_scenario_params.json"))

    # 5. 종료 처리
    try: signal.signal(signal.SIGTERM, self._on_sigterm)
    except ValueError: pass

    self._stop_file = Path(os.environ.get("AIC_STOP_FILE", "/tmp/aic_policy_stop"))
    threading.Thread(target=self._watch_stop_file, daemon=True).start()

    self.get_logger().info(
        f"[PortOffsetCollect] Ground Truth-guided Policy Initialized. "
        f"Root: {self.capture_root}"
    )

# ── 인프라 로직 ──────────────────────────────────────────────────────────

def _watch_stop_file(self) -> None:
    """AIC_STOP_FILE이 생기면 즉시 프로세스를 종료한다."""
    while True:
        if self._stop_file.exists(): os._exit(0)
        time.sleep(0.5)

def _on_sigterm(self, signum, frame) -> None:
    """SIGTERM을 받았을 때 정상 종료한다."""
    raise SystemExit(0)

def _wait_for_tf(self, target_frame: str, source_frame: str, timeout_sec: float = 10.0) -> bool:
    """target_frame 기준 source_frame TF가 timeout 안에 조회 가능해질 때까지 기다린다."""
    start = self.time_now()
    while (self.time_now() - start) < Duration(seconds=timeout_sec):
        try:
            self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time())
            return True
        except TransformException: self.sleep_for(0.1)
    return False

def _lookup_transform(self, target_frame: str, source_frame: str) -> Transform:
    """현재 시점의 target_frame 기준 source_frame transform을 조회한다."""
    return self._parent_node._tf_buffer.lookup_transform(target_frame, source_frame, Time()).transform

def _lookup_latest_transform_stamped(
    self,
    target_frame: str,
    source_frame: str,
) -> TransformStamped:
    """trial에서 고정된 frame을 snapshot하기 위해 최신 stamped TF를 조회한다."""
    return self._parent_node._tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        Time(),
    )

def _lookup_transform_at(self, target_frame: str, source_frame: str, stamp) -> TransformStamped:
    """지정한 ROS timestamp의 TF가 도착할 때까지 제한 시간 동안 기다려 조회한다."""
    query_time = Time.from_msg(stamp)
    buffer = self._parent_node._tf_buffer
    if not buffer.can_transform(target_frame, source_frame, query_time):
        self.get_logger().info(
            self._collect_log_text(
                "[PortOffsetCollect] Waiting for TF at capture timestamp: "
                f"target={target_frame}, source={source_frame}, "
                f"timeout={self.collect_sync_wait_timeout_sec:.3f}s",
                "cyan",
            )
        )
    return buffer.lookup_transform(
        target_frame,
        source_frame,
        query_time,
        timeout=Duration(seconds=self.collect_sync_wait_timeout_sec),
    )


def _record_tf_quality_message(
    self,
    message: TFMessage,
    source_topic: str,
    *,
    is_static: bool,
) -> None:
    """독립 TF 재구성과 보간 판정을 위해 raw TF와 ns stamp를 보존한다."""
    latest_stamp_ns = 0
    with self._tf_quality_lock:
        for transform in message.transforms:
            edge = (str(transform.header.frame_id), str(transform.child_frame_id))
            if is_static:
                self._tf_quality_static_edges.add(edge)
                self._tf_quality_buffer.set_transform_static(transform, source_topic)
                continue
            stamp_ns = int(Time.from_msg(transform.header.stamp).nanoseconds)
            latest_stamp_ns = max(latest_stamp_ns, stamp_ns)
            records = self._tf_quality_dynamic_stamps.setdefault(edge, {})
            records.setdefault(stamp_ns, set()).add(source_topic)
            self._tf_quality_buffer.set_transform(transform, source_topic)

        if latest_stamp_ns - self._tf_quality_last_prune_ns >= 1_000_000_000:
            cutoff_ns = latest_stamp_ns - 10_000_000_000
            for records in self._tf_quality_dynamic_stamps.values():
                for stamp_ns in [value for value in records if value < cutoff_ns]:
                    del records[stamp_ns]
            self._tf_quality_last_prune_ns = latest_stamp_ns


def _tf_path_edges(
    dynamic_edges: set[tuple[str, str]],
    static_edges: set[tuple[str, str]],
    target_frame: str,
    source_frame: str,
) -> list[tuple[str, str]] | None:
    """현재 raw TF graph에서 target과 source를 잇는 edge 경로를 찾는다."""
    graph: dict[str, list[tuple[str, tuple[str, str]]]] = {}
    for edge in dynamic_edges | static_edges:
        parent, child = edge
        graph.setdefault(parent, []).append((child, edge))
        graph.setdefault(child, []).append((parent, edge))
    queue = deque([(target_frame, [])])
    visited = {target_frame}
    while queue:
        frame, path = queue.popleft()
        if frame == source_frame:
            return path
        for neighbor, edge in graph.get(frame, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, edge]))
    return None


def _tf_interpolation_provenance(
    self,
    target_frame: str,
    source_frame: str,
    capture_stamp_ns: int,
) -> dict[str, Any]:
    """경로별 exact raw TF 또는 앞뒤 보간 stamp를 ns 단위로 기록한다."""
    with self._tf_quality_lock:
        dynamic_edges = {
            edge for edge, records in self._tf_quality_dynamic_stamps.items()
            if records
        }
        static_edges = set(self._tf_quality_static_edges)
        path = _tf_path_edges(
            dynamic_edges, static_edges, target_frame, source_frame
        )
        dynamic = {
            edge: {stamp: sorted(topics) for stamp, topics in
                   self._tf_quality_dynamic_stamps.get(edge, {}).items()}
            for edge in (path or [])
            if edge in dynamic_edges
        }
    if path is None:
        return {
            "capture_stamp_ns": int(capture_stamp_ns),
            "target_frame_id": target_frame,
            "source_frame_id": source_frame,
            "path_available": False,
            "all_dynamic_edges_bracket_capture": False,
            "exact_transform": False,
            "interpolated": False,
            "edges": [],
        }

    edge_records = []
    all_bracketed = True
    all_exact = True
    maximum_span_ns = 0
    any_interpolated = False
    for parent, child in path:
        edge = (parent, child)
        if edge in static_edges:
            edge_records.append({
                "parent_frame_id": parent,
                "child_frame_id": child,
                "is_static": True,
                "exact_sample": None,
                "interpolated": False,
            })
            continue
        records = dynamic.get(edge, {})
        stamps = sorted(records)
        index = bisect_left(stamps, capture_stamp_ns)
        exact = index < len(stamps) and stamps[index] == capture_stamp_ns
        previous_ns = capture_stamp_ns if exact else (stamps[index - 1] if index else None)
        next_ns = capture_stamp_ns if exact else (stamps[index] if index < len(stamps) else None)
        bracketed = exact or (previous_ns is not None and next_ns is not None)
        span_ns = (
            int(next_ns - previous_ns)
            if previous_ns is not None and next_ns is not None
            else None
        )
        ratio = (
            float(capture_stamp_ns - previous_ns) / float(span_ns)
            if span_ns not in (None, 0) and previous_ns is not None
            else 0.0 if exact else None
        )
        all_bracketed = all_bracketed and bracketed
        all_exact = all_exact and exact
        any_interpolated = any_interpolated or (bracketed and not exact)
        maximum_span_ns = max(maximum_span_ns, span_ns or 0)
        topics = set()
        if previous_ns is not None:
            topics.update(records.get(previous_ns, []))
        if next_ns is not None:
            topics.update(records.get(next_ns, []))
        edge_records.append({
            "parent_frame_id": parent,
            "child_frame_id": child,
            "is_static": False,
            "exact_sample": exact,
            "interpolated": bracketed and not exact,
            "previous_stamp_ns": previous_ns,
            "next_stamp_ns": next_ns,
            "interpolation_span_ns": span_ns,
            "interpolation_ratio": ratio,
            "source_topics": sorted(topics),
        })
    return {
        "capture_stamp_ns": int(capture_stamp_ns),
        "target_frame_id": target_frame,
        "source_frame_id": source_frame,
        "path_available": True,
        "all_dynamic_edges_bracket_capture": all_bracketed,
        "exact_transform": all_exact,
        "interpolated": any_interpolated,
        "maximum_interpolation_span_ns": int(maximum_span_ns),
        "edges": edge_records,
    }


def _transform_difference(
    left: TransformStamped,
    right: TransformStamped,
) -> tuple[float, float]:
    """두 stamped transform의 위치 norm과 quaternion 각도 차이를 반환한다."""
    left_t = left.transform.translation
    right_t = right.transform.translation
    position_difference_m = math.sqrt(
        (float(left_t.x) - float(right_t.x)) ** 2
        + (float(left_t.y) - float(right_t.y)) ** 2
        + (float(left_t.z) - float(right_t.z)) ** 2
    )
    left_q = np.array([
        left.transform.rotation.x,
        left.transform.rotation.y,
        left.transform.rotation.z,
        left.transform.rotation.w,
    ], dtype=float)
    right_q = np.array([
        right.transform.rotation.x,
        right.transform.rotation.y,
        right.transform.rotation.z,
        right.transform.rotation.w,
    ], dtype=float)
    left_q /= np.linalg.norm(left_q)
    right_q /= np.linalg.norm(right_q)
    dot = min(1.0, max(-1.0, abs(float(np.dot(left_q, right_q)))))
    return position_difference_m, float(2.0 * math.acos(dot))


def _capture_tf_quality_metadata(
    self,
    live_transform: TransformStamped,
    target_frame: str,
    source_frame: str,
    capture_stamp_ns: int,
) -> tuple[bool, dict[str, Any]]:
    """독립 raw TF 재구성과 live TF가 허용 오차 안인지 검사한다."""
    result = {
        "position_tolerance_m": TF_RECONSTRUCTION_POSITION_TOLERANCE_M,
        "angle_tolerance_rad": TF_RECONSTRUCTION_ANGLE_TOLERANCE_RAD,
        "valid": False,
    }
    try:
        reconstructed = self._tf_quality_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(nanoseconds=int(capture_stamp_ns)),
            timeout=Duration(seconds=self.collect_sync_wait_timeout_sec),
        )
    except TransformException as exc:
        result["provenance"] = _tf_interpolation_provenance(
            self, target_frame, source_frame, capture_stamp_ns
        )
        result["reconstruction_error"] = str(exc)
        result["rejection_reason"] = "raw_tf_reconstruction_unavailable"
        return False, result

    provenance = _tf_interpolation_provenance(
        self, target_frame, source_frame, capture_stamp_ns
    )
    result["provenance"] = provenance
    position_difference_m, angle_difference_rad = _transform_difference(
        live_transform, reconstructed
    )
    result.update({
        "position_difference_m": position_difference_m,
        "angle_difference_rad": angle_difference_rad,
        "reconstructed_transform": {
            "translation_m": {
                "x": float(reconstructed.transform.translation.x),
                "y": float(reconstructed.transform.translation.y),
                "z": float(reconstructed.transform.translation.z),
            },
            "rotation_xyzw": {
                "x": float(reconstructed.transform.rotation.x),
                "y": float(reconstructed.transform.rotation.y),
                "z": float(reconstructed.transform.rotation.z),
                "w": float(reconstructed.transform.rotation.w),
            },
        },
    })
    if not provenance["path_available"] or not provenance[
        "all_dynamic_edges_bracket_capture"
    ]:
        result["rejection_reason"] = "raw_tf_bracketing_unavailable"
        return False, result
    valid = (
        position_difference_m <= TF_RECONSTRUCTION_POSITION_TOLERANCE_M
        and angle_difference_rad <= TF_RECONSTRUCTION_ANGLE_TOLERANCE_RAD
    )
    result["valid"] = valid
    if not valid:
        result["rejection_reason"] = "tf_reconstruction_difference_exceeded"
    return valid, result


def _collect_log_text(self, message: str, color: str, *, bold: bool = True) -> str:
    """PortOffset 상태 메시지에 설정된 ANSI 색상과 굵기를 적용한다."""
    if not self.collect_color_log:
        return message
    prefix = (_LOG_COLORS["bold"] if bold else "") + _LOG_COLORS.get(color, "")
    return f"{prefix}{message}{_LOG_COLORS['reset']}"

def _select_port_frame(self, task: Task) -> str:
    """포트 입구 frame이 있으면 사용하고, 없으면 기본 port link frame으로 fallback한다."""
    port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
    entrance_frame = f"{port_frame}_entrance"
    if self._wait_for_tf("base_link", entrance_frame, timeout_sec=2.0):
        self.get_logger().info(f"[PortOffsetCollect] Using port entrance frame: {entrance_frame}")
        return entrance_frame
    self.get_logger().warn(f"[PortOffsetCollect] Port entrance TF unavailable, falling back to: {port_frame}")
    return port_frame

def _select_cable_tip_frame(self, task: Task) -> str:
    """케이블 끝단 제어를 위해 task 정보에서 사용 가능한 cable tip frame을 선택한다."""
    cable_prefix = task.cable_name
    plug_name = task.plug_name.strip()
    plug_type = task.plug_type.strip().lower()
    candidates = []

    if plug_name:
        candidates.append(f"{cable_prefix}/{plug_name}_link")
        if not plug_name.endswith("_tip"):
            candidates.append(f"{cable_prefix}/{plug_name}_tip_link")
    if plug_type:
        candidates.append(f"{cable_prefix}/{plug_type}_tip_link")

    for frame in dict.fromkeys(candidates):
        if self._wait_for_tf("base_link", frame, timeout_sec=0.5):
            self.get_logger().info(f"[PortOffsetCollect] Using cable tip frame: {frame}")
            return frame

    fallback_frame = f"{cable_prefix}/{plug_name}_link"
    self.get_logger().warn(
        f"[PortOffsetCollect] Cable tip TF unavailable, falling back to plug frame: {fallback_frame}"
    )
    return fallback_frame

def set_pose_target(self, move_robot, pose, stiffness=None, damping=None):
    """주어진 목표 pose와 impedance 값으로 controller에 MotionUpdate 명령을 보낸다."""
    _s = stiffness if stiffness is not None else STIFFNESS_DEFAULT
    _d = damping if damping is not None else DAMPING_DEFAULT
    mu = MotionUpdate(
        header=Header(frame_id="base_link", stamp=self.get_clock().now().to_msg()),
        pose=pose, target_stiffness=np.diag(_s).flatten(), target_damping=np.diag(_d).flatten(),
        feedforward_wrench_at_tip=Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0)),
        wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
        trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION),
    )
    try: move_robot(motion_update=mu)
    except Exception: pass
