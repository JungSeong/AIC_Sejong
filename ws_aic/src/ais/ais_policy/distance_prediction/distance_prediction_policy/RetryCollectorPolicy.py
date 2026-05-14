"""Debug distance policy variant that records retry-classifier features."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from std_msgs.msg import String

try:
    import cv2
except Exception:  # pragma: no cover - npy fallback is used if cv2 is unavailable.
    cv2 = None

from distance_prediction_policy.DebugSfpDistancePolicy import DebugSfpDistancePolicy
from distance_prediction_policy.config import DistancePredictionConfig, WS_ROOT


SUCCESS_CLASS = "complete_insert"
FEATURE_COLUMNS = (
    "pred_xy_offset_mm",
    "fz",
    "delta_fz",
    "fxy_norm",
    "cmd_insert_depth_mm",
)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader.fieldnames or [])
            existing_rows = list(reader)
        if existing:
            fieldnames = existing + [key for key in row.keys() if key not in existing]
            if len(fieldnames) != len(existing):
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(existing_rows)
        else:
            write_header = True

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class RetryCollectorPolicy(DebugSfpDistancePolicy):
    """Run DebugSfpDistancePolicy and save deployable align/insert features."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._retry_run_id = time.strftime("%Y%m%d_%H%M%S")
        self._retry_output_dir = Path(
            os.environ.get(
                "AIC_RETRY_OUTPUT_DIR",
                str(WS_ROOT / "data" / "retry_classifier"),
            )
        ).expanduser().resolve()
        self._retry_episode_prefix = os.environ.get("AIC_RETRY_EPISODE_PREFIX", "retry")
        self._retry_episode_id_override = os.environ.get("AIC_RETRY_EPISODE_ID", "")
        self._retry_intended_class = os.environ.get("AIC_RETRY_INTENDED_CLASS", "")
        self._retry_force_class = os.environ.get("AIC_RETRY_FORCE_CLASS", "")
        self._retry_sample_period_s = max(0.0, _env_float("AIC_RETRY_SAMPLE_PERIOD_S", 0.15))

        self._centered_xy_mm = _env_float("AIC_RETRY_CENTERED_XY_MM", 2.0)
        self._wall_xy_mm = _env_float("AIC_RETRY_WALL_XY_MM", 4.0)
        self._fxy_contact_n = _env_float("AIC_RETRY_FXY_CONTACT_N", 4.0)
        self._fz_contact_n = _env_float("AIC_RETRY_FZ_CONTACT_N", 4.0)
        self._fz_stuck_n = _env_float("AIC_RETRY_FZ_STUCK_N", 8.0)
        self._min_cmd_insert_depth_mm = _env_float("AIC_RETRY_MIN_CMD_INSERT_DEPTH_MM", 5.0)

        self._retry_episode_id = ""
        self._retry_episode_index = 0
        self._retry_samples: list[dict[str, float | str]] = []
        self._retry_feature_observation = None
        self._retry_success_event_observed = False
        self._retry_latest_insertion_event = ""
        self._retry_align_baseline_fz = 0.0
        self._retry_insert_start_z: float | None = None
        self._retry_last_sample_time = -1e9
        self._retry_prediction_fail_count = 0
        self._retry_finished = False

        self._retry_insertion_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._on_retry_insertion_event,
            10,
        )
        self.get_logger().info(
            "RetryCollectorPolicy ready: "
            f"output={self._retry_output_dir}, sample_period={self._retry_sample_period_s:.2f}s, "
            f"features={','.join(FEATURE_COLUMNS)}"
        )

    def _on_retry_insertion_event(self, msg: String) -> None:
        value = msg.data.strip().strip("/")
        if not value:
            return
        self._retry_latest_insertion_event = value
        self._retry_success_event_observed = True

    def _reset_retry_episode(self, task) -> None:
        self._retry_episode_index += 1
        task_id = str(getattr(task, "id", "") or "").strip()
        if self._retry_episode_id_override:
            episode_id = self._retry_episode_id_override
        elif task_id:
            episode_id = (
                f"{self._retry_episode_prefix}_{task_id}_"
                f"{getattr(task, 'target_module_name', '')}_"
                f"{getattr(task, 'port_name', '')}_"
                f"{self._retry_run_id}_{self._retry_episode_index:04d}"
            )
        else:
            episode_id = (
                f"{self._retry_episode_prefix}_{self._retry_run_id}_"
                f"{self._retry_episode_index:04d}"
            )
        self._retry_episode_id = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in episode_id
        )
        self._retry_samples = []
        self._retry_feature_observation = None
        self._retry_success_event_observed = False
        self._retry_latest_insertion_event = ""
        self._retry_align_baseline_fz = 0.0
        self._retry_insert_start_z = None
        self._retry_last_sample_time = -1e9
        self._retry_prediction_fail_count = 0
        self._retry_finished = False
        self.get_logger().info(
            "retry collector episode start: "
            f"episode={self._retry_episode_id}, output={self._retry_output_dir}"
        )

    def _elapsed_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _image_msg_to_rgb(self, image_msg) -> np.ndarray:
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = getattr(image_msg, "encoding", "").lower()
        if height <= 0 or width <= 0:
            raise ValueError("empty image")

        if encoding in {"rgba8", "bgra8"}:
            channels = 4
        elif encoding in {"rgb8", "bgr8"}:
            channels = 3
        elif encoding in {"mono8", "8uc1"}:
            channels = 1
        else:
            pixel_count = height * width
            data_len = len(image_msg.data)
            channels = 4 if data_len >= pixel_count * 4 else 3

        try:
            flat = np.frombuffer(image_msg.data, dtype=np.uint8)
        except TypeError:
            flat = np.asarray(image_msg.data, dtype=np.uint8)
        step = int(getattr(image_msg, "step", 0))
        if step > 0 and flat.size >= height * step:
            rows = flat[: height * step].reshape(height, step)
            image = rows[:, : width * channels].reshape(height, width, channels)
        else:
            image = flat[: height * width * channels].reshape(height, width, channels)

        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 4:
            image = image[:, :, :3]
        if encoding in {"bgr8", "bgra8"}:
            image = image[:, :, ::-1]
        return np.ascontiguousarray(image)

    def _save_retry_center_image(self) -> str:
        obs = self._retry_feature_observation
        if obs is None:
            return ""
        try:
            rgb = self._image_msg_to_rgb(obs.center_image)
        except Exception as exc:
            self.get_logger().warn(f"retry center image save skipped: {exc}")
            return ""

        image_dir = self._retry_output_dir / "images" / "center"
        image_dir.mkdir(parents=True, exist_ok=True)
        if cv2 is not None:
            path = image_dir / f"{self._retry_episode_id}_center.png"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                self.get_logger().warn(f"retry center image save failed: {path}")
                return ""
        else:
            path = image_dir / f"{self._retry_episode_id}_center.npy"
            np.save(path, rgb)
        return str(path)

    def _retry_predict_xy_offset_mm(self, obs) -> Optional[float]:
        try:
            offset_m = self._distance.predict_offset_m(obs, self._port_id())
        except Exception as exc:
            self._retry_prediction_fail_count += 1
            if self._retry_prediction_fail_count == 1:
                self.get_logger().warn(f"retry distance prediction failed: {exc}")
            return None
        if offset_m is None:
            return None
        offset = np.asarray(offset_m, dtype=np.float64)
        return float(np.linalg.norm(offset[:2]) * 1000.0)

    def _record_retry_sample(
        self,
        *,
        obs,
        stage: str,
        cmd_insert_depth_mm: float = 0.0,
        force: bool = False,
    ) -> None:
        if obs is None:
            return
        now = self._elapsed_s()
        if (
            not force
            and self._retry_sample_period_s > 0.0
            and now - self._retry_last_sample_time < self._retry_sample_period_s
        ):
            return

        pred_xy = self._retry_predict_xy_offset_mm(obs)
        if pred_xy is None:
            return
        wrench_force = obs.wrist_wrench.wrench.force
        fx = float(wrench_force.x)
        fy = float(wrench_force.y)
        fz = float(wrench_force.z)
        self._retry_samples.append(
            {
                "stage": stage,
                "t_s": now,
                "pred_xy_offset_mm": pred_xy,
                "fz": fz,
                "delta_fz": float(fz - self._retry_align_baseline_fz),
                "fxy_norm": float(math.hypot(fx, fy)),
                "cmd_insert_depth_mm": float(max(0.0, cmd_insert_depth_mm)),
            }
        )
        self._retry_last_sample_time = now
        self._retry_feature_observation = obs

    def _label_retry_row(self, row: dict[str, Any]) -> dict[str, str | int]:
        if self._retry_force_class:
            return {
                "class_name": self._retry_force_class,
                "binary_success": 1 if self._retry_force_class == SUCCESS_CLASS else 0,
                "label_reason": "forced_by_env",
            }
        if self._retry_success_event_observed:
            return {
                "class_name": SUCCESS_CLASS,
                "binary_success": 1,
                "label_reason": "insertion_event_observed",
            }

        pred_xy = float(row.get("pred_xy_offset_mm", 1e9))
        delta_fz = float(row.get("delta_fz", 0.0))
        fxy_norm = float(row.get("fxy_norm", 0.0))
        cmd_depth = float(row.get("cmd_insert_depth_mm", 0.0))

        if pred_xy <= self._centered_xy_mm and (
            delta_fz >= self._fz_stuck_n
            or (cmd_depth >= self._min_cmd_insert_depth_mm and delta_fz >= self._fz_contact_n)
        ):
            return {
                "class_name": "partial_insert",
                "binary_success": 0,
                "label_reason": "near_center_with_force_stuck",
            }
        if pred_xy >= self._wall_xy_mm and fxy_norm >= self._fxy_contact_n:
            return {
                "class_name": "side_wall_contact",
                "binary_success": 0,
                "label_reason": "far_from_center_with_lateral_force",
            }
        if pred_xy >= self._wall_xy_mm and delta_fz >= self._fz_contact_n:
            return {
                "class_name": "top_surface_contact",
                "binary_success": 0,
                "label_reason": "far_from_center_with_vertical_force",
            }
        return {
            "class_name": "timeout_or_unknown",
            "binary_success": 0,
            "label_reason": "no_success_event_observed",
        }

    def _finish_retry_episode(self, *, insert_result: bool, reason: str) -> None:
        if self._retry_finished:
            return
        self._retry_finished = True
        final_features: dict[str, Any] = {column: 0.0 for column in FEATURE_COLUMNS}
        final_stage = ""
        if self._retry_samples:
            final = self._retry_samples[-1]
            final_features.update({column: final[column] for column in FEATURE_COLUMNS})
            final_stage = str(final.get("stage", ""))
        center_image_path = self._save_retry_center_image()
        row = {
            "episode_id": self._retry_episode_id,
            "run_id": self._retry_run_id,
            "policy": self.__class__.__name__,
            "task_id": str(getattr(self._task, "id", "") or ""),
            "intended_class": self._retry_intended_class,
            "target_module_name": str(getattr(self._task, "target_module_name", "") or ""),
            "port_name": str(getattr(self._task, "port_name", "") or ""),
            "cable_name": str(getattr(self._task, "cable_name", "") or ""),
            "plug_name": str(getattr(self._task, "plug_name", "") or ""),
            "center_image_path": center_image_path,
            "sample_count": len(self._retry_samples),
            "final_stage": final_stage,
            "insert_result": int(bool(insert_result)),
            "finish_reason": reason,
            "success_event_observed": int(self._retry_success_event_observed),
            "latest_insertion_event": self._retry_latest_insertion_event,
            **final_features,
        }
        row.update(self._label_retry_row(row))
        _append_csv_row(self._retry_output_dir / "features.csv", row)
        _append_jsonl(self._retry_output_dir / "episodes.jsonl", row)
        self.get_logger().info(
            "retry collector episode saved: "
            f"episode={self._retry_episode_id}, class={row['class_name']}, "
            f"success={row['binary_success']}, samples={row['sample_count']}, "
            f"csv={self._retry_output_dir / 'features.csv'}"
        )

    def _stage_align(self, get_observation, move_robot) -> bool:
        obs = get_observation()
        force = self._force_vector(obs)
        self._retry_align_baseline_fz = float(force[2]) if force is not None else 0.0
        self._record_retry_sample(obs=obs, stage="align_start", force=True)

        def observed_get_observation():
            current = get_observation()
            self._record_retry_sample(obs=current, stage="align")
            return current

        result = super()._stage_align(observed_get_observation, move_robot)
        self._record_retry_sample(obs=get_observation(), stage="align_end", force=True)
        if not result:
            self._finish_retry_episode(insert_result=False, reason="align_failed")
        return result

    def _stage_insert(self, get_observation, move_robot) -> bool:
        self.get_logger().info("[stage 5/5] insert start + retry collection")
        obs = get_observation()
        pose = self._tcp_pose(obs)
        if pose is None:
            self.get_logger().error("insert failed: missing TCP pose")
            self._finish_retry_episode(insert_result=False, reason="missing_tcp_pose")
            return False

        max_depth = float(DistancePredictionConfig.MAX_INSERT_DEPTH_M)
        step_m = float(DistancePredictionConfig.MAX_DOWN_STEP_M)
        steps = min(
            int(math.ceil(max_depth / max(step_m, 1e-6))),
            DistancePredictionConfig.INSERT_MAX_STEPS,
        )
        start_z = float(pose.position.z)
        self._retry_insert_start_z = start_z
        baseline_force = self._force_norm(obs)
        self._record_retry_sample(obs=obs, stage="insert_start", cmd_insert_depth_mm=0.0, force=True)
        if baseline_force is None:
            self.get_logger().warn("insert force baseline unavailable; force guard disabled")
        else:
            self.get_logger().info(
                f"insert force baseline: {baseline_force:.2f}N, "
                f"align_baseline_fz={self._retry_align_baseline_fz:+.2f}N, "
                f"delta_limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
            )

        for step in range(steps):
            obs = get_observation()
            force = self._force_norm(obs)
            force_delta = None
            if force is not None and baseline_force is not None:
                force_delta = force - baseline_force
            if (
                force_delta is not None
                and force_delta > DistancePredictionConfig.FORCE_LIMIT_N
            ):
                self.get_logger().warn(
                    "insert force delta limit: "
                    f"force={force:.2f}N, baseline={baseline_force:.2f}N, "
                    f"delta={force_delta:.2f}N, "
                    f"limit={DistancePredictionConfig.FORCE_LIMIT_N:.2f}N"
                )
                self._record_retry_sample(
                    obs=obs,
                    stage="insert_force_limit",
                    cmd_insert_depth_mm=step * step_m * 1000.0,
                    force=True,
                )
                self._finish_retry_episode(insert_result=False, reason="force_delta_limit")
                return False

            current = self._tcp_pose(obs) or pose
            target_pose = self._copy_pose(current)
            target_pose.position.z = float(start_z - (step + 1) * step_m)
            if self._target_orientation is not None:
                target_pose.orientation = self._copy_quaternion(self._target_orientation)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=list(self._insertion_stiffness()),
                damping=list(self._insertion_damping()),
            )
            if step == 0 or step % 10 == 0:
                force_text = ""
                if force is not None and force_delta is not None:
                    force_text = f", force_delta={force_delta:+.2f}N"
                self.get_logger().info(
                    f"insert[{step:03d}]: dz={-(step + 1) * step_m * 1000:.1f}mm"
                    f"{force_text}"
                )
            self.sleep_for(DistancePredictionConfig.DT)
            self._record_retry_sample(
                obs=get_observation(),
                stage="insert",
                cmd_insert_depth_mm=(step + 1) * step_m * 1000.0,
            )

        if DistancePredictionConfig.SETTLE_AFTER_INSERT_S > 0:
            self.sleep_for(DistancePredictionConfig.SETTLE_AFTER_INSERT_S)
            self._record_retry_sample(
                obs=get_observation(),
                stage="settle",
                cmd_insert_depth_mm=steps * step_m * 1000.0,
                force=True,
            )
        self.get_logger().info("[stage 5/5] insert done")
        self._finish_retry_episode(insert_result=True, reason="insert_stage_done")
        return True

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._reset_retry_episode(task)
        result = super().insert_cable(task, get_observation, move_robot, send_feedback)
        if not self._retry_finished:
            self._finish_retry_episode(insert_result=bool(result), reason="policy_finished")
        return result
