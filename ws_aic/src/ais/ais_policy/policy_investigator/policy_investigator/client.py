from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np


CAMERAS = ("left", "center", "right")
REALTIME_EVENT_TYPES = {"align_command", "command_error"}
TASK_FIELDS = (
    "id",
    "cable_type",
    "cable_name",
    "plug_type",
    "plug_name",
    "port_type",
    "port_name",
    "target_module_name",
    "time_limit",
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _xyz_from_pose(pose: Any) -> dict[str, float]:
    return {
        "x": float(pose.position.x),
        "y": float(pose.position.y),
        "z": float(pose.position.z),
    }


def _quat_from_pose(pose: Any) -> dict[str, float]:
    return {
        "x": float(pose.orientation.x),
        "y": float(pose.orientation.y),
        "z": float(pose.orientation.z),
        "w": float(pose.orientation.w),
    }


def pose_to_dict(pose: Any | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "position": _xyz_from_pose(pose),
        "orientation": _quat_from_pose(pose),
    }


def vector_to_dict(values: Any) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    result = {"x": 0.0, "y": 0.0, "z": 0.0}
    for index, axis in enumerate(("x", "y", "z")):
        if index < vector.size:
            result[axis] = float(vector[index])
    return result


def image_msg_to_bgr(image_msg: Any) -> np.ndarray | None:
    if image_msg is None:
        return None
    height = int(getattr(image_msg, "height", 0))
    width = int(getattr(image_msg, "width", 0))
    if height <= 0 or width <= 0:
        return None

    encoding = str(getattr(image_msg, "encoding", "")).lower()
    if encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"rgb8", "bgr8"}:
        channels = 3
    else:
        pixel_count = height * width
        channels = 4 if len(image_msg.data) >= pixel_count * 4 else 3

    flat = np.frombuffer(image_msg.data, dtype=np.uint8)
    step = int(getattr(image_msg, "step", 0))
    if step > 0 and flat.size >= height * step:
        rows = flat[: height * step].reshape(height, step)
        image = rows[:, : width * channels].reshape(height, width, channels)
    else:
        expected = height * width * channels
        if flat.size < expected:
            return None
        image = flat[:expected].reshape(height, width, channels)

    if channels == 4:
        image = image[:, :, :3]
    if encoding in {"rgb8", "rgba8"}:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image)


class PolicyInvestigator:
    """Publish lightweight JSON events and save visual artifacts for a policy run."""

    def __init__(
        self,
        node: Any,
        *,
        policy_name: str,
        logger: Any = None,
        topic: str | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.node = node
        self.policy_name = policy_name
        self.logger = logger
        self.enabled = _env_bool("AIC_POLICY_INVESTIGATOR_ENABLE", True)
        self.topic = topic or os.environ.get(
            "AIC_POLICY_INVESTIGATOR_TOPIC",
            "/policy_investigator/events",
        )
        self.output_dir = Path(
            output_dir or os.environ.get(
                "AIC_POLICY_INVESTIGATOR_DIR",
                "/tmp/aic_policy_investigator",
            )
        ).expanduser()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_id
        self._seq = 0
        self._publisher = None
        self._msg_type = None
        self._gui_process: subprocess.Popen | None = None
        self._queue: queue.Queue[tuple[str, Any]] | None = None
        self._stop_event = threading.Event()
        self._seq_lock = threading.Lock()
        self._realtime_lock = threading.Lock()
        self._realtime_event = threading.Event()
        self._realtime_records: dict[str, tuple[dict[str, Any], Path]] = {}
        self._dropped_events = 0
        self._low_latency = _env_bool("AIC_POLICY_INVESTIGATOR_LOW_LATENCY", True)
        self._kill_gui_on_exit = _env_bool("AIC_POLICY_INVESTIGATOR_KILL_GUI_ON_EXIT", True)
        self._closed = False

        if not self.enabled:
            return

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.run_dir / "events.jsonl"
        self._init_publisher()
        if not self.enabled:
            return
        self._start_worker()
        self._maybe_launch_gui()
        atexit.register(self.close)
        self.event("investigator_ready", output_dir=str(self.run_dir), topic=self.topic)

    def _log_warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warn(message)

    def _init_publisher(self) -> None:
        try:
            from std_msgs.msg import String
        except Exception as exc:
            self._log_warn(f"Policy investigator disabled: std_msgs unavailable ({exc})")
            self.enabled = False
            return
        try:
            self._publisher = self.node.create_publisher(String, self.topic, 10)
            self._msg_type = String
        except Exception as exc:
            self._log_warn(f"Policy investigator publisher unavailable: {exc}")
            self.enabled = False

    def _start_worker(self) -> None:
        max_queue = int(os.environ.get("AIC_POLICY_INVESTIGATOR_QUEUE_SIZE", "50"))
        self._queue = queue.Queue(maxsize=max(1, max_queue))
        threading.Thread(
            target=self._worker_loop,
            name="policy_investigator_worker",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._realtime_worker_loop,
            name="policy_investigator_realtime_worker",
            daemon=True,
        ).start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._queue is None:
                return
            try:
                job_type, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if job_type == "event":
                    record, jsonl_path = payload
                    self._write_and_publish(record, jsonl_path)
                elif job_type == "snapshot":
                    record, jsonl_path, image_jobs = payload
                    self._save_snapshot_images(record, image_jobs)
                    self._write_and_publish(record, jsonl_path)
            except Exception as exc:
                self._log_warn(f"Policy investigator worker failed: {exc}")
            finally:
                self._queue.task_done()

    def _enqueue(self, job_type: str, payload: Any) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait((job_type, payload))
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
            self._queue.task_done()
        except queue.Empty:
            pass
        self._dropped_events += 1
        if self._dropped_events == 1 or self._dropped_events % 50 == 0:
            self._log_warn(
                "Policy investigator queue full; dropping old visualization events "
                f"(dropped={self._dropped_events})"
            )
        try:
            self._queue.put_nowait((job_type, payload))
        except queue.Full:
            self._dropped_events += 1

    def _enqueue_realtime(self, record: dict[str, Any], jsonl_path: Path) -> None:
        event_type = str(record.get("type", "event"))
        with self._realtime_lock:
            self._realtime_records[event_type] = (record, jsonl_path)
        self._realtime_event.set()

    def _realtime_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._realtime_event.wait(timeout=0.2)
            if self._stop_event.is_set():
                return
            with self._realtime_lock:
                pending = list(self._realtime_records.values())
                self._realtime_records.clear()
                self._realtime_event.clear()
            for record, jsonl_path in pending:
                self._write_and_publish(record, jsonl_path)

    def _write_and_publish(self, record: dict[str, Any], jsonl_path: Path) -> None:
        line = json.dumps(record, ensure_ascii=False)
        if self._publisher is not None and self._msg_type is not None:
            msg = self._msg_type()
            msg.data = line
            try:
                self._publisher.publish(msg)
            except Exception as exc:
                self._log_warn(f"Policy investigator publish failed: {exc}")
        try:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            self._log_warn(f"Policy investigator JSONL write failed: {exc}")

    def _save_snapshot_images(
        self,
        record: dict[str, Any],
        image_jobs: list[tuple[str, Any, Path]],
    ) -> None:
        images = record.get("images", {})
        for camera, image_msg, path in image_jobs:
            camera_info = images.get(camera, {})
            bgr = image_msg_to_bgr(image_msg)
            if bgr is None:
                camera_info["available"] = False
                camera_info["path"] = ""
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), bgr)
                camera_info["available"] = True
                camera_info["path"] = str(path)
            except Exception as exc:
                camera_info["available"] = False
                camera_info["path"] = ""
                camera_info["error"] = str(exc)

    def _maybe_launch_gui(self) -> None:
        if not _env_bool("AIC_POLICY_INVESTIGATOR_GUI", True):
            return
        if os.environ.get("AIC_POLICY_INVESTIGATOR_GUI_CHILD") == "1":
            return
        env = os.environ.copy()
        env["AIC_POLICY_INVESTIGATOR_GUI_CHILD"] = "1"
        command = [
            sys.executable,
            "-m",
            "policy_investigator.gui",
            "--topic",
            self.topic,
            "--output-dir",
            str(self.run_dir),
            "--parent-pid",
            str(os.getpid()),
        ]
        try:
            self._gui_process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self._log_warn(f"Policy investigator GUI launch failed: {exc}")

    def close(self, reason: str = "client_exit") -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        try:
            record = self._make_record("shutdown", reason=reason)
            self._write_and_publish(record, self._jsonl_path)
        except Exception:
            pass
        self._stop_event.set()
        self._realtime_event.set()
        if self._kill_gui_on_exit:
            self._terminate_gui_process()

    def _terminate_gui_process(self) -> None:
        process = self._gui_process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
        except Exception:
            pass

    def start_run(self, task: Any) -> None:
        task_id = str(getattr(task, "id", "") or "")
        safe_task = task_id or f"{getattr(task, 'target_module_name', 'task')}_{getattr(task, 'port_name', 'port')}"
        safe_task = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in safe_task)
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_task}"
        self.run_dir = self.output_dir / self.run_id
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self.run_dir / "events.jsonl"
        task_info = {field: str(getattr(task, field, "")) for field in TASK_FIELDS}
        self.event(
            "run_start",
            task_info=task_info,
            task_id=task_id,
            **task_info,
        )

    def event(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        record = {
            "type": event_type,
            "seq": seq,
            "time": time.time(),
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "policy": self.policy_name,
            "run_id": self.run_id,
            **payload,
        }
        record = _json_safe(record)
        if self._low_latency and event_type in REALTIME_EVENT_TYPES:
            self._enqueue_realtime(record, self._jsonl_path)
            return
        self._enqueue("event", (record, self._jsonl_path))

    def stage(self, name: str, status: str, detail: str = "") -> None:
        self.event("stage", stage=name, status=status, detail=detail)

    def log(self, stage: str, message: str, level: str = "info") -> None:
        self.event("log", stage=stage, level=level, message=message)

    def camera_snapshot(
        self,
        observation: Any,
        *,
        stage: str,
        note: str = "",
        subdir: str = "camera",
    ) -> dict[str, Any]:
        if not self.enabled or observation is None:
            return {}
        image_dir = self.run_dir / subdir / stage
        images: dict[str, Any] = {}
        image_jobs: list[tuple[str, Any, Path]] = []
        with self._seq_lock:
            next_seq = self._seq + 1
        for camera in CAMERAS:
            image_msg = getattr(observation, f"{camera}_image", None)
            width = int(getattr(image_msg, "width", 0)) if image_msg is not None else 0
            height = int(getattr(image_msg, "height", 0)) if image_msg is not None else 0
            has_image = image_msg is not None and width > 0 and height > 0
            path = image_dir / f"{next_seq:06d}_{camera}.jpg"
            info = {
                "available": has_image,
                "encoding": str(getattr(image_msg, "encoding", "")) if image_msg is not None else "",
                "width": width,
                "height": height,
                "path": str(path) if has_image else "",
            }
            if has_image:
                image_jobs.append((camera, image_msg, path))
            images[camera] = info
        record = self._make_record(
            "camera_snapshot",
            stage=stage,
            note=note,
            images=images,
        )
        self._enqueue("snapshot", (record, self._jsonl_path, image_jobs))
        return images

    def _make_record(self, event_type: str, **payload: Any) -> dict[str, Any]:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return _json_safe(
            {
                "type": event_type,
                "seq": seq,
                "time": time.time(),
                "created_at": datetime.now().isoformat(timespec="milliseconds"),
                "policy": self.policy_name,
                "run_id": self.run_id,
                **payload,
            }
        )

    def yolo_capture(
        self,
        observation: Any,
        *,
        stage: str,
        port_base_m: Any = None,
        debug_paths: Mapping[str, Any] | None = None,
    ) -> None:
        images = self.camera_snapshot(
            observation,
            stage=stage,
            note="yolo_capture",
            subdir="yolo_capture",
        )
        self.event(
            "yolo_capture",
            stage=stage,
            port_base_m=vector_to_dict(port_base_m) if port_base_m is not None else None,
            images=images,
            yolo_debug_paths=dict(debug_paths or {}),
        )

    def align_command(
        self,
        *,
        step_index: int,
        prediction_m: Any,
        command_step_m: Any,
        target_pose: Any,
        actual_pose: Any,
        xy_error_m: float,
    ) -> None:
        self.event(
            "align_command",
            stage="align",
            step_index=int(step_index),
            prediction_m=vector_to_dict(prediction_m),
            command_step_m=vector_to_dict(command_step_m),
            target_pose=pose_to_dict(target_pose),
            actual_pose=pose_to_dict(actual_pose),
            xy_error_m=float(xy_error_m),
        )

    def command_error(
        self,
        *,
        stage: str,
        step_index: int,
        commanded_pose: Any,
        actual_before_pose: Any,
        actual_after_pose: Any,
    ) -> None:
        commanded = pose_to_dict(commanded_pose)
        before = pose_to_dict(actual_before_pose)
        after = pose_to_dict(actual_after_pose)
        error = None
        actual_delta = None
        if commanded is not None and after is not None:
            cmd = commanded["position"]
            act = after["position"]
            error = {axis: float(act[axis] - cmd[axis]) for axis in ("x", "y", "z")}
        if before is not None and after is not None:
            start = before["position"]
            end = after["position"]
            actual_delta = {axis: float(end[axis] - start[axis]) for axis in ("x", "y", "z")}
        self.event(
            "command_error",
            stage=stage,
            step_index=int(step_index),
            commanded_pose=commanded,
            actual_before_pose=before,
            actual_after_pose=after,
            error_m=error,
            actual_delta_m=actual_delta,
        )
