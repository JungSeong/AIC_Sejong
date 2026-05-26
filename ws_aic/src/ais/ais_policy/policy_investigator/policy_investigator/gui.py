from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WIDTH = 1500
HEIGHT = 900
CAMERAS = ("left", "center", "right")
TASK_FIELDS = (
    "id",
    "target_module_name",
    "port_name",
    "port_type",
    "cable_name",
    "cable_type",
    "plug_name",
    "plug_type",
    "time_limit",
)


def _v3(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "position" in payload:
            return _v3(payload["position"])
        return np.array(
            [
                float(payload.get("x", 0.0)),
                float(payload.get("y", 0.0)),
                float(payload.get("z", 0.0)),
            ],
            dtype=np.float64,
        )
    values = np.asarray(payload, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return None
    return values[:3]


def _fmt_mm(value_m: float | None) -> str:
    if value_m is None or not math.isfinite(value_m):
        return "n/a"
    return f"{value_m * 1000.0:+.1f} mm"


class InvestigatorView:
    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.stage = "idle"
        self.stage_status = ""
        self.detail = ""
        self.policy = ""
        self.run_id = ""
        self.task_info: dict[str, str] = {}
        self.last_event_time = 0.0
        self.logs = deque(maxlen=16)
        self.images: dict[str, dict[str, Any]] = {}
        self.yolo_paths: dict[str, str] = {}
        self.actual_trail = deque(maxlen=240)
        self.command_trail = deque(maxlen=240)
        self.align_step = -1
        self.prediction_m: dict[str, float] | None = None
        self.command_step_m: dict[str, float] | None = None
        self.last_error_m: dict[str, float] | None = None
        self.last_actual_delta_m: dict[str, float] | None = None
        self.should_exit = False
        self._thumb_cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}

    def handle(self, event: dict[str, Any]) -> None:
        self.last_event_time = float(event.get("time", time.time()))
        self.policy = str(event.get("policy", self.policy))
        self.run_id = str(event.get("run_id", self.run_id))
        event_type = str(event.get("type", ""))
        if event_type == "stage":
            self.stage = str(event.get("stage", self.stage))
            self.stage_status = str(event.get("status", ""))
            self.detail = str(event.get("detail", ""))
            self.logs.appendleft(f"[stage] {self.stage}: {self.stage_status} {self.detail}".strip())
        elif event_type == "log":
            self.logs.appendleft(str(event.get("message", "")))
        elif event_type == "camera_snapshot":
            self.images = dict(event.get("images", {}))
            self.logs.appendleft(f"[camera] {event.get('stage', '')} {event.get('note', '')}".strip())
        elif event_type == "yolo_capture":
            self.images = dict(event.get("images", self.images))
            self.yolo_paths = dict(event.get("yolo_debug_paths", {}) or {})
            port = _v3(event.get("port_base_m"))
            if port is not None:
                self.command_trail.append(port)
            self.logs.appendleft("[yolo] capture saved")
        elif event_type == "align_command":
            self.align_step = int(event.get("step_index", self.align_step))
            self.prediction_m = dict(event.get("prediction_m", {}) or {})
            self.command_step_m = dict(event.get("command_step_m", {}) or {})
            actual = _v3(event.get("actual_pose"))
            target = _v3(event.get("target_pose"))
            if actual is not None:
                self.actual_trail.append(actual)
            if target is not None:
                self.command_trail.append(target)
        elif event_type == "command_error":
            self.last_error_m = dict(event.get("error_m", {}) or {})
            self.last_actual_delta_m = dict(event.get("actual_delta_m", {}) or {})
            actual = _v3(event.get("actual_after_pose"))
            target = _v3(event.get("commanded_pose"))
            if actual is not None:
                self.actual_trail.append(actual)
            if target is not None:
                self.command_trail.append(target)
        elif event_type == "run_start":
            self.stage = "run_start"
            self.stage_status = "start"
            task_info = dict(event.get("task_info", {}) or {})
            if not task_info:
                task_info = {field: str(event.get(field, "")) for field in TASK_FIELDS}
            self.task_info = {field: str(task_info.get(field, "")) for field in TASK_FIELDS}
            self.detail = f"{self.task_info.get('target_module_name', '')}/{self.task_info.get('port_name', '')}"
            self.yolo_paths = {}
            self.logs.appendleft(f"[run] {self.detail}")
        elif event_type == "shutdown":
            self.logs.appendleft(f"[shutdown] {event.get('reason', '')}")
            self.should_exit = True

    def _draw_text(self, canvas: np.ndarray, text: str, pos: tuple[int, int], scale: float = 0.55, color=(230, 230, 230), thickness: int = 1) -> None:
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _load_thumb(self, path: str, size: tuple[int, int]) -> np.ndarray:
        cache_key = (path, size)
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key].copy()
        thumb = np.full((size[1], size[0], 3), 35, dtype=np.uint8)
        if path:
            image = cv2.imread(path)
            if image is not None:
                h, w = image.shape[:2]
                scale = min(size[0] / max(1, w), size[1] / max(1, h))
                resized = cv2.resize(image, (int(w * scale), int(h * scale)))
                y = (size[1] - resized.shape[0]) // 2
                x = (size[0] - resized.shape[1]) // 2
                thumb[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        if path:
            if len(self._thumb_cache) > 24:
                self._thumb_cache.clear()
            self._thumb_cache[cache_key] = thumb.copy()
        return thumb

    def _draw_cameras(self, canvas: np.ndarray) -> None:
        x0, y0 = 30, 135
        w, h = 360, 205
        for index, camera in enumerate(CAMERAS):
            y = y0 + index * (h + 36)
            info = self.images.get(camera, {})
            overlay_path = str(self.yolo_paths.get(camera, ""))
            if overlay_path and not Path(overlay_path).is_file():
                overlay_path = ""
            raw_path = str(info.get("path", ""))
            image_path = overlay_path or raw_path
            thumb = self._load_thumb(image_path, (w, h))
            canvas[y : y + h, x0 : x0 + w] = thumb
            color = (80, 180, 110) if info.get("available") else (70, 70, 180)
            cv2.rectangle(canvas, (x0, y), (x0 + w, y + h), color, 2)
            source = "yolo+kpt" if overlay_path else str(info.get("encoding", ""))
            meta = f"{camera}  {info.get('width', 0)}x{info.get('height', 0)}  {source}"
            self._draw_text(canvas, meta, (x0, y - 10), 0.55, (220, 220, 220))

    def _project_iso(self, point: np.ndarray, center: tuple[int, int], scale: float, origin: np.ndarray) -> tuple[int, int]:
        p = (point - origin) * scale
        x = p[0] - 0.55 * p[1]
        y = -p[2] + 0.35 * p[1]
        return int(center[0] + x), int(center[1] + y)

    def _draw_trail(self, canvas: np.ndarray) -> None:
        area = (450, 125, 1015, 560)
        x, y, w, h = area
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (85, 85, 85), 1)
        self._draw_text(canvas, "3D command / actual", (x + 12, y + 28), 0.7, (245, 245, 245), 2)

        points = list(self.actual_trail) + list(self.command_trail)
        if not points:
            self._draw_text(canvas, "waiting for align commands", (x + 180, y + 245), 0.65, (130, 130, 130))
            return

        origin = np.mean(np.vstack(points), axis=0)
        span = np.max(np.linalg.norm(np.vstack(points) - origin, axis=1))
        scale = min(8500.0, 190.0 / max(0.002, float(span)))
        center = (x + w // 2, y + h // 2 + 60)

        axes = [
            (np.array([0.03, 0.0, 0.0]), (80, 80, 240), "x"),
            (np.array([0.0, 0.03, 0.0]), (80, 220, 80), "y"),
            (np.array([0.0, 0.0, 0.03]), (230, 210, 80), "z"),
        ]
        for vec, color, label in axes:
            p0 = self._project_iso(origin, center, scale, origin)
            p1 = self._project_iso(origin + vec, center, scale, origin)
            cv2.arrowedLine(canvas, p0, p1, color, 2, tipLength=0.18)
            self._draw_text(canvas, label, (p1[0] + 4, p1[1] + 4), 0.55, color, 2)

        def draw_polyline(trail, color):
            pts = [self._project_iso(point, center, scale, origin) for point in trail]
            for a, b in zip(pts, pts[1:]):
                cv2.line(canvas, a, b, color, 2)
            if pts:
                cv2.circle(canvas, pts[-1], 5, color, -1)

        draw_polyline(list(self.command_trail), (70, 180, 255))
        draw_polyline(list(self.actual_trail), (90, 235, 120))
        self._draw_text(canvas, "orange: command   green: actual", (x + 12, y + h - 18), 0.55, (220, 220, 220))

    def _draw_metrics(self, canvas: np.ndarray) -> None:
        x, y = 450, 715
        self._draw_text(canvas, f"align step: {self.align_step}", (x, y), 0.65, (235, 235, 235), 2)
        y += 34
        pred = self.prediction_m or {}
        self._draw_text(
            canvas,
            "prediction port: "
            f"x={_fmt_mm(pred.get('x'))}  y={_fmt_mm(pred.get('y'))}  z={_fmt_mm(pred.get('z'))}",
            (x, y),
        )
        y += 28
        step = self.command_step_m or {}
        self._draw_text(
            canvas,
            "command step:   "
            f"x={_fmt_mm(step.get('x'))}  y={_fmt_mm(step.get('y'))}  z={_fmt_mm(step.get('z'))}",
            (x, y),
        )
        y += 28
        error = self.last_error_m or {}
        self._draw_text(
            canvas,
            "actual-cmd err: "
            f"x={_fmt_mm(error.get('x'))}  y={_fmt_mm(error.get('y'))}  z={_fmt_mm(error.get('z'))}",
            (x, y),
            color=(170, 220, 255),
        )
        y += 28
        delta = self.last_actual_delta_m or {}
        self._draw_text(
            canvas,
            "actual delta:   "
            f"x={_fmt_mm(delta.get('x'))}  y={_fmt_mm(delta.get('y'))}  z={_fmt_mm(delta.get('z'))}",
            (x, y),
            color=(170, 255, 190),
        )

    def _draw_logs(self, canvas: np.ndarray) -> None:
        x, y = 1035, 125
        cv2.rectangle(canvas, (x, y), (WIDTH - 30, HEIGHT - 40), (85, 85, 85), 1)
        self._draw_text(canvas, "task", (x + 12, y + 28), 0.7, (245, 245, 245), 2)
        target = self.task_info.get("target_module_name", "")
        port = self.task_info.get("port_name", "")
        cable = self.task_info.get("cable_name", "")
        plug = self.task_info.get("plug_name", "")
        port_type = self.task_info.get("port_type", "")
        plug_type = self.task_info.get("plug_type", "")
        cable_type = self.task_info.get("cable_type", "")
        task_id = self.task_info.get("id", "")
        time_limit = self.task_info.get("time_limit", "")
        task_lines = [
            f"target: {target} / {port}",
            f"cable:  {cable} / {plug}",
            f"type:   cable={cable_type} plug={plug_type} port={port_type}",
            f"id:     {task_id}  limit={time_limit}s",
        ]
        for index, line in enumerate(task_lines):
            self._draw_text(canvas, line[:58], (x + 12, y + 62 + index * 24), 0.5, (215, 235, 235))

        log_y = y + 172
        self._draw_text(canvas, "log", (x + 12, log_y), 0.7, (245, 245, 245), 2)
        for index, line in enumerate(list(self.logs)):
            self._draw_text(canvas, line[:58], (x + 12, log_y + 34 + index * 26), 0.5, (220, 220, 220))

    def render(self) -> np.ndarray:
        canvas = np.full((HEIGHT, WIDTH, 3), 22, dtype=np.uint8)
        age = time.time() - self.last_event_time if self.last_event_time else 0.0
        status = f"{self.policy or 'policy'}  run={self.run_id or '-'}"
        self._draw_text(canvas, status, (30, 36), 0.75, (235, 235, 235), 2)
        stage_line = f"stage: {self.stage}  status: {self.stage_status}  age={age:.1f}s"
        self._draw_text(canvas, stage_line, (30, 73), 0.8, (120, 220, 255), 2)
        if self.detail:
            self._draw_text(canvas, self.detail, (30, 105), 0.6, (210, 210, 210))
        self._draw_cameras(canvas)
        self._draw_trail(canvas)
        self._draw_metrics(canvas)
        self._draw_logs(canvas)
        return canvas


def iter_replay_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/policy_investigator/events")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--window-name", default="AIC Policy Investigator")
    parser.add_argument("--replay", default="")
    parser.add_argument("--spin-burst", type=int, default=80)
    parser.add_argument("--wait-ms", type=int, default=10)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()

    view = InvestigatorView(Path(args.output_dir) if args.output_dir else None)

    replay_events = None
    if args.replay:
        replay_events = iter_replay_events(Path(args.replay))
    else:
        import rclpy
        from std_msgs.msg import String

        rclpy.init(args=None)
        node = rclpy.create_node("policy_investigator_gui")

        def on_event(msg: String) -> None:
            try:
                view.handle(json.loads(msg.data))
            except Exception as exc:
                view.logs.appendleft(f"[gui] bad event: {exc}")

        node.create_subscription(String, args.topic, on_event, 50)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, WIDTH, HEIGHT)

    try:
        while True:
            if args.parent_pid and not parent_alive(args.parent_pid):
                break
            if replay_events is not None:
                try:
                    view.handle(next(replay_events))
                except StopIteration:
                    pass
            else:
                rclpy.spin_once(node, timeout_sec=0.002)
                for _ in range(max(0, args.spin_burst - 1)):
                    rclpy.spin_once(node, timeout_sec=0.0)
            cv2.imshow(args.window_name, view.render())
            key = cv2.waitKey(max(1, args.wait_ms)) & 0xFF
            try:
                window_visible = cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                window_visible = 0
            if view.should_exit or key in {27, ord("q")} or window_visible < 1:
                break
    finally:
        if replay_events is None:
            node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
