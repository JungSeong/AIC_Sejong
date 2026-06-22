"""FinalPolicy align debug 이미지를 mp4 영상으로 묶는 유틸리티."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CAMERA_ORDER = ("left", "center", "right")
ALIGN_IMAGE_RE = re.compile(r"(?P<label>.+)__align_(?P<frame>\d+)\.jpg$")


def _resolve_project_root() -> Path:
    """현재 파일 위치에서 AIC_Sejong 프로젝트 루트를 역추적한다."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "ws_aic" / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[6]


def _collect_align_images(align_dir: Path) -> dict[str, dict[int, dict[str, Path]]]:
    """debug/align 하위 이미지를 task label, frame, camera 기준으로 모은다."""
    groups: dict[str, dict[int, dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for camera_dir in align_dir.iterdir() if align_dir.is_dir() else []:
        if not camera_dir.is_dir():
            continue
        camera_name = camera_dir.name
        for path in sorted(camera_dir.glob("*__align_*.jpg")):
            match = ALIGN_IMAGE_RE.match(path.name)
            if match is None:
                continue
            label = match.group("label")
            frame = int(match.group("frame"))
            groups[label][frame][camera_name] = path
    return groups


def _latest_label(groups: dict[str, dict[int, dict[str, Path]]]) -> str:
    """가장 최근에 저장된 align task label을 선택한다."""
    latest = None
    latest_mtime = -1.0
    for label, frames in groups.items():
        for camera_paths in frames.values():
            for path in camera_paths.values():
                mtime = path.stat().st_mtime
                if mtime > latest_mtime:
                    latest = label
                    latest_mtime = mtime
    if latest is None:
        raise RuntimeError("align debug 이미지가 없습니다.")
    return latest


def _read_image(path: Path | None, size: tuple[int, int]) -> np.ndarray:
    """이미지를 읽고 기준 size(width, height)에 맞춘다. 없으면 검은 화면을 만든다."""
    width, height = size
    if path is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _draw_camera_label(image: np.ndarray, camera_name: str, missing: bool) -> None:
    """합성 프레임에서 각 카메라 영역을 구분할 라벨을 그린다."""
    color = (0, 255, 255) if not missing else (0, 0, 255)
    label = camera_name if not missing else f"{camera_name} missing"
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _first_image_size(frames: dict[int, dict[str, Path]]) -> tuple[int, int]:
    """첫 번째 유효 이미지에서 영상 프레임 기준 size(width, height)를 얻는다."""
    for camera_paths in frames.values():
        for path in camera_paths.values():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                height, width = image.shape[:2]
                return width, height
    raise RuntimeError("읽을 수 있는 align debug 이미지가 없습니다.")


def _make_frame(
    camera_paths: dict[str, Path],
    size: tuple[int, int],
    camera: str,
) -> np.ndarray:
    """단일 카메라 또는 3카메라 합성 프레임을 만든다."""
    if camera != "combined":
        image = _read_image(camera_paths.get(camera), size)
        _draw_camera_label(image, camera, camera not in camera_paths)
        return image

    images = []
    for camera_name in CAMERA_ORDER:
        image = _read_image(camera_paths.get(camera_name), size)
        _draw_camera_label(image, camera_name, camera_name not in camera_paths)
        images.append(image)
    return np.hstack(images)


def make_video(
    align_dir: Path,
    output_path: Path | None,
    task_label: str | None,
    fps: float,
    camera: str,
) -> Path:
    """선택된 task label의 align debug 이미지를 mp4로 저장한다."""
    groups = _collect_align_images(align_dir)
    if not groups:
        raise RuntimeError(f"align debug 이미지가 없습니다: {align_dir}")

    label = task_label or _latest_label(groups)
    if label not in groups:
        available = ", ".join(sorted(groups.keys()))
        raise RuntimeError(f"task label을 찾을 수 없습니다: {label} (available: {available})")

    frames = groups[label]
    size = _first_image_size(frames)
    output_path = output_path or (align_dir / f"{label}__align_{camera}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    try:
        for frame_index in sorted(frames.keys()):
            frame = _make_frame(frames[frame_index], size, camera)
            height, width = frame.shape[:2]
            if width % 2 or height % 2:
                frame = frame[: height - height % 2, : width - width % 2]
                height, width = frame.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"video writer를 열 수 없습니다: {output_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return output_path


def main() -> int:
    project_root = _resolve_project_root()
    parser = argparse.ArgumentParser(
        description="FinalPolicy align debug 이미지를 mp4 영상으로 변환합니다."
    )
    parser.add_argument(
        "--align-dir",
        type=Path,
        default=project_root / "debug" / "align",
        help="debug/align 디렉토리 경로",
    )
    parser.add_argument("--output", type=Path, default=None, help="저장할 mp4 경로")
    parser.add_argument("--task-label", default=None, help="특정 task label만 영상화")
    parser.add_argument("--fps", type=float, default=2.0, help="출력 영상 fps")
    parser.add_argument(
        "--camera",
        choices=("combined", *CAMERA_ORDER),
        default="combined",
        help="합성 영상 또는 단일 카메라 선택",
    )
    parser.add_argument("--list", action="store_true", help="사용 가능한 task label 출력")
    args = parser.parse_args()

    groups = _collect_align_images(args.align_dir)
    if args.list:
        for label in sorted(groups.keys()):
            frame_count = len(groups[label])
            print(f"{label}\t{frame_count} frames")
        return 0

    output_path = make_video(
        align_dir=args.align_dir,
        output_path=args.output,
        task_label=args.task_label,
        fps=args.fps,
        camera=args.camera,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
