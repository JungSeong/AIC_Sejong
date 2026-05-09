#!/usr/bin/env python3
"""
YOLO 포트 검출 모델 학습
==========================

사용법:
  cd ~/AIC_Sejong/ws_aic/src
  pixi run python ais/ais_motion_planning/train_yolo.py

  # 데이터 경로 직접 지정 시:
  pixi run python ais/ais_motion_planning/train_yolo.py --data data/yolo/20260426/data.yaml

requirements:
  pixi run pip install ultralytics
"""

import argparse
from pathlib import Path

import yaml

_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/
_WS_ROOT = _SRC_ROOT.parent
_DEFAULT_DATA   = _SRC_ROOT / "data" / "yolo" / "20260507" / "data.yaml"
_DEFAULT_OUTPUT = _WS_ROOT / "model" / "ais_yolo_0507"

IMAGE_EXTS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _resolve_dataset_path(data_yaml: Path, dataset_cfg: dict, split: str) -> Path | None:
    split_value = dataset_cfg.get(split)
    if not split_value:
        return None

    split_path = Path(split_value).expanduser()
    if split_path.is_absolute():
        return split_path

    base_candidates = []
    base = dataset_cfg.get("path")
    if base:
        base_path = Path(base).expanduser()
        base_candidates.append(base_path if base_path.is_absolute() else data_yaml.parent / base_path)
    base_candidates.append(data_yaml.parent)

    for base_path in base_candidates:
        candidate = (base_path / split_path).resolve()
        if candidate.exists():
            return candidate

    return (base_candidates[0] / split_path).resolve()


def _count_images(image_dir: Path | None) -> int:
    if image_dir is None or not image_dir.exists():
        return 0
    return sum(
        1
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def _count_labels(image_dir: Path | None) -> int:
    if image_dir is None:
        return 0
    parts = list(image_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        label_dir = Path(*parts)
    else:
        label_dir = image_dir.parent.parent / "labels" / image_dir.name
    if not label_dir.exists():
        return 0
    return sum(1 for path in label_dir.rglob("*.txt") if path.is_file())


def print_dataset_counts(data_path: Path) -> None:
    with data_path.open("r", encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f) or {}

    print("데이터셋 개수:")
    for split, title in [("train", "훈련"), ("val", "검증")]:
        image_dir = _resolve_dataset_path(data_path, dataset_cfg, split)
        image_count = _count_images(image_dir)
        label_count = _count_labels(image_dir)
        print(f"  {title}: 이미지 {image_count}장, 라벨 {label_count}개")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--model",  type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume", type=str, default=None,
                        help="이어서 학습: last.pt 경로 지정 시 resume, "
                             "best.pt/epoch*.pt 지정 시 해당 가중치로 새 학습 시작")
    parser.add_argument("--val-only", type=str, default=None, metavar="MODEL_PT",
                        help="학습 없이 검증만 실행. 모델 .pt 경로를 지정하세요. "
                             "예: --val-only model/ais_yolo-2/weights/best.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.val_only:
        model_pt = Path(args.val_only).expanduser().resolve()
        data_yaml = Path(args.data).expanduser().resolve()
        data_path = str(data_yaml)
        print(f"검증 실행")
        print(f"  모델: {model_pt}")
        print(f"  데이터: {data_path}")
        print_dataset_counts(data_yaml)
        model = YOLO(str(model_pt))
        metrics = model.val(data=data_path, imgsz=args.imgsz, device=args.device)
        print(f"\n검증 결과:")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        return

    data_yaml    = Path(args.data).expanduser().resolve()
    data_path    = str(data_yaml)
    out          = Path(args.output).expanduser().resolve()
    project_path = str(out.parent)
    run_name     = out.name
    print_dataset_counts(data_yaml)

    if args.resume:
        resume_pt = Path(args.resume).expanduser().resolve()
        model     = YOLO(str(resume_pt))

        if resume_pt.name == "last.pt":
            # last.pt: epoch·optimizer 상태 포함 → 완전 재개
            print(f"학습 재개 (resume from last.pt)")
            print(f"  체크포인트: {resume_pt}")
            model.train(resume=True)
        else:
            # best.pt / epoch*.pt: 가중치만 로드 → 새 설정으로 재학습
            print(f"가중치 로드 후 재학습")
            print(f"  체크포인트: {resume_pt}")
            print(f"  데이터: {data_path}")
            print(f"  에폭: {args.epochs}  출력: {out}")
            model.train(
                data=data_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=project_path,
                name=run_name,
                patience=15,
                save=True,
                save_period=10,
                verbose=True,
                plots=True,
                device=args.device,
            )
    else:
        print(f"학습 시작 (scratch)")
        print(f"  데이터: {data_path}")
        print(f"  기본 모델: {args.model}")
        print(f"  에폭: {args.epochs}  출력: {out}")
        model = YOLO(args.model)
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=project_path,
            name=run_name,
            patience=15,
            save=True,
            save_period=10,
            verbose=True,
            plots=True,
            device=args.device,
        )

    print(f"\n학습 완료!")
    print(f"  최고 모델: {out}/weights/best.pt")

    metrics = model.val()
    print(f"\n검증 결과:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
