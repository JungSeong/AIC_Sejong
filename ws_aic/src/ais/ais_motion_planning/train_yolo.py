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

_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/
_WS_ROOT = _SRC_ROOT.parent

_DEFAULT_DATA   = _SRC_ROOT / "data" / "yolo" / "20260426" / "data.yaml"
_DEFAULT_OUTPUT = _WS_ROOT / "weight" / "ais_yolo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--model",  type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path    = str(Path(args.data).expanduser().resolve())
    out          = Path(args.output).expanduser().resolve()
    project_path = str(out.parent)
    run_name     = out.name

    print(f"학습 시작")
    print(f"  데이터: {data_path}")
    print(f"  기본 모델: {args.model}")
    print(f"  에폭: {args.epochs}")
    print(f"  출력 경로: {out}")

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
