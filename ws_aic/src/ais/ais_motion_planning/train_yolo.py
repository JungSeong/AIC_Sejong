#!/usr/bin/env python3
"""
YOLO 포트 검출 모델 학습
==========================

사용법:
  cd ~/AIC_Sejong/ws_aic/src/aic
  pixi run python /home/sch24/train_yolo.py

requirements:
  pixi run pip install ultralytics
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="~/aic_yolo_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=str, default="~/aic_yolo_runs/port_detector")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path    = str(Path(args.data).expanduser())
    out          = Path(args.output).expanduser().resolve()
    project_path = str(out.parent)
    run_name     = out.name

    print(f"학습 시작")
    print(f"  데이터: {data_path}")
    print(f"  기본 모델: {args.model}")
    print(f"  에폭: {args.epochs}")
    print(f"  출력 경로: {out}")

    model = YOLO(args.model)
    results = model.train(
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