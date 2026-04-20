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
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="base model: yolov8n/s/m/l/x.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="~/aic_yolo_runs")
    parser.add_argument("--name", type=str, default="port_detector")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics 패키지를 먼저 설치하세요:")
        print("  pixi run pip install ultralytics")
        return

    data_path = str(Path(args.data).expanduser())
    project_path = str(Path(args.project).expanduser())

    print(f"학습 시작")
    print(f"  데이터: {data_path}")
    print(f"  기본 모델: {args.model}")
    print(f"  에폭: {args.epochs}")
    print(f"  이미지 크기: {args.imgsz}")

    model = YOLO(args.model)
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=project_path,
        name=args.name,
        patience=15,        # early stopping
        save=True,
        save_period=10,
        verbose=True,
        plots=True,
    )

    print(f"\n학습 완료!")
    print(f"  최고 모델: {project_path}/{args.name}/weights/best.pt")

    # 검증
    metrics = model.val()
    print(f"\n검증 결과:")
    print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
