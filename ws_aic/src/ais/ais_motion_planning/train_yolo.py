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

_DEFAULT_DATA   = _SRC_ROOT / "data" / "yolo" / "20260426" / "data.yaml"
_DEFAULT_OUTPUT = _SRC_ROOT / "model" / "ais_yolo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=str(_DEFAULT_DATA))
    parser.add_argument("--model",  type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
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
        data_path = str(Path(args.data).expanduser().resolve())
        print(f"검증 실행")
        print(f"  모델: {model_pt}")
        print(f"  데이터: {data_path}")
        model = YOLO(str(model_pt))
        metrics = model.val(data=data_path, imgsz=args.imgsz, device=args.device)
        print(f"\n검증 결과:")
        print(f"  mAP@0.5:      {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  Precision:    {metrics.box.mp:.4f}")
        print(f"  Recall:       {metrics.box.mr:.4f}")
        return

    data_path    = str(Path(args.data).expanduser().resolve())
    out          = Path(args.output).expanduser().resolve()
    project_path = str(out.parent)
    run_name     = out.name

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
