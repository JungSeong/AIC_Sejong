#!/usr/bin/env python3
"""Export an AIC vision-offset checkpoint to a browser-friendly ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEMO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[5]
FINAL_POLICY_ROOT = (
    PROJECT_ROOT / "ws_aic" / "src" / "ais" / "ais_policy" / "final_policy"
)
DEFAULT_CHECKPOINTS = {
    "sfp": PROJECT_ROOT
    / "model"
    / "align"
    / "SFP"
    / "cross_attention_bilinear"
    / "cross_attention_bilinear_best.pt",
    "sc": PROJECT_ROOT
    / "model"
    / "align"
    / "SC"
    / "cross_attention_bilinear"
    / "cross_attention_bilinear_best.pt",
}
CANONICAL_LABELS = ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the FinalPolicy vision-offset PyTorch checkpoint to a static "
            "[1,V,3,H,W] ONNX graph for ONNX Runtime Web."
        )
    )
    parser.add_argument(
        "--port-type",
        choices=("sfp", "sc"),
        default="sfp",
        help="Select the repository's default SFP or SC checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the checkpoint selected by --port-type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. Defaults to public/models/vision_offset_<type>.onnx.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18).",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip onnx.checker validation after export.",
    )
    return parser.parse_args()


def require_export_dependencies():
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise SystemExit(
            "ONNX export dependency is missing. From ws_aic/src, run:\n"
            "  pixi add --pypi onnx"
        ) from exc
    return onnx, torch


def export_model(args: argparse.Namespace) -> Path:
    onnx, torch = require_export_dependencies()
    sys.path.insert(0, str(FINAL_POLICY_ROOT))

    from final_policy.vision_offset.predictor import VisionOffsetPredictor

    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINTS[args.port_type]).resolve()
    output = (
        args.output
        or DEMO_ROOT
        / "public"
        / "models"
        / f"vision_offset_{args.port_type}.onnx"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    predictor = VisionOffsetPredictor(
        checkpoint_path=checkpoint,
        device="cpu",
    )
    predictor.model.eval()
    label_to_index = {
        label: index for index, label in enumerate(predictor.label_order)
    }

    class CanonicalOutputModel(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, images):
            raw = self.model(images)
            zero = torch.zeros_like(raw[:, :1])
            columns = [
                raw[:, label_to_index[label] : label_to_index[label] + 1]
                if label in label_to_index
                else zero
                for label in CANONICAL_LABELS
            ]
            return torch.cat(columns, dim=1)

    image_size = predictor.image_size
    if image_size is None:
        raise ValueError("Checkpoint must define a fixed image_size for browser export.")
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = (int(value) for value in image_size)

    wrapper = CanonicalOutputModel(predictor.model).eval()
    example = torch.zeros(
        1,
        len(predictor.cameras),
        3,
        height,
        width,
        dtype=torch.float32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            output,
            input_names=["images"],
            output_names=["correction"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    if not args.skip_check:
        model = onnx.load(output, load_external_data=True)
        onnx.checker.check_model(model)

    metadata = {
        "checkpoint": str(checkpoint),
        "port_type": args.port_type,
        "input_name": "images",
        "input_shape": [1, len(predictor.cameras), 3, height, width],
        "camera_order": list(predictor.cameras),
        "output_name": "correction",
        "output_labels": list(CANONICAL_LABELS),
        "source_label_order": list(predictor.label_order),
        "opset": args.opset,
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Exported ONNX: {output}")
    print(f"Metadata:      {metadata_path}")
    print(f"Input shape:   {metadata['input_shape']}")
    print(f"Output labels: {metadata['output_labels']}")
    return output


def main() -> None:
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
