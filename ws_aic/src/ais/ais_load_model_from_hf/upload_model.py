#!/usr/bin/env python3
"""Upload a local model/weight directory to Hugging Face Hub."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/
DEFAULT_MODEL_DIR = _SRC_ROOT / "model" / "yolo" / "weight" / "ais_yolo"
DEFAULT_REPO_ID = "aic-sejong-team/port_detection_yolo"


def upload_model_folder(
    model_dir: Path,
    repo_id: str,
    revision: str,
    private: bool,
    path_in_repo: str,
    dry_run: bool,
) -> None:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    files = [p for p in model_dir.rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"No files to upload under: {model_dir}")

    print("[HF Model Upload]")
    print(f"  local_dir   : {model_dir}")
    print(f"  repo_id     : {repo_id}")
    print(f"  revision    : {revision}")
    print(f"  path_in_repo: {path_in_repo or '.'}")
    print(f"  private     : {private}")
    print(f"  files       : {len(files)}")

    if dry_run:
        print("\n[DRY-RUN] Upload skipped. First files:")
        for path in files[:20]:
            print(" -", path.relative_to(model_dir))
        return

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    if revision not in {"main", "master"}:
        api.create_branch(repo_id=repo_id, repo_type="model", branch=revision, exist_ok=True)

    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        path_in_repo=path_in_repo or None,
        revision=revision,
        ignore_patterns=[
            "**/.DS_Store",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.git/**",
        ],
    )
    print(f"\n[HF Model Upload] complete: https://huggingface.co/{repo_id}/tree/{revision}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local model weights to Hugging Face Hub."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Local model directory to upload. Default: {DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("AIC_HF_MODEL_REPO_ID", DEFAULT_REPO_ID),
        help=f"HF model repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("AIC_HF_MODEL_REVISION", "main"),
        help="HF branch/revision to upload to. Default: main",
    )
    parser.add_argument(
        "--path-in-repo",
        default=os.environ.get("AIC_HF_MODEL_PATH_IN_REPO", ""),
        help="Optional subdirectory inside the HF repo.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/use a public repo. Default is private.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded.",
    )
    args = parser.parse_args()

    upload_model_folder(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        private=not args.public,
        path_in_repo=args.path_in_repo.strip("/"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
