"""FinalPolicy가 필요한 모델 파일을 로컬 경로 또는 Hugging Face에서 찾는 모듈."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _resolve_project_root() -> Path:
    """현재 파일 위치에서 AIC_Sejong 프로젝트 루트를 역추적한다."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "ws_aic" / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[6]


PROJECT_ROOT = _resolve_project_root()
MODEL_ROOT = PROJECT_ROOT / "model"


@dataclass(frozen=True)
class ModelSpec:
    """모델별 로컬 저장 위치, 환경변수, 기본 HF repo 정보를 담는 설정 객체."""

    name: str
    env_key: str
    local_rel_path: Path
    hf_path_env_key: str
    default_hf_repo_id: str


DEFAULT_HF_REPO_ID = "aic-sejong-team/aic-final-policy-models"
DEFAULT_YOLO_HF_REPO_ID = "aic-sejong-team/detection"

SFP_YOLO_MODEL = ModelSpec(
    name="SFP YOLO",
    env_key="AIC_SFP_YOLO_MODEL_PATH",
    local_rel_path=Path("approach/SFP/weights/best.pt"),
    hf_path_env_key="AIC_SFP_YOLO_HF_PATH",
    default_hf_repo_id=DEFAULT_YOLO_HF_REPO_ID,
)
SC_YOLO_MODEL = ModelSpec(
    name="SC YOLO",
    env_key="AIC_SC_YOLO_MODEL_PATH",
    local_rel_path=Path("approach/SC/weights/best.pt"),
    hf_path_env_key="AIC_SC_YOLO_HF_PATH",
    default_hf_repo_id=DEFAULT_YOLO_HF_REPO_ID,
)
POSE_MODEL = ModelSpec(
    name="pose prediction",
    env_key="AIC_POSE_MODEL_PATH",
    local_rel_path=Path("ais_pose_prediction/pose_resnet50_v4.0/best.pt"),
    hf_path_env_key="AIC_POSE_HF_PATH",
    default_hf_repo_id=DEFAULT_HF_REPO_ID,
)


def _log(logger, level: str, message: str) -> None:
    """ROS logger가 전달된 경우에만 해당 레벨로 메시지를 남긴다."""
    if logger is None:
        return
    getattr(logger, level)(message)


def _repo_candidates(spec: ModelSpec) -> list[str]:
    """모델 스펙에서 실제 다운로드를 시도할 HF repo 후보 목록을 만든다."""
    candidates = [
        spec.default_hf_repo_id,
    ]
    result = []
    for repo_id in candidates:
        if repo_id and repo_id not in result:
            result.append(repo_id)
    return result


def _download_from_hugging_face(spec: ModelSpec, logger=None) -> Path:
    """로컬에 없는 모델을 HF snapshot으로 받아 PROJECT_ROOT/model 아래에 배치한다."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download missing model files. "
            "Run `pixi install` first."
        ) from exc

    hf_path = Path(
        os.environ.get(spec.hf_path_env_key, "").strip() or spec.local_rel_path
    )
    repo_type = os.environ.get("AIC_HF_MODEL_REPO_TYPE", "model")
    revision = os.environ.get("AIC_HF_MODEL_REVISION", "main")
    expected_path = MODEL_ROOT / spec.local_rel_path
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    errors = []
    for repo_id in _repo_candidates(spec):
        _log(
            logger,
            "info",
            f"{spec.name} model missing from {spec.env_key}; downloading {hf_path} "
            f"from Hugging Face repo {repo_id}",
        )
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(MODEL_ROOT),
                allow_patterns=[str(hf_path)],
            )
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
            continue

        downloaded_path = MODEL_ROOT / hf_path
        if downloaded_path.is_file():
            if downloaded_path != expected_path:
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded_path, expected_path)
            _log(logger, "info", f"{spec.name} model ready: {expected_path}")
            return expected_path
        if expected_path.is_file():
            _log(logger, "info", f"{spec.name} model ready: {expected_path}")
            return expected_path
        errors.append(
            f"{repo_id}: downloaded snapshot did not contain {hf_path}"
        )

    raise FileNotFoundError(
        f"{spec.name} model not found under {MODEL_ROOT} and Hugging Face "
        "download failed. Set "
        f"{spec.env_key}=<local file>. "
        f"Download attempts: {' | '.join(errors)}"
    )


def resolve_model_path(spec: ModelSpec, logger=None) -> str:
    """환경변수 경로를 우선 사용하고, 없으면 기본 HF repo에서 모델을 내려받는다."""
    env_path = os.environ.get(spec.env_key)
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            _log(logger, "info", f"{spec.name} model from {spec.env_key}: {path}")
            return str(path)
        _log(
            logger,
            "warn",
            f"{spec.env_key} is set but file does not exist: {path}; "
            "downloading from the default Hugging Face repo",
        )
    else:
        _log(
            logger,
            "info",
            f"{spec.env_key} is not set; downloading from the default Hugging Face repo",
        )

    return str(_download_from_hugging_face(spec, logger=logger))
