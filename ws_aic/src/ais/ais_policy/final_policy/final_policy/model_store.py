"""FinalPolicy가 필요한 모델 파일을 로컬 경로 또는 Hugging Face에서 찾는 모듈."""

from __future__ import annotations

import os
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
MODEL_LOG_COLOR = "\033[1;96m"
MODEL_LOG_RESET = "\033[0m"
MODEL_LOG_DISABLED_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ModelSpec:
    """모델별 로컬 저장 위치, 환경변수, 기본 HF repo 정보를 담는 설정 객체."""

    name: str
    env_key: str
    local_rel_path: Path
    hf_path_env_key: str
    default_hf_repo_id: str
    default_hf_rel_path: Path | None = None
    download_root_rel_path: Path | None = None


DEFAULT_HF_REPO_ID = "aic-sejong-team/aic-final-policy-models"
DEFAULT_YOLO_HF_REPO_ID = "aic-sejong-team/detection"
DEFAULT_VISION_OFFSET_HF_REPO_ID = "aic-sejong-team/aic-vision-offset-models"

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
SFP_VISION_OFFSET_MODEL = ModelSpec(
    name="SFP vision offset",
    env_key="AIC_SFP_VISION_OFFSET_MODEL_PATH",
    local_rel_path=Path("align/SFP/cross_attention_bilinear/cross_attention_bilinear_best.pt"),
    hf_path_env_key="AIC_SFP_VISION_OFFSET_HF_PATH",
    default_hf_repo_id=DEFAULT_VISION_OFFSET_HF_REPO_ID,
    default_hf_rel_path=Path("SFP/cross_attention_bilinear/cross_attention_bilinear_best.pt"),
    download_root_rel_path=Path("align"),
)
SC_VISION_OFFSET_MODEL = ModelSpec(
    name="SC vision offset",
    env_key="AIC_SC_VISION_OFFSET_MODEL_PATH",
    local_rel_path=Path("align/SC/cross_attention_bilinear/cross_attention_bilinear_best.pt"),
    hf_path_env_key="AIC_SC_VISION_OFFSET_HF_PATH",
    default_hf_repo_id=DEFAULT_VISION_OFFSET_HF_REPO_ID,
    default_hf_rel_path=Path("SC/cross_attention_bilinear/cross_attention_bilinear_best.pt"),
    download_root_rel_path=Path("align"),
)


def format_model_log(message: str) -> str:
    """모델 다운로드/로드 관련 로그를 터미널에서 눈에 띄도록 포맷한다."""
    text = f"[MODEL] {message}"
    color_enabled = (
        os.environ.get("AIC_MODEL_LOG_COLOR", "1").strip().lower()
        not in MODEL_LOG_DISABLED_VALUES
    )
    if not color_enabled:
        return text
    return f"{MODEL_LOG_COLOR}{text}{MODEL_LOG_RESET}"


def _log(logger, level: str, message: str) -> None:
    """ROS logger가 전달된 경우에만 해당 레벨로 메시지를 남긴다."""
    if logger is None:
        return
    getattr(logger, level)(format_model_log(message))


def _local_model_path(spec: ModelSpec) -> Path:
    """PROJECT_ROOT/model 아래 모델별 상대 경로의 실제 파일 위치를 반환한다."""
    return MODEL_ROOT / spec.local_rel_path


def _default_hf_model_path(spec: ModelSpec) -> Path:
    """HF repo 안에서 기본으로 찾을 모델 상대 경로를 반환한다."""
    return spec.default_hf_rel_path or spec.local_rel_path


def _download_root(spec: ModelSpec) -> Path:
    """HF snapshot을 실제로 받을 로컬 디렉토리를 반환한다."""
    if spec.download_root_rel_path is None:
        return MODEL_ROOT
    return MODEL_ROOT / spec.download_root_rel_path


def _hf_model_path(spec: ModelSpec, logger=None) -> Path:
    """HF snapshot에서 받을 모델 경로를 repo 상대 경로로 정규화한다."""
    default_hf_path = _default_hf_model_path(spec)
    raw_path = os.environ.get(spec.hf_path_env_key, "").strip()
    if not raw_path:
        return default_hf_path

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        try:
            rel_path = path.relative_to(MODEL_ROOT)
        except ValueError:
            _log(
                logger,
                "warn",
                f"{spec.hf_path_env_key} is an absolute local path outside {MODEL_ROOT}: "
                f"{path}; using default HF path {default_hf_path}",
            )
            return default_hf_path

        if rel_path == spec.local_rel_path or rel_path in spec.local_rel_path.parents:
            return default_hf_path
        return rel_path

    if path == spec.local_rel_path or path in spec.local_rel_path.parents:
        return default_hf_path
    return path


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

    hf_path = _hf_model_path(spec, logger=logger)
    repo_type = os.environ.get("AIC_HF_MODEL_REPO_TYPE", "model")
    revision = os.environ.get("AIC_HF_MODEL_REVISION", "main")
    expected_path = _local_model_path(spec)
    download_root = _download_root(spec)
    download_root.mkdir(parents=True, exist_ok=True)

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
                local_dir=str(download_root),
                allow_patterns=[str(hf_path)],
            )
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
            continue

        downloaded_path = download_root / hf_path
        if expected_path.is_file():
            _log(logger, "info", f"{spec.name} model ready: {expected_path}")
            return expected_path
        if downloaded_path.is_file():
            _log(logger, "info", f"{spec.name} model ready: {downloaded_path}")
            return downloaded_path
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
    """환경변수, PROJECT_ROOT/model 상대 경로, HF 다운로드 순서로 모델 경로를 찾는다."""
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
            "falling back to the project model directory",
        )
    else:
        _log(
            logger,
            "info",
            f"{spec.env_key} is not set; checking the project model directory",
        )

    local_path = _local_model_path(spec)
    if local_path.is_file():
        _log(logger, "info", f"{spec.name} model from project model directory: {local_path}")
        return str(local_path)

    _log(
        logger,
        "info",
        f"{spec.name} model not found at {local_path}; "
        "downloading from the default Hugging Face repo",
    )
    return str(_download_from_hugging_face(spec, logger=logger))
