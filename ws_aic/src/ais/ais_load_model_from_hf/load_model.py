from pathlib import Path
from huggingface_hub import snapshot_download

_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/

snapshot_download(
    repo_id="aic-sejong-team/sc-distance-resnet50-left-center-right-concat",
    local_dir=str(_SRC_ROOT / "model" / "sc-distance-resnet50-left-center-right-concat"),
    revision="v1.0",
)