from pathlib import Path
from huggingface_hub import snapshot_download

_SRC_ROOT = Path(__file__).resolve().parents[6]  # ws_aic/src/

snapshot_download(
    repo_id="aic-sejong-team/aic-vision-offset-models",
    local_dir=str(_SRC_ROOT / "model" / "align"),
    revision="main",
)