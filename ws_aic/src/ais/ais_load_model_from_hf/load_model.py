from pathlib import Path
from huggingface_hub import snapshot_download

_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/

snapshot_download(
    repo_id="aic-sejong-team/port_detection_yolo",
    local_dir=str(_SRC_ROOT / "model" / "ais_yolo-2"),
)