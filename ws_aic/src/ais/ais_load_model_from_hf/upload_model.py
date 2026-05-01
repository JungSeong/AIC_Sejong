from huggingface_hub import HfApi, login
from pathlib import Path
import os

login(token=os.environ["HF_TOKEN"])
_SRC_ROOT = Path(__file__).resolve().parents[2]  # ws_aic/src/

api = HfApi()

# # 레포 먼저 생성
# api.create_repo(
#     repo_id="aic-sejong-team/port_detection_yolo",
#     repo_type="model",
#     private=True
# )

# 폴더 업로드
api.upload_folder(
    repo_id="aic-sejong-team/port_detection_yolo",
    repo_type="model",
    folder_path=str(_SRC_ROOT / "model" / "ais_yolo-2"),
)