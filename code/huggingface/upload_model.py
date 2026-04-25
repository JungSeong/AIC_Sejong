from huggingface_hub import HfApi, login
import os

login(token=os.environ["HF_TOKEN"])

api = HfApi()

# # 레포 먼저 생성
# api.create_repo(
#     repo_id="aic-sejong-team/baseline",
#     repo_type="model",
#     private=True
# )

# 폴더 업로드
api.upload_folder(
    folder_path="../../aic_data/model/yolo",
    repo_id="aic-sejong-team/baseline",
    repo_type="model",
)