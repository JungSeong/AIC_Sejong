import os
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. 타임아웃 시간을 10분(600초)으로 대폭 연장
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "600"
# 2. 고속 전송 라이브러리 활성화
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "aic-sejong-team/aic-dataset"
folder_path = "runs/20260421_144551/set_0001"
local_dir = Path("./my_data")

try:
    print(f"다운로드 시작")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[f"{folder_path}/*"],
        # 워커 수를 늘려 병렬로 다운로드
        max_workers=16, 
        local_dir_use_symlinks=False
    )
    
    print("다운로드 완료")

except Exception as e:
    print(f"오류 발생: {e}")