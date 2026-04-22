import os
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# 고속 다운로드를 위해 hf_transfer 사용 설정 (pip install hf_transfer 필요)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 설정 정보
repo_id = "aic-sejong-team/aic-dataset"
folder_path = "runs/20260421_144551/set_0001"
local_dir = "./my_data"
# Private 저장소인 경우 토큰을 직접 입력하거나 huggingface-cli login이 되어 있어야 합니다.
token = None 

api = HfApi(token=token)

try:
    # 1. 전체 파일 목록 로드
    print(f"저장소 '{repo_id}'에서 파일 목록을 가져오는 중...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # 2. 특정 경로로 시작하는 파일만 필터링
    target_files = [f for f in files if f.startswith(folder_path)]
    
    print(f"전체 파일 수: {len(files)}")
    print(f"다운로드 대상 파일 수: {len(target_files)}")

    if not target_files:
        print("검색된 파일이 없습니다. folder_path의 대소문자와 경로를 다시 확인하세요.")
        print(f"참고용 실제 경로 예시: {files[:3]}")
    else:
        # 3. 개별 파일 다운로드
        for file in tqdm(target_files, desc="Downloading"):
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                local_dir=local_dir,
                token=token,
                local_dir_use_symlinks=False  # Windows 환경 권한 문제 방지
            )
        print("다운로드가 완료되었습니다.")

except Exception as e:
    print(f"오류 발생: {e}")