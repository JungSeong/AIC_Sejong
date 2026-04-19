#!/usr/bin/env python3
"""
debug_hf_upload.py
──────────────────
collect_data_native.py 의 HuggingFace Hub 업로드 기능만 독립적으로 테스트.

사용법:
  python3 debug_hf_upload.py                          # 토큰 확인 + 더미 업로드
  python3 debug_hf_upload.py --repo-id your-org/repo  # 레포 지정
  python3 debug_hf_upload.py --token hf_xxx           # 토큰 직접 지정 (테스트용)
  python3 debug_hf_upload.py --check-only             # 토큰/인증 확인만 (업로드 X)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path


# ── 1. 패키지 설치 여부 확인 ─────────────────────────────
try:
    from huggingface_hub import HfApi, create_repo, whoami
    print("[OK] huggingface_hub 임포트 성공")
except ImportError:
    print("[ERROR] huggingface_hub 가 설치되어 있지 않습니다.")
    print("        pip install huggingface_hub")
    sys.exit(1)


# ── 2. 토큰 감지 함수 ─────────────────────────────────────

def resolve_token(cli_token: str | None) -> str | None:
    """우선순위: CLI 인수 > HF_TOKEN > HUGGING_FACE_HUB_TOKEN > ~/.cache/huggingface/token"""
    if cli_token:
        return cli_token

    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_var, "").strip()
        if val:
            print(f"[INFO] 토큰 출처: 환경변수 {env_var}")
            return val

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        val = token_file.read_text().strip()
        if val:
            print(f"[INFO] 토큰 출처: {token_file}")
            return val

    return None


# ── 3. 토큰 유효성 확인 ───────────────────────────────────

def check_token(token: str | None) -> bool:
    """whoami() 로 토큰 유효성/권한 확인."""
    try:
        info = whoami(token=token)
        print(f"[OK] 인증 성공!")
        print(f"     사용자명 : {info['name']}")
        print(f"     이메일   : {info.get('email', '(비공개)')}")
        orgs = [o['name'] for o in info.get('orgs', [])]
        if orgs:
            print(f"     소속 org  : {', '.join(orgs)}")
        else:
            print(f"     소속 org  : (없음)")
        return True
    except Exception as e:
        print(f"[ERROR] 인증 실패: {e}")
        print()
        print("해결 방법:")
        print("  1) huggingface-cli login  → Write 권한 토큰 입력")
        print("  2) export HF_TOKEN=hf_xxx  → 환경변수 설정")
        print("  3) python3 debug_hf_upload.py --token hf_xxx  → 직접 지정")
        return False


# ── 4. 더미 파일 생성 ─────────────────────────────────────

def make_dummy_upload_dir() -> Path:
    """임시 디렉토리에 작은 더미 파일들을 생성."""
    tmp = Path(tempfile.mkdtemp(prefix="hf_debug_"))
    (tmp / "dummy_episode_summary.json").write_text(
        '{"episode": "debug_test", "status": "ok"}\n'
    )
    (tmp / "README.md").write_text(
        "# Debug Upload\nThis is a test upload from `debug_hf_upload.py`.\n"
    )
    print(f"[INFO] 더미 업로드 디렉토리: {tmp}")
    return tmp


# ── 5. 레포 생성 테스트 ───────────────────────────────────

def test_create_repo(repo_id: str, token: str | None, private: bool) -> bool:
    print(f"\n[STEP] 레포 확인/생성: {repo_id} (private={private})")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        print(f"[OK] 레포 준비 완료: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as e:
        print(f"[ERROR] 레포 생성 실패: {e}")
        return False


# ── 6. 폴더 업로드 테스트 ─────────────────────────────────

def test_upload_folder(
    upload_dir: Path,
    repo_id: str,
    token: str | None,
    path_in_repo: str = "debug_test",
) -> bool:
    print(f"\n[STEP] 폴더 업로드: {upload_dir} → {repo_id}/{path_in_repo}")
    api = HfApi(token=token)
    try:
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
        )
        print(f"[OK] 업로드 완료!")
        print(f"     URL: https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}")
        return True
    except Exception as e:
        print(f"[ERROR] 업로드 실패: {e}")
        return False


# ── 메인 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Hub 업로드 디버그 스크립트"
    )
    parser.add_argument("--repo-id",    type=str, default="aic-sejong-team/aic-dataset",
                        help="테스트할 Hub 레포 ID (기본: aic-sejong-team/aic-dataset)")
    parser.add_argument("--token",      type=str, default=None,
                        help="HF 토큰 직접 지정 (없으면 자동 감지)")
    parser.add_argument("--check-only", action="store_true",
                        help="토큰/인증 확인만 하고 업로드는 건너뜀")
    parser.add_argument("--public",     action="store_true",
                        help="레포를 공개로 생성 (기본: 비공개)")
    args = parser.parse_args()

    print("=" * 55)
    print(" HuggingFace Hub 업로드 디버그")
    print("=" * 55)

    # Step 1: 토큰 감지
    print("\n[STEP 1] 토큰 감지")
    token = resolve_token(args.token)
    if token:
        masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "***"
        print(f"[INFO] 토큰: {masked}")
    else:
        print("[WARN] 저장된 토큰 없음. 로그인 필요:")
        print("       huggingface-cli login")

    # Step 2: 인증 확인
    print("\n[STEP 2] 인증 확인 (whoami)")
    auth_ok = check_token(token)
    if not auth_ok:
        sys.exit(1)

    if args.check_only:
        print("\n--check-only 옵션: 인증 확인 완료. 업로드 건너뜀.")
        sys.exit(0)

    # Step 3: 레포 생성
    print("\n[STEP 3] 레포 생성/확인")
    repo_ok = test_create_repo(args.repo_id, token, private=not args.public)
    if not repo_ok:
        print("\n[실패] 레포 생성 단계에서 중단.")
        print("  → org 멤버 권한이 없거나 토큰에 Write 권한이 없을 수 있습니다.")
        sys.exit(1)

    # Step 4: 더미 파일 업로드
    print("\n[STEP 4] 더미 파일 업로드")
    dummy_dir = make_dummy_upload_dir()
    upload_ok = test_upload_folder(dummy_dir, args.repo_id, token)

    print("\n" + "=" * 55)
    if upload_ok:
        print(" [SUCCESS] 모든 테스트 통과! collect_data_native.py 에서도 정상 동작합니다.")
    else:
        print(" [FAILED] 업로드 실패. 위 에러 메시지를 확인하세요.")
    print("=" * 55)

    # 임시 디렉토리 정리
    import shutil
    shutil.rmtree(dummy_dir, ignore_errors=True)

    sys.exit(0 if upload_ok else 1)


if __name__ == "__main__":
    main()
