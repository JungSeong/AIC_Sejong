#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi
import sys

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

def upload_dataset(dataset_root: Path, repo_id: str, branch: str, private: bool = True):
    if not LEROBOT_AVAILABLE:
        print("Error: 'lerobot' package is not installed.")
        return

    if not (dataset_root / "meta" / "info.json").exists():
        print(f"Error: No LeRobot dataset found at {dataset_root}")
        return

    print(f"[*] Resuming dataset from {dataset_root}...")
    try:
        # Pass repo_id as it is required in this version
        dataset = LeRobotDataset.resume(repo_id=repo_id, root=dataset_root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"[*] Ensuring branch '{branch}' exists in {repo_id}...")
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        # Create branch if it doesn't exist
        branches = [b.name for b in api.list_repo_refs(repo_id, repo_type="dataset").branches]
        if branch not in branches:
            print(f"[*] Creating branch '{branch}'...")
            api.create_branch(repo_id=repo_id, repo_type="dataset", branch=branch)
    except Exception as e:
        print(f"[!] Warning during repo/branch preparation: {e}")
        print("[*] Continuing anyway, push_to_hub might handle it...")

    print(f"[*] Pushing dataset to {repo_id} (branch: {branch})...")
    try
        # Update repo_id in metadata before pushing
        dataset.meta.repo_id = repo_id
        dataset.push_to_hub(branch=branch, private=private)
        print(f"[+] Successfully uploaded to https://huggingface.co/datasets/{repo_id}/tree/{branch}")
    except Exception as e:
        print(f"Error pushing dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a LeRobot dataset to HuggingFace Hub")
    parser.add_argument("--root", type=Path, required=True, help="Path to the local LeRobot dataset directory (the one containing meta/)")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID (e.g., org/name)")
    parser.add_argument("--branch", type=str, default="main", help="Branch/revision to push to")
    parser.add_argument("--public", action="store_true", help="Make the repository public (default is private)")
    
    args = parser.parse_args()
    
    upload_dataset(
        dataset_root=args.root,
        repo_id=args.repo_id,
        branch=args.branch,
        private=not args.public
    )
