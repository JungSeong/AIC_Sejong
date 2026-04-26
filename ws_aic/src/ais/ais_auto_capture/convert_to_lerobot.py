#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation

# [수정 이유] lerobot 0.5.1 이상 버전 구조에 맞춘 임포트 경로.
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

def compute_plug_to_port(port_tf: dict, plug_tf: dict) -> np.ndarray:
    """plug의 포즈를 port 좌표계로 표현한 상대 포즈 (7차원: translation xyz + quaternion xyzw)."""
    t_port = np.array([port_tf["translation"]["x"], port_tf["translation"]["y"], port_tf["translation"]["z"]])
    q_port = np.array([port_tf["rotation"]["x"], port_tf["rotation"]["y"], port_tf["rotation"]["z"], port_tf["rotation"]["w"]])
    t_plug = np.array([plug_tf["translation"]["x"], plug_tf["translation"]["y"], plug_tf["translation"]["z"]])
    q_plug = np.array([plug_tf["rotation"]["x"], plug_tf["rotation"]["y"], plug_tf["rotation"]["z"], plug_tf["rotation"]["w"]])

    R_port = Rotation.from_quat(q_port)
    R_plug = Rotation.from_quat(q_plug)
    t_rel = R_port.inv().apply(t_plug - t_port)
    q_rel = (R_port.inv() * R_plug).as_quat()  # [x, y, z, w]
    return np.concatenate([t_rel, q_rel]).astype(np.float32)


def convert_episodes_to_lerobot(
    capture_dir: Path,
    out_dir: Path,
    repo_id: str,
    fps: int = 10,
    push_to_hub: bool = False,
    private: bool = True
):
    if not LEROBOT_AVAILABLE:
        print("Error: 'lerobot' package is not installed.")
        return

    features = {
        "observation.state": {"dtype": "float32", "shape": (35,), "names": [
            "tcp_pose.position.x", "tcp_pose.position.y", "tcp_pose.position.z",
            "tcp_pose.orientation.x", "tcp_pose.orientation.y", "tcp_pose.orientation.z", "tcp_pose.orientation.w",
            "tcp_velocity.linear.x", "tcp_velocity.linear.y", "tcp_velocity.linear.z",
            "tcp_velocity.angular.x", "tcp_velocity.angular.y", "tcp_velocity.angular.z",
            "tcp_error.x", "tcp_error.y", "tcp_error.z", "tcp_error.rx", "tcp_error.ry", "tcp_error.rz",
            "joint_positions.0", "joint_positions.1", "joint_positions.2", "joint_positions.3",
            "joint_positions.4", "joint_positions.5", "joint_positions.6",
            "force.x", "force.y", "force.z",
            "torque.x", "torque.y", "torque.z",
            "gripper_offset.x", "gripper_offset.y", "gripper_offset.z"
        ]},
        "action": {"dtype": "float32", "shape": (7,), "names": [
            "position.x", "position.y", "position.z",
            "orientation.x", "orientation.y", "orientation.z", "orientation.w"
        ]},
        "observation.plug_to_port": {"dtype": "float32", "shape": (7,), "names": [
            "translation.x", "translation.y", "translation.z",
            "rotation.x", "rotation.y", "rotation.z", "rotation.w"
        ]},
        "observation.images.left_camera": {"dtype": "video", "shape": (256, 288, 3), "names": ["height", "width","channels"]},
        "observation.images.center_camera": {"dtype": "video", "shape": (256, 288, 3), "names": ["height","width", "channels"]},
        "observation.images.right_camera": {"dtype": "video", "shape": (256, 288, 3), "names": ["height", "width","channels"]},
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # lerobot은 root/repo_id 경로에 저장하므로, 로컬 저장 시 org prefix 제거.
    # HuggingFace 업로드는 별도로 full repo_id 사용.
    local_name = repo_id.split("/")[-1]
    dataset = LeRobotDataset.create(
        repo_id=local_name,
        root=out_dir,
        fps=fps,
        features=features,
        use_videos=True,
    )

    # [수정 사항] 태스크 이름 리스트를 메타데이터에 등록 (0: sfp, 1: sc)
    # task_names = ["sfp_insertion", "sc_insertion"]
    # dataset.meta.tasks = task_names

    # 2. 에피소드 루프
    episode_dirs = sorted([d for d in capture_dir.iterdir() if d.is_dir() and (d / "steps.jsonl").exists()])
    
    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        steps_file = ep_dir / "steps.jsonl"
        with open(steps_file, "r") as f:
            steps_data = [json.loads(line) for line in f]

        if not steps_data:
            continue

        # gripper_offset is stored once per episode in meta.json
        meta_file = ep_dir / "meta.json"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        gripper_offset = meta.get("ground_truth_gripper_offset", {"x": 0.0, "y": 0.0, "z": 0.0})

        for step in steps_data:
            obs_state = step["observation"]["state"]
            wrist_wrench = step["observation"]["wrist_wrench"]
            state_vector = [
                obs_state["tcp_pose.position.x"], obs_state["tcp_pose.position.y"], obs_state["tcp_pose.position.z"],
                obs_state["tcp_pose.orientation.x"], obs_state["tcp_pose.orientation.y"], obs_state["tcp_pose.orientation.z"], obs_state["tcp_pose.orientation.w"],
                obs_state["tcp_velocity.linear.x"], obs_state["tcp_velocity.linear.y"], obs_state["tcp_velocity.linear.z"],
                obs_state["tcp_velocity.angular.x"], obs_state["tcp_velocity.angular.y"], obs_state["tcp_velocity.angular.z"],
                obs_state["tcp_error.x"], obs_state["tcp_error.y"], obs_state["tcp_error.z"],
                obs_state["tcp_error.rx"], obs_state["tcp_error.ry"], obs_state["tcp_error.rz"],
                obs_state["joint_positions.0"], obs_state["joint_positions.1"], obs_state["joint_positions.2"],
                obs_state["joint_positions.3"], obs_state["joint_positions.4"], obs_state["joint_positions.5"], obs_state["joint_positions.6"],
                wrist_wrench["force"]["x"], wrist_wrench["force"]["y"], wrist_wrench["force"]["z"],
                wrist_wrench["torque"]["x"], wrist_wrench["torque"]["y"], wrist_wrench["torque"]["z"],
                gripper_offset["x"], gripper_offset["y"], gripper_offset["z"]
            ]

            action_data = step["lerobot_action"]
            action_vector = [
                action_data["position.x"], action_data["position.y"], action_data["position.z"],
                action_data["orientation.x"], action_data["orientation.y"], action_data["orientation.z"], action_data["orientation.w"]
            ]

            def load_and_resize(img_info):
                img_path = ep_dir / img_info["path"]
                img = cv2.imread(str(img_path))
                if img is None:
                    return np.zeros((256, 288, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return cv2.resize(img, (288, 256), interpolation=cv2.INTER_AREA)

            left_img = load_and_resize(step["observation"]["left_image"])
            center_img = load_and_resize(step["observation"]["center_image"])
            right_img = load_and_resize(step["observation"]["right_image"])

            # peg-hole 상대 포즈 (port 좌표계 기준 plug 포즈)
            transforms = step.get("transforms", {})
            plug_to_port = compute_plug_to_port(transforms["port"], transforms["plug"])

            # [수정 사항] 'task'는 반드시 정수(int64)여야 함. 태스크 타입에 따라 인덱스 부여.
            task_type = step["task"]["port_type"].lower()
            task_idx = 0 if "sfp" in task_type else 1

            dataset.add_frame({
                "observation.state": np.array(state_vector, dtype=np.float32),
                "observation.plug_to_port": plug_to_port,
                "action": np.array(action_vector, dtype=np.float32),
                "observation.images.left_camera": left_img,
                "observation.images.center_camera": center_img,
                "observation.images.right_camera": right_img,
                "task": task_idx,
            })

        dataset.save_episode()

    dataset.finalize()
    
    if push_to_hub:
        dataset.push_to_hub(repo_id=repo_id, private=private)
        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out-dir",     type=Path, required=True)
    parser.add_argument("--repo-id",     type=str,  required=True,
                        help="HuggingFace Hub repo ID (예: aic-sejong-team/aic-dataset)")
    parser.add_argument("--run-id",      type=str,  default=None,
                        help="업로드 구분자. 미지정 시 현재 시각(YYYYMMDD_HHMMSS)으로 자동 생성. "
                             "최종 repo_id = {repo-id}-{run-id}")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    final_repo_id = f"{args.repo_id}-{run_id}"
    print(f"[업로드] repo_id: {final_repo_id}")

    convert_episodes_to_lerobot(
        args.capture_dir,
        args.out_dir,
        final_repo_id,
        push_to_hub=args.push
    )
