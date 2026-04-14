# 훈련된 가중치로 나만의 Policy 노드 만들기

> **요약 (2-3줄):** `lerobot-train` 결과물(`model.safetensors` + `config.json` + 정규화 통계)을 로드하여 Policy 노드를 만든다. 노드는 `ws_aic/src/aic/` 안에 `ros2 pkg create`로 생성하고, 패키지별 `pixi.toml`과 루트 `pixi.toml` 양쪽에 등록한 뒤 `pixi reinstall`로 빌드한다. 핵심 흐름은 "파일 로드 → 관측값 전처리(이미지+상태 정규화) → 추론 → 역정규화 → 로봇 명령"이다.

---

## 1. 훈련 결과물 파일 구조

```
data/outputs/train/act_AIC/
├── checkpoints/
│   ├── 005000/
│   │   └── pretrained_model/
│   │       ├── config.json                                       ← 모델 구조 설정
│   │       ├── model.safetensors                                 ← 가중치
│   │       └── policy_preprocessor_step_3_normalizer_processor.safetensors  ← 정규화 통계
│   └── ...
└── train_config.json                                             ← 훈련 전체 설정
```

> **주의:** HF Hub에 push하면 위 파일들이 `JungSeong2/act_AIC` 리포에 올라간다.

---

## 2. 핵심 흐름 (RunACT.py 기준)

```
①  파일 로드 (config.json + model.safetensors + normalizer.safetensors)
        ↓
②  관측값 전처리
    ├── 이미지: bytes → numpy → resize(×0.25) → tensor(CHW) → (x-mean)/std
    └── 상태벡터(26차원): TCP pose/vel/error + joint positions → (x-mean)/std
        ↓
③  모델 추론
    policy.select_action(obs)  →  normalized_action [1, 7]
        ↓
④  역정규화
    raw_action = normalized_action * std + mean
        ↓
⑤  로봇 명령 전송
    Twist(linear=[ax,ay,az], angular=[arx,ary,arz]) → MotionUpdate → move_robot()
```

---

## 3. 상태벡터(State) 26차원 구성

| 인덱스 | 항목 | 차원 | 설명 |
|--------|------|------|------|
| 0-2 | `tcp_pose.position` | 3 | TCP 위치 (x, y, z) |
| 3-6 | `tcp_pose.orientation` | 4 | TCP 자세 (quaternion: x, y, z, w) |
| 7-9 | `tcp_velocity.linear` | 3 | TCP 선속도 (x, y, z) |
| 10-12 | `tcp_velocity.angular` | 3 | TCP 각속도 (x, y, z) |
| 13-18 | `tcp_error` | 6 | TCP 위치/자세 오차 (x, y, z, rx, ry, rz) |
| 19-25 | `joint_states.position` | 7 | 관절 위치 (joint 0~6) |
| **합계** | | **26** | |

---

## 4. 액션(Action) 7차원 구성

| 인덱스 | 항목 | 설명 |
|--------|------|------|
| 0-2 | `linear.x/y/z` | TCP 선속도 명령 |
| 3-5 | `angular.x/y/z` | TCP 각속도 명령 |
| *(6번째는 정책에 따라 다름)* | | |

---

## 5. ROS 2 패키지 생성 전체 과정

### 5-1. 패키지 생성 위치

반드시 AIC 소스 디렉토리 **안**에서 생성해야 한다.

```bash
cd ~/LLM_TUNE/AIC_Sejong/ws_aic/src/aic
source ~/LLM_TUNE/AIC_Sejong/ws_aic/install/setup.bash

ros2 pkg create my_policy_node --build-type ament_python
```

생성 결과:

```
ws_aic/src/aic/
└── my_policy_node/               ← 이 디렉토리가 ROS 2 패키지
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/
    │   └── my_policy_node        ← ament_index용 빈 파일 (건드리지 말 것)
    ├── test/                     ← 테스트 (필요 없으면 무시)
    └── my_policy_node/           ← 실제 Python 패키지
        └── __init__.py
```

> **핵심:** `ros2 pkg create`는 현재 디렉토리에 패키지를 만든다. 경로를 따로 지정하는 플래그는 없으므로 **실행 위치가 곧 패키지 위치**다.

---

### 5-2. `package.xml` 수정

AIC 인터페이스 의존성을 추가한다. (`ros2 pkg create`가 생성한 파일에서 `<depend>` 추가)

```xml
<?xml version="1.0"?>
<package format="3">
  <name>my_policy_node</name>
  <version>0.0.1</version>
  <description>My custom policy for AIC</description>
  <maintainer email="your@email.com">YourName</maintainer>
  <license>Apache-2.0</license>

  <depend>aic_control_interfaces</depend>
  <depend>aic_model</depend>
  <depend>aic_model_interfaces</depend>
  <depend>aic_task_interfaces</depend>
  <depend>geometry_msgs</depend>
  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

---

### 5-3. `pixi.toml` 생성 (패키지 디렉토리 내부)

`ros2 pkg create`는 `pixi.toml`을 만들어주지 않는다. **수동으로 생성**해야 한다.

`ws_aic/src/aic/my_policy_node/pixi.toml`:

```toml
[package.build.backend]
name = "pixi-build-ros"
version = "==0.3.3.20260113.c8b6a54"
channels = [
  "https://prefix.dev/pixi-build-backends",
  "robostack-kilted",
  "conda-forge",
]

[package.host-dependencies]
ros-kilted-aic-control-interfaces = { path = "../aic_interfaces/aic_control_interfaces" }
ros-kilted-aic-model              = { path = "../aic_model" }
ros-kilted-aic-model-interfaces   = { path = "../aic_interfaces/aic_model_interfaces" }
ros-kilted-aic-task-interfaces    = { path = "../aic_interfaces/aic_task_interfaces" }

[package.build-dependencies]
ros-kilted-aic-control-interfaces = { path = "../aic_interfaces/aic_control_interfaces" }
ros-kilted-aic-model              = { path = "../aic_model" }
ros-kilted-aic-model-interfaces   = { path = "../aic_interfaces/aic_model_interfaces" }
ros-kilted-aic-task-interfaces    = { path = "../aic_interfaces/aic_task_interfaces" }
```

> **경로 기준:** `../`은 `ws_aic/src/aic/`를 기준으로 한다. `aic_example_policies/pixi.toml`과 동일한 구조다.

---

### 5-4. 루트 `pixi.toml`에 패키지 등록

`ws_aic/src/aic/pixi.toml`의 `[dependencies]` 섹션에 추가:

```toml
[dependencies]
# ... 기존 항목들 ...
ros-kilted-my-policy-node = { path = "my_policy_node" }
```

> **주의:** ROS 패키지 이름(`package.xml`의 `<name>`)의 언더스코어(`_`)가 하이픈(`-`)으로 변환된다. `my_policy_node` → `ros-kilted-my-policy-node`

---

### 5-5. 정책 코드 작성

`my_policy_node/my_policy_node/` 안에 Python 파일 생성:

```
my_policy_node/
└── my_policy_node/
    ├── __init__.py
    └── Baseline.py    ← 여기에 Baseline 클래스 작성 (파일명 = 클래스명)
```

---

### 5-6. 빌드 및 실행

```bash
# 루트 aic 디렉토리에서 패키지 재빌드
cd ~/LLM_TUNE/AIC_Sejong/ws_aic/src/aic
pixi reinstall ros-kilted-my-policy-node

# 실행
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.Baseline
```

> **`policy` 파라미터 형식:** `<ROS패키지명>.<파일명>.<클래스명>` — **파일명과 클래스명이 반드시 일치해야 함**
> 예: `Baseline.py` 안에 `class Baseline(Policy)` → `-p policy:=my_policy_node.Baseline`

---

## 6. 나만의 Policy 노드 전체 코드 구조

```python
# my_policy_node/my_policy_node/my_policy.py

import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download  # HF 사용 시

from aic_model.policy import Policy
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Twist, Vector3, Wrench

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig


class MyPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 1. 파일 경로 ──────────────────────────────────────────
        # (A) HuggingFace에서 다운로드
        policy_path = Path(snapshot_download(repo_id="JungSeong2/act_AIC"))

        # (B) 로컬 체크포인트 사용 시
        # policy_path = Path("/home/vsc/LLM_TUNE/AIC_Sejong/data/outputs/train/act_AIC/checkpoints/005000/pretrained_model")

        # ── 2. config.json 로드 ───────────────────────────────────
        with open(policy_path / "config.json") as f:
            config_dict = json.load(f)
            config_dict.pop("type", None)  # draccus 오류 방지
        config = draccus.decode(ACTConfig, config_dict)

        # ── 3. 가중치 로드 ────────────────────────────────────────
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"))
        self.policy.eval().to(self.device)

        # ── 4. 정규화 통계 로드 ───────────────────────────────────
        stats = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left":   {"mean": stat("observation.images.left_camera.mean",   (1,3,1,1)),
                       "std":  stat("observation.images.left_camera.std",    (1,3,1,1))},
            "center": {"mean": stat("observation.images.center_camera.mean", (1,3,1,1)),
                       "std":  stat("observation.images.center_camera.std",  (1,3,1,1))},
            "right":  {"mean": stat("observation.images.right_camera.mean",  (1,3,1,1)),
                       "std":  stat("observation.images.right_camera.std",   (1,3,1,1))},
        }
        self.state_mean  = stat("observation.state.mean", (1, -1))
        self.state_std   = stat("observation.state.std",  (1, -1))
        self.action_mean = stat("action.mean", (1, -1))
        self.action_std  = stat("action.std",  (1, -1))

        self.image_scale = 0.25  # 훈련 시 사용한 스케일과 반드시 동일하게

    # ── 이미지 전처리 ─────────────────────────────────────────────
    def _img_to_tensor(self, raw_img, mean, std):
        img = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if self.image_scale != 1.0:
            img = cv2.resize(img, None, fx=self.image_scale, fy=self.image_scale,
                             interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2,0,1).float().div(255.0).unsqueeze(0).to(self.device)
        return (t - mean) / std

    # ── 관측값 전처리 ─────────────────────────────────────────────
    def prepare_obs(self, obs_msg):
        tcp   = obs_msg.controller_state.tcp_pose
        vel   = obs_msg.controller_state.tcp_velocity
        state = np.array([
            tcp.position.x, tcp.position.y, tcp.position.z,
            tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w,
            vel.linear.x,  vel.linear.y,  vel.linear.z,
            vel.angular.x, vel.angular.y, vel.angular.z,
            *obs_msg.controller_state.tcp_error,           # 6차원
            *obs_msg.joint_states.position[:7],            # 7차원
        ], dtype=np.float32)

        raw_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        return {
            "observation.images.left_camera":   self._img_to_tensor(obs_msg.left_image,   self.img_stats["left"]["mean"],   self.img_stats["left"]["std"]),
            "observation.images.center_camera": self._img_to_tensor(obs_msg.center_image, self.img_stats["center"]["mean"], self.img_stats["center"]["std"]),
            "observation.images.right_camera":  self._img_to_tensor(obs_msg.right_image,  self.img_stats["right"]["mean"],  self.img_stats["right"]["std"]),
            "observation.state": (raw_t - self.state_mean) / self.state_std,
        }

    # ── 메인 루프 ─────────────────────────────────────────────────
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.policy.reset()

        import time
        start = time.time()
        while time.time() - start < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                continue

            obs = self.prepare_obs(obs_msg)

            with torch.inference_mode():
                norm_action = self.policy.select_action(obs)  # [1, 7]

            action = ((norm_action * self.action_std) + self.action_mean)[0].cpu().numpy()

            twist = Twist(
                linear=Vector3(x=float(action[0]), y=float(action[1]), z=float(action[2])),
                angular=Vector3(x=float(action[3]), y=float(action[4]), z=float(action[5])),
            )

            msg = MotionUpdate()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.velocity = twist
            msg.target_stiffness = np.diag([100.0]*3 + [50.0]*3).flatten()
            msg.target_damping   = np.diag([40.0]*3  + [15.0]*3).flatten()
            msg.feedforward_wrench_at_tip = Wrench()
            msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
            msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY

            move_robot(motion_update=msg)
            send_feedback("running")

            time.sleep(max(0, 0.25 - (time.time() - loop_start)))  # ~4Hz

        return True
```

---

## 7. 로컬 체크포인트 vs HuggingFace 사용

| 방식 | 코드 | 장단점 |
|------|------|--------|
| **HF Hub** | `snapshot_download(repo_id="JungSeong2/act_AIC")` | 어디서나 실행, 인터넷 필요 |
| **로컬 경로** | `policy_path = Path(".../checkpoints/005000/pretrained_model")` | 빠름, 오프라인 가능 |

---

## 8. 실행 순서 (Docker eval 워크플로우)

> **`source ~/ws_aic/install/setup.bash` 불필요** — `pixi run`이 ROS 2 환경을 자동 구성.
> 자세한 내용은 `CLAUDE-OUTPUTS/summaries/0413/summary_docker-eval-run-guide_v1.md` 참고.

```bash
# Terminal 1 — eval 컨테이너 (Zenoh + Gazebo + aic_engine 포함)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Terminal 2 — 내 정책 (30초 이내 실행)
cd ~/aic_sejong/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=my_policy_node.Baseline
```

---

## 9. 주의사항 체크리스트

| 항목 | 확인 내용 |
|------|----------|
| **패키지 생성 위치** | `ws_aic/src/aic/` 안에서 `ros2 pkg create` 실행 |
| **`pixi.toml` 수동 생성** | `ros2 pkg create`는 `pixi.toml`을 만들어주지 않음 |
| **루트 `pixi.toml` 등록** | `ros-kilted-my-policy-node = { path = "my_policy_node" }` 추가 필수 |
| **`pixi reinstall`** | 코드 변경 후 반드시 `pixi reinstall ros-kilted-my-policy-node` 실행 |
| **`policy` 파라미터** | `파일명 = 클래스명` 필수. 형식: `ROS패키지명.파일명.클래스명` (예: `my_policy_node.Baseline.Baseline`) |
| **image_scale** | 훈련 시 `camera_image_scaling`(기본 0.25)과 반드시 일치 |
| **state 차원** | 훈련 데이터와 동일한 순서·차원(26) |
| **config.json `type` 필드** | `draccus.decode` 전에 반드시 `pop("type", None)` |
| **`policy.reset()`** | `insert_cable()` 진입 시 매번 호출 (ACT 내부 KV캐시 초기화) |
| **정규화 파일명** | `policy_preprocessor_step_3_normalizer_processor.safetensors` (버전마다 다를 수 있음) |
| **실행 순서** | 정책 노드 → 시뮬레이션 (30초 이내) |

