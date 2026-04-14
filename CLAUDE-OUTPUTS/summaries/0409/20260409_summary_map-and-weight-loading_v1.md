# AIC 맵 파일 저장 위치 & RunACT 가중치 로딩 정리

> `ws_aic/src/aic/docs/` 하위 문서 및 `RunACT.py` 소스 분석 결과.
> 시나리오 생성 시 맵 파일이 저장되는 위치와 수동 변경 방법, RunACT.py의 가중치 로딩 방식을 정리한다.

---

## 1. 맵 파일 저장 위치 및 수동 변경

### 자동 저장 위치

| 항목 | 내용 |
|------|------|
| **기본 저장 경로** | `/tmp/aic.sdf` |
| **저장 트리거** | 시뮬레이션 실행 시 모든 엔티티(로봇, task board, 케이블)가 스폰된 후 자동 저장 |
| **저장 플러그인 설정 파일** | `aic_description/world/aic.sdf` |

`aic.sdf` 내 world 플러그인 파라미터:

```xml
<save_world_path>/tmp/aic.sdf</save_world_path>   <!-- 저장 경로 -->
<save_world_delay_s>0.0</save_world_delay_s>        <!-- 스폰 후 딜레이 (초) -->
```

---

### 시나리오 생성 예시 (파라미터로 맵 구성)

**eval container 기준:**
```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_yaw:=0.785 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    spawn_cable:=true cable_type:=sfp_sc_cable \
    ground_truth:=true start_aic_engine:=false
```

**source build 기준:**
```bash
ros2 launch aic_bringup aic_gz_bringup.launch.py [동일 파라미터]
```

---

### 수동 보존 방법

생성 직후 `/tmp/aic.sdf`를 원하는 경로로 복사하면 됩니다.

```bash
# 복사로 보존
cp /tmp/aic.sdf ~/training_scenarios/scenario_001.sdf

# 여러 시나리오 생성 예시
# Scenario 1: NIC card in slot 2
/entrypoint.sh spawn_task_board:=true nic_card_mount_2_present:=true \
    spawn_cable:=true cable_type:=sfp_sc_cable ground_truth:=true start_aic_engine:=false
cp /tmp/aic.sdf ~/training_scenarios/nic_slot_2.sdf

# Scenario 2: SC connector, 다른 yaw
/entrypoint.sh spawn_task_board:=true task_board_yaw:=1.57 \
    sc_mount_rail_1_present:=true spawn_cable:=true ground_truth:=true start_aic_engine:=false
cp /tmp/aic.sdf ~/training_scenarios/sc_right_rotated.sdf
```

> 저장된 `.sdf` 파일은 **IsaacLab, MuJoCo 등 다른 시뮬레이터로 임포트** 가능.

---

### 맵 파일 수동 변경 방법

저장된 `.sdf` 파일을 **텍스트 에디터로 직접 수정**하면 됩니다.

| 수정 대상 | 위치 |
|-----------|------|
| task board 위치/자세 | `<pose>` 태그 (x y z roll pitch yaw) |
| 케이블 타입/위치 | 케이블 모델 `<pose>` 태그 |
| NIC/SC/SFP 마운트 유무 | 해당 엔티티 블록 삭제 or 추가 |
| 조명/물리 설정 | world 레벨 태그 |

**저장 경로 자체를 바꾸고 싶다면:** `aic_description/world/aic.sdf` 내 `<save_world_path>` 값을 수정.

---

## 2. RunACT.py 가중치 로딩

> 파일 경로: `aic_example_policies/aic_example_policies/ros/RunACT.py`

### 로딩 출처

**HuggingFace Hub** (`huggingface_hub.snapshot_download`)에서 자동 다운로드.

```python
repo_id = "grkw/aic_act_policy"

policy_path = Path(
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "*.safetensors"],
    )
)
```

- 다운로드 위치: HuggingFace 로컬 캐시 (`~/.cache/huggingface/hub/...`)
- **인터넷 연결 필요**, 최초 실행 시 자동 다운로드

---

### 다운로드되는 파일

| 파일명 | 용도 |
|--------|------|
| `config.json` | ACTConfig (모델 아키텍처 설정) |
| `model.safetensors` | **메인 모델 가중치** |
| `policy_preprocessor_step_3_normalizer_processor.safetensors` | 정규화 통계 (mean/std) |

---

### 가중치 로딩 순서

```python
# 1. Config 로드 (unknown 'type' 필드 제거 후 draccus로 디코딩)
with open(policy_path / "config.json", "r") as f:
    config_dict = json.load(f)
    if "type" in config_dict:
        del config_dict["type"]
config = draccus.decode(ACTConfig, config_dict)

# 2. 모델 아키텍처 생성 + 가중치 로드
self.policy = ACTPolicy(config)
model_weights_path = policy_path / "model.safetensors"
self.policy.load_state_dict(load_file(model_weights_path))  # safetensors
self.policy.eval()
self.policy.to(self.device)  # CUDA or CPU 자동 선택
```

---

### 정규화 통계 로딩

```python
stats_path = policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
stats = load_file(stats_path)
```

| 통계 키 | shape | 용도 |
|---------|-------|------|
| `observation.images.{left,center,right}_camera.mean/std` | `(1, 3, 1, 1)` | 이미지 정규화 |
| `observation.state.mean/std` | `(1, 26)` | 로봇 상태 정규화 |
| `action.mean/std` | `(1, 7)` | 액션 역정규화 (출력 복원) |

**정규화 공식:**

$$\hat{x} = \frac{x - \mu}{\sigma}$$

**역정규화 공식 (액션 출력 복원):**

$$x = \hat{x} \cdot \sigma + \mu$$

---

## 요약

| 항목 | 핵심 내용 |
|------|-----------|
| **맵 자동 저장 위치** | `/tmp/aic.sdf` |
| **저장 설정 파일** | `aic_description/world/aic.sdf` (`<save_world_path>`) |
| **수동 보존 방법** | `cp /tmp/aic.sdf <원하는경로>` |
| **맵 직접 수정** | `.sdf` 파일을 텍스트 에디터로 편집 |
| **RunACT 가중치 출처** | HuggingFace Hub (`grkw/aic_act_policy`) |
| **가중치 파일** | `model.safetensors` (safetensors 라이브러리로 로드) |
| **정규화 통계 파일** | `policy_preprocessor_step_3_normalizer_processor.safetensors` |
