# AIC Example Policies 코드 요약

> `aic_example_policies/ros/` 하위 5개 정책 파일(WaveArm 제외) 코드 분석.
> 각 Policy는 `aic_model.Policy`를 상속하고 `insert_cable()` 메서드를 구현한다.
> 제어 방식(관절/Cartesian), 강성/감쇠 파라미터, 목적(정상 작동 vs 패널티 유발)이 파일마다 다르다.

---

## 전체 비교 요약

| 클래스 | 제어 방식 | 목적 | 특이사항 |
|--------|-----------|------|----------|
| `CheatCode` | **Cartesian (Pose)** | 케이블 삽입 성공 (치트) | ground truth TF 필요, PI 제어기 포함 |
| `RunACT` | **Cartesian (Twist)** | ACT 모델 추론 및 삽입 | HuggingFace 가중치, 30초 루프 |
| `GentleGiant` | **관절 공간 (Position)** | 저크 최소화 (부드러운 움직임) | 낮은 강성, 높은 감쇠 |
| `SpeedDemon` | **관절 공간 (Position)** | 저크 최대화 (빠르고 거친 움직임) | 높은 강성, 낮은 감쇠 |
| `WallToucher` | **관절 공간 (Position)** | 벽 접촉 → off-limit 감지 유발 | shoulder_pan=1.57 방향 |
| `WallPresser` | **관절 공간 (Position)** | 벽 지속 가압 → 삽입력 패널티 유발 | shoulder_pan=-1.57 방향 |

---

## 1. CheatCode.py

**목적:** `ground_truth` TF 프레임을 사용해 실제 포트/플러그 위치를 알고 삽입하는 "치트" 정책. 디버깅용.

**핵심 로직:**

```
1. 포트/케이블 TF 프레임 대기 (ground_truth:=true 필수)
2. 100스텝 동안 포트 위 z_offset=0.2m 위치로 Slerp 보간 이동
3. z_offset을 -0.015m까지 0.0005씩 감소 → 삽입
4. 5초 대기 후 종료
```

**핵심 함수 `calc_gripper_pose()`:**

- 포트 방향과 플러그 방향 사이의 **쿼터니언 차분**으로 그리퍼 목표 자세 계산
- X/Y 위치 오차에 대해 **적분기(I gain)** 적용 (windup 클리핑 ±0.05)

$$q_{target} = q_{diff} \cdot q_{gripper}, \quad q_{diff} = q_{port} \cdot q_{plug}^{-1}$$

**주요 파라미터:**

| 파라미터 | 값 |
|----------|----|
| `z_offset` 초기값 | 0.2 m |
| 삽입 종료 조건 | `z_offset < -0.015` m |
| z_offset 감소 스텝 | 0.0005 m |
| I gain | 0.15 |
| max_integrator_windup | ±0.05 |

**의존성:** `tf2_ros`, `transforms3d` (쿼터니언 연산)

---

## 2. RunACT.py

**목적:** ACT(Action Chunking with Transformers) 모델을 HuggingFace에서 받아 실시간 추론으로 케이블 삽입.

**핵심 로직:**

```
1. HuggingFace snapshot_download("grkw/aic_act_policy")
2. config.json → ACTConfig, model.safetensors → ACTPolicy 로드
3. 정규화 통계(safetensors)로 이미지/상태/액션 mean/std 준비
4. 30초간 루프 (약 4Hz):
   - 카메라 3채널 + 로봇 상태(26차원) 정규화 → 텐서
   - policy.select_action() 추론 → 7차원 액션
   - 역정규화 후 Cartesian Twist로 로봇 이동
```

**입력 상태 벡터 (26차원):**

| 구성 | 차원 |
|------|------|
| TCP 위치 (x,y,z) | 3 |
| TCP 방향 (quaternion) | 4 |
| TCP 선속도 | 3 |
| TCP 각속도 | 3 |
| TCP 오차 | 6 |
| 관절 위치 | 7 |

**정규화 공식:**

$$\hat{x} = \frac{x - \mu}{\sigma}, \quad x_{out} = \hat{x}_{pred} \cdot \sigma_{action} + \mu_{action}$$

**임피던스 파라미터 (Cartesian Twist):**

| 항목 | Linear (x,y,z) | Angular (rx,ry,rz) |
|------|----------------|---------------------|
| Stiffness | 100.0 | 50.0 |
| Damping | 40.0 | 15.0 |
| Wrench feedback gains | 0.5, 0.5, 0.5 | 0.0, 0.0, 0.0 |

---

## 3. GentleGiant.py

**목적:** **낮은 강성 + 높은 감쇠** → 느리고 부드러운 움직임 → **저크 점수 최대화** 테스트용.

**핵심 로직:**

```
home  = [-0.16, -1.35, -1.66, -1.69, 1.57, 1.41]  (홈)
target = [0.6, -1.3, -1.9, -1.57, 1.57, 0.6]      (목표)

3 사이클 반복:
  - target으로 50스텝 이동 (0.1s 간격)
  - home으로 50스텝 복귀
홈에서 30스텝 안정화
```

**임피던스 파라미터:**

| 항목 | 대형 관절 (1~3) | 소형 관절 (4~6) |
|------|----------------|----------------|
| **Stiffness** | **50.0** (낮음) | **20.0** |
| Damping | 40.0 | 20.0 |

> `SpeedDemon`과 목표 경로는 동일하나 강성/감쇠가 정반대 → 움직임 특성 비교용 쌍.

---

## 4. SpeedDemon.py

**목적:** **높은 강성 + 낮은 감쇠** → 빠르고 급격한 움직임 → **저크 점수 최소화** (패널티 유발) 테스트용.

**핵심 로직:**

```
3 사이클 반복:
  - target으로 50스텝 "스냅" (0.1s 간격)
  - home으로 50스텝 "스냅"
마지막에 moderate 파라미터로 홈 안정화 (30스텝)
```

**임피던스 파라미터:**

| 단계 | 강성 (대/소) | 감쇠 (대/소) |
|------|-------------|-------------|
| 이동 중 | **500.0 / 200.0** (매우 높음) | **5.0 / 2.0** (매우 낮음) |
| 안정화 | 200.0 / 50.0 | 40.0 / 15.0 |

---

## 5. WallToucher.py

**목적:** 로봇 팔을 측벽에 가볍게 접촉 → **off-limit 접촉 감지** 유발 테스트.

**전략:** `shoulder_pan = +1.57` (오른쪽 방향) 후 팔 신장.

```
retracted = [1.57, -1.35, -1.66, -1.69, 1.57, 1.41]  (팔 접힘)
extended  = [1.57, -0.5,   0.0,  -1.69, 1.57, 1.41]  (팔 신장 → 벽 접촉)

3 사이클:
  - low_stiffness로 retracted 이동 (30스텝)
  - high_stiffness로 extended 이동 (50스텝) → 벽 가압
홈 복귀 (50스텝, low_stiffness)
```

**임피던스 파라미터:**

| 상태 | Stiffness (대/소) | Damping |
|------|-------------------|---------|
| 수축 (low) | 200.0 / 50.0 | 40.0 / 15.0 |
| 신장 (high) | 300.0 / 50.0 | 40.0 / 15.0 |

---

## 6. WallPresser.py

**목적:** 로봇 팔을 측벽에 **지속적으로 강하게 가압** → **삽입력 패널티** 유발 테스트.

**전략:** `shoulder_pan = -1.57` (왼쪽 방향, WallToucher와 반대), 더 높은 강성으로 강압.

```
retracted = [-1.57, -1.35, -1.66, -1.69, 1.57, 1.41]
extended  = [-1.57, -0.5,   0.0,  -1.69, 1.57, 1.41]

3 사이클:
  - retracted 이동 (30스텝)
  - extended로 50스텝 가압 (지속 접촉 → F/T 임계 초과)
홈 복귀 (50스텝)
```

**임피던스 파라미터:**

| 항목 | 값 (대/소) |
|------|-----------|
| Stiffness | **300.0 / 50.0** (WallToucher high와 동일) |
| Damping | 40.0 / 15.0 |

> **WallToucher vs WallPresser:** 방향만 다름 (±1.57). WallPresser는 F/T 임계를 충분히 초과하는 지속 접촉을 유발하도록 설계.

---

## 공통 구조 정리

```python
class SomePolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        ...
        return True
```

**제어 메시지 타입:**

| 메시지 | 사용 정책 | 제어 공간 |
|--------|-----------|-----------|
| `JointMotionUpdate` | GentleGiant, SpeedDemon, WallToucher, WallPresser | 관절 공간 |
| `MotionUpdate` (Twist) | RunACT | Cartesian 속도 |
| `set_pose_target()` | CheatCode | Cartesian 위치 |

**홈 포지션 (공통):**

```
[-0.16, -1.35, -1.66, -1.69, 1.57, 1.41]
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
```
