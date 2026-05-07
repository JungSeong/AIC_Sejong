# Camera Intrinsic / Extrinsic / Calibration 정리
**날짜:** 2026-05-06

---

## 1. 배경

평가 환경에서 TF가 제공되지 않는다면, policy는 `tip <-> port`의 상대 `x, y, z`를 직접 알 수 없다.

현재 `Observation.msg`에서 받을 수 있는 것은 주로 다음과 같다.

| 항목 | 의미 |
|---|---|
| left / center / right image | 세 카메라 RGB 이미지 |
| left / center / right CameraInfo | 각 카메라 intrinsic |
| joint_states | 로봇 관절 상태 |
| controller_state | TCP pose, velocity, error 등 |
| wrist_wrench | wrist F/T sensor |

`observation.plug_to_port.translation.z` 같은 값은 데이터셋 recording 시 TF로 계산해 저장한 label에 가깝다. 평가 시 policy 입력으로 기대하면 안 된다.

따라서 vision으로 `port`와 `cable tip`의 3D 위치를 추정하려면 camera intrinsic과 extrinsic의 의미를 정확히 구분해야 한다.

---

## 2. Intrinsic

Intrinsic은 카메라 내부 파라미터다.

즉, 3D 점이 해당 카메라 이미지의 어느 pixel에 찍히는지를 설명한다.

대표 값:

| 파라미터 | 의미 |
|---|---|
| `fx`, `fy` | pixel 단위 초점거리 |
| `cx`, `cy` | 이미지 principal point, 보통 이미지 중심 근처 |
| distortion coefficients | 렌즈 왜곡 계수 |

ROS `CameraInfo`의 `K` 행렬은 보통 다음 형태다.

```text
K = [ fx  0  cx
      0  fy  cy
      0   0   1 ]
```

카메라 좌표계에서 3D 점이 `(X, Y, Z)`에 있을 때, 왜곡을 무시하면 pixel 좌표는 대략 다음과 같다.

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

정리하면 intrinsic은 다음 질문에 답한다.

```text
이 카메라 렌즈와 센서가 3D 점을 2D pixel로 어떻게 투영하는가?
```

---

## 3. Extrinsic

Extrinsic은 카메라 외부 파라미터다.

즉, 카메라 좌표계가 `base_link`, world, 또는 다른 카메라 좌표계 기준으로 어디에 있고 어느 방향을 보는지를 설명한다.

예:

```text
T_base_left_camera
T_base_center_camera
T_base_right_camera
```

각 transform은 보통 다음을 포함한다.

| 항목 | 의미 |
|---|---|
| translation `(x, y, z)` | 기준 좌표계에서 카메라 위치 |
| rotation `(roll, pitch, yaw)` 또는 quaternion | 기준 좌표계에서 카메라 방향 |

정리하면 extrinsic은 다음 질문에 답한다.

```text
이 카메라는 로봇 base_link 기준 어디에 붙어 있고, 어느 방향을 보고 있는가?
```

---

## 4. 왜 Intrinsic만으로는 3D 위치를 알 수 없는가

한 카메라에서 object center pixel `(u, v)`를 알면, 그 pixel이 가리키는 3D ray는 알 수 있다.

하지만 깊이 `Z`는 아직 모른다.

```text
한 pixel = 카메라에서 뻗는 하나의 3D 직선(ray)
```

한 카메라에서 object center pixel `(u, v)`를 알면, 그 pixel이 가리키는 3D ray는 알 수 있다.
따라서 단일 RGB 카메라 + intrinsic만으로는 정확한 3D 좌표를 바로 얻을 수 없다.

3D 위치를 얻으려면 보통 다음 중 하나가 필요하다.

| 방법 | 필요한 것 |
|---|---|
| RGB-D | depth image |
| multi-view triangulation | 2개 이상 카메라의 intrinsic + 카메라 간 extrinsic |
| known object size | bbox 크기와 실제 물체 크기 기반 depth 근사 |
| learned model | 이미지에서 직접 `tip_to_port xyz` 회귀 |

---

## 5. Multi-view Triangulation에서 필요한 정보

left camera와 right camera에서 같은 3D point를 검출했다고 하자.

예:

```text
left image:  port center = (u_l, v_l)
right image: port center = (u_r, v_r)
```

이때 3D 위치를 계산하려면 다음이 필요하다.

1. left camera intrinsic
2. right camera intrinsic
3. left camera pose
4. right camera pose

즉:

```text
K_left, K_right
T_base_left_camera, T_base_right_camera
```

또는 카메라 간 transform:

```text
T_left_right
```

이 정보가 있어야 두 카메라 ray가 3D 공간에서 어디에서 만나는지 계산할 수 있다.

---

## 6. 고정 Calibration의 의미

여기서 말한 고정 calibration은 넓은 의미로 camera calibration이 맞다.

다만 두 종류를 구분해야 한다.

### 6.1 Intrinsic Calibration

각 카메라 자체의 렌즈/센서 파라미터를 구하는 과정.

구하는 값:

```text
fx, fy, cx, cy, distortion coefficients
```

현재 AIC 환경에서는 이 값이 `CameraInfo`로 제공된다.

### 6.2 Extrinsic Calibration

각 카메라가 로봇 기준 어디에 있고 어느 방향을 보는지 구하는 과정.

구하는 값:

```text
T_base_left_camera
T_base_center_camera
T_base_right_camera
```

또는:

```text
T_center_left
T_center_right
```

평가 환경에서 TF를 쓸 수 없다면, 이 extrinsic 값을 미리 측정해 코드 상수로 넣어야 한다.

예:

```python
CAMERA_EXTRINSICS = {
    "left": T_base_left_camera,
    "center": T_base_center_camera,
    "right": T_base_right_camera,
}
```

이것이 "고정 calibration 값을 코드에 넣는다"는 말의 핵심이다.

---

## 7. TF가 있을 때와 없을 때

### TF가 있을 때

기존 코드처럼 runtime에 transform을 조회할 수 있다.

```python
tf_buffer.lookup_transform("base_link", "left_camera_link", Time())
```

이 경우 extrinsic을 TF tree에서 얻는다.

### TF가 없을 때

위 lookup이 불가능하다.

따라서 다음 중 하나가 필요하다.

| 선택지 | 설명 |
|---|---|
| 고정 extrinsic 사용 | 카메라가 로봇에 고정되어 있다는 가정 하에 사전 측정한 transform을 코드에 넣음 |
| 직접 xyz 회귀 모델 | 이미지 3장과 robot state를 넣고 `tip_to_port xyz`를 바로 예측 |

---

## 8. 현재 AIC 상황에서의 해석

현재 policy observation에는 3개 RGB camera와 각 camera의 `CameraInfo`가 있다.

따라서:

```text
Intrinsic은 있음.
```

하지만 평가 환경에서 TF가 없다면:

```text
Extrinsic은 runtime에서 얻을 수 없음.
```

따라서 3D 기하 기반 접근을 하려면 다음이 필요하다.

1. YOLO 또는 keypoint detector로 port / cable tip의 2D 위치 검출
2. left / center / right 중 최소 2개 view에서 같은 point 매칭
3. 각 camera intrinsic 사용
4. 사전에 고정해 둔 camera extrinsic 사용
5. triangulation으로 port 3D, tip 3D 추정
6. `tip_xyz - port_xyz`로 offset 계산

---

## 9. 중요한 주의점

같은 point matching이 되려면 세 카메라가 같은 물체를 안정적으로 검출해야 한다.

검증해야 할 항목:

| 항목 | 의미 |
|---|---|
| detection rate | 각 카메라에서 object가 얼마나 자주 잡히는가 |
| confidence | confidence가 카메라별로 크게 다르지 않은가 |
| bbox area | view마다 크기가 지나치게 작거나 불안정하지 않은가 |
| class mismatch | 같은 물체를 다른 class로 잡지 않는가 |
| reprojection consistency | triangulated 3D point를 다른 카메라에 다시 투영했을 때 bbox와 맞는가 |

현재 추가한 EDA의 목적은 이 중 detection rate, confidence, bbox area를 먼저 확인하는 것이다.

---

## 10. 결론

`CameraInfo`는 intrinsic만 준다.

카메라 간 또는 `base_link` 기준 extrinsic은 `CameraInfo`만으로는 알 수 없다.

평가 환경에서 TF가 없다면, `tip <-> port`의 3D offset을 얻기 위해서는 다음 중 하나가 필요하다.

1. camera extrinsic을 사전에 calibration해서 고정값으로 코드에 넣고 multi-view triangulation 수행
2. 이미지 기반 모델이 `tip_to_port xyz`를 직접 예측하도록 학습

정확한 기하 기반 추정을 원하면 1번이 필요하다.

데이터가 충분하고 시뮬레이션 조건이 안정적이면 2번도 가능하지만, 이 경우 모델이 depth와 geometry를 암묵적으로 학습해야 한다.

