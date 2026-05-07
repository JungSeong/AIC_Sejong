# LeRobot 데이터셋 메타데이터 — 파라미터 정리

> 데이터셋 포맷: `codebase_version v3.0`
> 작성일: 2026-04-26

---

## 전체 개요

| 파라미터 | 값 | 설명 |
| --- | --- | --- |
| `codebase_version` | v3.0 | LeRobot 데이터셋 포맷 버전 |
| `robot_type` | null | 로봇 종류 (미지정) |
| `total_episodes` | 2 | 전체 에피소드(시연) 수 |
| `total_frames` | 1062 | 전체 프레임 수 (에피소드 합산) |
| `total_tasks` | 1 | 태스크 종류 수 |
| `chunks_size` | 1000 | parquet 파일당 최대 프레임 수 (분할 단위) |
| `data_files_size_in_mb` | 100 MB | parquet 데이터 파일 총 용량 |
| `video_files_size_in_mb` | 200 MB | 영상 파일 총 용량 |
| `fps` | 10 | 초당 프레임 수 |
| `splits` | `train: 0:2` | 학습/검증 분할 (에피소드 0~1 → train) |
| `data_path` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` | parquet 파일 경로 템플릿 |
| `video_path` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` | 영상 파일 경로 템플릿 |

---

## Features (채널별 상세)

### `observation.state` — shape `[35]`, dtype `float32`

로봇의 현재 상태 벡터. 35차원.

| 그룹 | 필드 | 차원 | 설명 |
| --- | --- | --- | --- |
| TCP 위치/자세 | `tcp_pose.position.{x,y,z}` | 3 | 엔드이펙터 위치 (미터) |
| TCP 위치/자세 | `tcp_pose.orientation.{x,y,z,w}` | 4 | 엔드이펙터 자세 (쿼터니언) |
| TCP 속도 | `tcp_velocity.linear.{x,y,z}` | 3 | 엔드이펙터 선속도 |
| TCP 속도 | `tcp_velocity.angular.{x,y,z}` | 3 | 엔드이펙터 각속도 |
| TCP 오차 | `tcp_error.{x,y,z,rx,ry,rz}` | 6 | 목표 대비 위치/자세 오차 |
| 관절 각도 | `joint_positions.{0~6}` | 7 | 7축 관절 각도 (라디안) |
| 힘/토크 | `force.{x,y,z}` | 3 | F/T 센서 — 힘 (N) |
| 힘/토크 | `torque.{x,y,z}` | 3 | F/T 센서 — 토크 (Nm) |
| 그리퍼 오프셋 | `gripper_offset.{x,y,z}` | 3 | 그리퍼 위치 보정 오프셋 |

---

### `action` — shape `[7]`, dtype `float32`

로봇에 전달하는 제어 명령. 목표 TCP 위치/자세.

| 필드 | 차원 | 설명 |
| --- | --- | --- |
| `position.{x,y,z}` | 3 | 목표 엔드이펙터 위치 |
| `orientation.{x,y,z,w}` | 4 | 목표 엔드이펙터 자세 (쿼터니언) |

---

### `observation.plug_to_port` — shape `[7]`, dtype `float32`

플러그 → 포트 간 상대 변환. 조립 태스크의 목표 관계를 나타냄.

| 필드 | 차원 | 설명 |
| --- | --- | --- |
| `translation.{x,y,z}` | 3 | 플러그에서 포트까지의 상대 위치 |
| `rotation.{x,y,z,w}` | 4 | 상대 자세 (쿼터니언) |

---

### `observation.images.*` — shape `[256, 288, 3]`, dtype `video`

10fps로 촬영된 RGB 카메라 영상 3채널.

| 키 | 카메라 |
| --- | --- |
| `observation.images.left_camera` | 왼쪽 카메라 |
| `observation.images.center_camera` | 중앙 카메라 |
| `observation.images.right_camera` | 오른쪽 카메라 |

**영상 정보:**

| 항목 | 값 |
| --- | --- |
| 해상도 | 256 × 288 |
| 채널 | 3 (RGB) |
| 코덱 | av1 |
| 픽셀 포맷 | yuv420p |
| FPS | 10 |
| 오디오 | 없음 |

---

### `observation.scenario_params` — shape `[11]`, dtype `float32`

에피소드별 실험 조건 파라미터. 씬 구성을 수치로 인코딩.

| 필드 | 설명 |
| --- | --- |
| `trial_type` | 시험 유형 / 태스크 변형 분류 |
| `rail_idx` | 사용된 레일 인덱스 |
| `board_x` | 보드 위치 — X축 |
| `board_y` | 보드 위치 — Y축 |
| `board_yaw` | 보드 회전 — Yaw 각도 |
| `gripper_offset_x` | 초기 그리퍼 오프셋 — X |
| `gripper_offset_y` | 초기 그리퍼 오프셋 — Y |
| `gripper_offset_z` | 초기 그리퍼 오프셋 — Z |
| `nic_translation` | NIC 커넥터 위치 변위 |
| `nic_yaw` | NIC 커넥터 Yaw 회전 |
| `sc_translation` | SC(슬롯/커넥터) 위치 변위 |

---

### 인덱스 및 타임스탬프 채널

| Feature | dtype | 설명 |
| --- | --- | --- |
| `timestamp` | float32 `[1]` | 프레임 타임스탬프 (초) |
| `frame_index` | int64 `[1]` | 에피소드 내 프레임 번호 |
| `episode_index` | int64 `[1]` | 에피소드 번호 |
| `index` | int64 `[1]` | 데이터셋 전체 기준 절대 프레임 번호 |
| `task_index` | int64 `[1]` | 태스크 유형 번호 |

---

## 구조 요약

```
observation.state [35]           →  로봇 운동학 + F/T 센서 + 그리퍼
action [7]                       →  목표 TCP 위치/자세 명령
observation.plug_to_port [7]     →  조립 목표 상대 변환
observation.images.* [256×288]   →  RGB 영상 3채널 (10fps, av1)
observation.scenario_params [11] →  에피소드별 씬 구성 파라미터
timestamp / *_index              →  동기화 메타데이터
```
