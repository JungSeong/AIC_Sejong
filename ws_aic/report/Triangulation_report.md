# Triangulation Development Status Report

- 작성일: 2026-08-02
- 코드 기준: HEAD <code>38d0619</code> + 2026-08-02 working tree
- 대상: randomized case 생성, FinalPolicy multi-camera triangulation, ROS 평가, OpenCV/RViz debug
- 결론: **고정 home의 camera common-FOV 생성 조건과 strict timestamp SFP smoke는 통과했지만, YOLO 검출률과 정확도를 production 수준으로 검증한 단계는 아니다.**

### Why?

기존 워크플로우는 randomized case 생성, multi-camera 3D 계산, policy 사용, GT 평가와 시각 디버깅이 하나의 검증 가능한 경로로 연결돼 있지 않았다. 특히 기존 case generator는 PortOffset 범위에서 만든 candidate를 camera 가시성 검사 없이 모두 채택했고, evaluator와 debug 경로도 prediction의 frame과 관측 timestamp를 일관되게 증명하기 어려웠다.

따라서 실제 target entrance가 고정 observation pose의 공통 FOV에 들어오는 case를 생성하고, 동일한 triangulation 결과를 policy·ROS topic·OpenCV/RViz debug에 공유하며, prediction timestamp와 frame을 기준으로 GT를 비교하는 통합 경로가 필요했다. 이 작업의 목적은 기하 가시성과 시간·좌표계가 명시된 평가 가능성을 확보하는 것이며, YOLO 검출이나 production 정확도를 이미 보장한다고 주장하는 것이 아니다.

### What I Made

현재 triangulation은 단순 계산 함수만 있는 상태가 아니라 다음 실행 경로까지 연결돼 있다.

1. PortOffsetCollector 범위에서 candidate를 생성하고 fixed-home common FOV를 통과한 case만 YAML에 저장
2. 생성한 YAML을 Distrobox simulator에 전달
3. 세 camera의 YOLO 관측으로 포트 3D 위치 계산
4. 결과를 FinalPolicy 접근·정렬 단계에 사용
5. 결과를 timestamp와 frame이 있는 ROS topic으로 발행
6. OpenCV와 RViz에서 detection, 재투영, GT 오차를 실시간 확인하고 마지막 image를 1 Hz로 재발행
7. evaluator가 prediction과 같은 시각의 GT TF를 비교하고 CSV/JSONL/summary 저장

현재 구현 수준은 다음과 같다.

| 영역 | 현재 상태 | 근거 |
|---|---|---|
| Random case 생성 | 구현 완료 | <code>--seed</code>, <code>--num_cases</code>, PortOffset generator와 bounded rejection sampling |
| Simulator 실행 | 구현 완료 | Distrobox entrypoint parameter 전달 및 추가 <code>--sim-arg</code> 지원 |
| YOLO detection | 구현 완료 | left/center/right detection과 port index filtering |
| 3D triangulation | 구현 완료 | 모든 two-view pair DLT와 전체 camera 재투영 RMS 비교 |
| Camera timestamp 검사 | 구현 완료 | center 기준, 세 camera 전체 span threshold 검사 |
| ControllerState timestamp 검사 | 미구현 | extrinsic에는 TCP pose를 쓰지만 camera와 controller 시각 차이 gate가 없음 |
| Prediction topic | 구현 완료 | camera 관측 timestamp와 configured frame의 <code>PointStamped</code> |
| Evaluator frame 변환 | 구현 완료 | prediction <code>header.frame_id</code>를 읽어 base frame으로 변환 |
| Evaluator GT 동기화 | 구현 완료 | prediction timestamp의 GT TF 조회, 제한된 nearby fallback |
| OpenCV/RViz debug | 구현 완료 | detection image, triangulation reprojection image, 3D MarkerArray |
| Policy 제어 연결 | 구현 완료 | approach target 및 align 검증/보정에 triangulation 사용 |
| 수학 정확도 회귀 테스트 | 미구현 | synthetic camera geometry의 정답 3D 복원 테스트 없음 |
| Random batch 성능 검증 | 초기 | strict timestamp SFP smoke 1건 검증, SC 결과 없음 |
| 완전 자동 E2E 평가 | 미구현 | runner와 evaluator를 별도 terminal에서 실행하며 case는 수신 순서로 대응 |
| 초기 target 기하 가시성 | 구현 완료 | fixed home에서 target entrance가 margin 내부 camera 2개 이상인 case만 채택 |
| 초기 YOLO 검출 보장 | 미구현 | 가림, 조명, confidence는 생성 단계 pinhole projection으로 보장할 수 없음 |

#### 2026-08-02 Fixed Observation Common-FOV

Robot home joint noise를 0으로 고정하고, YAML의 board/module/port asset transform으로 실제 target entrance의 <code>base_link</code> XYZ를 계산한다. 이 점을 URDF에서 산출한 fixed-home camera optical transform으로 투영해 기본 64 px margin 안에 들어오는 camera가 두 개 이상인 candidate만 채택한다.

~~~text
# ais_triangulation/run_triangulation_cases.py | target_camera_projections()
# ais_triangulation/run_triangulation_cases.py | generate_cases()
p_camera = inverse(T_base_camera) · p_base
u = fx · X / Z + cx
v = fy · Y / Z + cy

accept ⇔ visible_camera_count ≥ 2
~~~

조건에 맞지 않으면 같은 case index를 다시 sampling하며 10,000회 실패 시 오류로 종료한다. Seed 30으로 생성한 SFP/SC 20건은 19건이 세 camera, 1건이 center-right 공통 FOV를 만족했다. 100건 회귀 검사도 모두 camera 두 개 이상 조건을 통과했고, YAML transform 계산은 기존 smoke simulator GT와 각 축 1 µm 이내에서 일치했다.

상세 계산식과 asset 출처는 <code>report/map_generation_with_ports_visible.md</code>에 정리했다.

#### 전체 실행 워크플로우

~~~mermaid
flowchart TB
    A["run_triangulation_cases.py"] --> B["PortOffset 범위에서 candidate 생성"]
    B --> B2{"fixed-home common FOV camera ≥ 2?"}
    B2 -->|No| B
    B2 -->|Yes| C["YAML 저장 후 Distrobox 실행"]
    C --> D["AIC simulator + FinalPolicy"]

    D --> E["left / center / right Image"]
    D --> F["ControllerState.tcp_pose"]
    E --> G{"camera timestamp span 통과?"}
    G -->|No| H["Observation 폐기"]
    G -->|Yes| I["camera별 YOLO keypoint 검출"]
    F --> J["TCP pose + 고정 extrinsic으로 camera projection matrix 계산"]
    I --> K["두 camera OpenCV DLT"]
    J --> K
    K --> L["board/Z 범위 filter"]
    L --> M["전체 camera 재투영 30 px 검사와 RMS 계산"]
    M --> N["task hint로 target port 선택"]

    N --> O["FinalPolicy cache"]
    O --> P["approach control"]
    N -. "align 시 새 Observation으로 재계산" .-> V["align 검증 / XY 보정"]
    O --> Q["PointStamped publish"]
    O --> R["OpenCV/RViz PRED 재투영"]

    Q --> S["evaluate_triangulation_euclidean.py"]
    S --> T["prediction timestamp의 GT entrance TF 조회"]
    T --> U["dx / dy / dz / Euclidean error 저장"]
~~~

#### 1. Case 생성과 simulator 실행

<code>ais/ais_triangulation/run_triangulation_cases.py</code>는 다음을 지원한다.

- seed 재현성
- case 개수 가변 생성
- SFP/SC port type 및 순서 선택
- Task Board translation/yaw randomization
- module rail과 cable pose randomization
- <code>BASE_ROBOT_HOME</code> 고정과 target entrance common-FOV 검사
- <code>--min-visible-cameras</code>, <code>--visibility-margin-px</code>
- <code>gazebo_gui</code>, <code>launch_rviz</code> 등 명시적 simulator parameter 전달
- 반복 가능한 <code>--sim-arg NAME=VALUE</code>
- YAML만 생성하는 <code>--generate-only</code>
- YAML 생성은 bold magenta, simulator 시작은 bold green 로그

Multi-trial schema의 전역 robot section은 고정 <code>BASE_ROBOT_HOME</code>을 모든 trial이 공유한다. Task Board, module, cable은 trial마다 다시 sampling하며 common-FOV 조건을 통과한 조합만 채택한다.

#### 2. 현재 3D 계산 로직

핵심 구현은 <code>ais_policy/final_policy/final_policy/vision.py | _estimate_all_sync()</code>와 <code>ais_policy/final_policy/final_policy/vision.py | _triangulate()</code>다.

~~~python
# ais_policy/final_policy/final_policy/vision.py | _triangulate()
p_a = k_a @ t_base_to_optA[:3, :]
p_b = k_b @ t_base_to_optB[:3, :]

pts_4d = cv2.triangulatePoints(
    p_a,
    p_b,
    [[u_a], [v_a]],
    [[u_b], [v_b]],
)
port_3d = (pts_4d[:3] / pts_4d[3]).flatten()
~~~

실제 후보 생성 순서는 다음과 같다.

1. left/center/right image timestamp가 모두 존재하는지 확인한다.
2. center timestamp를 reference로 두고 세 timestamp의 전체 span을 계산한다.
3. span이 기본 30 ms를 넘으면 Observation을 폐기한다.
4. 각 camera에서 YOLO detection과 keypoint 중심 UV를 얻는다.
5. 같은 <code>point_name</code>의 detection만 서로 대응시킨다.
6. 사용할 수 있는 모든 pair와 detection 조합을 DLT로 계산한다.
7. OpenCV DLT로 base-link 기준 3D 위치를 계산한다.
8. <code>BOARD_CENTER</code> 반경 0.5 m와 Z 범위 -0.1~0.5 m로 후보를 거른다.
9. 각 3D 점을 해당 point가 검출된 모든 camera에 재투영하고 camera별 30 px 이내인지 확인한다.
10. 전체 camera 재투영 RMS가 작은 순서로 정렬하고 기존 heuristic score를 보조 기준으로 사용한다.
11. 10 mm 이내 후보는 중복으로 제거해 가장 낮은 RMS의 pair 결과를 남긴다.
12. task의 port/module 이름과 port index로 최종 target을 선택한다.

카메라 projection matrix는 TF2에서 optical frame을 직접 조회하지 않는다. Observation의 <code>controller_state.tcp_pose</code>와 코드에 고정된 <code>tool0 -&gt; optical</code> calibration을 조합한다.

~~~text
# ais_policy/final_policy/final_policy/vision.py | _base_to_camera_optical_matrix()
T_base_tcp
→ T_base_tool0
→ T_base_optical
→ inverse
→ T_optical_base
~~~

#### 3. FinalPolicy 제어 연결

초기 <code>lift_up_detect</code>에서 triangulation 성공 시 다음 하나의 cache에 저장한다.

~~~python
# ais_policy/final_policy/final_policy/FinalPolicy.py | _cache_detected_port()
self._cached_port_base = np.asarray(port, dtype=np.float64)
~~~

같은 cache 값이 다음 경로로 전달된다.

- <code>ais_policy/final_policy/final_policy/FinalPolicy.py | _publish_triangulated_port_xyz()</code>: evaluator 입력 topic
- <code>ais_policy/final_policy/final_policy/debug.py | _save_triangulation_debug_images()</code>: OpenCV/RViz의 PRED
- <code>ais_policy/final_policy/final_policy/FinalPolicy.py | _stage_approach()</code>: 포트 앞 접근 목표

Align은 초기 cache를 재사용하지 않는다. Vision-offset 모델이 안정화된 뒤 새 Observation으로 port와 tip triangulation을 다시 계산해 잔차를 확인한다. 기본 설정은 triangulation 사용이 켜져 있고, 잔차 6 mm 이하이면 정렬 완료로 인정한다. 6 mm를 넘으면 gain 0.5, 최대 3 mm step으로 XY를 추가 이동한다. Triangulation을 얻지 못했을 때 반드시 실패시키는 옵션은 기본적으로 꺼져 있다.

#### 4. Topic과 debug

FinalPolicy는 옵션이 켜진 경우 <code>/final_policy/triangulated_port_xyz</code>에 <code>PointStamped</code>를 발행한다.

- <code>header.stamp</code>: triangulation에 사용한 center camera timestamp 우선
- <code>header.frame_id</code>: 기본 <code>base_link</code>
- <code>point</code>: <code>_cached_port_base</code>의 XYZ

숫자 XYZ는 항상 base-link cache다. <code>AIC_TRIANGULATED_PORT_XYZ_FRAME_ID</code>를 다른 frame으로 설정해도 좌표를 그 frame으로 변환하지 않고 header 문자열만 바뀐다. 따라서 현재 publisher에서는 frame ID를 <code>base_link</code>로 유지해야 한다.

Debug는 다음을 제공한다.

- YOLO detection overlay: camera별 bbox/keypoint/UV/confidence
- Triangulation overlay: 동일한 최종 3D 값의 PRED 재투영
- GT overlay: center timestamp에서 조회한 port entrance TF 재투영
- MarkerArray: base-link 기준 PRED, GT, 두 점의 오차선
- image header: 원본 camera timestamp와 optical frame 유지
- late subscriber 대응: 마지막 detection/triangulation image를 원본 header 그대로 cache하고 subscriber가 있으면 기본 1 Hz로 재발행

따라서 evaluator가 받는 prediction과 OpenCV debug의 PRED는 같은 3D cache를 사용한다.

#### 5. Evaluator

<code>evaluate_triangulation_euclidean.py</code>는 YAML task의 port type, module name, port name으로 entrance frame을 구성한다.

~~~text
# ais_triangulation/evaluate_triangulation_euclidean.py | target_frame_from_task()
SC:
task_board/{target_module}/sc_port_base_link_entrance

SFP:
task_board/{target_module}/{port_name}_link_entrance
~~~

ROS mode에서 evaluator는 다음을 수행한다.

1. prediction의 <code>header.frame_id</code>와 <code>header.stamp</code>를 필수로 확인한다.
2. prediction frame이 base frame과 다르면 prediction timestamp의 TF로 XYZ를 변환한다.
3. 같은 prediction timestamp에서 base frame 기준 GT entrance TF를 조회한다.
4. <code>ais_triangulation/evaluate_triangulation_euclidean.py | lookup_synced_transform()</code>은 exact lookup이 실패하면 TF2 full lookup을 시도한다.
5. 그래도 실패하면 기본 30 ms 이내의 latest TF만 <code>nearby_latest</code>로 허용한다.
6. <code>dx/dy/dz</code>와 3D Euclidean error를 mm 단위로 저장한다.

<code>--sync-threshold-ms 0</code>이면 <code>nearby_latest</code> fallback을 허용하지 않으므로 exact timestamp 조회가 성공한 sample만 저장한다.

CSV, JSONL과 summary JSON에는 prediction frame, TF 변환 mode, prediction/GT TF 시간차도 기록된다. CSV/JSONL/JSON offline 입력도 지원하지만 offline record에는 prediction과 GT XYZ가 모두 있어야 한다.

### What was problem

현재까지 해결된 문제는 다음과 같다.

| 과거 문제 | 현재 처리 |
|---|---|
| case 수가 고정됨 | <code>--num_cases</code>로 가변화 |
| randomization 범위가 별도 관리됨 | PortOffset scenario generator 직접 재사용 |
| simulator parameter 전달 제한 | 명시적 option과 반복 <code>--sim-arg</code> 지원 |
| prediction frame을 무시함 | <code>header.frame_id</code> 기준 base frame 변환 |
| GT를 prediction과 다른 시각에 조회할 수 있음 | prediction timestamp로 GT TF 조회 |
| OpenCV PRED와 publish XYZ 경로가 다를 수 있음 | 하나의 <code>_cached_port_base</code> 공유 |
| 실시간 debug 확인이 어려움 | camera Image topic과 MarkerArray 제공 |
| 결과 발행 뒤 RViz를 연결하면 Image가 비어 있음 | 마지막 Image를 원본 header로 cache하고 subscriber가 있을 때 기본 1 Hz 재발행 |
| 사용 가능한 pair 중 마지막 pair가 선택됨 | 모든 camera pair의 전체-view 재투영 RMS를 비교해 최저 오차 pair 선택 |
| 기존 generator가 candidate 가시성을 검사하지 않음 | fixed home에서 entrance가 margin 내부 camera 2개 이상인 candidate만 채택 |
| GPU 호환 오류 시 실행 불가 | <code>AIC_YOLO_DEVICE=cpu</code>를 inference device로 전달 |

그러나 다음 문제는 아직 남아 있다.

#### 1. ControllerState와 camera의 시간차를 검사하지 않음

Triangulation의 camera extrinsic은 <code>controller_state.tcp_pose</code>에서 계산하지만 <code>ais_policy/final_policy/final_policy/vision.py | _estimate_all_sync()</code>은 세 image timestamp만 검사한다.

Adapter는 newest-first deque에서 image보다 늦지 않은 첫 ControllerState를 선택하므로 가장 가까운 과거 sample을 고른다. 하지만 최대 허용 시간차는 검사하지 않는다.

~~~text
# aic/aic_adapter/src/aic_adapter.cpp | AicAdapterNode::image_callback()
center image at T
+ controller TCP pose at T - delta
→ camera extrinsic 계산

delta에 명시적 상한 없음
~~~

로봇이 이동하는 중이면 이 차이가 곧 extrinsic 오차와 triangulation 오차로 이어질 수 있다. 현재 시간 정합성은 camera끼리는 검사하지만 camera-controller까지 완성된 상태는 아니다.

#### 2. Lens distortion 보정이 없음

Triangulation과 재투영은 CameraInfo의 K matrix를 사용하지만 distortion coefficient D와 OpenCV distortion 보정 API를 사용하지 않는다. 입력 image가 이미 rectified라는 명시적 보장이 없다면 화면 가장자리에서 오차가 커질 수 있다.

#### 3. 실제 계산은 pair별 two-view이며 세 camera 동시 최적화는 없음

세 camera가 모두 검출되면 left-center, center-right, left-right를 모두 DLT로 계산하고 전체 camera 재투영 RMS가 가장 낮은 pair를 선택한다. 하지만 각 계산 자체는 two-view이며 세 view 전체를 이용한 least-squares refinement는 없다.

Pair 선택은 재투영 RMS를 기준으로 개선됐지만 baseline이나 ray angle에 따른 depth conditioning은 아직 비교하지 않는다.

#### 4. 품질 기준이 heuristic과 hard-coded 값에 의존

- third-camera pixel threshold: 30 px
- candidate duplicate threshold: 10 mm
- board center/radius: 고정 위치와 0.5 m
- Z range: -0.1~0.5 m
- candidate score: meter 단위 거리에서 confidence 합의 0.1배를 차감

이 값들은 randomization 결과와 실제 calibration 오차에 대해 통계적으로 튜닝됐다는 근거가 아직 없다.

#### 5. 수학 정확도 test와 E2E test가 없음

현재 20개 test는 배선, metadata, frame, timestamp gate, camera pair 선택과 common-FOV 생성을 확인한다. 하지만 알려진 camera matrix와 3D 정답을 사용해 <code>ais_policy/final_policy/final_policy/vision.py | _triangulate()</code>가 정답을 복원하는 synthetic test는 없다.

또한 Gazebo를 실행해 SFP/SC randomized case 전체를 생성부터 evaluator 결과까지 자동 검증하는 test도 없다.

#### 6. All-cases 대응이 message 순서에 의존

Evaluator의 <code>--all-cases</code>는 YAML trial 순서와 prediction 수신 순서를 1:1로 대응한다. Prediction 하나가 누락되거나 중복 발행되면 이후 case가 전부 한 칸씩 어긋날 수 있다. Prediction message에는 <code>case_name</code>이 포함되지 않는다.

#### 7. 정확도 표본이 부족함

Strict timestamp smoke에서 SFP 1건을 검증했고 3D 오차는 1.175 mm였다. 단일 sample이므로 다음을 아직 판단할 수 없다.

- seed와 board pose 변화에 대한 오차 분포
- SFP port 0/1 선택 정확도
- SC port 정확도
- camera pair별 실패율
- detection 실패율과 triangulation reject 비율
- 평균, P95, 최대 3D 오차
- align 보정이 실제 insertion 성공률을 높이는지

#### 8. Prediction frame 설정이 좌표 변환을 수행하지 않음

Publisher는 <code>_cached_port_base</code> 값을 그대로 쓰면서 <code>AIC_TRIANGULATED_PORT_XYZ_FRAME_ID</code>를 header에 넣는다. Frame ID를 <code>world</code> 등으로 바꾸면 base-link XYZ가 world 값으로 변환되는 것이 아니라 잘못된 frame label이 붙는다. Evaluator는 header를 신뢰하므로 이 설정 오류는 평가 오차로 이어진다.

#### 9. Simulator clean teardown이 완료되지 않음

Smoke trial은 engine 기준 성공했고 evaluator 결과도 저장됐지만, trial 종료 뒤 launch stack이 스스로 끝나지 않았다. 수동 SIGINT 정리 중 Gazebo component에서 <code>corrupted size vs. prev_size</code>가 발생해 runner exit code를 성공으로 볼 수 없었다. 계산 결과 검증과 별개로, 자동 batch 운용 전에 정상 종료 조건과 Gazebo process cleanup을 수정해야 한다.

#### 10. Common-FOV는 YOLO 검출을 보장하지 않음

생성 filter는 target entrance 중심점의 pinhole projection만 확인한다. Port mesh 전체 포함, 다른 물체에 의한 가림, 조명, distortion, YOLO confidence는 판정하지 않는다. 실제 검출까지 보장하려면 simulator 시작 직후 동일 target이 camera 두 개 이상에서 검출되는지 확인하는 runtime gate가 별도로 필요하다.

### How it changed

현재 개발 흐름은 다음 단계까지 진행됐다.

~~~mermaid
flowchart LR
    A["초기: two-camera triangulation 계산"] --> B["YOLO port index와 task hint 연결"]
    B --> C["PortOffset 범위 + fixed-home common-FOV case runner"]
    C --> D["PointStamped publish + evaluator"]
    D --> E["prediction frame/timestamp 기반 TF 동기화"]
    E --> F["OpenCV/RViz PRED·GT debug + latest Image 1 Hz"]
    F --> G["FinalPolicy approach/align 제어 통합"]
    G --> H["현재: strict SFP smoke 통과, batch 검증 초기"]
~~~

파일별 현재 책임은 다음과 같다.

| 파일 | 현재 책임 |
|---|---|
| <code>run_triangulation_cases.py</code> | target entrance projection, common-FOV rejection sampling, YAML 생성과 simulator 실행 |
| <code>vision.py</code> | camera timing, YOLO, DLT, 후보 검증과 target 선택 |
| <code>geometry.py</code> | base 3D 점의 camera pixel 재투영 |
| <code>FinalPolicy.py</code> | cache, publish, approach와 align 제어 연결 |
| <code>debug.py</code> | synchronized GT 조회, OpenCV overlay, RViz Image/Marker 발행과 마지막 Image 1 Hz 재발행 |
| <code>evaluate_triangulation_euclidean.py</code> | frame/timestamp-aware GT 비교와 결과 통계 |
| <code>test_triangulation_cases.py</code> | runner, timestamp, frame, debug 배선 회귀 검사 |

현재 바로 사용할 수 있는 범위:

- 단일 또는 소수 case를 생성해 simulator 실행
- fixed home에서 target entrance가 camera 두 개 이상의 안전 margin 안에 있는 case 생성
- triangulation 결과를 RViz/OpenCV에서 실시간 확인
- prediction과 같은 timestamp/base frame의 GT 오차 저장
- FinalPolicy의 접근과 align 보조값으로 사용

아직 완료로 판단하면 안 되는 범위:

- 전체 PortOffset randomization 범위에서의 정확도 보장
- SC/SFP 공통 성능 기준 충족
- 이동 중 camera-controller 완전 timestamp alignment
- lens distortion까지 포함한 calibrated multi-view reconstruction
- 누락/중복 prediction에 안전한 자동 batch 평가
- insertion 성공률 개선의 통계적 증명
- common-FOV를 통과한 case의 YOLO 검출 성공 보장

따라서 현재 상태는 **prototype을 넘어 실제 정책과 평가 도구까지 연결된 integration-complete 단계**이지만, **accuracy-validated 또는 production-ready 단계는 아니다.**

다음 우선순위는 다음 순서가 적절하다.

1. successful trial 뒤 launch stack과 Gazebo가 정상 종료되도록 teardown 경로를 수정한다.
2. center image와 ControllerState timestamp 차이에 threshold를 적용하거나, center timestamp의 camera optical TF를 직접 조회한다.
3. synthetic geometry test로 DLT 좌표계와 projection convention을 고정한다.
4. image rectification 여부를 확인하고 필요하면 distortion 보정을 추가한다.
5. SFP/SC randomized batch에서 최소 수십 case를 평가해 mean/P95/max와 reject rate를 기록한다.
6. prediction에 case identifier를 포함하거나 runner가 evaluator lifecycle을 함께 관리하도록 해 all-cases 순서 의존성을 제거한다.
7. Publisher frame ID를 base-link로 고정하거나, 다른 frame을 허용할 경우 XYZ도 실제로 변환한다.
8. 필요성이 측정된 경우에만 three-view refinement를 추가한다.
