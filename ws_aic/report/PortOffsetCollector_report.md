# PortOffsetCollector Camera–Plug Timestamp Synchronization Report

- 작성일: 2026-07-30
- 대상: PortOffsetCollector의 camera, ControllerState, plug TF 수집 흐름
- 결론: center camera timestamp를 기준으로 메인 TF2 buffer에서 plug TF를 한 번만 조회한다. 별도 raw TF cache와 동일 transform 재조회는 제거했다.

### Why?

학습 sample의 image와 plug label은 같은 시점의 상태를 나타내야 한다. 기존 최초 구현은 TCP 정지 후 Observation을 얻고 조회 시점의 최신 plug TF를 저장했기 때문에, image timestamp와 label timestamp가 달라질 수 있었다. 이후 같은 center timestamp를 두 TF2 buffer에서 중복 조회하는 검사가 추가됐지만 두 buffer가 같은 TF source와 TF2 계산을 사용해 독립적인 정확성 증명이 되지 않았다.

따라서 center image timestamp를 sample 기준 시각으로 고정하고, 그 시각의 plug TF를 기존 메인 TF2 buffer에서 직접 조회하는 단일 경로가 필요했다. 이 목적은 camera와 label의 시간 기준을 일치시키는 것이며, 별도 raw TF 재구성이나 MCAP 사후 증명을 유지하는 것이 아니다.

### What I Made

PortOffsetCollector의 sample 기준 시각은 center image의 <code>header.stamp</code>다. left/right image와 ControllerState가 설정된 허용오차 안에 있는 Observation을 선택한 뒤, 같은 center timestamp를 지정해 <code>base_link &lt;- cable_tip_frame</code> TF를 메인 TF2 buffer에서 조회한다.

| 구분 | 이전 | 현재 |
|---|---|---|
| 기준 시각 | center image timestamp | center image timestamp |
| 실제 plug TF | 메인 TF2에서 기준 시각으로 조회 | 메인 TF2에서 기준 시각으로 조회 |
| 추가 처리 | 별도 raw TF cache에서 같은 계산을 반복 | 없음 |
| 저장 승인 | 두 TF2 결과 비교까지 통과 | 지정 시각 TF 조회와 기존 timestamp gate 통과 |
| 별도 TF subscriber | /tf, /scoring/tf, /tf_static 구독 | 제거 |
| metadata | raw TF provenance와 재구성 오차 포함 | 실제 저장에 사용한 source timestamp와 transform 포함 |

전체 워크플로우 변화는 다음과 같다.

~~~mermaid
flowchart TB
    subgraph After["After"]
        A1["left / center / right Image + ControllerState"] --> A2["center timestamp T 선택"]
        A2 --> A3{"Camera·Controller 허용오차 통과?"}
        A3 -->|No| A7["Sample 폐기"]
        A3 -->|Yes| A4["메인 TF2 buffer에서 plug TF(T) 한 번 조회"]
        A4 -->|조회 실패| A7
        A4 -->|조회 성공| A5["TF timestamp metadata 확인"]
        A5 -->|통과| A6["Image + Label 저장"]
        A5 -->|실패| A7
    end
    
    subgraph Before["Before"]
        B1["left / center / right Image + ControllerState"] --> B2["center timestamp T 선택"]
        B2 --> B3["메인 TF2 buffer에서 plug TF(T) 조회"]
        B3 --> B4["별도 raw TF buffer에서 plug TF(T) 재조회"]
        B4 --> B5["두 결과의 위치·회전 비교"]
        B5 --> B6{"모든 gate 통과?"}
        B6 -->|Yes| B7["Image + Label 저장"]
        B6 -->|No| B8["Sample 폐기"]
    end
~~~

관련 변경 파일은 다음과 같다.

- <code>port_offset_runtime.py</code>: 별도 TF buffer/subscriber와 raw 재구성 함수 제거
- <code>port_offset_stage_motion.py</code>: 두 번째 TF 품질 gate 제거
- <code>PortOffsetCollect.py</code>, <code>port_offset_base.py</code>: 삭제된 함수 binding/import 제거
- <code>test_port_offset_timestamp_sync.py</code>: 지정 camera timestamp가 메인 TF2 조회에 전달되는 회귀 테스트로 교체
- <code>ais_auto_capture/README.md</code>: 현재 단일 TF2 조회 흐름으로 설명 수정

### What was problem

최초 구현은 TCP 정지 후 Observation을 얻고 최신 TF를 별도로 조회했다.

~~~python
# ais_policy/data_gen_node/data_gen_node/port_offset_stage_motion.py | _stage_collect()
save_obs = get_observation()
plug_tf = self._lookup_transform(
    "base_link",
    cable_tip_frame,
)  # Time(): 조회 시점의 최신 TF
~~~

이 방식은 image timestamp와 TF 조회 시점이 달라질 수 있었다. TCP가 정지했다는 조건도 camera와 plug TF가 같은 시각이라는 뜻은 아니었다.

이를 해결하는 과정에서 center image timestamp의 TF를 메인 buffer에서 조회한 다음, 동일한 TF topic을 받는 별도 buffer에서도 같은 timestamp를 다시 조회하는 로직이 추가됐다.

~~~python
# ais_policy/data_gen_node/data_gen_node/port_offset_stage_motion.py | _stage_collect()
live_tf = self._lookup_transform_at(
    "base_link",
    cable_tip_frame,
    center_image.header.stamp,
)

reconstructed_tf = self._tf_quality_buffer.lookup_transform(
    "base_link",
    cable_tip_frame,
    Time.from_msg(center_image.header.stamp),
)

position_error, angle_error = _transform_difference(
    live_tf,
    reconstructed_tf,
)
~~~

이 두 번째 조회는 다음 이유로 현재 목적에 필요하지 않았다.

1. 두 buffer가 같은 TF source를 입력으로 사용했다.
2. 두 결과 모두 TF2의 같은 보간·경로 합성 구현으로 계산됐다.
3. 첫 번째 지정 timestamp 조회가 실패하면 이미 sample을 저장하지 않는다.
4. MCAP 사후 재구성을 사용하지 않으므로 별도 raw provenance를 유지할 요구가 없다.
5. 별도 subscriber, 10초 cache, lock, edge timestamp graph와 추가 실패 경로만 늘어났다.

따라서 같은 입력과 같은 계산기를 이용한 두 번째 조회는 독립적인 GT 검증이 아니라 중복된 buffer 일관성 검사였다.

### How it changed

현재 핵심 로직은 다음과 같다.

~~~python
# ais_policy/data_gen_node/data_gen_node/port_offset_stage_motion.py | _stage_collect()
capture_stamp = save_obs.center_image.header.stamp

save_raw_plug_stamped = self._lookup_transform_at(
    "base_link",
    ctx["cable_tip_frame"],
    capture_stamp,
)
~~~

<code>ais_policy/data_gen_node/data_gen_node/port_offset_runtime.py | _lookup_transform_at()</code>은 전달받은 stamp를 ROS time으로 변환해 policy가 이미 사용하는 메인 TF2 buffer에 직접 전달한다.

~~~python
# ais_policy/data_gen_node/data_gen_node/port_offset_runtime.py | _lookup_transform_at()
def _lookup_transform_at(
    self,
    target_frame: str,
    source_frame: str,
    stamp,
) -> TransformStamped:
    query_time = Time.from_msg(stamp)
    return self._parent_node._tf_buffer.lookup_transform(
        target_frame,
        source_frame,
        query_time,
        timeout=Duration(seconds=self.collect_sync_wait_timeout_sec),
    )
~~~

최종 저장 흐름은 다음 순서다.

1. <code>ais_policy/data_gen_node/data_gen_node/port_offset_dataset.py | _wait_for_synchronized_observation()</code>이 left/center/right image와 ControllerState timestamp를 검사한다.
2. center image timestamp를 <code>capture_stamp_ns</code>로 선택한다.
3. camera 간 최대 차이와 ControllerState 대 center 차이가 기본 30 ms를 넘으면 sample을 폐기한다.
4. <code>ais_policy/data_gen_node/data_gen_node/port_offset_runtime.py | _lookup_transform_at()</code>이 메인 TF2 buffer에서 center timestamp의 plug TF를 최대 설정 timeout 동안 조회한다.
5. 조회가 실패하면 예외가 수집 단계의 기존 실패 처리로 전달되어 sample을 저장하지 않는다.
6. 조회에 성공하면 <code>ais_policy/data_gen_node/data_gen_node/port_offset_dataset.py | _tf_sync_metadata()</code>가 실제 저장할 transform과 timestamp metadata를 구성한다.
7. 모든 조건을 통과한 Observation, plug TF와 label만 함께 저장한다.

현재 보장 범위는 다음과 같다.

- camera 기준 시각과 plug label 계산 시각은 동일한 center image ROS timestamp를 사용한다.
- TF2가 해당 시각의 transform을 만들 수 없으면 sample을 저장하지 않는다.
- left/right image와 ControllerState는 center timestamp와 설정된 허용오차 안에 있어야 한다.
- TF2가 정확한 raw sample을 사용했는지 또는 앞뒤 sample을 보간했는지는 별도로 기록하지 않는다.
- 세 camera의 물리적인 hardware trigger 동시 노출까지 보장하지는 않는다.
- MCAP 기반 사후 재구성은 이 workflow의 승인 조건으로 사용하지 않는다.
