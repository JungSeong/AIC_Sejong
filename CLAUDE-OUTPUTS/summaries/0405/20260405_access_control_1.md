# 접근 제어 (Zenoh ACL)

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/access_control.md`

---

## 개요

Zenoh ACL(접근 제어 목록)로 참가자가 시뮬레이션 내부 포즈 데이터(`/gz_server` 등)에 접근하지 못하도록 차단해 치팅을 방지한다.

---

## 수동 테스트 (터미널 3개)

### Terminal 1 — Zenoh 라우터 (ACL 적용)
```bash
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_router.sh
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 2 — 시뮬레이션 환경
```bash
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_eval_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  nic_card_mount_0_present:=true sc_port_0_present:=true \
  ground_truth:=false spawn_task_board:=true spawn_cable:=true \
  attach_cable_to_gripper:=true sfp_mount_rail_0_present:=true \
  cable_type:=sfp_sc_cable
```

### Terminal 3 — 접근 차단 확인 (모델 세션)
```bash
. install/setup.bash
. src/aic/docker/aic_model/zenoh_config_model_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# 아래 호출은 ACL에 의해 차단됨
ros2 service call /gz_server/get_entities_states simulation_interfaces/srv/GetEntitiesStates
```

> `eval` 세션 환경변수를 사용하면 엔티티 포즈/속도 전체 조회 가능. 하지만 실제 제출 환경에서는 `eval` 패스워드가 다르게 설정됨.

---

## Docker Compose로 테스트

```bash
# 빌드
docker compose -f docker/docker-compose.yaml build

# 실행 (평가·모델 컨테이너 상호작용 테스트)
docker compose -f docker/docker-compose.yaml up
```

---

*관련 문서: `submission.md` / `custom_dockerfile.md`*
