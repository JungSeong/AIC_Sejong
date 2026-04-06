# 커스텀 Dockerfile 가이드

> 날짜: 2026-04-05
> 원본: `ws_aic/src/aic/docs/custom_dockerfile.md`

---

## 대상

`aic_model` 프레임워크를 사용하지 않고 자체 ROS 2 노드를 직접 구현하는 고급 사용자.

---

## 필수 요건 3가지

### 1. RMW — `rmw_zenoh_cpp` 사용 필수

```dockerfile
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
```

`rmw_zenoh_cpp`가 이미지 내에 설치되어 있어야 하며, 이 환경변수를 덮어쓰면 안 됨.

---

### 2. Zenoh 연결 설정

평가 실행 시 주입되는 환경변수:

| 변수 | 설명 |
|------|------|
| `RMW_IMPLEMENTATION` | 항상 `rmw_zenoh_cpp` |
| `ZENOH_ROUTER_CHECK_ATTEMPTS` | 항상 `-1` (라우터 준비 전 오류 방지) |
| `AIC_MODEL_ROUTER_ADDR` | 연결할 Zenoh 라우터 주소 |
| `AIC_MODEL_PASSWD` | `model` 사용자 인증 패스워드 |

엔트리포인트에서 `ZENOH_CONFIG_OVERRIDE` 설정:
```bash
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_MODEL_ROUTER_ADDR"'"];transport/auth/usrpwd/user="model";transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'";transport/auth/usrpwd/dictionary_file="/credentials.txt"'
```

크리덴셜 파일 생성:
```bash
echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
```

---

### 3. 엔트리포인트

이미지 실행 시 자동으로 정책 노드를 시작해야 함. **추가 인자 없이** 동작해야 함.

```dockerfile
ENTRYPOINT ["/entrypoint_my_policy.sh"]
```

---

*참고: `submission.md` / Zenoh 공식 문서: https://zenoh.io/docs/manual/access-control/*
