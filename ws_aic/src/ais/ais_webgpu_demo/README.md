# AIC Sejong Vision Offset WebGPU Demo

AIC Sejong의 `FinalPolicy` vision-offset PyTorch 체크포인트를 ONNX로 내보내고, 브라우저의 GPU에서 추론하는 데모입니다.

```text
PyTorch checkpoint (.pt)
        ↓ export_vision_offset_onnx.py
ONNX model (.onnx)
        ↓ Vite local server
Chrome / Edge + ONNX Runtime Web
        ↓ WebGPU
[x, y, z, roll, pitch, yaw]
```

> 데모·모델 검증용입니다. 브라우저의 예측값을 실제 로봇 제어 명령으로 사용하지 마세요.

## 모델 인터페이스

| 항목 | 값 |
|---|---|
| 입력 이름 | `images` |
| 입력 형상 | `[1, 3, 3, 224, 224]` |
| 카메라 순서 | `left`, `center`, `right` |
| 색상 순서 | RGB |
| 정규화 | ImageNet mean/std |
| 출력 이름 | `correction` |
| 출력 순서 | `x_m`, `y_m`, `z_m`, `roll_rad`, `pitch_rad`, `yaw_rad` |

체크포인트가 5D 출력인 경우 exporter가 `z_m = 0`을 삽입해 브라우저에는 항상 6D 결과를 제공합니다.

## 디렉터리 구조

```text
ais_webgpu_demo/
├── index.html
├── package.json
├── README.md
├── public/
│   ├── models/             # 생성한 ONNX 모델
│   └── ort/                # npm script가 복사하는 ORT WASM 파일
├── scripts/
│   ├── copy-ort-wasm.mjs
│   └── export_vision_offset_onnx.py
└── src/
    ├── main.js
    └── styles.css
```

ONNX 모델과 빌드 결과는 Git에 포함되지 않습니다.

## 요구 사항

- AIC Sejong Pixi 환경
- Node.js 20 이상
- Chrome 또는 Edge 최신 버전
- WebGPU를 지원하는 클라이언트 GPU와 드라이버
- `localhost` 또는 HTTPS 보안 컨텍스트

여기서 **클라이언트**는 Moonlight가 아니라 이 데모 웹페이지를 여는
`운영체제 + 브라우저 + GPU/드라이버` 조합을 의미합니다. 추론에는 서버가
아닌 클라이언트의 GPU가 사용됩니다.

### WebGPU 클라이언트 지원 범위

| 클라이언트 환경 | 지원 상태 | 권장도 |
|---|---|---|
| Windows 10/11 + 최신 Chrome 또는 Edge | D3D12 기반 기본 지원 | 매우 권장 |
| macOS + 최신 Chrome 또는 Edge | 기본 지원 | 권장 |
| macOS/iPhone/iPad + Safari 26 이상 | WebGPU 지원 | 권장 |
| ChromeOS + Chrome | Vulkan 지원 기기에서 사용 가능 | 권장 |
| Android 12 이상 + Chrome | Qualcomm/ARM GPU 중심 지원, 기기별 차이 있음 | 기기별 확인 |
| Windows + Firefox 142 이상 | 기본 활성화 | 사용 가능 |
| Apple Silicon Mac + Firefox 147 이상 | 기본 활성화 | 사용 가능 |
| Linux + Chrome/Firefox | Vulkan, 드라이버 및 브라우저 설정에 따라 다름 | 사전 테스트 필요 |
| Intel Mac + Firefox | 제한적·실험적 | 권장하지 않음 |

이 데모에는 다음 조합이 가장 안정적입니다.

```text
Windows 10/11
+ 최신 Chrome 또는 Edge
+ 최신 NVIDIA/AMD/Intel 그래픽 드라이버
```

일반적으로 NVIDIA RTX, AMD Radeon RX, Intel Arc/Iris Xe 및 Apple Silicon
계열이 적합합니다. 내장 그래픽과 모바일 GPU도 WebGPU를 지원할 수 있지만,
약 295MB 모델의 초기화 및 추론 성능은 데스크톱 GPU가 유리합니다.

지원 범위는 브라우저 업데이트에 따라 달라질 수 있으므로 다음 공식 문서를
함께 확인합니다.

- [Chrome WebGPU 지원 개요](https://developer.chrome.com/docs/web-platform/webgpu/overview)
- [Firefox WebGPU 지원 현황](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Experimental_features#webgpu_api)
- [Safari 26 WebGPU 안내](https://webkit.org/blog/17640/webkit-features-for-safari-26-2/)
- [MDN WebGPU API](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)

### 브라우저에서 지원 여부 확인

브라우저 개발자 도구의 Console에서 다음 코드를 실행합니다.

```javascript
const adapter = await navigator.gpu?.requestAdapter({
  powerPreference: "high-performance",
});

console.log(adapter ? "WebGPU 지원" : "WebGPU 미지원");
console.log(adapter?.info);
```

`navigator.gpu`가 `undefined`이거나 `adapter`가 `null`이면 해당
브라우저·GPU 조합에서는 WebGPU가 활성화되지 않은 것입니다. 데모 화면
상단의 `WebGPU 사용 가능` 상태로도 확인할 수 있습니다.

WebGPU는 보안 컨텍스트에서만 사용할 수 있으므로 외부 PC에서는 일반 HTTP
주소 대신 HTTPS를 사용하거나, 아래 설명처럼 SSH 터널을 통해
`http://localhost`로 접속합니다.

## 1. Python 환경 준비

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src
pixi install
```

ONNX는 `ws_aic/src/pixi.toml`의 PyPI 의존성에 포함되어 있습니다.

## 2. PyTorch 체크포인트를 ONNX로 변환

### SFP 모델

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src

pixi run python \
  ais/ais_webgpu_demo/scripts/export_vision_offset_onnx.py \
  --port-type sfp
```

생성 파일:

```text
ais/ais_webgpu_demo/public/models/vision_offset_sfp.onnx
ais/ais_webgpu_demo/public/models/vision_offset_sfp.metadata.json
```

### SC 모델

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src

pixi run python \
  ais/ais_webgpu_demo/scripts/export_vision_offset_onnx.py \
  --port-type sc
```

생성 파일:

```text
ais/ais_webgpu_demo/public/models/vision_offset_sc.onnx
ais/ais_webgpu_demo/public/models/vision_offset_sc.metadata.json
```

### 임의 체크포인트 지정

```bash
pixi run python \
  ais/ais_webgpu_demo/scripts/export_vision_offset_onnx.py \
  --port-type sfp \
  --checkpoint /path/to/checkpoint.pt \
  --output ais/ais_webgpu_demo/public/models/custom.onnx
```

`--port-type`은 기본 체크포인트와 출력 파일 이름을 정합니다. `--checkpoint`와 `--output`을 지정하면 해당 값을 우선 사용합니다.

## 3. 웹 의존성 설치

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src/ais/ais_webgpu_demo
npm install
```

주요 패키지:

- `onnxruntime-web`: ONNX 모델 로딩과 WebGPU 실행
- `vite`: 로컬 개발 서버와 프로덕션 빌드

## 4. 개발 서버 실행

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src/ais/ais_webgpu_demo
npm run dev
```

호스트 PC에서 다음 주소를 엽니다.

```text
http://localhost:5173
```

Vite는 `0.0.0.0`에 바인딩되지만, 다른 PC에서 `http://192.168.0.200:5173`으로 직접 열면 보안 컨텍스트가 아니므로 WebGPU가 비활성화될 수 있습니다.

원격 PC에서는 SSH 터널을 사용해 브라우저 주소를 `localhost`로 유지합니다.

```bash
ssh -p 2122 \
  -L 5173:127.0.0.1:5173 \
  swlinux@selpa.vums.co.kr
```

SSH 연결을 유지한 상태로 원격 PC 브라우저에서 엽니다.

```text
http://localhost:5173
```

이 경우 WebGPU 추론은 Sunshine 호스트가 아니라 **원격 브라우저를 실행한 PC의 GPU**에서 수행됩니다.

## 5. 데모 사용

1. 브라우저 상단에서 `WebGPU 사용 가능` 상태를 확인합니다.
2. `SFP` 또는 `SC`를 선택합니다.
3. `모델 로드`를 누릅니다.
4. `left`, `center`, `right` 카메라 이미지를 각각 선택합니다.
5. `WebGPU 추론 실행`을 누릅니다.
6. 결과 카드에서 위치는 `mm`, 회전은 `deg` 단위로 확인합니다.

초기 모델 로딩은 모델 크기 때문에 시간이 걸릴 수 있습니다. 한 번 로드한 뒤의 추론 시간이 결과 패널에 표시됩니다.

## 6. 프로덕션 빌드

```bash
cd /home/swlinux/Desktop/workspace/AIC_Sejong/ws_aic/src/ais/ais_webgpu_demo
npm run build
```

빌드 결과:

```text
dist/
```

로컬에서 빌드 결과를 확인합니다.

```bash
npm run preview
```

```text
http://localhost:4173
```

## 7. 전처리와 출력 변환

브라우저는 기존 `VisionOffsetPredictor`와 동일하게 다음 전처리를 수행합니다.

1. 각 카메라 이미지를 `224×224`로 bilinear resize
2. RGBA Canvas 데이터를 RGB로 변환
3. `[0,255]` 값을 `[0,1]`로 변환
4. ImageNet mean/std로 채널별 정규화
5. `[1,3,3,224,224]` NCHW 텐서 생성

위치 결과는 meter에서 millimeter로, 회전 결과는 radian에서 degree로 바꿔 표시합니다. ONNX 원본 출력값 자체는 meter/radian 단위입니다.

## 8. 검증 권장 사항

브라우저 데모를 모델 품질 평가에 사용하기 전 다음을 비교하세요.

- 같은 이미지 3장에 대한 PyTorch 출력
- 같은 이미지 3장에 대한 ONNX/WebGPU 출력
- 축별 절대 오차
- 첫 실행을 제외한 추론 지연시간
- Chrome/Edge 및 서로 다른 GPU에서의 결과

ONNX export는 고정 입력 형상 `[1,3,3,224,224]`을 사용합니다. 카메라 개수, 순서 또는 입력 크기를 바꾸면 다시 export해야 합니다.

## 9. 알려진 제약

- 실제 체크포인트가 약 295MB이므로 ONNX 다운로드와 초기화가 느릴 수 있습니다.
- 브라우저와 GPU 드라이버에 따라 WebGPU 지원 연산과 성능이 다릅니다.
- Firefox/Safari에서는 지원 상태가 다를 수 있으므로 Chrome/Edge를 우선 사용합니다.
- 브라우저 탭이 비활성화되면 실행 스케줄링이 제한될 수 있습니다.
- ROS 2 토픽과 직접 연결하지 않으며 사용자가 선택한 로컬 이미지 파일만 처리합니다.
- 실제 제어 경로는 기존 ROS 2 + PyTorch + CUDA를 유지합니다.
