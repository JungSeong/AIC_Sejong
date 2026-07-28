import * as ort from "onnxruntime-web/webgpu";

import "./styles.css";

const IMAGE_SIZE = 224;
const VIEW_ORDER = ["left", "center", "right"];
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const OUTPUT_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"];

const state = {
  session: null,
  files: new Map(),
  previewUrls: new Map(),
};

const elements = {
  gpuDot: document.querySelector("#gpu-dot"),
  gpuStatus: document.querySelector("#gpu-status"),
  gpuDetail: document.querySelector("#gpu-detail"),
  portType: document.querySelector("#port-type"),
  modelUrl: document.querySelector("#model-url"),
  loadModel: document.querySelector("#load-model"),
  modelStatus: document.querySelector("#model-status"),
  runInference: document.querySelector("#run-inference"),
  inferenceStatus: document.querySelector("#inference-status"),
  latency: document.querySelector("#latency"),
};

ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
ort.env.wasm.numThreads = 1;

function setStatus(element, message, kind = "neutral") {
  element.textContent = message;
  element.dataset.kind = kind;
}

function updateRunButton() {
  const hasAllImages = VIEW_ORDER.every((view) => state.files.has(view));
  elements.runInference.disabled = !(state.session && hasAllImages);
  if (!state.session) {
    setStatus(elements.inferenceStatus, "먼저 ONNX 모델을 로드하세요.");
  } else if (!hasAllImages) {
    setStatus(elements.inferenceStatus, "left, center, right 이미지 3장을 선택하세요.");
  } else {
    setStatus(elements.inferenceStatus, "WebGPU 추론을 실행할 준비가 됐습니다.", "ready");
  }
}

async function detectWebGpu() {
  if (!window.isSecureContext) {
    elements.gpuDot.dataset.kind = "error";
    elements.gpuStatus.textContent = "보안 컨텍스트 필요";
    elements.gpuDetail.textContent = "localhost 또는 HTTPS로 접속하세요.";
    return;
  }
  if (!navigator.gpu) {
    elements.gpuDot.dataset.kind = "error";
    elements.gpuStatus.textContent = "WebGPU 미지원";
    elements.gpuDetail.textContent = "최신 Chrome 또는 Edge에서 다시 확인하세요.";
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("GPU adapter를 찾지 못했습니다.");
    }
    const info = adapter.info;
    const adapterName =
      info?.description || info?.device || info?.vendor || "사용 가능한 GPU";
    elements.gpuDot.dataset.kind = "ready";
    elements.gpuStatus.textContent = "WebGPU 사용 가능";
    elements.gpuDetail.textContent = adapterName;
  } catch (error) {
    elements.gpuDot.dataset.kind = "error";
    elements.gpuStatus.textContent = "WebGPU 초기화 실패";
    elements.gpuDetail.textContent = error.message;
  }
}

async function loadModel() {
  if (!navigator.gpu) {
    setStatus(elements.modelStatus, "이 브라우저는 WebGPU를 지원하지 않습니다.", "error");
    return;
  }

  const modelUrl = elements.modelUrl.value.trim();
  if (!modelUrl) {
    setStatus(elements.modelStatus, "모델 URL을 입력하세요.", "error");
    return;
  }

  elements.loadModel.disabled = true;
  state.session = null;
  updateRunButton();
  setStatus(
    elements.modelStatus,
    "모델을 다운로드하고 WebGPU 세션을 초기화하는 중입니다…",
    "working",
  );

  const startedAt = performance.now();
  try {
    state.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: [
        {
          name: "webgpu",
          preferredLayout: "NCHW",
        },
      ],
      graphOptimizationLevel: "all",
    });
    const elapsed = performance.now() - startedAt;
    setStatus(
      elements.modelStatus,
      `모델 로드 완료 · ${elapsed.toFixed(0)} ms · input: ${state.session.inputNames[0]}`,
      "ready",
    );
  } catch (error) {
    console.error(error);
    setStatus(
      elements.modelStatus,
      `모델 로드 실패: ${error.message}`,
      "error",
    );
  } finally {
    elements.loadModel.disabled = false;
    updateRunButton();
  }
}

function updatePreview(view, file) {
  const previousUrl = state.previewUrls.get(view);
  if (previousUrl) {
    URL.revokeObjectURL(previousUrl);
  }
  const nextUrl = URL.createObjectURL(file);
  state.previewUrls.set(view, nextUrl);

  const preview = document.querySelector(`[data-preview="${view}"]`);
  preview.src = nextUrl;
  preview.closest(".camera-card").dataset.ready = "true";
}

async function fileToNormalizedChw(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(IMAGE_SIZE, IMAGE_SIZE);
  const context = canvas.getContext("2d", {
    alpha: false,
    willReadFrequently: true,
  });
  if (!context) {
    bitmap.close();
    throw new Error("이미지 전처리용 Canvas 컨텍스트를 만들지 못했습니다.");
  }

  context.drawImage(bitmap, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  bitmap.close();
  const rgba = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const pixelCount = IMAGE_SIZE * IMAGE_SIZE;
  const chw = new Float32Array(3 * pixelCount);

  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const rgbaOffset = pixel * 4;
    for (let channel = 0; channel < 3; channel += 1) {
      const normalized =
        (rgba[rgbaOffset + channel] / 255 - IMAGENET_MEAN[channel]) /
        IMAGENET_STD[channel];
      chw[channel * pixelCount + pixel] = normalized;
    }
  }
  return chw;
}

async function createInputTensor() {
  const pixelCountPerView = 3 * IMAGE_SIZE * IMAGE_SIZE;
  const data = new Float32Array(VIEW_ORDER.length * pixelCountPerView);
  const views = await Promise.all(
    VIEW_ORDER.map((view) => fileToNormalizedChw(state.files.get(view))),
  );
  views.forEach((viewData, index) => {
    data.set(viewData, index * pixelCountPerView);
  });
  return new ort.Tensor("float32", data, [
    1,
    VIEW_ORDER.length,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
  ]);
}

function renderOutput(values) {
  if (values.length !== OUTPUT_LABELS.length) {
    throw new Error(
      `출력 크기가 ${values.length}입니다. exporter가 만든 6D 모델이 필요합니다.`,
    );
  }

  const formatted = {
    x: values[0] * 1000,
    y: values[1] * 1000,
    z: values[2] * 1000,
    roll: (values[3] * 180) / Math.PI,
    pitch: (values[4] * 180) / Math.PI,
    yaw: (values[5] * 180) / Math.PI,
  };

  for (const label of OUTPUT_LABELS) {
    document.querySelector(`#result-${label}`).textContent =
      formatted[label].toFixed(3);
  }
}

async function runInference() {
  if (!state.session) {
    return;
  }

  elements.runInference.disabled = true;
  setStatus(elements.inferenceStatus, "이미지를 전처리하고 추론 중입니다…", "working");

  try {
    const tensor = await createInputTensor();
    const inputName = state.session.inputNames[0];
    const outputName = state.session.outputNames[0];

    const startedAt = performance.now();
    const outputMap = await state.session.run({ [inputName]: tensor });
    const elapsed = performance.now() - startedAt;
    const output = outputMap[outputName];
    renderOutput(Array.from(output.data));

    elements.latency.textContent = `${elapsed.toFixed(1)} ms`;
    setStatus(elements.inferenceStatus, "WebGPU 추론이 완료됐습니다.", "ready");
  } catch (error) {
    console.error(error);
    setStatus(
      elements.inferenceStatus,
      `추론 실패: ${error.message}`,
      "error",
    );
  } finally {
    updateRunButton();
  }
}

elements.portType.addEventListener("change", () => {
  elements.modelUrl.value = `/models/vision_offset_${elements.portType.value}.onnx`;
  state.session = null;
  setStatus(elements.modelStatus, "포트 유형이 변경되었습니다. 모델을 다시 로드하세요.");
  updateRunButton();
});

document.querySelectorAll(".camera-input").forEach((input) => {
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    const view = input.dataset.view;
    state.files.set(view, file);
    updatePreview(view, file);
    updateRunButton();
  });
});

elements.loadModel.addEventListener("click", loadModel);
elements.runInference.addEventListener("click", runInference);

detectWebGpu();
updateRunButton();
