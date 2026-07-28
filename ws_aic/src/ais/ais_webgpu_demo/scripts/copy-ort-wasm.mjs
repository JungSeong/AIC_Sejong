import { cp, mkdir, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const demoRoot = dirname(scriptDirectory);
const sourceDirectory = join(
  demoRoot,
  "node_modules",
  "onnxruntime-web",
  "dist",
);
const targetDirectory = join(demoRoot, "public", "ort");

await mkdir(targetDirectory, { recursive: true });
const files = await readdir(sourceDirectory);
const wasmFiles = files.filter((file) => file.endsWith(".wasm"));

if (wasmFiles.length === 0) {
  throw new Error(`ONNX Runtime WASM 파일을 찾지 못했습니다: ${sourceDirectory}`);
}

await Promise.all(
  wasmFiles.map((file) =>
    cp(join(sourceDirectory, file), join(targetDirectory, file)),
  ),
);

console.log(`Copied ${wasmFiles.length} ONNX Runtime WASM file(s).`);
