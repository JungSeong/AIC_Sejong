from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CAMERAS = ("left", "center", "right")
REPO_ID = "aic-sejong-team/aic-entrance-dataset"
REVISION = "v1.2"
DATASET_DIRNAME = "vision_offset_dataset"


def _ws_aic_root() -> Path:
    return Path(__file__).resolve().parents[4]


DEFAULT_DATASET_ROOT = (
    _ws_aic_root() / "data" / "aic-entrance-dataset" / REVISION / DATASET_DIRNAME
)


def _normalize_dataset_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if (root / "samples.jsonl").exists():
        return root
    nested = root / DATASET_DIRNAME
    if (nested / "samples.jsonl").exists():
        return nested
    return root


def _episode_name_from_sample_id(sample_id: str) -> str:
    match = re.match(r"(.+)_collect_\d+$", sample_id)
    return match.group(1) if match else sample_id


def _version_roots(
    dataset_root: Path,
    versions: Sequence[str] | None,
) -> list[Path]:
    if versions is not None:
        return [dataset_root / version for version in versions]
    if (dataset_root / "metadata").exists():
        return [dataset_root]
    roots = [
        path
        for path in sorted(dataset_root.iterdir())
        if path.is_dir() and (path / "metadata").exists()
    ]
    return roots or [dataset_root]


def _load_camera_metadata(version_root: Path, splits: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        metadata_dir = version_root / "metadata" / split
        if not metadata_dir.exists():
            continue
        for path in sorted(metadata_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as file:
                record = json.load(file)
            record["_metadata_path"] = str(path)
            record["_dataset_root"] = str(version_root)
            record["_version"] = version_root.name
            record["_split"] = split
            records.append(record)
    return records


def _group_camera_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        camera = str(record.get("camera", ""))
        if camera not in CAMERAS:
            continue
        sample = grouped.get(sample_id)
        if sample is None:
            sample = dict(record)
            sample["episode_name"] = _episode_name_from_sample_id(sample_id)
            sample["version"] = record.get("_version", "")
            sample["split"] = record.get("_split", "")
            task = record.get("task", {})
            sample["port_type"] = task.get("port_type", "")
            sample["port_name"] = task.get("port_name", "")
            sample["target_module_name"] = task.get("target_module_name", "")
            sample["images"] = {}
            sample["metadata_by_camera"] = {}
            grouped[sample_id] = sample
        sample["images"][camera] = str(record["image"])
        sample["metadata_by_camera"][camera] = str(record["_metadata_path"])
    return list(grouped.values())


def load_rpy_randomization_samples(
    dataset_root: str | Path,
    *,
    versions: Sequence[str] | None = None,
    splits: Sequence[str] = ("train", "val"),
) -> list[dict[str, Any]]:
    """Load ais_rpy_randomization samples grouped across camera views."""
    root = Path(dataset_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    for version_root in _version_roots(root, versions):
        records.extend(_load_camera_metadata(version_root, splits))
    return _group_camera_records(records)


def download_vision_offset_dataset(
    local_dir: str | Path | None = None,
    *,
    revision: str = REVISION,
    max_workers: int = 16,
    fallback_to_direct: bool = True,
) -> Path:
    """Download the Hugging Face vision offset dataset into ``ws_aic/data``."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HUGGINGFACE_HUB_READ_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if local_dir is None:
        local_dir = _ws_aic_root() / "data" / "aic-entrance-dataset" / revision
    local_dir = Path(local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = local_dir / DATASET_DIRNAME
    if fallback_to_direct and (dataset_root / "samples.jsonl").exists():
        _download_missing_files_direct(dataset_root, revision=revision, max_workers=max_workers)
        return dataset_root

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=[f"{DATASET_DIRNAME}/*"],
            max_workers=max_workers,
            local_dir_use_symlinks=False,
        )
    except Exception:
        if not fallback_to_direct:
            raise
        _download_missing_files_direct(dataset_root, revision=revision, max_workers=max_workers)
    else:
        if fallback_to_direct:
            _download_missing_files_direct(dataset_root, revision=revision, max_workers=max_workers)
    return dataset_root


def _repo_file_url(path: str, revision: str) -> str:
    encoded_path = quote(f"{DATASET_DIRNAME}/{path}", safe="/")
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{encoded_path}"


def _download_url(url: str, destination: Path, *, retries: int = 5) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 120)) as response:
                if response.status_code == 429:
                    time.sleep(30 * attempt)
                    continue
                response.raise_for_status()
                with partial.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            partial.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _download_missing_files_direct(
    dataset_root: Path,
    *,
    revision: str,
    max_workers: int,
) -> None:
    samples_path = dataset_root / "samples.jsonl"
    if not samples_path.exists():
        _download_url(_repo_file_url("samples.jsonl", revision), samples_path)

    samples = load_samples(dataset_root)
    expected_paths = sorted(
        {image_path for sample in samples for image_path in sample["images"].values()}
    )
    missing_paths = [
        image_path for image_path in expected_paths if not (dataset_root / image_path).exists()
    ]
    if not missing_paths:
        return

    workers = max(1, min(max_workers, 8))
    print(f"Downloading {len(missing_paths)} missing images with {workers} workers.")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_url,
                _repo_file_url(image_path, revision),
                dataset_root / image_path,
            ): image_path
            for image_path in missing_paths
        }
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed while downloading {image_path}") from exc
            completed += 1
            if completed == len(missing_paths) or completed % 100 == 0:
                print(f"Downloaded {completed}/{len(missing_paths)} missing images.")


def load_samples(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[dict[str, Any]]:
    """Load samples.jsonl records as dictionaries."""
    dataset_root = _normalize_dataset_root(dataset_root)
    samples_path = dataset_root / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Missing {samples_path}. Run download_vision_offset_dataset() first."
        )

    samples: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def filter_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    task_types: Iterable[str] | None = None,
    port_types: Iterable[str] | None = None,
    episode_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter metadata records without touching image files."""
    task_types = set(task_types) if task_types is not None else None
    port_types = set(port_types) if port_types is not None else None
    episode_names = set(episode_names) if episode_names is not None else None

    filtered: list[dict[str, Any]] = []
    for sample in samples:
        if task_types is not None and sample.get("task_type") not in task_types:
            continue
        if port_types is not None and sample.get("port_type") not in port_types:
            continue
        if episode_names is not None and sample.get("episode_name") not in episode_names:
            continue
        filtered.append(dict(sample))
    return filtered


def port_id_from_text(value: str) -> int | None:
    matches = re.findall(r"port_(\d+)", value)
    if matches:
        return int(matches[-1])
    match = re.search(r"_(\d+)$", value)
    if match is not None:
        return int(match.group(1))
    return None


def port_id_from_sample(sample: Mapping[str, Any]) -> int:
    """Return the local port index used by the multi-head distance model."""
    explicit_port_id = sample.get("_port_id")
    if explicit_port_id is not None:
        return int(explicit_port_id)
    scenario = sample.get("scenario", {})
    if "sfp_port_idx" in scenario:
        return int(scenario["sfp_port_idx"])
    for key in ("port_name", "target_module_name"):
        port_id = port_id_from_text(str(sample.get(key, "")))
        if port_id is not None:
            return port_id
    return 0


def _quat_xyzw_to_matrix(qx: float, qy: float, qz: float, qw: float) -> torch.Tensor:
    norm = float((qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5)
    if norm < 1e-12:
        return torch.eye(3, dtype=torch.float32)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return torch.tensor(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=torch.float32,
    )


def _rpy_from_quaternion_record(record: Mapping[str, Any]) -> torch.Tensor:
    rotation = _quat_xyzw_to_matrix(
        float(record["qx"]),
        float(record["qy"]),
        float(record["qz"]),
        float(record["qw"]),
    )
    sy = float(torch.linalg.vector_norm(rotation[:2, 0]))
    if sy >= 1e-9:
        roll = torch.atan2(rotation[2, 1], rotation[2, 2])
        pitch = torch.atan2(-rotation[2, 0], torch.tensor(sy, dtype=rotation.dtype))
        yaw = torch.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = torch.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = torch.atan2(-rotation[2, 0], torch.tensor(sy, dtype=rotation.dtype))
        yaw = torch.tensor(0.0, dtype=rotation.dtype)
    return torch.stack([roll, pitch, yaw]).float()


def _selected_port_info(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ports = sample.get("label", {}).get("ports", {})
    port_key = sample.get("_port_key")
    if port_key in ports:
        return ports[port_key]

    target_key = f"{sample.get('target_module_name', '')}/{sample.get('port_name', '')}"
    if target_key in ports:
        return ports[target_key]

    for port_info in ports.values():
        if port_info.get("is_target", False):
            return port_info
    return None


def rpy_from_sample(sample: Mapping[str, Any], *, unit: str = "rad") -> torch.Tensor:
    """Return selected-port RPY from metadata."""
    if unit not in {"rad", "deg"}:
        raise ValueError("unit must be 'rad' or 'deg'.")

    actual = sample.get("actual", {})
    record = actual.get("plug_reference_in_port")
    if record is None:
        port_info = _selected_port_info(sample)
        record = None if port_info is None else port_info.get("port_in_plug_tip")
    if record is None or not all(key in record for key in ("qx", "qy", "qz", "qw")):
        raise KeyError("sample has no selected-port RPY quaternion")

    values = _rpy_from_quaternion_record(record)
    if unit == "deg":
        values = values * (180.0 / torch.pi)
    return values


def sample_has_rpy(sample: Mapping[str, Any]) -> bool:
    try:
        rpy_from_sample(sample)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def distance_sample_has_target(
    sample: Mapping[str, Any],
    *,
    target_source: str = "label",
    target_mode: str = "plug_tip_to_port",
    target_keys: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
) -> bool:
    try:
        distance_target_from_sample(
            sample,
            target_source=target_source,
            target_mode=target_mode,
            target_keys=target_keys,
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True


def distance_target_from_sample(
    sample: Mapping[str, Any],
    *,
    target_source: str = "label",
    target_mode: str = "plug_tip_to_port",
    target_keys: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
) -> torch.Tensor:
    if target_source == "label":
        label = sample.get("_port_label") or sample["label"][target_mode]
    elif target_source == "actual":
        label = sample["actual"][target_mode]
    else:
        raise ValueError("target_source must be 'label' or 'actual'.")
    return torch.tensor([float(label[key]) for key in target_keys], dtype=torch.float32)


def expand_samples_by_available_ports(
    samples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand v1.2 samples so one image can supervise each available port head."""
    expanded: list[dict[str, Any]] = []
    for sample in samples:
        ports = sample.get("label", {}).get("ports", {})
        added = False
        for port_key, port_info in ports.items():
            if not port_info.get("available", False):
                continue
            label = port_info.get("plug_tip_in_port")
            if label is None:
                continue
            row = dict(sample)
            row["_port_id"] = port_id_from_text(str(port_key)) or 0
            row["_port_key"] = port_key
            row["_port_label"] = dict(label)
            expanded.append(row)
            added = True
        if not added:
            expanded.append(dict(sample))
    return expanded


class VisionOffsetDataset(Dataset):
    """Image dataset for predicting plug-tip-to-port offsets.

    The labels come from ``label.plug_tip_to_port`` in ``samples.jsonl``.
    For SFP data generated by ``collect_sfp_distance_dataset.py``, that vector
    is expressed in ``label.frame``; the default frame is
    ``task_board/<module>/<port>_link_entrance``.
    Use ``target_keys=("x_mm", "y_mm", "z_mm")`` for millimeter regression
    or ``("x_m", "y_m", "z_m")`` for meter regression.
    """

    def __init__(
        self,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        *,
        samples: Sequence[Mapping[str, Any]] | None = None,
        cameras: Sequence[str] = ("center",),
        target_keys: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
        target_source: str = "label",
        target_mode: str = "plug_tip_to_port",
        transform: Any | None = None,
        target_mean: Sequence[float] | torch.Tensor | None = None,
        target_std: Sequence[float] | torch.Tensor | None = None,
        always_return_views: bool = False,
        expand_all_ports: bool = True,
        use_rpy: bool = False,
        rpy_unit: str = "rad",
        rpy_mean: Sequence[float] | torch.Tensor | None = None,
        rpy_std: Sequence[float] | torch.Tensor | None = None,
        validate_files: bool = False,
    ) -> None:
        self.dataset_root = _normalize_dataset_root(dataset_root)
        raw_samples = (
            [dict(sample) for sample in samples]
            if samples is not None
            else load_samples(self.dataset_root)
        )
        self.samples = (
            expand_samples_by_available_ports(raw_samples)
            if expand_all_ports
            else raw_samples
        )
        self.cameras = tuple(cameras)
        self.target_keys = tuple(target_keys)
        self.target_source = target_source
        self.target_mode = target_mode
        self.transform = transform
        self.always_return_views = always_return_views
        self.use_rpy = bool(use_rpy)
        self.rpy_unit = rpy_unit

        if not self.cameras:
            raise ValueError("At least one camera must be selected.")
        invalid_cameras = set(self.cameras) - set(CAMERAS)
        if invalid_cameras:
            raise ValueError(f"Unknown camera names: {sorted(invalid_cameras)}")
        if self.rpy_unit not in {"rad", "deg"}:
            raise ValueError("rpy_unit must be 'rad' or 'deg'.")

        self.target_mean = self._as_tensor(target_mean)
        self.target_std = self._as_tensor(target_std)
        if (self.target_mean is None) != (self.target_std is None):
            raise ValueError("target_mean and target_std must be provided together.")
        self.rpy_mean = self._as_tensor(rpy_mean)
        self.rpy_std = self._as_tensor(rpy_std)
        if (self.rpy_mean is None) != (self.rpy_std is None):
            raise ValueError("rpy_mean and rpy_std must be provided together.")
        if self.use_rpy:
            missing_rpy = [sample["sample_id"] for sample in self.samples if not sample_has_rpy(sample)]
            if missing_rpy:
                preview = "\n".join(missing_rpy[:5])
                raise ValueError(
                    f"{len(missing_rpy)} samples are missing selected-port RPY. "
                    f"First missing samples:\n{preview}"
                )

        if validate_files:
            missing = self.missing_image_paths()
            if missing:
                preview = "\n".join(str(path) for path in missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} image files are missing. First missing files:\n{preview}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        images = [self._load_image(sample, camera) for camera in self.cameras]
        image_tensor = (
            torch.stack(images, dim=0)
            if len(images) > 1 or self.always_return_views
            else images[0]
        )

        raw_target = self._target_tensor(sample)
        target = raw_target
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        item = {
            "image": image_tensor,
            "port_id": torch.tensor(port_id_from_sample(sample), dtype=torch.long),
            "target": target.float(),
            "raw_target": raw_target.float(),
            "sample_id": sample["sample_id"],
            "episode_name": sample["episode_name"],
            "task_type": sample.get("task_type", sample.get("task", {}).get("id", "")),
            "port_type": sample.get("port_type", sample.get("task", {}).get("port_type", "")),
            "port_name": sample.get("port_name", ""),
            "port_key": sample.get("_port_key", ""),
            "cameras": self.cameras,
        }
        if self.use_rpy:
            rpy = rpy_from_sample(sample, unit=self.rpy_unit)
            if self.rpy_mean is not None and self.rpy_std is not None:
                rpy = (rpy - self.rpy_mean) / self.rpy_std
            item["rpy"] = rpy.float()
        return item

    def missing_image_paths(self) -> list[Path]:
        missing: list[Path] = []
        for sample in self.samples:
            for camera in self.cameras:
                path = self._sample_root(sample) / sample["images"][camera]
                if not path.exists():
                    missing.append(path)
        return missing

    def _load_image(self, sample: Mapping[str, Any], camera: str) -> torch.Tensor:
        path = self._sample_root(sample) / sample["images"][camera]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                return self.transform(image)
            return TF.to_tensor(image)

    def _target_tensor(self, sample: Mapping[str, Any]) -> torch.Tensor:
        return distance_target_from_sample(
            sample,
            target_source=self.target_source,
            target_mode=self.target_mode,
            target_keys=self.target_keys,
        )

    def _sample_root(self, sample: Mapping[str, Any]) -> Path:
        return Path(sample.get("_dataset_root", self.dataset_root)).expanduser().resolve()

    @staticmethod
    def _as_tensor(values: Sequence[float] | torch.Tensor | None) -> torch.Tensor | None:
        if values is None:
            return None
        return torch.as_tensor(values, dtype=torch.float32)
