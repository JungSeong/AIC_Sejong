from __future__ import annotations

import numpy as np

from ais_ground_truth_guided_residual_visual_servoing.core.config import (
    SfpGrvsConfig,
    sample_epsilon_m,
)


def test_default_artifact_paths_include_package_version_and_targets():
    package = SfpGrvsConfig.PACKAGE_NAME
    version = SfpGrvsConfig.VERSION

    assert SfpGrvsConfig.YOLO_DATASET_DIR.parts[-3:] == (package, version, "yolo")
    assert SfpGrvsConfig.DISTANCE_DATASET_DIR.parts[-3:] == (
        package,
        version,
        "distance_prediction",
    )
    assert SfpGrvsConfig.ROTATION_DATASET_DIR.parts[-3:] == (
        package,
        version,
        "rotation_prediction",
    )
    assert SfpGrvsConfig.YOLO_MODEL_DIR.parts[-3:] == (package, version, "yolo")
    assert SfpGrvsConfig.DISTANCE_MODEL_DIR.parts[-3:] == (
        package,
        version,
        "distance_prediction",
    )
    assert SfpGrvsConfig.ROTATION_MODEL_DIR.parts[-3:] == (
        package,
        version,
        "rotation_prediction",
    )
    assert SfpGrvsConfig.BATCH_DIR.parts[-3:] == (package, version, "batches")
    assert SfpGrvsConfig.REPLAY_BUFFER_DIR.parts[-3:] == (
        package,
        version,
        "replay_buffer",
    )


def test_sample_epsilon_uses_sfp_ci99_ranges():
    rng = np.random.default_rng(1)
    epsilon = sample_epsilon_m(rng)

    assert SfpGrvsConfig.GT_FALLBACK_EPSILON_X_RANGE_M[0] <= epsilon[0]
    assert epsilon[0] <= SfpGrvsConfig.GT_FALLBACK_EPSILON_X_RANGE_M[1]
    assert SfpGrvsConfig.GT_FALLBACK_EPSILON_Y_RANGE_M[0] <= epsilon[1]
    assert epsilon[1] <= SfpGrvsConfig.GT_FALLBACK_EPSILON_Y_RANGE_M[1]
    assert SfpGrvsConfig.GT_FALLBACK_EPSILON_Z_RANGE_M[0] <= epsilon[2]
    assert epsilon[2] <= SfpGrvsConfig.GT_FALLBACK_EPSILON_Z_RANGE_M[1]
