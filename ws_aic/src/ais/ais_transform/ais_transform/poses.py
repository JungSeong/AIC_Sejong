"""Pose value objects used by ``ais_transform``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ._math import VectorLike, _quat_xyzw, _rotation_matrix_xyzw, _vec3


@dataclass(frozen=True)
class Pose3D:
    """``base_link`` 또는 parent frame 기준 pose.

    position: xyz, meter 단위
    orientation: quaternion xyzw, ROS/scipy 순서
    """

    position: np.ndarray
    orientation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vec3(self.position))
        object.__setattr__(self, "orientation", _quat_xyzw(self.orientation))

    @classmethod
    def from_xyz_quat(cls, xyz: VectorLike, quat_xyzw: VectorLike) -> "Pose3D":
        return cls(position=_vec3(xyz), orientation=_quat_xyzw(quat_xyzw))

    @classmethod
    def from_transform_dict(cls, transform: Mapping[str, Mapping[str, float]]) -> "Pose3D":
        """JSON/기록 데이터의 translation/rotation dict를 Pose3D로 변환."""

        t = transform["translation"]
        q = transform["rotation"]
        return cls.from_xyz_quat(
            [t["x"], t["y"], t["z"]],
            [q["x"], q["y"], q["z"], q["w"]],
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _rotation_matrix_xyzw(self.orientation)

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 동차 변환행렬."""

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.position
        return matrix


@dataclass(frozen=True)
class PlugToPort:
    """plug가 port 원점에 맞기 위해 움직여야 하는 ``base_link`` 기준 벡터."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", _vec3(self.vector))


__all__ = ["PlugToPort", "Pose3D"]
