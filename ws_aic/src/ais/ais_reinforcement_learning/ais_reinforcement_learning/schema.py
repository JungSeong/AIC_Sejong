from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ActionSample:
    sample_id: str
    run_id: str
    task_target: str
    task_port: str
    task_cable: str
    task_plug: str
    phase: str
    step: int
    state: list[float]
    action: list[float]
    extras: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
