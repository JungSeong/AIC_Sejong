from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JsonlActionDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError(f"No samples found: {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        state = torch.tensor(row["state"], dtype=torch.float32)
        action = torch.tensor(row["action"], dtype=torch.float32)
        return state, action

    @property
    def input_dim(self) -> int:
        return len(self.rows[0]["state"])

    @property
    def action_dim(self) -> int:
        return len(self.rows[0]["action"])
