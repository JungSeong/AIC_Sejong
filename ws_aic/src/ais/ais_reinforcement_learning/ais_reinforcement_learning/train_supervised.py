from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from ais_reinforcement_learning.config import SfpRlConfig
from ais_reinforcement_learning.dataset import JsonlActionDataset
from ais_reinforcement_learning.models import ActionMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFP action predictor from JSONL rollouts.")
    parser.add_argument("--data", type=Path, default=SfpRlConfig.SUPERVISED_DATA)
    parser.add_argument("--output", type=Path, default=SfpRlConfig.SUPERVISED_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = JsonlActionDataset(args.data)
    val_size = max(1, int(len(dataset) * float(args.val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ActionMLP(
        input_dim=dataset.input_dim,
        action_dim=dataset.action_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for state, action in train_loader:
            state = state.to(args.device)
            action = action.to(args.device)
            pred = model(state)
            loss = loss_fn(pred, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * state.shape[0]
        train_loss /= max(1, len(train_ds))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for state, action in val_loader:
                state = state.to(args.device)
                action = action.to(args.device)
                pred = model(state)
                loss = loss_fn(pred, action)
                val_loss += float(loss.item()) * state.shape[0]
        val_loss /= max(1, len(val_ds))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": dataset.input_dim,
                    "action_dim": dataset.action_dim,
                    "hidden_dim": args.hidden_dim,
                    "depth": args.depth,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "data": str(args.data),
                },
                args.output,
            )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best={best_val:.6f}"
        )


if __name__ == "__main__":
    main()
