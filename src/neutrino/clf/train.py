from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainParams:
    lr: float = 1e-3
    max_epochs: int = 20
    device: str = "cpu"


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_simple(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: TrainParams,
    run_name: str = "mlp_baseline",
    runs_dir: Path = Path("runs"),
):
    model.to(params.device)
    opt = torch.optim.Adam(model.parameters(), lr=params.lr)
    loss_fn = nn.CrossEntropyLoss()

    run_dir = runs_dir / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, params.max_epochs + 1):
        # train
        model.train()
        tr_loss, tr_acc, n_tr = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(params.device)
            yb = yb.to(params.device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            bs = yb.size(0)
            tr_loss += loss.item() * bs
            tr_acc += accuracy(logits, yb) * bs
            n_tr += bs
        tr_loss /= max(n_tr, 1)
        tr_acc /= max(n_tr, 1)

        # val
        model.eval()
        va_loss, va_acc, n_va = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(params.device)
                yb = yb.to(params.device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                bs = yb.size(0)
                va_loss += loss.item() * bs
                va_acc += accuracy(logits, yb) * bs
                n_va += bs
        va_loss /= max(n_va, 1)
        va_acc /= max(n_va, 1)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
        history.append(row)
        print(
            f"[{epoch:03d}] loss {tr_loss:.4f}/{va_loss:.4f}  acc {tr_acc:.3f}/{va_acc:.3f}"
        )

    # save tiny artifacts
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")
    return run_dir, history
