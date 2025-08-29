from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# You already have these:
# from neutrino.clf.config.io_config import ClfIoConfig
# from neutrino.clf.config.feature_config import ClfFeatureConfig


def _read_columns(path: Path) -> list[str]:
    return [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def load_np_split(
    split_dir: Path,
    split_prefix: str,
    a_suffix: str,
    b_suffix: str,
    columns_filename: str,
    feature_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    cols = _read_columns(split_dir / columns_filename)

    A = np.load(split_dir / f"{split_prefix}{a_suffix}")
    B = np.load(split_dir / f"{split_prefix}{b_suffix}")

    idx = [cols.index(c) for c in feature_order]
    XA = A[:, idx]
    XB = B[:, idx]

    X = np.vstack([XA, XB]).astype(np.float32, copy=False)
    y = np.concatenate(
        [np.zeros(len(XA), dtype=np.int64), np.ones(len(XB), dtype=np.int64)]
    )
    return X, y, cols


@dataclass
class StandardizeStats:
    mean: torch.Tensor
    std: torch.Tensor


def make_torch_loaders(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    batch_size: int = 512,
    seed: int = 42,
    standardize: bool = True,
):
    # tensors
    X_t = torch.from_numpy(X)  # [N, D]
    y_t = torch.from_numpy(y)  # [N]

    # split
    N = X_t.shape[0]
    n_val = int(round(N * val_frac))
    n_train = N - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(
        TensorDataset(X_t, y_t), [n_train, n_val], generator=g
    )

    stats = None
    if standardize:
        Xtr, _ = train_ds[:]
        mean = Xtr.mean(dim=0)
        std = Xtr.std(dim=0).clamp_min(1e-8)
        stats = StandardizeStats(mean=mean, std=std)

        def _norm(ds: TensorDataset) -> TensorDataset:
            Xd, yd = ds[:]
            Xd = (Xd - mean) / std
            return TensorDataset(Xd, yd)

        train_ds = _norm(train_ds)
        val_ds = _norm(val_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, stats
