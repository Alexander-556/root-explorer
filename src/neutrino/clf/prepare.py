from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch

from neutrino.clf.config.io_config import ClfIoConfig


@dataclass
class TensorPair:
    A: torch.Tensor  # shape: [NA, D_all], dtype: float32
    B: torch.Tensor  # shape: [NB, D_all], dtype: float32
    columns: List[str]  # length D_all

    @classmethod
    def load_tensor(cls) -> "TensorPair":
        """Load .npy A/B and columns.txt, convert to float32 tensors, return TensorPair."""

        cfg: ClfIoConfig = ClfIoConfig.load_config()
        split_dir: Path = cfg.split_dir
        a_path: Path = split_dir / f"{cfg.split_prefix}{cfg.a_suffix}"
        b_path: Path = split_dir / f"{cfg.split_prefix}{cfg.b_suffix}"
        cols_path: Path = split_dir / cfg.columns_filename

        # numpy → tensors
        A_np = np.load(a_path)
        B_np = np.load(b_path)
        columns = [
            ln.strip()
            for ln in cols_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

        A_t = torch.from_numpy(A_np).float()
        B_t = torch.from_numpy(B_np).float()

        return cls(
            A=A_t,
            B=B_t,
            columns=columns,
        )

    @property
    def shapes(self) -> tuple[torch.Size, torch.Size]:
        """Return (A.shape, B.shape)"""
        return self.A.shape, self.B.shape
