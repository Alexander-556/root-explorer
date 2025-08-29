# src/neutrino/clf/config/model_config.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ClfModelConfig:
    # Config Keys (not shared across instances)
    type: str
    params: dict[str, Any]
    config_path: Path

    # Class Defaults (shared across instances)
    DEFAULT_CONFIG_PATH: ClassVar[Path] = (
        Path("configs") / "model" / "model_config.json"
    )

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClfModelConfig":

        # Resolve path (allow override)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # Load raw JSON
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # --- Parse each field explicitly ---

        # 1) Model type
        model_type: str = str(raw["type"])

        # 2) Params (keep generic dict, but coerce common fields)
        raw_params: dict[str, Any] = dict(raw.get("params", {}))

        # Hidden sizes: list[int]
        if "hidden_sizes" in raw_params:
            hidden_any = raw_params["hidden_sizes"]
            # Keep it simple (you control inputs): assume it's a list
            hidden_sizes: list[int] = [int(x) for x in hidden_any]
            raw_params["hidden_sizes"] = hidden_sizes

        # Dropout: float
        if "dropout" in raw_params:
            raw_params["dropout"] = float(raw_params["dropout"])

        # --- Construct dataclass ---
        return cls(
            type=model_type,
            params=raw_params,
            config_path=path,
        )
