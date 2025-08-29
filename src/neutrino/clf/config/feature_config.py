import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class ClfFeatureConfig:
    # Config Keys
    # not shared across instances
    feature_order: list[str]

    config_path: Path

    # Class Defaults
    # shared across instances
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "model" / "feature_config.json"

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClfFeatureConfig":

        # Resolve path (allow override)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # Load raw JSON
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3) List[str] for feature_order
        raw_targets: list[Any] = raw["feature_order"]
        feature_order: list[str] = []
        for x in raw_targets:
            feature_order.append(str(x))

        # --- Construct dataclass ---
        return cls(
            feature_order=feature_order,
            config_path=path,
        )
