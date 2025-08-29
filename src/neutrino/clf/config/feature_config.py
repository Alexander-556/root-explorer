# src/neutrino/clf/config/feature_config.py
import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class ClfFeatureConfig:
    """
    Dataclass wrapper for feature configuration.

    This loader handles JSON that specifies the order of features (columns)
    to be used in the classifier. The config ensures consistent feature
    selection across training and evaluation.
    """

    # -------------------------------------------------------------------------
    # Instance attributes (unique per config object)
    # -------------------------------------------------------------------------
    feature_order: list[str]  # List of feature/column names in the desired order
    config_path: Path  # Path to the JSON file actually used

    # -------------------------------------------------------------------------
    # Class attributes (shared across all instances)
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG_PATH: ClassVar[Path] = (
        Path("configs") / "model" / "feature_config.json"
    )

    # -------------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------------
    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClfFeatureConfig":
        """
        Load a ClfFeatureConfig instance from JSON.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to a config JSON file. If None, uses DEFAULT_CONFIG_PATH.

        Returns
        -------
        ClfFeatureConfig
            Dataclass instance populated with config values.
        """

        # 1. Resolve path (either user-specified or default)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # 2. Load raw JSON dict
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3. Parse fields explicitly

        # List[str] for feature order
        raw_targets: list[Any] = raw["feature_order"]
        feature_order: list[str] = [str(x) for x in raw_targets]

        # 4. Construct dataclass and return
        return cls(
            feature_order=feature_order,
            config_path=path,
        )
