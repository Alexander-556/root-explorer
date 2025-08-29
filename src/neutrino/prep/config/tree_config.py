# src/neutrino/prep/config/tree_config.py
import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class TreeConfig:
    """
    Dataclass wrapper for tree-level configuration.

    This loader handles a JSON file that specifies the name of the tree to be read
    from a ROOT file (or other tree-structured data source). The loader provides a
    clean API for accessing config values inside Python code.
    """

    # -------------------------------------------------------------------------
    # Instance attributes (unique per config object)
    # -------------------------------------------------------------------------
    tree_name: str  # Name of the tree to load
    config_path: Path  # Path to the JSON file actually used

    # -------------------------------------------------------------------------
    # Class attributes (shared across all instances)
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "data" / "tree_config.json"

    # -------------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------------
    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "TreeConfig":
        """
        Load a TreeConfig instance from JSON.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to a config JSON file. If None, uses DEFAULT_CONFIG_PATH.

        Returns
        -------
        TreeConfig
            Dataclass instance populated with config values.
        """
        # 1. Resolve path (either user-specified or default)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # 2. Load raw JSON dict
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3. Extract fields from dict
        tree_name = raw["tree_name"]

        # 4. Construct dataclass and return
        return cls(
            tree_name=tree_name,
            config_path=path,
        )
