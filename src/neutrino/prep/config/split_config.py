# src/neutrino/prep/config/split_config.py
import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """
    Dataclass wrapper for split configuration.

    This loader handles JSON that describes how the dataset should be split
    into different subsets (e.g., A vs B). The config specifies which branches
    to use for splitting, how to map categories to numeric labels, and what
    target branches to extract.
    """

    # -------------------------------------------------------------------------
    # Instance attributes (unique per config object)
    # -------------------------------------------------------------------------
    flag_branch: str  # Name of the branch that defines the split
    flag_values: dict[str, int]  # Mapping of flag labels → numeric codes
    cat_branch: str  # Branch that defines interaction categories
    target_branches: list[str]  # Features to extract from the tree
    type_group: dict[str, list[str]]  # Grouping of categories into A/B
    type_map: dict[str, int]  # Mapping of interaction types → numeric codes
    config_path: Path  # Path to the JSON file actually used

    # -------------------------------------------------------------------------
    # Class attributes (shared across all instances)
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "data" / "split_config.json"

    # -------------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------------
    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "SplitConfig":
        """
        Load a SplitConfig instance from JSON.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to a config JSON file. If None, uses DEFAULT_CONFIG_PATH.

        Returns
        -------
        SplitConfig
            Dataclass instance populated with config values.
        """

        # 1. Resolve path (either user-specified or default)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # 2. Load raw JSON dict
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3. Parse fields explicitly

        # Simple strings
        flag_branch: str = str(raw["flag_branch"])
        cat_branch: str = str(raw["cat_branch"])

        # Dict[str, int] for flag values
        raw_flag_values: dict[str, Any] = raw["flag_values"]
        flag_values: dict[str, int] = {
            str(k): int(v) for k, v in raw_flag_values.items()
        }

        # List[str] for target branches
        raw_targets: list[Any] = raw["target_branches"]
        target_branches: list[str] = [str(x) for x in raw_targets]

        # Dict[str, list[str]] for type groups
        raw_type_group: dict[str, Any] = raw["type_group"]
        type_group: dict[str, list[str]] = {
            str(group): [str(lbl) for lbl in labels]
            for group, labels in raw_type_group.items()
        }

        # Dict[str, int] for type map
        raw_type_map: dict[str, Any] = raw["type_map"]
        type_map: dict[str, int] = {
            str(lbl): int(code) for lbl, code in raw_type_map.items()
        }

        # 4. Construct dataclass and return
        return cls(
            flag_branch=flag_branch,
            flag_values=flag_values,
            cat_branch=cat_branch,
            target_branches=target_branches,
            type_group=type_group,
            type_map=type_map,
            config_path=path,
        )
