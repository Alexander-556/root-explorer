# src/neutrino/clf/config/io_config.py
import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class ClfIoConfig:
    """
    Dataclass wrapper for classifier I/O configuration.

    This loader handles JSON that specifies where to read prepared splits
    (A/B .npy files and the columns text file) and where to write run outputs.
    """

    # -------------------------------------------------------------------------
    # Instance attributes (unique per config object)
    # -------------------------------------------------------------------------
    output_dir: Path  # Directory to save run artifacts (e.g., runs/)
    split_prefix: str  # Base name for split files (e.g., "data")
    split_dir: Path  # Directory containing npy splits (e.g., output/split1)
    a_suffix: str  # Suffix for class A file (e.g., "_A.npy")
    b_suffix: str  # Suffix for class B file (e.g., "_B.npy")
    columns_filename: str  # File listing column names (e.g., "data_columns.txt")
    config_path: Path  # Path to the JSON file actually used

    # -------------------------------------------------------------------------
    # Class attributes (shared across all instances)
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "model" / "io_config.json"

    # -------------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------------
    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClfIoConfig":
        """
        Load a ClfIoConfig instance from JSON.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to a config JSON file. If None, uses DEFAULT_CONFIG_PATH.

        Returns
        -------
        ClfIoConfig
            Dataclass instance populated with config values.
        """

        # 1. Resolve path (either user-specified or default)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # 2. Load raw JSON dict
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3. Parse fields explicitly

        # Simple strings
        a_suffix: str = str(raw["a_suffix"])
        b_suffix: str = str(raw["b_suffix"])
        split_prefix: str = str(raw["split_prefix"])
        columns_filename: str = str(raw["columns_filename"])

        # Paths
        output_dir: Path = Path(raw["output_dir"])
        split_dir: Path = Path(raw["split_dir"])

        # 4. Construct dataclass and return
        return cls(
            a_suffix=a_suffix,
            b_suffix=b_suffix,
            split_prefix=split_prefix,
            columns_filename=columns_filename,
            output_dir=output_dir,
            split_dir=split_dir,
            config_path=path,
        )
