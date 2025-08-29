# src/neutrino/prep/config/file_config.py
import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class FileConfig:
    """
    Dataclass wrapper for file-level configuration.

    This loader handles JSON that specifies the location of the main ROOT file
    (or other data file). It provides a simple API to access the file path inside
    Python code.
    """

    # -------------------------------------------------------------------------
    # Instance attributes (unique per config object)
    # -------------------------------------------------------------------------
    file_path: Path  # Path to the ROOT file specified in JSON
    config_path: Path  # Path to the JSON file actually used

    # -------------------------------------------------------------------------
    # Class attributes (shared across all instances)
    # -------------------------------------------------------------------------
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "data" / "file_config.json"

    # -------------------------------------------------------------------------
    # Config loader
    # -------------------------------------------------------------------------
    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "FileConfig":
        """
        Load a FileConfig instance from JSON.

        Parameters
        ----------
        path : Path | str | None, optional
            Path to a config JSON file. If None, uses DEFAULT_CONFIG_PATH.

        Returns
        -------
        FileConfig
            Dataclass instance populated with config values.
        """

        # 1. Resolve path (either user-specified or default)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # 2. Load raw JSON dict
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # 3. Extract fields
        file_path = Path(raw["file_path"])

        # 4. Construct dataclass and return
        return cls(
            file_path=file_path,
            config_path=path,
        )
