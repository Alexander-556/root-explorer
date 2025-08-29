import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class ClfIoConfig:
    # Config Keys
    # not shared across instances

    output_dir: Path
    split_prefix: str
    split_dir: Path
    a_suffix: str
    b_suffix: str
    columns_filename: str
    config_path: Path

    # Class Defaults
    # shared across instances
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "model" / "io_config.json"

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClfIoConfig":

        # Resolve path (allow override)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # Load raw JSON
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # --- Parse each field explicitly ---

        # 1) Simple strings
        a_suffix: str = str(raw["a_suffix"])
        b_suffix: str = str(raw["b_suffix"])
        split_prefix: str = str(raw["split_prefix"])
        columns_filename: str = str(raw["columns_filename"])

        # 2) Paths
        output_dir=Path(raw["output_dir"])
        split_dir=Path(raw["split_dir"])


        # --- Construct dataclass ---
        return cls(
            a_suffix=a_suffix,
            b_suffix=b_suffix,
            split_prefix=split_prefix,
            columns_filename=columns_filename,
            output_dir=output_dir,
            split_dir=split_dir,
            config_path=path,
        )
