import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class TreeConfig:
    # Config Keys
    # not shared across instances
    tree_name: str
    config_path: Path

    # Class Defaults
    # shared across instances
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "data" / "tree_config.json"

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "TreeConfig":

        # conditional that handles optional input override
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # load config from json
        # ? Optional: add try and catch
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # Map dict to config obj
        return cls(
            tree_name=raw["tree_name"],
            config_path=path,
        )
