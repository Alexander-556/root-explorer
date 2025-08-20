import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class TreeConfig:
    # Config Keys
    # not shared across instances
    Tree_Name: str
    Config_Path: Path

    # Class Defaults
    # shared across instances
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "tree_config.json"

    @classmethod
    def load_config(cls, path: Path | str | None = None) -> "TreeConfig":

        # conditional that handles optional input override
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # load config from json
        # ? Optional: add try and catch
        with open(path, "r", encoding="utf-8") as f:
            raw_config: dict[str, Any] = json.load(f)

        # Map dict to config obj
        return cls(
            Tree_Name=raw_config["Tree_Name"],
            Config_Path=path,
        )
