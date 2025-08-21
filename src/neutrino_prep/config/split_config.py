import json
from pathlib import Path
from typing import Any, ClassVar
from dataclasses import dataclass


@dataclass
class SplitConfig:
    # Config Keys
    # not shared across instances
    flag_branch: str
    flag_values: dict[str, int]
    cat_branch: str
    target_branches: list[str]
    type_group: dict[str, list[str]]
    type_map: dict[str, int]
    config_path: Path

    # Class Defaults
    # shared across instances
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "split_config.json"

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "SplitConfig":
        
        # Resolve path (allow override)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # Load raw JSON
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # --- Parse each field explicitly ---

        # 1) Simple strings
        flag_branch: str = str(raw["flag_branch"])
        cat_branch: str = str(raw["cat_branch"])

        # 2) Dict[str, int] for flag values
        raw_flag_values: dict[str, Any] = raw["flag_values"]
        flag_values: dict[str, int] = {}
        for k, v in raw_flag_values.items():
            key: str = str(k)
            val: int = int(v)
            flag_values[key] = val

        # 3) List[str] for target branches
        raw_targets: list[Any] = raw["target_branches"]
        target_branches: list[str] = []
        for x in raw_targets:
            target_branches.append(str(x))

        # 4) Dict[str, list[str]] for type groups
        raw_type_group: dict[str, Any] = raw["type_group"]
        type_group: dict[str, list[str]] = {}
        for group_name, labels in raw_type_group.items():
            group_key: str = str(group_name)
            label_list: list[str] = []
            for label in labels:
                label_list.append(str(label))
            type_group[group_key] = label_list

        # 5) Dict[str, int] for type map
        raw_type_map: dict[str, Any] = raw["type_map"]
        type_map: dict[str, int] = {}
        for label, code in raw_type_map.items():
            type_map[str(label)] = int(code)

        # --- Construct dataclass ---
        return cls(
            flag_branch=flag_branch,
            flag_values=flag_values,
            cat_branch=cat_branch,
            target_branches=target_branches,
            type_group=type_group,
            type_map=type_map,
            config_path=path,
        )
