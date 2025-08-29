# src/neutrino/clf/config/classify_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional


# ---------------- Nested config sections ---------------- #

@dataclass
class ModelConfig:
    type: Literal["sk_random_forest", "sk_logreg", "torch_mlp"]
    params: dict[str, Any]


@dataclass
class TrainConfig:
    test_size: float
    random_state: int
    stratify: bool


@dataclass
class PreprocessConfig:
    standardize: bool
    scaler: Literal["standard", "none"]


@dataclass
class IOConfig:
    output_dir: Path
    split_prefix: str
    split_dir: Path
    a_suffix: str
    b_suffix: str
    columns_filename: str


# ---------------- Top-level config ---------------- #

@dataclass
class ClassifyConfig:
    # User-configurable keys
    feature_order: list[str]
    run_name: str
    model: ModelConfig
    train: TrainConfig
    preprocess: PreprocessConfig
    io: IOConfig

    # Book-keeping
    config_path: Path

    # Default location (adjust if you store it elsewhere)
    DEFAULT_CONFIG_PATH: ClassVar[Path] = Path("configs") / "model" / "classify_config.json"

    @classmethod
    def load_config(
        cls,
        path: Path | str | None = None,
    ) -> "ClassifyConfig":
        """
        Strict, explicit parser that mirrors your SplitConfig style:
        - Converts types explicitly
        - Resolves paths
        - Does not mutate/guess missing keys (fails fast)
        """
        # Resolve path (allow override)
        path = Path(path) if path else cls.DEFAULT_CONFIG_PATH

        # Load raw JSON
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)

        # -------- Parse top-level simple fields --------
        # feature_order: list[str]
        raw_features: list[Any] = raw["feature_order"]
        feature_order: list[str] = [str(x) for x in raw_features]

        # run_name: str
        run_name: str = str(raw["run_name"])

        # -------- Parse model block --------
        raw_model: dict[str, Any] = raw["model"]
        model_type: str = str(raw_model["type"])
        # Allow only known values; raise otherwise
        if model_type not in {"sk_random_forest", "sk_logreg", "torch_mlp"}:
            raise ValueError(f"Unsupported model.type: {model_type!r}")
        # params: arbitrary dict with explicit basic conversions when useful
        raw_params: dict[str, Any] = dict(raw_model.get("params", {}))

        # Convert JSON null -> Python None for common keys
        def _maybe_none(v: Any) -> Any:
            return None if v is None else v

        # Example normalizations (safe if keys missing)
        if "max_depth" in raw_params:
            raw_params["max_depth"] = _maybe_none(raw_params["max_depth"])
        if "n_jobs" in raw_params and isinstance(raw_params["n_jobs"], int) or raw_params.get("n_jobs") == -1:
            # keep as-is; sklearn accepts -1
            pass
        if "random_state" in raw_params:
            raw_params["random_state"] = int(raw_params["random_state"])

        model = ModelConfig(
            type=model_type,  # type: ignore[arg-type]
            params=raw_params,
        )

        # -------- Parse train block --------
        raw_train: dict[str, Any] = raw["train"]
        test_size: float = float(raw_train["test_size"])
        random_state: int = int(raw_train["random_state"])
        stratify: bool = bool(raw_train.get("stratify", True))
        train_cfg = TrainConfig(
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        # -------- Parse preprocess block --------
        raw_pre: dict[str, Any] = raw["preprocess"]
        standardize: bool = bool(raw_pre.get("standardize", True))
        scaler_raw: str = str(raw_pre.get("scaler", "standard"))
        if scaler_raw not in {"standard", "none"}:
            raise ValueError(f"Unsupported preprocess.scaler: {scaler_raw!r}")
        preprocess_cfg = PreprocessConfig(
            standardize=standardize,
            scaler=scaler_raw,  # type: ignore[arg-type]
        )

        # -------- Parse io block --------
        raw_io: dict[str, Any] = raw["io"]
        output_dir = Path(str(raw_io["output_dir"]))
        split_prefix = str(raw_io["split_prefix"])
        split_dir = Path(str(raw_io["split_dir"]))  # handles backslashes from JSON on Windows
        a_suffix = str(raw_io["a_suffix"])
        b_suffix = str(raw_io["b_suffix"])
        columns_filename = str(raw_io["columns_filename"])
        io_cfg = IOConfig(
            output_dir=output_dir,
            split_prefix=split_prefix,
            split_dir=split_dir,
            a_suffix=a_suffix,
            b_suffix=b_suffix,
            columns_filename=columns_filename,
        )

        # -------- Construct and return dataclass --------
        return cls(
            feature_order=feature_order,
            run_name=run_name,
            model=model,
            train=train_cfg,
            preprocess=preprocess_cfg,
            io=io_cfg,
            config_path=path,
        )


# ---------------- Convenience: resolved file paths ---------------- #

@dataclass
class SplitFiles:
    """
    Helper to resolve concrete file paths based on IOConfig and split_prefix.
    """
    a_path: Path
    b_path: Path
    columns_path: Path

    @classmethod
    def from_io(cls, io: IOConfig) -> "SplitFiles":
        """
        Build absolute file paths like:
        split_dir / f"{split_prefix}{a_suffix}"  (e.g., data_A.npy)
        """
        a = io.split_dir / f"{io.split_prefix}{io.a_suffix}"
        b = io.split_dir / f"{io.split_prefix}{io.b_suffix}"
        cols = io.split_dir / io.columns_filename
        return cls(a_path=a, b_path=b, columns_path=cols)
