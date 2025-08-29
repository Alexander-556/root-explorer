from pathlib import Path
import torch

from neutrino.clf.config.io_config import ClfIoConfig
from neutrino.clf.config.feature_config import ClfFeatureConfig

from neutrino.clf.prepare import load_np_split, make_torch_loaders
from neutrino.clf.model import MLP
from neutrino.clf.train import TrainParams, train_simple

# choose configs you already have
IO_CFG_PATH = "configs/model/io_config.json"
FEAT_CFG_PATH = "configs/model/feature_config.json"


def main():
    io = ClfIoConfig.load_config(IO_CFG_PATH)
    ft = ClfFeatureConfig.load_config(FEAT_CFG_PATH)

    X, y, _ = load_np_split(
        split_dir=io.split_dir,
        split_prefix=io.split_prefix,
        a_suffix=io.a_suffix,
        b_suffix=io.b_suffix,
        columns_filename=io.columns_filename,
        feature_order=ft.feature_order,
    )

    train_loader, val_loader, stats = make_torch_loaders(
        X, y, val_frac=0.2, batch_size=512, seed=42, standardize=True
    )

    model = MLP(in_dim=len(ft.feature_order), hidden=(64, 64), dropout=0.0)
    params = TrainParams(lr=1e-3, max_epochs=20, device="cpu")
    run_dir, hist = train_simple(
        model,
        train_loader,
        val_loader,
        params,
        run_name="mlp_baseline",
        runs_dir=Path("runs"),
    )
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
