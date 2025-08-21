from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
from pathlib import Path

from neutrino_prep.io.tree_ref import TreeRef
from neutrino_prep.io.tree_reader import TreeReader
from neutrino_prep.config.split_config import SplitConfig


@dataclass(frozen=True)
class SplitPair:

    a: dict[str, np.ndarray]
    b: dict[str, np.ndarray]

    @staticmethod
    def _combine_dict_to_matrix(
        d: dict[str, np.ndarray],
        order: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, list[str]]:

        # Decide the column order
        if order is None:
            # preserve insertion order from the dict
            col_names = list(d.keys())
        else:
            col_names = list(order)

        if not col_names:
            raise ValueError("No features selected to combine.")

        # Sanity: same length for all arrays
        lengths: list[int] = []

        for name in col_names:
            # 1) Get the per-branch array (could already be a NumPy array)
            arr = d[name]

            # 2) Convert to a NumPy array (no copy if arr is already np.ndarray)
            arr_np = np.asarray(arr)

            # 3) Get number of rows (first axis length)
            #    Assumes 1D arrays shaped like (N,), which matches your data.
            row_count = arr_np.shape[0]

            # 4) Collect it
            lengths.append(row_count)

        N0 = lengths[0]

        if any(L != N0 for L in lengths):
            raise ValueError(
                f"Inconsistent lengths across columns: {dict(zip(col_names, lengths))}"
            )

        # Combine as columns → (N, D).  (np.column_stack avoids a transpose.)
        X = np.column_stack(
            [np.asarray(d[name]).reshape(-1) for name in col_names],
        )

        return X, col_names

    # --- public API ---
    def combined_a(
        self,
        order: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, list[str]]:
        """Return (Xa, columns) where Xa has shape (N_a, D)."""
        return self._combine_dict_to_matrix(self.a, order)

    def combined_b(
        self,
        order: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, list[str]]:
        """Return (Xb, columns) where Xb has shape (N_b, D)."""
        return self._combine_dict_to_matrix(self.b, order)

    def combined_both(
        self,
        order: Iterable[str] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Return (Xa, Xb, columns). Ensures both use the *same* column order.
        If `order` is None, uses the order from `self.a`.
        """
        if order is None:
            order = list(self.a.keys())

        Xa, cols = self._combine_dict_to_matrix(self.a, order)
        Xb, _ = self._combine_dict_to_matrix(self.b, cols)
        return Xa, Xb, cols

    def save_npy(
        self,
        out_prefix: str | Path,
        order: Iterable[str] | None = None,
        dtype: np.dtype | str | None = None,
        group_suffix: tuple[str, str] = ("A", "B"),
    ) -> tuple[Path, Path, Path, list[str]]:
        """
        Save:
        - {prefix}_A.npy : (N_a, D) matrix
        - {prefix}_B.npy : (N_b, D) matrix
        - {prefix}_columns.txt : one column name per line (same order as matrices)

        If `out_prefix` is relative and doesn't start with 'output', it will be saved under 'output/'.
        Returns (path_a, path_b, path_cols, columns).
        """
        
        Xa, Xb, columns = self.combined_both(order)

        if dtype is not None:
            Xa = Xa.astype(dtype, copy=False)
            Xb = Xb.astype(dtype, copy=False)

        base = Path(out_prefix)

        # Anchor under output/ when a relative path not already starting with 'output'
        if not base.is_absolute():
            if not base.parts or base.parts[0].lower() != "output":
                base = Path("output") / base

        # strip any extension to make a clean prefix
        base_no_ext = base if base.suffix == "" else base.with_suffix("")
        base_no_ext.parent.mkdir(parents=True, exist_ok=True)

        path_a = base_no_ext.with_name(
            base_no_ext.name + f"_{group_suffix[0]}"
        ).with_suffix(".npy")

        path_b = base_no_ext.with_name(
            base_no_ext.name + f"_{group_suffix[1]}"
        ).with_suffix(".npy")

        path_cols = base_no_ext.with_name(base_no_ext.name + "_columns.txt")

        np.save(path_a, Xa)
        np.save(path_b, Xb)

        with open(path_cols, "w", encoding="utf-8") as f:
            for name in columns:
                f.write(f"{name}\n")

        return path_a, path_b, path_cols, columns


class DataSep:
    def __init__(
        self,
        ref: TreeRef,
    ) -> None:

        self.reader = TreeReader(ref)
        self.config = SplitConfig.load_config()
        self._default_flag = self.config.flag_branch

    def _mask_eq(
        self,
        values: np.ndarray,
        target: int,
    ) -> np.ndarray:
        """Build a boolean mask where values == target."""

        return values == target

    def split_by_flag(
        self,
        branches: Iterable[str] | None = None,
        flag_branch: str | None = None,
        a_value: int | None = None,
        b_value: int | None = None,
    ) -> SplitPair:

        if flag_branch is None or flag_branch == "":
            flag = self._default_flag
        else:
            flag = flag_branch

        if branches is None:
            branches = self.config.target_branches
        else:
            branches = list(branches)

        if a_value is None:
            a_value = self.config.flag_values["A"]
        if b_value is None:
            b_value = self.config.flag_values["B"]

        # Read user-requested branches + the flag branch
        cols: list[str] = list(dict.fromkeys([*branches, flag]))
        data = self.reader.read_multiple(cols)

        # Separate the flag from the rest (so it’s not returned in A/B sets).
        flag_values = data.pop(flag)

        # Build masks and slice arrays.
        mask_a = self._mask_eq(flag_values, a_value)
        mask_b = self._mask_eq(flag_values, b_value)

        # data: dict[str, np.ndarray]  # e.g., {"energy": ..., "q2": ...}
        # mask_a, mask_b: np.ndarray[bool]  # same length as the arrays in `data`

        a: dict[str, np.ndarray] = {}
        b: dict[str, np.ndarray] = {}

        for name, arr in data.items():
            # Keep only rows where mask_a is True (e.g., Sample_Flag == 0)
            a[name] = arr[mask_a]

            # Keep only rows where mask_b is True (e.g., Sample_Flag == 1)
            b[name] = arr[mask_b]

        # Now `a` and `b` are dicts with the same keys as `data`,
        # but each array contains only the selected rows.

        print({k: v.shape for k, v in a.items()})
        print({k: v.shape for k, v in b.items()})

        return SplitPair(a=a, b=b)

    def split_by_categories(
        self,
        branches: Iterable[str] | None = None,
        cat_branch: str | None = None,
        groups: dict[str, list[str]] | None = None,
    ) -> SplitPair:

        # Load configs and prepare for override

        if branches is None:
            branches = self.config.target_branches
        else:
            branches = list(branches)

        if cat_branch is None or cat_branch == "":
            cat_branch = self.config.cat_branch

        if groups is None:
            groups = self.config.type_group

        type_map = self.config.type_map  # {"QE":0, "RES":1, ...}

        # Read requested branches + the categorical branch
        cols: list[str] = list(dict.fromkeys([*branches, cat_branch]))
        data = self.reader.read_multiple(cols)

        # Remove from payload; keep only user branches
        cats = data.pop(cat_branch)

        # Define output dict
        out: dict[str, dict[str, np.ndarray]] = {}

        for group_name, labels in groups.items():
            # Map each string label (e.g. "QE") to its integer code via type_map
            codes: list[int] = []
            for label in labels:
                code = type_map[label]  # KeyError if label not in the map
                codes.append(code)

            # Build a single mask for this group: cats ∈ codes
            mask = np.isin(cats, codes)

            # Slice every requested branch with this mask
            group_data: dict[str, np.ndarray] = {}
            for name, arr in data.items():
                group_data[name] = arr[mask]

            out[group_name] = group_data

        # Todo: Make this more elgaent
        result: SplitPair = SplitPair(a=out["A"], b=out["B"])

        print({k: v.shape for k, v in out["A"].items()})
        print({k: v.shape for k, v in out["B"].items()})

        return result
