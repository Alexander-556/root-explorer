from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
from pathlib import Path


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
