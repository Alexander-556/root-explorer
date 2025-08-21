from typing import Iterable
import numpy as np

from neutrino_prep.io.tree_ref import TreeRef
from neutrino_prep.io.tree_reader import TreeReader
from neutrino_prep.config.split_config import SplitConfig
from neutrino_prep.pipeline.data_pair import SplitPair


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
