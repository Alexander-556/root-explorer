from neutrino_prep.io.tree_ref import TreeRef

import numpy as np


class TreeReader:

    def __init__(
        self,
        ref: TreeRef,
    ) -> None:
        
        self.ref = ref
        self.io = ref.io
        self.tree_name = ref.tree_name

    def _get_tree(self):

        if self.io._handle is None:
            raise RuntimeError("RootIO is not open.")

        return self.io._handle[self.tree_name]
    
    def read_one(self, branch: str) -> np.ndarray:
        
        arrs = self._get_tree().arrays([branch], library="np")
        return arrs[branch]
