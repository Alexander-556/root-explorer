from neutrino_prep.io.tree_ref import TreeRef


class TreeMeta:
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

    def get_num_entries(self) -> int:
        return int(self._get_tree().num_entries)

    def get_branch_names(self) -> list[str]:
        return list(self._get_tree().keys())

    def has_branch(self, name: str) -> bool:
        return name in self._get_tree().keys()
