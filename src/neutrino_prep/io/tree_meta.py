from neutrino_prep.io.root_io import RootIO
from neutrino_prep.config.tree_config import TreeConfig


class TreeMeta:
    def __init__(self, io: RootIO, input_tree_name: str | None = None) -> None:

        self.config:TreeConfig = TreeConfig.load_config()

        self.io = io

        if input_tree_name is None or input_tree_name == "":
            self.tree_name = self.config.Tree_Name
        else:
            self.tree_name = input_tree_name

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
