from dataclasses import dataclass
from neutrino_prep.io.root_io import RootIO
from neutrino_prep.config.tree_config import TreeConfig


@dataclass(frozen=True)
class TreeRef:
    io: RootIO
    tree_name: str

    @classmethod
    def load_ref(
        cls,
        io: RootIO,
        input_tree_name: str | None = None,
    ) -> "TreeRef":

        if isinstance(input_tree_name, str):
            input_tree_name = input_tree_name.strip()

        temp_tree_name: str | None = None

        if input_tree_name is None or input_tree_name == "":
            temp_tree_name = TreeConfig.load_config().tree_name
        else:
            temp_tree_name = input_tree_name

        return cls(io=io, tree_name=temp_tree_name)
