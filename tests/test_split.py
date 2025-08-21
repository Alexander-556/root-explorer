from neutrino_prep.io.tree_meta import TreeMeta
from neutrino_prep.io.tree_reader import TreeReader
from neutrino_prep.io.root_io import RootIO
from neutrino_prep.io.tree_ref import TreeRef
from neutrino_prep.pipeline.data_sep import DataSep, SplitPair


rt1: RootIO = RootIO()

rt1.open_root()

ref1: TreeRef = TreeRef.load_ref(rt1)

sp1 = DataSep(ref1)

# sp1.split_by_flag()
sp1.split_by_categories()

rt1.close_root()
