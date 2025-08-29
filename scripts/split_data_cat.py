from neutrino.prep.io.root_io import RootIO
from neutrino.prep.io.tree_ref import TreeRef
from neutrino.prep.pipeline.data_sep import DataSep
from neutrino.prep.pipeline.data_pair import SplitPair


with RootIO() as rio:
    ref: TreeRef = TreeRef.load_ref(rio)
    sep: DataSep = DataSep(ref)

    pair: SplitPair = sep.split_by_categories()
    pair.save_npy("split2/data")