from neutrino_prep.io.root_io import RootIO
from neutrino_prep.io.tree_ref import TreeRef
from neutrino_prep.pipeline.data_sep import DataSep


with RootIO() as rio:
    ref = TreeRef.load_ref(rio)
    sep = DataSep(ref)
    pair = sep.split_by_flag()
    pair.save_npy("split1/data")
