from neutrino.prep.io.root_io import RootIO
from neutrino.prep.io.tree_ref import TreeRef
from neutrino.prep.io.tree_meta import TreeMeta

with RootIO() as rio:
    ref: TreeRef = TreeRef.load_ref(rio)
    meta: TreeMeta = TreeMeta(ref)
    print(meta.get_branch_names())
    print(meta.get_num_entries())
