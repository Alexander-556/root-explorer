from neutrino.prep.io.tree_meta import TreeMeta
from neutrino.prep.io.root_io import RootIO

rt1: RootIO = RootIO()

rt1.open_root()
print(rt1.is_open)

tt1:TreeMeta = TreeMeta(rt1)

print(tt1.get_num_entries())
print(tt1.get_branch_names())
print(tt1.has_branch("random"))

rt1.close_root()
print(rt1.is_open)