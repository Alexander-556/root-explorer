from neutrino_prep.io.tree_meta import TreeMeta
from neutrino_prep.io.tree_reader import TreeReader
from neutrino_prep.io.root_io import RootIO
from neutrino_prep.io.tree_ref import TreeRef


rt1: RootIO = RootIO()

rt1.open_root()
print(rt1.is_open)

ref1: TreeRef = TreeRef.load_ref(rt1)

tt1 = TreeMeta(ref1)

print(tt1.get_num_entries())
print(tt1.get_branch_names())
print(tt1.has_branch("random"))

tr1 = TreeReader(ref1)

print(tr1.read_one("Init_Nu_Energy"))
print(tr1.read_multiple(["Init_Nu_Energy","Transfer_qSq"]))

rt1.close_root()
print(rt1.is_open)