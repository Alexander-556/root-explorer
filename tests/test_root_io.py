from neutrino_prep.io.root_io import RootIO

rt1: RootIO = RootIO()

rt1.open_root()
print(rt1.is_open)
rt1.close_root()