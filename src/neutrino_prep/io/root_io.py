from pathlib import Path
from typing import Any, Optional

import uproot
from neutrino_prep.config.file_config import FileConfig


class RootIO:
    def __init__(
        self,
        input_path: Path | str | None = None,
    ) -> None:
        self.config: FileConfig = FileConfig.load_config()

        # normalize path selection
        if isinstance(input_path, str):
            input_path = input_path.strip()
            
        self.root_path: Path = Path(input_path) if input_path else self.config.file_path

        self._handle: Optional[Any] = None  # uproot file/dir handle when open

    # ---------- context manager methods ----------
    def __enter__(self) -> "RootIO":
        """Allow: with RootIO(...) as rio: ..."""
        if not self.is_open:
            self.open_root()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Always close when leaving the with-block (even on exception)."""
        self.close_root()
        # Returning False means: do NOT suppress exceptions.
        return False

    # ---------- explicit open/close (still available) ----------
    def open_root(self) -> None:
        if not self.is_open:
            self._handle = uproot.open(self.root_path)

    def close_root(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    @property
    def is_open(self) -> bool:
        return self._handle is not None
