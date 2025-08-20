from neutrino_prep.config.file_config import FileConfig
from pathlib import Path
from typing import Any

import uproot


class RootIO:

    def __init__(self, input_path: Path | str | None = None) -> None:

        self.config: FileConfig = FileConfig.load_config()

        # normalize
        if isinstance(input_path, str):
            input_path = input_path.strip()

        if input_path is None or input_path == "":
            temp_path = self.config.File_Path
        else:
            temp_path = Path(input_path)

        self.root_path: Path = temp_path

        self._handle: Any = None

    def open_root(self):
        self._handle = uproot.open(self.root_path)

    def close_root(self):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None

    @property
    def is_open(self) -> bool:
        return self._handle is not None
