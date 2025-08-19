"""
root_config_loader.py

Loads the root_config_file
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class RootFileConfig:

    def __init__(
            self,
            init_file_path: str = "Configs/RootFileConfig.json"
            ) -> None:
        
        self.config_path = init_file_path
        # self.config
        pass

    def load_config(self):
        print(f"{self.config_path}")

        if (self.config_path):
            pass