"""
file_loader.py

Purpose:
    Handles the file operations related to the root file.

Class:
    Creates a class called ``LoadedFileObj''

"""

import uproot
import numpy as np

class LoadedFileObj:

    def __init__(self) -> None:
        self.root_file_path = None
        self.file = None
        
    def open_file(self)->bool:
        try:
            self.file = uproot.open(self.root_file_path)
        except Exception as e:
            print(f"Error opening file: {e}")
            return False
        finally:
            return True