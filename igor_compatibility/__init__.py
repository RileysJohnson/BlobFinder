# ========================================================================
# igor_compatibility/__init__.py
# ========================================================================

"""
Igor Pro Compatibility Layer

Provides Igor Pro-like functions and data management to maintain
compatibility with the original Igor Pro workflow.
"""

from .data_management import DataManager
from .wave_operations import (GetBrowserSelection, GetDataFolder, SetDataFolder,
                             DataFolderExists, CountObjects, WaveRefIndexedDFR,
                             NameOfWave, NewDataFolder, UniqueName)