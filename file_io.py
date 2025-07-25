"""
File I/O Module
Handles reading .ibw files and managing data folder structures
"""

import numpy as np
import os
from igor_compatibility import Wave
import struct
from pathlib import Path


class DataFolder:
    """
    Mimics Igor Pro data folder structure
    """

    def __init__(self, name="root"):
        self.name = name
        self.waves = {}
        self.subfolders = {}
        self.parent = None
        self.current = True  # For root folder

    def add_wave(self, wave, name=None):
        if name is None:
            name = wave.name
        self.waves[name] = wave
        wave.name = name

    def add_subfolder(self, folder_name):
        new_folder = DataFolder(folder_name)
        new_folder.parent = self
        self.subfolders[folder_name] = new_folder
        return new_folder

    def get_wave(self, path):
        """Get wave by path (e.g., 'Images:YEG_1')"""
        parts = path.split(':')
        current_folder = self

        # Navigate to the correct folder
        for part in parts[:-1]:
            if part in current_folder.subfolders:
                current_folder = current_folder.subfolders[part]
            else:
                return None

        # Get the wave
        wave_name = parts[-1]
        return current_folder.waves.get(wave_name)

    def get_folder(self, path):
        """Get folder by path"""
        if not path or path == "root":
            return self

        parts = path.split(':')
        current_folder = self

        for part in parts:
            if part in current_folder.subfolders:
                current_folder = current_folder.subfolders[part]
            else:
                return None

        return current_folder

    def list_waves(self):
        """List all waves in this folder"""
        return list(self.waves.keys())

    def list_folders(self):
        """List all subfolders"""
        return list(self.subfolders.keys())


# Global data browser (simulates Igor's data browser)
data_browser = DataFolder("root")


def GetBrowserSelection(index=0):
    """
    Igor GetBrowserSelection equivalent
    Returns the path to the currently selected folder
    """
    global current_selected_folder
    if hasattr(GetBrowserSelection, 'selected'):
        return GetBrowserSelection.selected
    return "root:"


def SetBrowserSelection(path):
    """Set the currently selected folder in the browser"""
    GetBrowserSelection.selected = path


def GetDataFolder(flag):
    """
    Igor GetDataFolder equivalent
    flag=1 returns current data folder path
    """
    global data_browser
    if flag == 1:
        return "root:"  # Simplified for now
    return data_browser


def DataFolderExists(path):
    """
    Igor DataFolderExists equivalent
    """
    global data_browser
    folder = data_browser.get_folder(path.rstrip(':'))
    return folder is not None


def CountObjects(folder_path, object_type):
    """
    Igor CountObjects equivalent
    object_type: 1 for waves, 4 for data folders
    """
    global data_browser
    folder = data_browser.get_folder(folder_path.rstrip(':'))
    if folder is None:
        return 0

    if object_type == 1:  # Waves
        return len(folder.waves)
    elif object_type == 4:  # Data folders
        return len(folder.subfolders)
    return 0


def NewDataFolder(path):
    """
    Igor NewDataFolder equivalent
    """
    global data_browser
    parts = path.split(':')
    current_folder = data_browser

    for part in parts:
        if part and part not in current_folder.subfolders:
            current_folder.add_subfolder(part)
        if part:
            current_folder = current_folder.subfolders[part]


def DuplicateDataFolder(source_path, dest_path):
    """
    Igor DuplicateDataFolder equivalent
    """
    global data_browser
    source_folder = data_browser.get_folder(source_path.rstrip(':'))
    if source_folder is None:
        return

    # Create destination folder
    NewDataFolder(dest_path.rstrip(':'))
    dest_folder = data_browser.get_folder(dest_path.rstrip(':'))

    # Copy all waves
    for wave_name, wave in source_folder.waves.items():
        new_wave = Wave(wave.data.copy(), wave_name, wave.note)
        new_wave.scaling = wave.scaling.copy()
        dest_folder.add_wave(new_wave)

    # Recursively copy subfolders
    for subfolder_name, subfolder in source_folder.subfolders.items():
        DuplicateDataFolder(f"{source_path}:{subfolder_name}",
                            f"{dest_path}:{subfolder_name}")


def WaveRefIndexedDFR(folder_path, index):
    """
    Igor WaveRefIndexedDFR equivalent
    Returns wave reference by index in a data folder
    """
    global data_browser
    folder = data_browser.get_folder(folder_path.rstrip(':'))
    if folder is None:
        return None

    wave_names = list(folder.waves.keys())
    if 0 <= index < len(wave_names):
        return folder.waves[wave_names[index]]
    return None


def GetIndexedObjNameDFR(folder_path, obj_type, index):
    """
    Igor GetIndexedObjNameDFR equivalent
    Returns name of indexed object in data folder
    """
    global data_browser
    folder = data_browser.get_folder(folder_path.rstrip(':'))
    if folder is None:
        return ""

    if obj_type == 1:  # Waves
        wave_names = list(folder.waves.keys())
        if 0 <= index < len(wave_names):
            return wave_names[index]
    elif obj_type == 4:  # Data folders
        folder_names = list(folder.subfolders.keys())
        if 0 <= index < len(folder_names):
            return folder_names[index]

    return ""


def UniqueName(base_name, obj_type, mode):
    """
    Igor UniqueName equivalent
    Generates a unique name for waves or data folders
    """
    global data_browser
    counter = 0
    while True:
        if counter == 0:
            test_name = base_name
        else:
            test_name = f"{base_name}_{counter}"

        # Check if name exists
        if obj_type == 1:  # Waves
            if test_name not in data_browser.waves:
                return test_name
        elif obj_type == 11:  # Data folders
            if test_name not in data_browser.subfolders:
                return test_name

        counter += 1
        if counter > 1000:  # Prevent infinite loop
            break

    return base_name


def GetWavesDataFolder(wave, flag):
    """
    Igor GetWavesDataFolder equivalent
    Returns the data folder path containing the wave
    """
    # Simplified - would need full path tracking in real implementation
    if flag == 1:
        return "root:"
    elif flag == 2:
        return f"root:{wave.name}"
    return "root:"


def NameOfWave(wave):
    """
    Igor NameOfWave equivalent
    """
    return wave.name


class IBWReader:
    """
    Reads Igor Pro .ibw (Igor Binary Wave) files
    Simplified version - full implementation would need complete IBW format parsing
    """

    @staticmethod
    def read_ibw_file(filename):
        """
        Read an Igor .ibw file and return a Wave object
        This is a simplified implementation
        """
        try:
            # Try using igor package if available
            import igor.binarywave as bw

            data = bw.load(filename)
            wave_data = data['wave']['wData']

            # Get wave name from filename
            wave_name = os.path.splitext(os.path.basename(filename))[0]

            # Create Wave object
            wave = Wave(wave_data, wave_name)

            # Set scaling if available
            if 'wave' in data and 'dimension_units' in data['wave']:
                # Extract scaling information
                pass  # Implement scaling extraction

            return wave

        except ImportError:
            # Fallback implementation
            return IBWReader._read_ibw_manual(filename)

    @staticmethod
    def _read_ibw_manual(filename):
        """
        Manual IBW file reading (simplified)
        """
        with open(filename, 'rb') as f:
            # This is a very simplified IBW reader
            # Full implementation would need complete format specification

            # Skip header (simplified)
            f.seek(64)  # Skip basic header

            # Read data dimensions (this is simplified)
            # Real IBW format is more complex
            try:
                # Try to read as float32 array
                data = np.fromfile(f, dtype=np.float32)

                # Try to reshape if it looks like 2D
                if len(data) > 100:
                    side = int(np.sqrt(len(data)))
                    if side * side == len(data):
                        data = data.reshape((side, side))

                wave_name = os.path.splitext(os.path.basename(filename))[0]
                return Wave(data, wave_name)

            except Exception:
                # If all else fails, return empty wave
                wave_name = os.path.splitext(os.path.basename(filename))[0]
                return Wave(np.array([]), wave_name)


def LoadWaves(file_paths, dest_folder="root"):
    """
    Load waves from files into specified data folder
    """
    global data_browser
    folder = data_browser.get_folder(dest_folder.rstrip(':'))
    if folder is None:
        NewDataFolder(dest_folder.rstrip(':'))
        folder = data_browser.get_folder(dest_folder.rstrip(':'))

    loaded_waves = []

    # Handle both single file and list of files
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        try:
            if file_path.lower().endswith('.ibw'):
                wave = IBWReader.read_ibw_file(file_path)
                folder.add_wave(wave)
                loaded_waves.append(wave)
            else:
                # Handle other file types as needed
                print(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue

    return loaded_waves


def Note(wave, note_text=None):
    """
    Igor Note function equivalent
    If note_text is provided, sets the note
    If not provided, returns the current note
    """
    if note_text is None:
        return wave.note
    else:
        if note_text.startswith('/K'):
            # Kill existing note
            wave.note = ""
        else:
            wave.note += note_text


def StringByKey(key, key_list, key_separator=":", list_separator="\r"):
    """
    Igor StringByKey equivalent
    """
    items = key_list.split(list_separator)
    for item in items:
        if key_separator in item:
            item_key, item_value = item.split(key_separator, 1)
            if item_key.strip() == key:
                return item_value.strip()
    return ""


def NumberByKey(key, key_list, key_separator=":", list_separator="\r"):
    """
    Igor NumberByKey equivalent
    """
    value_str = StringByKey(key, key_list, key_separator, list_separator)
    try:
        return float(value_str)
    except ValueError:
        return np.nan


def ReplaceStringByKey(key, key_list, new_value, key_separator=":", list_separator="\r"):
    """
    Igor ReplaceStringByKey equivalent
    """
    items = key_list.split(list_separator)
    new_items = []
    found = False

    for item in items:
        if key_separator in item:
            item_key, item_value = item.split(key_separator, 1)
            if item_key.strip() == key:
                new_items.append(f"{key}{key_separator}{new_value}")
                found = True
            else:
                new_items.append(item)
        else:
            new_items.append(item)

    if not found:
        new_items.append(f"{key}{key_separator}{new_value}")

    return list_separator.join(new_items)


def Num2Str(number):
    """
    Igor Num2Str equivalent
    """
    return str(number)


def Str2Num(string):
    """
    Igor Str2Num equivalent
    """
    try:
        return float(string)
    except ValueError:
        return np.nan


def Date():
    """
    Igor Date function equivalent
    """
    import datetime
    return datetime.datetime.now().strftime("%a, %b %d, %Y")