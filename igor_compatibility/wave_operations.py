# ========================================================================
# igor_compatibility/wave_operations.py
# ========================================================================

"""
Wave Operations

Igor Pro-compatible wave and data folder operations.
Provides the same interface as Igor Pro for data management.
"""

import os
from tkinter import filedialog
from core.error_handling import handle_error, safe_print

# Global variable to track current data folder
current_data_folder = None

def get_script_directory():
    """Get the directory where the script is located"""
    import sys
    try:
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return script_dir
    except Exception:
        return os.getcwd()

def GetBrowserSelection(index):
    """Get folder selection (simulates Igor Pro browser selection)."""
    try:
        folder = filedialog.askdirectory(title="Select folder containing images")
        return folder if folder else ""
    except Exception as e:
        handle_error("GetBrowserSelection", e)
        return ""

def SetDataFolder(folder_path):
    """Set current data folder"""
    try:
        global current_data_folder
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            if not os.path.isabs(folder_path):
                folder_path = os.path.join(get_script_directory(), folder_path)
            current_data_folder = folder_path
            safe_print(f"Set data folder to: {folder_path}")
        else:
            current_data_folder = get_script_directory()
            safe_print(f"Set data folder to script directory: {current_data_folder}")
    except Exception as e:
        handle_error("SetDataFolder", e)
        current_data_folder = get_script_directory()

def GetDataFolder(level):
    """Get current data folder"""
    global current_data_folder
    if current_data_folder is None:
        current_data_folder = get_script_directory()
    return current_data_folder

def DataFolderExists(folder_path):
    """Check if data folder exists."""
    try:
        return os.path.exists(folder_path) if folder_path else False
    except Exception as e:
        handle_error("DataFolderExists", e)
        return False

def CountObjects(folder_path, object_type):
    """Count objects in folder."""
    try:
        if not os.path.exists(folder_path):
            return 0
        if object_type == 1:  # Count image files
            count = 0
            for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
                count += len([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])
            return count
        return 0
    except Exception as e:
        handle_error("CountObjects", e)
        return 0

def WaveRefIndexedDFR(folder_path, index):
    """Get wave reference by index."""
    try:
        from igor_compatibility.data_management import DataManager
        if not os.path.exists(folder_path):
            return None
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])
        image_files.sort()
        if index >= len(image_files):
            return None
        image_path = os.path.join(folder_path, image_files[index])
        return DataManager.load_image_file(image_path)
    except Exception as e:
        handle_error("WaveRefIndexedDFR", e, f"index {index}")
        return None

def NameOfWave(wave):
    """Get name of wave (Igor Pro compatible)."""
    if hasattr(wave, 'name'):
        return wave.name
    elif isinstance(wave, np.ndarray):
        return "image"
    else:
        return "wave"

def NewDataFolder(folder_name):
    """Create new data folder"""
    try:
        current_dir = GetDataFolder(1)
        folder_name = folder_name.replace(":", "_").strip()
        full_path = os.path.join(current_dir, folder_name)
        os.makedirs(full_path, exist_ok=True)
        safe_print(f"Created data folder: {full_path}")
        SetDataFolder(full_path)
        return full_path
    except Exception as e:
        handle_error("NewDataFolder", e)
        return ""

def UniqueName(base_name, type_num, mode):
    """Generate unique name"""
    try:
        current_dir = GetDataFolder(1)
        base_name = base_name.replace(":", "_").strip()
        counter = 0
        while True:
            if counter == 0:
                test_name = base_name
            else:
                test_name = f"{base_name}_{counter}"
            test_path = os.path.join(current_dir, test_name)
            if not os.path.exists(test_path):
                return test_name
            counter += 1
            if counter > 1000:
                break
    except Exception as e:
        handle_error("UniqueName", e)
        return f"{base_name}_error"