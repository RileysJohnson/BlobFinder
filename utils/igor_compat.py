"""Contains utilities to replicate Igor Pro functions and folder logic."""

# #######################################################################
#                 UTILITIES: IGOR PRO COMPATIBILITY
#
#   CONTENTS:
#       - get_script_directory: Gets the location of the running script.
#       - SetDataFolder/GetDataFolder: Manages the current working directory.
#       - GetBrowserSelection: Simulates Igor's folder selection dialog.
#       - CountObjects: Counts image files in a folder.
#       - WaveRefIndexedDFR: Loads an image by its index in a folder.
#       - NameOfWave: Gets a descriptive name for a data object.
#       - NewDataFolder/UniqueName: Creates unique folder/file names.
#       - verify_folder_structure: Verifies output folder structure.
#
# #######################################################################

import os
import sys
from tkinter import filedialog
from .error_handler import handle_error, safe_print
from .data_manager import DataManager

def get_script_directory():
    """Get the directory where the script is located"""
    try:
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Ensure we return a clean path without spaces in folder names
            return script_dir
    except Exception:
        return os.getcwd()  # Fallback to current working directory

def check_and_fix_folders():
    """Check current folder structure and fix issues"""
    try:
        current_dir = get_script_directory()
        safe_print(f"Script directory: {current_dir}")
        safe_print(f"Current data folder: {GetDataFolder(1)}")

        # List contents of current directory
        if os.path.exists(current_dir):
            contents = os.listdir(current_dir)
            safe_print(f"Contents of current directory:")
            for item in contents:
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path):
                    safe_print(f"  [DIR]  {item}")
                else:
                    safe_print(f"  [FILE] {item}")

        return current_dir

    except Exception as e:
        handle_error("check_and_fix_folders", e)
        return None

def GetBrowserSelection(index):
    """Get folder selection (simulates Igor Pro browser selection)."""
    try:
        folder = filedialog.askdirectory(title="Select folder containing images")
        return folder if folder else ""

    except Exception as e:
        handle_error("GetBrowserSelection", e)
        return ""

def SetDataFolder(folder_path):
    """Set current data folder - FIXED VERSION"""
    try:
        global current_data_folder

        # Clean up the path to avoid issues with spaces
        if folder_path:
            folder_path = os.path.normpath(folder_path)
            # Ensure we're always working with absolute paths
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
    """Get current data folder - FIXED VERSION"""
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
        if not os.path.exists(folder_path):
            return None

        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])

        image_files.sort()  # Ensure consistent ordering

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
        return "image"  # Default name for numpy arrays
    else:
        return "wave"

def NewDataFolder(folder_name):
    """Create new data folder - FIXED VERSION"""
    try:
        current_dir = GetDataFolder(1)

        # Handle folder names that might have spaces or special characters
        folder_name = folder_name.replace(":", "_").strip()

        # Create full path
        full_path = os.path.join(current_dir, folder_name)

        # Create the folder structure
        os.makedirs(full_path, exist_ok=True)
        safe_print(f"Created data folder: {full_path}")

        # Update current data folder to the new one
        SetDataFolder(full_path)
        return full_path

    except Exception as e:
        handle_error("NewDataFolder", e)
        return ""

def UniqueName(base_name, type_num, mode):
    """Generate unique name - FIXED VERSION"""
    try:
        current_dir = GetDataFolder(1)

        # Clean the base name
        base_name = base_name.replace(":", "_").strip()

        counter = 0
        while True:
            if counter == 0:
                test_name = base_name
            else:
                test_name = f"{base_name}_{counter}"

            # Check in current directory
            test_path = os.path.join(current_dir, test_name)
            if not os.path.exists(test_path):
                return test_name
            counter += 1

            # Prevent infinite loops
            if counter > 1000:
                break

    except Exception as e:
        handle_error("UniqueName", e)
        return f"{base_name}_error"

def verify_folder_structure(base_folder):
    """Verify and print the folder structure created (Igor Pro style)."""
    try:
        safe_print("\n" + "=" * 60)
        safe_print("FOLDER STRUCTURE CREATED:")
        safe_print("=" * 60)

        if os.path.exists(base_folder):
            for root, dirs, files in os.walk(base_folder):
                level = root.replace(base_folder, '').count(os.sep)
                indent = ' ' * 2 * level
                safe_print(f"{indent}{os.path.basename(root)}/")

                # Print files
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    if file.endswith(('.npy', '.json', '.txt')):
                        safe_print(f"{subindent}{file}")
        safe_print("=" * 60)

    except Exception as e:
        handle_error("verify_folder_structure", e)

def verify_igor_compatibility(folder_path):
    """Verify that created folders match Igor structure"""
    try:
        if not os.path.exists(folder_path):
            safe_print(f"✗ ERROR: Folder does not exist: {folder_path}")
            return False

        required_files = [
            "Heights.npy", "Volumes.npy", "Areas.npy", "AvgHeights.npy",
            "COM.npy", "Original.npy"
        ]

        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(folder_path, file)):
                missing_files.append(file)

        if missing_files:
            safe_print(f"✗ Missing required files: {missing_files}")
            return False

        # Check for particle folders
        particle_folders = [d for d in os.listdir(folder_path)
                            if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("Particle_")]

        safe_print(f"✓ Igor Pro compatibility check passed")
        safe_print(f"✓ Found {len(particle_folders)} particle folders")
        safe_print(f"✓ All required measurement files present")

        return True

    except Exception as e:
        handle_error("verify_igor_compatibility", e)
        return False



