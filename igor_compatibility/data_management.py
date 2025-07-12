"""
Igor Pro Data Management Compatibility Layer

Provides Igor Pro-compatible data folder management:
- SetDataFolder/GetDataFolder: Current folder tracking
- DataFolderExists: Check folder existence
- NewDataFolder: Create Igor-style folders
- UniqueName: Generate unique names
- Data organization matching Igor Pro exactly

This maintains the exact folder structure expected by Igor Pro users.
"""

import os
import sys
from core.error_handling import handle_error, safe_print


def get_script_directory():
    """Get the directory where the script is located"""
    try:
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return script_dir
    except Exception:
        return os.getcwd()


# Global variable to track current data folder
current_data_folder = None


def SetDataFolder(folder_path):
    """
    Set current data folder - EXACT IGOR PRO BEHAVIOR

    Args:
        folder_path: Path to set as current data folder
    """
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
    """
    Get current data folder - EXACT IGOR PRO BEHAVIOR

    Args:
        level: Data folder level (1 for current)

    Returns:
        Current data folder path
    """
    global current_data_folder
    if current_data_folder is None:
        current_data_folder = get_script_directory()
    return current_data_folder


def DataFolderExists(folder_path):
    """
    Check if data folder exists - EXACT IGOR PRO BEHAVIOR

    Args:
        folder_path: Path to check

    Returns:
        True if folder exists, False otherwise
    """
    try:
        return os.path.exists(folder_path) if folder_path else False

    except Exception as e:
        handle_error("DataFolderExists", e)
        return False


def NewDataFolder(folder_path):
    """
    Create new data folder - EXACT IGOR PRO BEHAVIOR

    Args:
        folder_path: Path to create

    Returns:
        Created folder path
    """
    try:
        # If it's not an absolute path, make it relative to script directory
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(get_script_directory(), folder_path)

        os.makedirs(folder_path, exist_ok=True)
        safe_print(f"Created data folder: {folder_path}")
        return folder_path

    except Exception as e:
        handle_error("NewDataFolder", e)
        return ""


def UniqueName(base_name, type_num, mode):
    """
    Generate unique name - EXACT IGOR PRO BEHAVIOR

    Args:
        base_name: Base name for uniqueness
        type_num: Type number (11 for folder)
        mode: Mode (2 for folder creation)

    Returns:
        Unique name string
    """
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

        return f"{base_name}_error"

    except Exception as e:
        handle_error("UniqueName", e)
        return f"{base_name}_error"


def CountObjects(folder_path, object_type):
    """
    Count objects in folder - EXACT IGOR PRO BEHAVIOR

    Args:
        folder_path: Path to folder
        object_type: Type of objects to count (1 for waves/images)

    Returns:
        Number of objects found
    """
    try:
        if not os.path.exists(folder_path):
            return 0

        if object_type == 1:  # Count image files
            count = 0
            for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
                count += len([f for f in os.listdir(folder_path)
                              if f.lower().endswith(ext.lower())])
            return count
        return 0

    except Exception as e:
        handle_error("CountObjects", e)
        return 0