"""
File I/O Module
Handles reading various image file formats and managing data folder structures
Complete implementation matching Igor Pro data management
Cross-platform compatible version with proper error handling
"""

import numpy as np
import os
import sys
import struct
from pathlib import Path
from igor_compatibility import Wave

# Optional imports for different file formats
try:
    import igor.binarywave as bw

    IGOR_AVAILABLE = True
except ImportError:
    IGOR_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import tifffile

    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from skimage import io as skimage_io

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class DataFolder:
    """
    Mimics Igor Pro data folder structure
    Complete implementation matching Igor Pro functionality
    Cross-platform compatible implementation
    """

    def __init__(self, name="root"):
        self.name = name
        self.waves = {}
        self.subfolders = {}
        self.parent = None
        self.current = True  # For root folder

    def add_wave(self, wave, name=None):
        """Add a wave to this folder"""
        if name is None:
            name = wave.name
        self.waves[name] = wave
        wave.name = name

    def add_subfolder(self, folder_name):
        """Add a subfolder"""
        new_folder = DataFolder(folder_name)
        new_folder.parent = self
        self.subfolders[folder_name] = new_folder
        return new_folder

    def get_wave(self, path):
        """Get wave by path (e.g., 'Images:YEG_1')"""
        if not path:
            return None

        parts = path.split(':')
        current_folder = self

        # Navigate to the correct folder
        for part in parts[:-1]:
            if part and part in current_folder.subfolders:
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

        parts = [p for p in path.split(':') if p]  # Remove empty parts
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

    def get_full_path(self):
        """Get full path from root"""
        if self.parent is None:
            return "root"
        else:
            return self.parent.get_full_path() + ":" + self.name

    def count_waves(self):
        """Count total waves in this folder and subfolders"""
        count = len(self.waves)
        for subfolder in self.subfolders.values():
            count += subfolder.count_waves()
        return count

    def find_wave(self, wave_name):
        """Find a wave by name recursively"""
        if wave_name in self.waves:
            return self.waves[wave_name]

        for subfolder in self.subfolders.values():
            found = subfolder.find_wave(wave_name)
            if found:
                return found

        return None

    def remove_wave(self, wave_name):
        """Remove a wave from this folder"""
        if wave_name in self.waves:
            del self.waves[wave_name]
            return True
        return False

    def remove_subfolder(self, folder_name):
        """Remove a subfolder"""
        if folder_name in self.subfolders:
            del self.subfolders[folder_name]
            return True
        return False


# Global data browser (simulates Igor Pro data browser)
data_browser = DataFolder("root")


def load_image(file_path):
    """
    Load an image file and return as Wave object
    Supports multiple formats: IGor IBW, TIFF, PNG, JPEG
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        file_ext = file_path.suffix.lower()

        if file_ext == '.ibw':
            # Igor Binary Wave format
            if IGOR_AVAILABLE:
                try:
                    data = bw.load(str(file_path))
                    wave_data = data['wave']['wData']

                    # Handle different data types
                    if wave_data.dtype == np.complex64 or wave_data.dtype == np.complex128:
                        wave_data = np.abs(wave_data)  # Take magnitude for complex data

                    wave = Wave(wave_data, file_path.stem)

                    # Set scaling if available
                    if 'wave' in data and 'sfA' in data['wave']:
                        sf = data['wave']['sfA']
                        if len(sf) >= 2:
                            wave.SetScale('x', sf[0][1], sf[0][0])  # offset, delta
                        if len(sf) >= 4:
                            wave.SetScale('y', sf[1][1], sf[1][0])

                    print(f"Loaded Igor wave: {wave_data.shape}, {wave_data.dtype}")
                    return wave

                except Exception as e:
                    print(f"Error loading Igor file: {e}")
                    return None
            else:
                print("Igor package not available for .ibw files")
                return None

        elif file_ext in ['.tif', '.tiff']:
            # TIFF format
            if TIFFFILE_AVAILABLE:
                try:
                    data = tifffile.imread(str(file_path))
                    # Handle different TIFF formats
                    if data.dtype == np.uint8:
                        data = data.astype(np.float64) / 255.0
                    elif data.dtype == np.uint16:
                        data = data.astype(np.float64) / 65535.0
                    elif data.dtype == np.uint32:
                        data = data.astype(np.float64) / (2 ** 32 - 1)
                    else:
                        data = data.astype(np.float64)

                    # Handle RGB/RGBA images
                    if len(data.shape) == 3:
                        if data.shape[2] == 3:  # RGB
                            data = np.mean(data, axis=2)  # Convert to grayscale
                        elif data.shape[2] == 4:  # RGBA
                            data = np.mean(data[:, :, :3], axis=2)  # Ignore alpha

                    wave = Wave(data, file_path.stem)
                    print(f"Loaded TIFF: {data.shape}, {data.dtype}")
                    return wave

                except Exception as e:
                    print(f"Error loading TIFF with tifffile: {e}")
                    # Fallback to PIL
                    pass

            # Fallback to PIL for TIFF
            if PIL_AVAILABLE:
                try:
                    img = Image.open(file_path)
                    if img.mode != 'L':
                        img = img.convert('L')  # Convert to grayscale
                    data = np.array(img, dtype=np.float64) / 255.0
                    wave = Wave(data, file_path.stem)
                    print(f"Loaded TIFF with PIL: {data.shape}")
                    return wave
                except Exception as e:
                    print(f"Error loading TIFF with PIL: {e}")

        elif file_ext in ['.png', '.jpg', '.jpeg']:
            # Standard image formats
            if PIL_AVAILABLE:
                try:
                    img = Image.open(file_path)
                    if img.mode != 'L':
                        img = img.convert('L')  # Convert to grayscale
                    data = np.array(img, dtype=np.float64) / 255.0
                    wave = Wave(data, file_path.stem)
                    print(f"Loaded image: {data.shape}")
                    return wave
                except Exception as e:
                    print(f"Error loading with PIL: {e}")

            # Fallback to scikit-image
            if SKIMAGE_AVAILABLE:
                try:
                    data = skimage_io.imread(str(file_path), as_gray=True)
                    data = data.astype(np.float64)
                    wave = Wave(data, file_path.stem)
                    print(f"Loaded with scikit-image: {data.shape}")
                    return wave
                except Exception as e:
                    print(f"Error loading with scikit-image: {e}")

        else:
            print(f"Unsupported file format: {file_ext}")
            return None

    except Exception as e:
        print(f"Error loading image {file_path}: {str(e)}")
        return None

    print(f"Failed to load image: {file_path}")
    return None


def save_image(wave, file_path, format='auto'):
    """
    Save a Wave object as an image file
    Supports TIFF, PNG, JPEG formats
    """
    try:
        file_path = Path(file_path)

        if format == 'auto':
            format = file_path.suffix.lower()

        # Normalize data to 0-1 range
        data = wave.data.copy()
        if np.max(data) > np.min(data):
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            data = np.zeros_like(data)

        if format in ['.tif', '.tiff']:
            if TIFFFILE_AVAILABLE:
                # Save as 32-bit float TIFF
                tifffile.imwrite(str(file_path), data.astype(np.float32))
            elif PIL_AVAILABLE:
                # Convert to 16-bit for PIL
                image_data = (data * 65535).astype(np.uint16)
                img = Image.fromarray(image_data, mode='I;16')
                img.save(file_path)
            else:
                print("No TIFF library available")
                return False
        else:
            # Convert to uint8 for other formats
            image_data = (data * 255).astype(np.uint8)
            if PIL_AVAILABLE:
                img = Image.fromarray(image_data, mode='L')
                img.save(file_path)
            else:
                print("PIL not available for saving images.")
                return False

        return True

    except Exception as e:
        print(f"Error saving image {file_path}: {str(e)}")
        return False


# Igor Pro compatibility functions
def GetBrowserSelection(index=0):
    """
    Igor GetBrowserSelection equivalent
    Returns the path to the currently selected folder
    """
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
    parts = [p for p in path.split(':') if p]  # Remove empty parts
    current_folder = data_browser

    for part in parts:
        if part not in current_folder.subfolders:
            current_folder.add_subfolder(part)
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

    if dest_folder is None:
        return

    # Copy all waves
    for wave_name, wave in source_folder.waves.items():
        # Create a copy of the wave
        from igor_compatibility import Duplicate
        new_wave = Duplicate(wave, wave_name)
        dest_folder.add_wave(new_wave)


def KillDataFolder(path):
    """
    Igor KillDataFolder equivalent
    """
    global data_browser
    if path == "/Z":  # Special flag to not show errors
        return

    parts = [p for p in path.split(':') if p]
    if not parts:
        return

    # Navigate to parent folder
    current_folder = data_browser
    for part in parts[:-1]:
        if part in current_folder.subfolders:
            current_folder = current_folder.subfolders[part]
        else:
            return  # Folder doesn't exist

    # Remove the target folder
    folder_to_remove = parts[-1]
    if folder_to_remove in current_folder.subfolders:
        del current_folder.subfolders[folder_to_remove]


def UniqueName(base_name, object_type, index):
    """
    Igor UniqueName equivalent
    Returns a unique name for a new object
    """
    global data_browser
    counter = 0
    while True:
        if counter == 0:
            candidate_name = base_name.rstrip('_')
        else:
            candidate_name = f"{base_name.rstrip('_')}_{counter}"

        # Check if name exists
        if object_type == 11:  # Data folder
            if candidate_name not in data_browser.subfolders:
                return candidate_name
        else:  # Wave
            if candidate_name not in data_browser.waves:
                return candidate_name

        counter += 1
        if counter > 1000:  # Prevent infinite loop
            break

    return f"{base_name}_{counter}"


def LoadWave(file_path):
    """
    Igor LoadWave equivalent
    Load a wave from file into the data browser
    """
    global data_browser

    wave = load_image(file_path)
    if wave is not None:
        data_browser.add_wave(wave)
        return True
    return False


def SaveWave(wave, file_path):
    """
    Igor Save equivalent for waves
    """
    return save_image(wave, file_path)


def WaveList(match_string="*", separator=";", folder_path=""):
    """
    Igor WaveList equivalent
    Returns a list of wave names matching the pattern
    """
    global data_browser

    if folder_path:
        folder = data_browser.get_folder(folder_path)
        if folder is None:
            return ""
    else:
        folder = data_browser

    import fnmatch
    wave_names = []

    for name in folder.waves.keys():
        if fnmatch.fnmatch(name, match_string):
            wave_names.append(name)

    return separator.join(wave_names)


def DataFolderDir(flag=1):
    """
    Igor DataFolderDir equivalent
    Returns information about the current data folder
    """
    global data_browser

    if flag == 1:  # List data folders
        folder_names = list(data_browser.subfolders.keys())
        return ";".join(folder_names)
    elif flag == 4:  # List waves
        wave_names = list(data_browser.waves.keys())
        return ";".join(wave_names)
    else:
        return ""


def GetWavesDataFolder(wave_ref, flag):
    """
    Igor GetWavesDataFolder equivalent
    """
    if flag == 1:
        return "root:"
    return "root"


def MoveWave(wave, dest_folder_path):
    """
    Igor MoveWave equivalent
    Move a wave to a different data folder
    """
    global data_browser

    # Remove from current location
    found = False
    for folder in [data_browser] + list(data_browser.subfolders.values()):
        if wave.name in folder.waves:
            del folder.waves[wave.name]
            found = True
            break

    if not found:
        return False

    # Add to destination
    dest_folder = data_browser.get_folder(dest_folder_path.rstrip(':'))
    if dest_folder is not None:
        dest_folder.add_wave(wave)
        return True

    return False


def check_file_io_dependencies():
    """Check what file I/O libraries are available"""
    print("File I/O Library Status:")
    print(f"  Igor package: {'Available' if IGOR_AVAILABLE else 'Not available (pip install igor)'}")
    print(f"  PIL/Pillow: {'Available' if PIL_AVAILABLE else 'Not available (pip install Pillow)'}")
    print(f"  tifffile: {'Available' if TIFFFILE_AVAILABLE else 'Not available (pip install tifffile)'}")
    print(f"  scikit-image: {'Available' if SKIMAGE_AVAILABLE else 'Not available (pip install scikit-image)'}")
    print()

    if not any([IGOR_AVAILABLE, PIL_AVAILABLE, TIFFFILE_AVAILABLE, SKIMAGE_AVAILABLE]):
        print("WARNING: No image reading libraries available!")
        print("Please install at least one of: igor, Pillow, tifffile, or scikit-image")
        return False

    return True


def GetFileFolderInfo(file_path):
    """
    Get information about a file or folder
    Igor GetFileFolderInfo equivalent
    """
    path = Path(file_path)

    if not path.exists():
        return {
            'exists': False,
            'isFolder': False,
            'isFile': False,
            'size': 0,
            'modification': 0
        }

    import os
    stat = path.stat()

    return {
        'exists': True,
        'isFolder': path.is_dir(),
        'isFile': path.is_file(),
        'size': stat.st_size,
        'modification': stat.st_mtime
    }


def IndexedFile(base_path, index, extension=""):
    """
    Generate indexed file names
    Igor IndexedFile equivalent
    """
    if extension and not extension.startswith('.'):
        extension = '.' + extension

    base = Path(base_path)
    if index == -1:
        # Find next available index
        index = 0
        while True:
            candidate = base.parent / f"{base.stem}_{index:03d}{extension}"
            if not candidate.exists():
                return str(candidate)
            index += 1
            if index > 9999:  # Prevent infinite loop
                break
    else:
        candidate = base.parent / f"{base.stem}_{index:03d}{extension}"
        return str(candidate)


def PathInfo(path_string):
    """
    Get path information
    Igor PathInfo equivalent
    """
    path = Path(path_string)

    return {
        'path': str(path),
        'name': path.name,
        'extension': path.suffix,
        'folder': str(path.parent),
        'exists': path.exists()
    }


# Initialize on import
if __name__ == "__main__":
    check_file_io_dependencies()