"""
File I/O Module
Handles reading various image file formats and managing data folder structures
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


# Global data browser (simulates Igor's data browser)
data_browser = DataFolder("root")


def load_image_file(file_path):
    """
    Load an image file and return a Wave object
    Supports .ibw, .tif/.tiff, .png, .jpg/.jpeg formats
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return None

    try:
        print(f"Loading file: {file_path}")

        # Determine file type and load accordingly
        suffix = file_path.suffix.lower()

        if suffix == '.ibw':
            wave = load_ibw_file(file_path)
        elif suffix in ['.tif', '.tiff']:
            wave = load_tiff_file(file_path)
        elif suffix in ['.png', '.jpg', '.jpeg']:
            wave = load_standard_image(file_path)
        else:
            # Try to load as generic image
            wave = load_standard_image(file_path)

        if wave is not None:
            # Ensure unique wave name based on file
            wave.name = file_path.stem  # Use just the filename without extension
            print(f"Successfully loaded: {wave.name}, shape: {wave.data.shape}")

        return wave

    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None


def load_ibw_file(file_path):
    """Load Igor binary wave (.ibw) file"""
    if not IGOR_AVAILABLE:
        print("Igor package not available. Cannot load .ibw files.")
        print("Install with: pip install igor")
        return None

    try:
        # Load using igor package
        ibw_data = bw.load(str(file_path))
        wave_data = ibw_data['wave']['wData']

        # Get wave name from filename
        wave_name = file_path.stem

        # Create Wave object
        wave = Wave(wave_data, wave_name)

        # Set scaling if available
        if 'wave' in ibw_data and 'sfA' in ibw_data['wave']:
            sf_a = ibw_data['wave']['sfA']
            if len(sf_a) >= 2:
                wave.SetScale('x', sf_a[0][1], sf_a[0][0])  # offset, delta
                if len(sf_a) >= 2 and len(wave_data.shape) >= 2:
                    wave.SetScale('y', sf_a[1][1], sf_a[1][0])

        return wave

    except Exception as e:
        print(f"Error loading .ibw file {file_path}: {str(e)}")
        # Try manual loading as fallback
        return load_ibw_manual(file_path)


def load_ibw_manual(file_path):
    """Manual .ibw file loading (simplified version)"""
    try:
        with open(file_path, 'rb') as f:
            # Read header
            header = f.read(64)

            # This is a very simplified .ibw reader
            # For production use, the igor package is recommended
            data_type = struct.unpack('<h', header[4:6])[0]

            if data_type == 2:  # 32-bit float
                data = np.frombuffer(f.read(), dtype=np.float32)
            elif data_type == 4:  # 64-bit float
                data = np.frombuffer(f.read(), dtype=np.float64)
            else:
                # Try as float32 by default
                data = np.frombuffer(f.read(), dtype=np.float32)

            # Try to determine dimensions (this is a guess)
            if len(data) == 256 * 256:
                data = data.reshape(256, 256)
            elif len(data) == 512 * 512:
                data = data.reshape(512, 512)
            else:
                # Try square root for square images
                side = int(np.sqrt(len(data)))
                if side * side == len(data):
                    data = data.reshape(side, side)

            wave_name = file_path.stem
            return Wave(data, wave_name)

    except Exception as e:
        print(f"Manual .ibw loading failed for {file_path}: {str(e)}")
        return None


def load_tiff_file(file_path):
    """Load TIFF file"""
    try:
        # Try tifffile first (better for scientific images)
        if TIFFFILE_AVAILABLE:
            data = tifffile.imread(str(file_path))
        elif SKIMAGE_AVAILABLE:
            data = skimage_io.imread(str(file_path))
        elif PIL_AVAILABLE:
            with Image.open(file_path) as img:
                data = np.array(img)
        else:
            print("No TIFF reading library available.")
            return None

        # Ensure we have at least 2D data
        if len(data.shape) == 1:
            # Try to guess dimensions
            side = int(np.sqrt(len(data)))
            if side * side == len(data):
                data = data.reshape(side, side)
            else:
                print(f"Cannot determine dimensions for 1D data in {file_path}")
                return None

        # Convert to float for consistency
        if data.dtype == np.uint8:
            data = data.astype(np.float64) / 255.0
        elif data.dtype == np.uint16:
            data = data.astype(np.float64) / 65535.0
        else:
            data = data.astype(np.float64)

        wave_name = file_path.stem
        wave = Wave(data, wave_name)

        # Set default scaling (1 pixel = 1 unit)
        wave.SetScale('x', 0, 1)
        wave.SetScale('y', 0, 1)

        return wave

    except Exception as e:
        print(f"Error loading TIFF file {file_path}: {str(e)}")
        return None


def load_standard_image(file_path):
    """Load standard image formats (PNG, JPEG, etc.)"""
    try:
        if PIL_AVAILABLE:
            with Image.open(file_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                data = np.array(img, dtype=np.float64) / 255.0
        elif SKIMAGE_AVAILABLE:
            data = skimage_io.imread(str(file_path))
            if len(data.shape) == 3:
                # Convert RGB to grayscale
                data = np.mean(data, axis=2)
            data = data.astype(np.float64) / 255.0
        else:
            print("No image reading library available.")
            return None

        wave_name = file_path.stem
        wave = Wave(data, wave_name)

        # Set default scaling (1 pixel = 1 unit)
        wave.SetScale('x', 0, 1)
        wave.SetScale('y', 0, 1)

        return wave

    except Exception as e:
        print(f"Error loading image file {file_path}: {str(e)}")
        return None


def save_wave_as_image(wave, file_path, format='tiff'):
    """Save a wave as an image file"""
    try:
        data = wave.data

        # Normalize data to 0-1 range
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if data_max > data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data

        # Convert to uint16 for better precision in TIFF
        if format.lower() == 'tiff':
            image_data = (normalized_data * 65535).astype(np.uint16)
            if TIFFFILE_AVAILABLE:
                tifffile.imwrite(str(file_path), image_data)
            elif PIL_AVAILABLE:
                img = Image.fromarray(image_data, mode='I;16')
                img.save(file_path)
            else:
                print("No TIFF writing library available.")
                return False
        else:
            # Convert to uint8 for other formats
            image_data = (normalized_data * 255).astype(np.uint8)
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


# Initialize on import
if __name__ == "__main__":
    check_file_io_dependencies()