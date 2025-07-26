"""
File I/O Module
Handles loading and saving of various image file formats
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete IBW support and proper error handling
"""

import numpy as np
from pathlib import Path
import warnings

# Try to import optional packages for different file formats
# The igor package can be imported in different ways depending on version
IGOR_AVAILABLE = False
bw = None

try:
    # Try newer igor package structure first
    import igor.binarywave as bw

    IGOR_AVAILABLE = True
except ImportError:
    try:
        # Try older structure
        import binarywave as bw

        IGOR_AVAILABLE = True
    except ImportError:
        try:
            # Try direct igor import
            import igor

            if hasattr(igor, 'load'):
                bw = igor
                IGOR_AVAILABLE = True
        except ImportError:
            IGOR_AVAILABLE = False
            bw = None

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

from igor_compatibility import Wave

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


class DataFolder:
    """Mimics Igor Pro data folder structure"""

    def __init__(self, name=""):
        self.name = name
        self.waves = {}
        self.subfolders = {}
        self.variables = {}
        self.strings = {}

    def add_wave(self, wave):
        """Add a wave to this folder"""
        self.waves[wave.name] = wave

    def get_wave(self, name):
        """Get a wave by name"""
        return self.waves.get(name, None)

    def add_subfolder(self, name):
        """Add a subfolder"""
        if name not in self.subfolders:
            self.subfolders[name] = DataFolder(name)
        return self.subfolders[name]

    def get_folder(self, path):
        """Get a folder by path (e.g., 'folder1:subfolder2')"""
        if not path or path == "" or path == ":":
            return self

        parts = path.split(':')
        current = self

        for part in parts:
            if part and part in current.subfolders:
                current = current.subfolders[part]
            elif part:
                # Create folder if it doesn't exist
                current = current.add_subfolder(part)

        return current


# Global data browser (Igor Pro root folder equivalent)
data_browser = DataFolder("root")


def LoadWave(file_path):
    """
    Load an image file and return as Wave object
    Supports multiple formats: Igor IBW, TIFF, PNG, JPEG
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    try:
        file_ext = file_path.suffix.lower()

        if file_ext == '.ibw':
            # Igor Binary Wave format - only show error once per session
            if not hasattr(LoadWave, '_igor_error_shown'):
                LoadWave._igor_error_shown = False

            if IGOR_AVAILABLE and bw is not None:
                try:
                    print(f"Loading Igor binary wave: {file_path}")

                    # Try different loading methods based on igor package version
                    data = None
                    if hasattr(bw, 'load'):
                        data = bw.load(str(file_path))
                    elif hasattr(bw, 'read'):
                        data = bw.read(str(file_path))
                    else:
                        if not LoadWave._igor_error_shown:
                            print(f"Error: Igor package loaded but no load/read method found")
                            LoadWave._igor_error_shown = True
                        return None

                    # Extract wave data with multiple fallback methods
                    wave_data = None

                    # Method 1: Standard igor format
                    if isinstance(data, dict) and 'wave' in data and 'wData' in data['wave']:
                        wave_data = data['wave']['wData']
                    # Method 2: Direct wData
                    elif isinstance(data, dict) and 'wData' in data:
                        wave_data = data['wData']
                    # Method 3: Object with data attribute
                    elif hasattr(data, 'data'):
                        wave_data = data.data
                    # Method 4: Direct numpy array
                    elif isinstance(data, np.ndarray):
                        wave_data = data
                    # Method 5: Try to find any array-like data
                    elif isinstance(data, dict):
                        # Look for any array-like data in the dictionary
                        for key, value in data.items():
                            if isinstance(value, np.ndarray) and value.size > 1:
                                wave_data = value
                                print(f"Found data in key: {key}")
                                break

                    if wave_data is None:
                        if not LoadWave._igor_error_shown:
                            print("Error: Could not extract wave data from IBW file")
                            print(f"Data structure: {type(data)}")
                            if isinstance(data, dict):
                                print(f"Available keys: {list(data.keys())}")
                            LoadWave._igor_error_shown = True
                        return None

                    # Handle different data types
                    if wave_data.dtype == np.complex64 or wave_data.dtype == np.complex128:
                        print("Converting complex data to magnitude")
                        wave_data = np.abs(wave_data)  # Take magnitude for complex data

                    # Ensure 2D for images
                    if wave_data.ndim == 1:
                        # Try to reshape if it's a flattened 2D image
                        size = wave_data.shape[0]
                        sqrt_size = int(np.sqrt(size))
                        if sqrt_size * sqrt_size == size:
                            wave_data = wave_data.reshape(sqrt_size, sqrt_size)
                            print(f"Reshaped 1D data to 2D: {wave_data.shape}")
                        else:
                            if not LoadWave._igor_error_shown:
                                print(f"Warning: 1D wave with {size} points, cannot auto-reshape to 2D")
                                LoadWave._igor_error_shown = True
                            return None
                    elif wave_data.ndim > 2:
                        print(f"Warning: {wave_data.ndim}D data, using first 2D slice")
                        if wave_data.ndim == 3:
                            wave_data = wave_data[:, :, 0]
                        else:
                            wave_data = wave_data.squeeze()
                            if wave_data.ndim > 2:
                                wave_data = wave_data[:, :, 0]

                    # Convert to float64 for consistency
                    wave_data = wave_data.astype(np.float64)

                    wave = Wave(wave_data, file_path.stem)

                    # Set scaling if available (with better error handling)
                    try:
                        if isinstance(data, dict) and 'wave' in data:
                            wave_info = data['wave']

                            # Try to get scaling factors
                            if 'sfA' in wave_info and wave_info['sfA'] is not None:
                                sf = wave_info['sfA']
                                if len(sf) >= 2 and len(sf[0]) >= 2:
                                    # sf[0] = [delta, offset] for X dimension
                                    wave.SetScale('x', sf[0][1], sf[0][0])  # offset, delta
                                if len(sf) >= 4 and len(sf[1]) >= 2:
                                    # sf[1] = [delta, offset] for Y dimension
                                    wave.SetScale('y', sf[1][1], sf[1][0])  # offset, delta

                            # Try to get units
                            if 'dimUnits' in wave_info and wave_info['dimUnits'] is not None:
                                units = wave_info['dimUnits']
                                if len(units) >= 1 and units[0]:
                                    unit_str = str(units[0][0]) if isinstance(units[0], (list, tuple)) else str(
                                        units[0])
                                    wave.SetScale('x', wave.GetScale('x')['offset'],
                                                  wave.GetScale('x')['delta'], unit_str)
                                if len(units) >= 2 and units[1]:
                                    unit_str = str(units[1][0]) if isinstance(units[1], (list, tuple)) else str(
                                        units[1])
                                    wave.SetScale('y', wave.GetScale('y')['offset'],
                                                  wave.GetScale('y')['delta'], unit_str)
                    except Exception as e:
                        print(f"Warning: Could not set scaling from IBW file: {e}")

                    print(f"Successfully loaded Igor wave: {wave_data.shape}, {wave_data.dtype}")
                    return wave

                except Exception as e:
                    if not LoadWave._igor_error_shown:
                        print(f"Error loading Igor file: {e}")
                        print(f"File may be corrupted or unsupported IBW version")
                        LoadWave._igor_error_shown = True
                    return None
            else:
                if not LoadWave._igor_error_shown:
                    print(f"Error: Igor package not available for loading {file_path.name}")
                    print("Install with: pip install igor")
                    LoadWave._igor_error_shown = True
                return None

        elif file_ext in ['.tif', '.tiff']:
            # TIFF format
            if TIFFFILE_AVAILABLE:
                try:
                    print(f"Loading TIFF file: {file_path}")
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

                    # Handle multi-dimensional TIFF
                    if data.ndim > 2:
                        if data.shape[2] == 3 or data.shape[2] == 4:  # RGB or RGBA
                            # Convert to grayscale
                            if data.shape[2] == 3:
                                data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
                            else:  # RGBA
                                data = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
                        else:
                            data = data[:, :, 0]  # Take first channel

                    wave = Wave(data, file_path.stem)
                    print(f"Successfully loaded TIFF: {data.shape}, {data.dtype}")
                    return wave

                except Exception as e:
                    print(f"Error loading TIFF with tifffile: {e}")
                    # Fallback to PIL

            # Try PIL as fallback for TIFF
            if PIL_AVAILABLE:
                try:
                    print(f"Trying PIL for TIFF: {file_path}")
                    img = Image.open(file_path)
                    if img.mode != 'L':
                        img = img.convert('L')  # Convert to grayscale
                    data = np.array(img, dtype=np.float64) / 255.0
                    wave = Wave(data, file_path.stem)
                    print(f"Successfully loaded TIFF with PIL: {data.shape}")
                    return wave
                except Exception as e:
                    print(f"Error loading TIFF with PIL: {e}")

        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
            # Standard image formats
            if PIL_AVAILABLE:
                try:
                    print(f"Loading image file: {file_path}")
                    img = Image.open(file_path)

                    # Convert to grayscale if needed
                    if img.mode != 'L':
                        img = img.convert('L')

                    # Convert to numpy array
                    data = np.array(img, dtype=np.float64) / 255.0

                    wave = Wave(data, file_path.stem)
                    print(f"Successfully loaded image: {data.shape}, {data.dtype}")
                    return wave

                except Exception as e:
                    print(f"Error loading image with PIL: {e}")

            # Try scikit-image as fallback
            elif SKIMAGE_AVAILABLE:
                try:
                    print(f"Trying scikit-image for: {file_path}")
                    data = skimage_io.imread(str(file_path), as_gray=True)
                    data = data.astype(np.float64)
                    if data.max() > 1.0:
                        data = data / data.max()  # Normalize

                    wave = Wave(data, file_path.stem)
                    print(f"Successfully loaded with scikit-image: {data.shape}")
                    return wave
                except Exception as e:
                    print(f"Error loading with scikit-image: {e}")

        else:
            print(f"Unsupported file format: {file_ext}")
            print("Supported formats: .ibw, .tif, .tiff, .png, .jpg, .jpeg")
            return None

    except Exception as e:
        print(f"Unexpected error loading file {file_path}: {e}")
        return None

    print(f"Failed to load file: {file_path}")
    return None


def SaveWave(wave, file_path, format="tiff"):
    """
    Save a wave to file
    """
    file_path = Path(file_path)

    try:
        if format.lower() in ["tiff", "tif"]:
            if TIFFFILE_AVAILABLE:
                # Convert to appropriate format for saving
                data = wave.data.copy()
                if data.dtype == np.float64 or data.dtype == np.float32:
                    # Scale to 16-bit if floating point
                    data = (data * 65535).astype(np.uint16)
                elif data.dtype == np.bool_:
                    data = data.astype(np.uint8) * 255

                tifffile.imwrite(str(file_path), data)
                print(f"Saved wave to: {file_path}")
                return True
            else:
                print("tifffile package required for TIFF saving")
                return False

        elif format.lower() in ["png", "jpg", "jpeg"]:
            if PIL_AVAILABLE:
                # Convert to PIL Image
                data = wave.data.copy()
                if data.dtype == np.float64 or data.dtype == np.float32:
                    data = (data * 255).astype(np.uint8)

                img = Image.fromarray(data, mode='L')
                img.save(str(file_path))
                print(f"Saved wave to: {file_path}")
                return True
            else:
                print("PIL package required for PNG/JPEG saving")
                return False
        else:
            print(f"Unsupported save format: {format}")
            return False

    except Exception as e:
        print(f"Error saving wave: {e}")
        return False


def NewDataFolder(folder_path):
    """
    Igor NewDataFolder equivalent
    Create a new data folder
    """
    global data_browser

    # Remove leading/trailing colons and split path
    clean_path = folder_path.strip(':')
    if not clean_path:
        return data_browser

    return data_browser.get_folder(clean_path)


def SetDataFolder(folder_path):
    """
    Igor SetDataFolder equivalent (simplified)
    """
    global data_browser
    folder = data_browser.get_folder(folder_path.strip(':'))
    return folder is not None


def Duplicate(source_wave, new_name):
    """
    Igor Duplicate equivalent
    Create a copy of a wave with a new name
    """
    new_wave = Wave(source_wave.data.copy(), new_name, source_wave.note)

    # Copy scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


def KillWaves(*wave_names):
    """
    Igor KillWaves equivalent
    Remove waves from memory (simplified - just removes from global references)
    """
    global data_browser

    for name in wave_names:
        # Remove from all folders
        for folder in [data_browser] + list(data_browser.subfolders.values()):
            if name in folder.waves:
                del folder.waves[name]


def WaveExists(wave_name):
    """
    Igor WaveExists equivalent
    Check if a wave exists
    """
    global data_browser

    # Check in root and all subfolders
    for folder in [data_browser] + list(data_browser.subfolders.values()):
        if wave_name in folder.waves:
            return True
    return False


def DataFolderDir(flag):
    """
    Igor DataFolderDir equivalent
    """
    global data_browser

    if flag == 1:  # List folders
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

    # Check Igor package more thoroughly
    igor_status = "Not available"
    if IGOR_AVAILABLE and bw is not None:
        try:
            if hasattr(bw, 'load') or hasattr(bw, 'read'):
                igor_status = "Available"
            else:
                igor_status = "Available but no load method"
        except:
            igor_status = "Available but not functional"

    print(f"  Igor package: {igor_status} {'(pip install igor)' if igor_status == 'Not available' else ''}")
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

    return f"{base_path}{index:04d}{extension}"


def test_igor_package():
    """Test igor package functionality"""
    print("Testing Igor package...")

    if not IGOR_AVAILABLE or bw is None:
        print("❌ Igor package not available")
        return False

    print(f"✓ Igor package imported successfully")
    print(f"  Module: {bw.__name__ if hasattr(bw, '__name__') else 'unknown'}")
    print(f"  Has load method: {hasattr(bw, 'load')}")
    print(f"  Has read method: {hasattr(bw, 'read')}")

    if hasattr(bw, 'load'):
        print("✓ Using bw.load() method")
    elif hasattr(bw, 'read'):
        print("✓ Using bw.read() method")
    else:
        print("❌ No suitable load method found")
        return False

    return True


def Testing(string_input, number_input):
    """Testing function for file_io module"""
    print(f"File I/O testing: {string_input}, {number_input}")
    # Also test igor package if this is called
    if not test_igor_package():
        print("⚠️  Igor package test failed")
    return len(string_input) + number_input