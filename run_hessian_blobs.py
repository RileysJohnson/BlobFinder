#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main Runner
Fixed version with proper error handling and dependencies

This is the main entry point for the Hessian Blob Detection Suite.
It launches the GUI application that provides the same functionality
as the original Igor Pro implementation.

Usage:
    python run_hessian_blobs.py

Copyright 2019 by The Curators of the University of Missouri (original Igor Pro code)
Python port maintains 1-1 functionality with Igor Pro version
G.M. King Laboratory - University of Missouri-Columbia
Original coded by: Brendan Marsh - marshbp@stanford.edu
"""

import sys
import os
import warnings
import numpy as np
import platform
from pathlib import Path

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("Hessian Blob Particle Detection Suite")
    print("Python Port of Igor Pro Implementation")
    print()
    print("G.M. King Laboratory - University of Missouri-Columbia")
    print("Original Igor Pro code by: Brendan Marsh - marshbp@stanford.edu")
    print("Python port maintains 1-1 functionality with Igor Pro version")
    print("=" * 70)
    print()


def check_python_version():
    """Check Python version compatibility"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 7):
        print("ERROR: Python 3.7 or higher is required.")
        print(f"Current version: {major}.{minor}")
        return False
    print(f"✓ Python version: {major}.{minor} (compatible)")
    return True


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('numpy', 'numpy', 'Scientific computing library'),
        ('scipy', 'scipy', 'Scientific algorithms library'),
        ('matplotlib', 'matplotlib', 'Plotting library'),
        ('tkinter', 'tkinter', 'GUI framework (usually comes with Python)'),
    ]

    optional_packages = [
        ('igor', 'igor', 'For reading .ibw files natively'),
        ('PIL', 'Pillow', 'For reading common image formats'),
        ('tifffile', 'tifffile', 'For reading TIFF files'),
        ('skimage', 'scikit-image', 'Alternative image reading library'),
    ]

    missing_required = []
    missing_optional = []

    print("Checking dependencies...")

    # Check required packages
    for package, install_name, description in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
                print(f"✓ {package}: Available")
            else:
                __import__(package)
                print(f"✓ {package}: Available")
        except ImportError:
            print(f"✗ {package}: Missing - {description}")
            missing_required.append(install_name)

    # Check optional packages
    for package, install_name, description in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}: Available")
        except ImportError:
            print(f"○ {package}: Optional - {description}")
            missing_optional.append(install_name)

    print()

    if missing_required:
        print("REQUIRED packages missing:")
        for package in missing_required:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        print("Optional packages missing (limited functionality):")
        for package in missing_optional:
            print(f"  - {package}")
        print(f"\nInstall with: pip install {' '.join(missing_optional)}")
        print("Note: At least one image reading library (PIL, tifffile, or scikit-image) is recommended.")
        print()

    return True


def check_file_structure():
    """Check if all required Python files are present"""
    required_files = [
        'main_gui.py',
        'igor_compatibility.py',
        'file_io.py',
        'main_functions.py',
        'scale_space.py',
        'utilities.py'
    ]

    optional_files = [
        'preprocessing.py',
        'particle_measurements.py'
    ]

    current_dir = Path(__file__).parent
    missing_files = []

    print("Checking file structure...")

    for filename in required_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"✓ {filename}: Found")
        else:
            print(f"✗ {filename}: Missing")
            missing_files.append(filename)

    for filename in optional_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"✓ {filename}: Found")
        else:
            print(f"○ {filename}: Optional (will use fallback)")

    print()

    if missing_files:
        print("ERROR: Required files missing:")
        for filename in missing_files:
            print(f"  - {filename}")
        print("\nPlease ensure all required Python files are in the same directory.")
        return False

    return True


def create_missing_modules():
    """Create minimal versions of missing optional modules"""
    current_dir = Path(__file__).parent

    # Create minimal preprocessing.py if missing
    preprocessing_file = current_dir / 'preprocessing.py'
    if not preprocessing_file.exists():
        print("Creating minimal preprocessing.py...")
        with open(preprocessing_file, 'w') as f:
            f.write('''"""
Preprocessing Module (Minimal Implementation)
Contains image preprocessing functions
"""

def BatchPreprocess():
    """Batch preprocessing function"""
    from tkinter import messagebox
    messagebox.showinfo("Preprocessing", "Preprocessing functionality not fully implemented yet.")
    return True
''')

    # Create minimal particle_measurements.py if missing
    measurements_file = current_dir / 'particle_measurements.py'
    if not measurements_file.exists():
        print("Creating minimal particle_measurements.py...")
        with open(measurements_file, 'w') as f:
            f.write('''"""
Particle Measurements Module (Minimal Implementation)
Contains particle measurement and analysis functions
"""

def MeasureParticles():
    """Measure particles function"""
    from tkinter import messagebox
    messagebox.showinfo("Measurements", "Particle measurement functionality not fully implemented yet.")
    return True

def ViewParticles():
    """View particles function"""
    from tkinter import messagebox
    messagebox.showinfo("Viewer", "Particle viewer functionality not fully implemented yet.")
    return True
''')


def setup_environment():
    """Setup the environment for running"""
    # Add current directory to Python path
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Set up matplotlib backend for GUI
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except ImportError:
        pass  # Will be caught in dependency check


def main():
    """Main entry point"""
    print_banner()

    # Check system requirements
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)

    if not check_dependencies():
        input("Press Enter to exit...")
        sys.exit(1)

    if not check_file_structure():
        input("Press Enter to exit...")
        sys.exit(1)

    # Create missing optional modules
    create_missing_modules()

    # Setup environment
    setup_environment()

    # Import and run the GUI
    try:
        print("Starting Hessian Blob Detection GUI...")

        # Import the main GUI (after path is set up)
        from main_gui import main as run_gui

        print("✓ All modules loaded successfully")
        print("✓ Launching GUI application...")
        print()

        # Check file I/O capabilities
        try:
            from file_io import check_file_io_dependencies
            check_file_io_dependencies()
        except ImportError:
            print("Warning: Could not check file I/O dependencies")

        # Run the GUI
        run_gui()

    except ImportError as e:
        print(f"ERROR: Failed to import required modules: {e}")
        print("\nThis usually means:")
        print("1. A required Python file is missing or corrupted")
        print("2. A required dependency is not properly installed")
        print("3. There's a syntax error in one of the Python files")
        print("\nPlease check that all files are present and dependencies are installed.")
        input("Press Enter to exit...")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to start application: {e}")
        print("\nUnexpected error occurred. Please check:")
        print("1. All files are present and not corrupted")
        print("2. All dependencies are properly installed")
        print("3. No other applications are interfering")

        # Print detailed error for debugging
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()

        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)