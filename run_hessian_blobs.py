#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main Runner

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

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('skimage', 'scikit-image'),
        ('tkinter', 'tkinter (usually comes with Python)'),
        ('PIL', 'Pillow')
    ]

    missing_packages = []

    for package, install_name in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(install_name)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install " + " ".join(missing_packages))
        return False

    return True


def check_optional_dependencies():
    """Check optional dependencies and warn if missing"""
    optional_packages = [
        ('igor', 'igor - for reading .ibw files natively')
    ]

    missing_optional = []

    for package, description in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(description)

    if missing_optional:
        print("Optional packages not found (functionality may be limited):")
        for package in missing_optional:
            print(f"  - {package}")
        print("Install with: pip install igor")
        print()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Hessian Blob Particle Detection Suite")
    print("Python Port of Igor Pro Implementation")
    print("G.M. King Laboratory - University of Missouri-Columbia")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    check_optional_dependencies()

    # Import and run the GUI
    try:
        print("Starting Hessian Blob Detection GUI...")
        print("Loading modules...")

        # Import the main GUI
        from main_gui import main as run_gui

        print("Launching GUI application...")
        print()

        # Run the GUI
        run_gui()

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all Python files are in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()