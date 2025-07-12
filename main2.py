"""
Main entry point for Hessian Blob Particle Detection Suite
Complete Igor Pro port with GUI application
"""

# !/usr/bin/env python3

import sys
import os
import warnings

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend before any GUI imports
import matplotlib

matplotlib.use('TkAgg')
matplotlib.rcParams['figure.raise_window'] = False

# Import main application
from hessian_blobs.main_functions import HessianBlobGUI


def main():
    """Main function to run the Hessian Blob Detection Suite"""
    try:
        print("Starting Hessian Blob Particle Detection Suite...")
        print("Python Port of Igor Pro Code by Brendan Marsh")
        print("G.M. King Laboratory, University of Missouri-Columbia")
        print("=" * 60)

        # Create and run application
        app = HessianBlobGUI()
        app.run()

    except Exception as e:
        print(f"Application Error: {e}")
        import traceback
        traceback.print_exc()

        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", str(e))
        except:
            pass


if __name__ == "__main__":
    main()
