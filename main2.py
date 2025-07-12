# ========================================================================
# main.py - Main entry point
# ========================================================================

# !/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Python Port
Copyright 2019 by The Curators of the University of Missouri, a public corporation

G.M. King Laboratory
University of Missouri-Columbia
Originally created by: Brendan Marsh
Email: marshbp@stanford.edu
Ported by: Riley Johnson

Main entry point for the Hessian Blob Detection Suite.
Run this file to start the GUI application.
"""

import sys
import os
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure matplotlib for thread safety
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['figure.raise_window'] = False
plt.ioff()  # Turn off interactive mode

# Suppress warnings
warnings.filterwarnings('ignore')

from gui.main_window import HessianBlobGUI
from core.error_handling import handle_error, safe_print


def main():
    """Main function to run the Hessian Blob Detection Suite."""
    try:
        safe_print("Starting Hessian Blob Detection Suite...")

        # Create and run application
        app = HessianBlobGUI()
        app.run()

    except Exception as e:
        error_msg = handle_error("main", e)
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Application Error", error_msg)
        except:
            print(error_msg)


if __name__ == "__main__":
    main()