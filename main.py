"""
Hessian Blob Particle Detection Suite - Python Port

This is the main entry point for the application. It applies necessary
patches for library compatibility and launches the main GUI.
"""

# #######################################################################
# #######################################################################
#
#             Hessian Blob Particle Detection Suite
#
# G.M. King Laboratory
# University of Missouri-Columbia
#
# Originally created by: Brendan Marsh
# Python Port by: Riley Johnson
#
#   CONTENTS:
#       - Numpy Deprecation Monkey Patch
#       - main(): The primary function to launch the application.
#
# #######################################################################
# #######################################################################

import numpy as np
import matplotlib
import tkinter as tk
from tkinter import messagebox

if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Now, import the rest of the application
from gui.main_window import HessianBlobGUI
from utils.error_handler import handle_error


def main():
    """Main function to run the Hessian Blob Detection Suite."""
    try:
        print("Starting Hessian Blob Detection Suite...")

        # Set matplotlib to use a thread-safe backend before any plotting
        matplotlib.use('TkAgg')

        # Create and run the application
        app = HessianBlobGUI()
        app.run()

    except Exception as e:
        error_msg = handle_error("main", e)
        try:
            # Use tkinter messagebox if GUI is available
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", error_msg)
        except:
            # Fallback to console print if GUI fails
            print(error_msg)


if __name__ == "__main__":
    main()