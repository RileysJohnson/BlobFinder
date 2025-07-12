#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Python Port
Copyright 2019 by The Curators of the University of Missouri, a public corporation

G.M. King Laboratory
University of Missouri-Columbia
Originally created by: Brendan Marsh
Email: marshbp@stanford.edu
Ported by: Riley Johnson
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import pickle
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.fft import fft2, ifft2, fftfreq
from skimage import filters
from PIL import Image
import shutil
import json
import datetime
from typing import Tuple, Optional, Dict, Any, List
import sys
import os
import threading
import queue
import warnings
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

# Suppress matplotlib threading warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Agg.*')

# Fix numpy deprecation warnings
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

try:
    import igor.binarywave as bw
except ImportError:
    print("Warning: igor.binarywave not available. IBW files cannot be loaded.")
    bw = None

warnings.filterwarnings('ignore')


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


# Global variables
current_data_folder = "root:"
experiment_data = {}


# ========================================================================
# ERROR HANDLING AND LOGGING
# ========================================================================

class HessianBlobError(Exception):
    """Custom exception for Hessian Blob operations"""
    pass


def safe_print(message):
    """Thread-safe printing for GUI applications"""
    try:
        print(message)
    except Exception:
        pass


def handle_error(func_name: str, error: Exception, context: str = ""):
    """Centralized error handling with context"""
    error_msg = f"Error in {func_name}"
    if context:
        error_msg += f" ({context})"
    error_msg += f": {str(error)}"
    safe_print(error_msg)
    return error_msg


# ========================================================================
# PARAMETER VALIDATION
# ========================================================================

def validate_hessian_parameters(params):
    """Validate Hessian blob parameters"""
    required_params = 7
    if len(params) < required_params:
        raise HessianBlobError(f"Parameter array must contain at least {required_params} parameters")

    scale_start, layers, scale_factor, det_h_thresh, particle_type, subpixel_mult, allow_overlap = params[:7]

    # Validate ranges
    if scale_start <= 0:
        raise HessianBlobError("Minimum size must be positive")
    if layers <= 0:
        raise HessianBlobError("Maximum size must be positive")
    if scale_factor <= 1.0:
        raise HessianBlobError("Scaling factor must be greater than 1.0")
    if particle_type not in [-1, 0, 1]:
        raise HessianBlobError("Particle type must be -1, 0, or 1")
    if subpixel_mult < 1:
        raise HessianBlobError("Subpixel ratio must be >= 1")
    if allow_overlap not in [0, 1]:
        raise HessianBlobError("Allow overlap must be 0 or 1")

    return True


def validate_constraints(constraints):
    """Validate particle constraints"""
    if len(constraints) != 6:
        raise HessianBlobError("Constraints must contain 6 values: minH, maxH, minA, maxA, minV, maxV")

    min_h, max_h, min_a, max_a, min_v, max_v = constraints

    if min_h >= max_h and max_h != np.inf:
        raise HessianBlobError("Minimum height must be less than maximum height")
    if min_a >= max_a and max_a != np.inf:
        raise HessianBlobError("Minimum area must be less than maximum area")
    if min_v >= max_v and max_v != np.inf:
        raise HessianBlobError("Minimum volume must be less than maximum volume")

    return True


def validate_and_convert_parameters(params):
    """Validate and convert parameters exactly like Igor Pro"""
    try:
        # Extract parameters
        scaleStart = params[0]
        layers = params[1]
        scaleFactor = params[2]
        detHResponseThresh = params[3]
        particleType = int(params[4])
        subPixelMult = int(params[5])
        allowOverlap = int(params[6])

        # Additional validation matching Igor Pro exactly
        if scaleStart <= 0:
            raise HessianBlobError("Minimum Size must be positive")
        if layers <= 0:
            raise HessianBlobError("Maximum Size must be positive")
        if scaleFactor <= 1.0:
            raise HessianBlobError("Scaling Factor must be greater than 1.0")
        if particleType not in [-1, 0, 1]:
            raise HessianBlobError("Particle Type must be -1, 0, or 1")
        if subPixelMult < 1:
            raise HessianBlobError("Subpixel Ratio must be >= 1")
        if allowOverlap not in [0, 1]:
            raise HessianBlobError("Allow Overlap must be 0 or 1")

        # Parameter conversion
        dimDelta_im_0 = 1.0  # Pixel spacing

        # Convert scale parameters
        scaleStart_converted = (scaleStart * dimDelta_im_0) ** 2 / 2
        layers_converted = int(
            np.ceil(np.log((layers * dimDelta_im_0) ** 2 / (2 * scaleStart_converted)) / np.log(scaleFactor)))
        subPixelMult_converted = max(1, round(subPixelMult))
        scaleFactor_converted = max(1.1, scaleFactor)

        safe_print(f"Parameter conversion:")
        safe_print(f"  Original scaleStart: {scaleStart} -> {scaleStart_converted}")
        safe_print(f"  Original layers: {layers} -> {layers_converted}")
        safe_print(f"  Original scaleFactor: {scaleFactor} -> {scaleFactor_converted}")
        safe_print(f"  Original subPixelMult: {subPixelMult} -> {subPixelMult_converted}")

        return (scaleStart_converted, layers_converted, scaleFactor_converted,
                detHResponseThresh, particleType, subPixelMult_converted, allowOverlap)

    except Exception as e:
        handle_error("validate_and_convert_parameters", e)
        raise


def print_analysis_parameters(params):
    """Print analysis parameters like Igor Pro"""
    try:
        safe_print("\n" + "=" * 60)
        safe_print("HESSIAN BLOB ANALYSIS PARAMETERS")
        safe_print("=" * 60)

        param_names = [
            "Minimum Size in Pixels",
            "Maximum Size in Pixels",
            "Scaling Factor",
            "Minimum Blob Strength",
            "Particle Type",
            "Subpixel Ratio",
            "Allow Overlap",
            "Min Height Constraint",
            "Max Height Constraint",
            "Min Area Constraint",
            "Max Area Constraint",
            "Min Volume Constraint",
            "Max Volume Constraint"
        ]

        for i, (name, value) in enumerate(zip(param_names, params[:13])):
            if i == 3:  # Blob strength
                if value == -1:
                    safe_print(f"{i + 1:2d}. {name:<25}: Otsu's Method")
                elif value == -2:
                    safe_print(f"{i + 1:2d}. {name:<25}: Interactive Selection")
                else:
                    safe_print(f"{i + 1:2d}. {name:<25}: {value:.3e}")
            elif i == 4:  # Particle type
                type_str = {-1: "Negative only", 0: "Both", 1: "Positive only"}
                safe_print(f"{i + 1:2d}. {name:<25}: {value} ({type_str.get(value, 'Unknown')})")
            elif i == 6:  # Allow overlap
                overlap_str = {0: "No", 1: "Yes"}
                safe_print(f"{i + 1:2d}. {name:<25}: {value} ({overlap_str.get(value, 'Unknown')})")
            elif i >= 7:  # Constraints
                if value == -np.inf:
                    safe_print(f"{i + 1:2d}. {name:<25}: -∞ (no limit)")
                elif value == np.inf:
                    safe_print(f"{i + 1:2d}. {name:<25}: +∞ (no limit)")
                else:
                    safe_print(f"{i + 1:2d}. {name:<25}: {value:.3e}")
            else:
                safe_print(f"{i + 1:2d}. {name:<25}: {value}")

        safe_print("=" * 60)

    except Exception as e:
        handle_error("print_analysis_parameters", e)


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


# ========================================================================
# FILE I/O AND DATA MANAGEMENT
# ========================================================================

class DataManager:
    """Manages file I/O and data folder organization"""

    @staticmethod
    def create_igor_folder_structure(base_path: str, folder_type: str = "particles") -> str:
        """Create Igor Pro-compatible folder structure"""
        try:
            os.makedirs(base_path, exist_ok=True)

            if folder_type == "particles":
                # Create standard particle analysis structure
                subdirs = []
            elif folder_type == "series":
                # Series analysis doesn't need subdirs initially
                subdirs = []

            for subdir in subdirs:
                os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

            safe_print(f"Created Igor Pro folder structure: {base_path}")
            return base_path

        except Exception as e:
            raise HessianBlobError(f"Failed to create folder structure: {e}")

    @staticmethod
    def save_wave_data(data: np.ndarray, filepath: str, wave_info: Dict = None) -> bool:
        """Save wave data in Igor Pro-compatible format"""
        try:
            # Save as .npy for Python compatibility
            np.save(filepath, data)

            # Save metadata if provided
            if wave_info:
                metadata_file = filepath.replace('.npy', '_info.json')
                with open(metadata_file, 'w') as f:
                    json.dump(wave_info, f, indent=2, default=str)

            return True

        except Exception as e:
            handle_error("save_wave_data", e, f"file: {filepath}")
            return False

    @staticmethod
    def load_image_file(filepath: str) -> Optional[np.ndarray]:
        """Load image from various formats"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.ibw' and bw is not None:
                wave = bw.load(filepath)
                image_data = wave['wave']['wData']
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            elif file_ext == '.npy':
                image_data = np.load(filepath)
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            elif file_ext in ['.tiff', '.tif', '.png', '.jpg', '.jpeg']:
                img = Image.open(filepath)
                image_data = np.array(img)
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            else:
                raise HessianBlobError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            handle_error("load_image_file", e, f"file: {filepath}")
            return None

    @staticmethod
    def create_particle_info(particle_data: Dict, particle_id: int) -> Dict:
        """Create particle information dictionary matching Igor format"""
        info = {
            'Parent': particle_data.get('parent', 'image'),
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Height': float(particle_data.get('height', 0)),
            'Avg Height': float(particle_data.get('avg_height', 0)),
            'Volume': float(particle_data.get('volume', 0)),
            'Area': float(particle_data.get('area', 0)),
            'Perimeter': float(particle_data.get('perimeter', 0)),
            'Scale': float(particle_data.get('scale', 0)),
            'xCOM': float(particle_data.get('com', [0, 0])[0]),
            'yCOM': float(particle_data.get('com', [0, 0])[1]),
            'pSeed': int(particle_data.get('p_seed', 0)),
            'qSeed': int(particle_data.get('q_seed', 0)),
            'rSeed': int(particle_data.get('r_seed', 0)),
            'subPixelXCenter': float(particle_data.get('p_seed', 0)),
            'subPixelYCenter': float(particle_data.get('q_seed', 0))
        }
        return info


# ========================================================================
# IMPROVED GUI DIALOGS
# ========================================================================

class ParameterDialog:
    """Parameter dialog with validation"""

    @staticmethod
    def get_hessian_parameters() -> Optional[Tuple]:
        """Get Hessian blob parameters"""
        root = tk.Tk()
        root.title("Hessian Blob Parameters")
        root.geometry("600x500")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (600 // 2)
        y = (root.winfo_screenheight() // 2) - (500 // 2)
        root.geometry(f"600x500+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Hessian Blob Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Default values
        default_values = {
            'scaleStart': 1,
            'layers': 256,
            'scaleFactor': 1.5,
            'detHResponseThresh': -2,
            'particleType': 1,
            'subPixelMult': 1,
            'allowOverlap': 0
        }

        vars_dict = {}
        for key, default in default_values.items():
            if isinstance(default, int):
                vars_dict[key] = tk.IntVar(value=default)
            else:
                vars_dict[key] = tk.DoubleVar(value=default)

        labels = [
            "Minimum Size in Pixels",
            "Maximum Size in Pixels",
            "Scaling Factor",
            "Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)",
            "Particle Type (-1 for negative, +1 for positive, 0 for both)",
            "Subpixel Ratio",
            "Allow Hessian Blobs to Overlap? (1=yes 0=no)"
        ]

        # Create parameter input fields
        entries = {}
        for i, (key, var) in enumerate(vars_dict.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=5)

            label = tk.Label(frame, text=labels[i], width=50, anchor='w',
                             font=('Arial', 10), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=15, font=('Arial', 10))
            entry.pack(side='right', padx=(10, 0))
            entries[key] = entry

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def validate_and_continue():
            try:
                params = [
                    vars_dict['scaleStart'].get(),
                    vars_dict['layers'].get(),
                    vars_dict['scaleFactor'].get(),
                    vars_dict['detHResponseThresh'].get(),
                    vars_dict['particleType'].get(),
                    vars_dict['subPixelMult'].get(),
                    vars_dict['allowOverlap'].get()
                ]

                validate_hessian_parameters(params)
                result[0] = tuple(params)
                root.destroy()

            except Exception as e:
                error_label.config(text=str(e))

        def on_cancel():
            root.destroy()

        def show_help():
            help_text = """
Hessian Blob Parameters Help:

1. Minimum Size: Minimum radius of particles to detect (pixels)
2. Maximum Size: Maximum radius of particles to detect (pixels)  
3. Scaling Factor: Scale-space precision (1.2-2.0, default 1.5)
4. Blob Strength: Threshold (-2=interactive, -1=Otsu, >0=manual)
5. Particle Type: +1=positive blobs, -1=negative, 0=both
6. Subpixel Ratio: Subpixel precision multiplier (1=pixel accuracy)
7. Allow Overlap: 1=allow overlapping particles, 0=no overlap
"""
            messagebox.showinfo("Parameter Help", help_text)

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=validate_and_continue,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Help", command=show_help,
                  bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]

    @staticmethod
    def get_constraints_dialog() -> Optional[Tuple]:
        """Get particle constraints"""
        root = tk.Tk()
        root.title("Constraints")
        root.geometry("500x400")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (500 // 2)
        y = (root.winfo_screenheight() // 2) - (400 // 2)
        root.geometry(f"500x400+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Particle Constraints",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        subtitle_label = tk.Label(root,
                                  text="Limit analysis to particles within certain bounds\n(use -inf and inf for no constraints)",
                                  font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 15))

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20)

        # Default constraint values
        vars_dict = {
            'minHeight': tk.StringVar(value="-inf"),
            'maxHeight': tk.StringVar(value="inf"),
            'minArea': tk.StringVar(value="-inf"),
            'maxArea': tk.StringVar(value="inf"),
            'minVolume': tk.StringVar(value="-inf"),
            'maxVolume': tk.StringVar(value="inf")
        }

        labels = [
            "Minimum height",
            "Maximum height",
            "Minimum area",
            "Maximum area",
            "Minimum volume",
            "Maximum volume"
        ]

        # Create constraint input fields
        for i, (key, var) in enumerate(vars_dict.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=8)

            label = tk.Label(frame, text=labels[i], width=20, anchor='w',
                             font=('Arial', 11), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=15, font=('Arial', 11))
            entry.pack(side='right', padx=(10, 0))

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def parse_value(val_str):
            """Parse constraint values, handling inf/-inf."""
            val_str = val_str.strip()
            if val_str.lower() in ['-inf', '-infinity']:
                return -np.inf
            elif val_str.lower() in ['inf', 'infinity']:
                return np.inf
            else:
                return float(val_str)

        def validate_and_continue():
            try:
                constraints = [
                    parse_value(vars_dict['minHeight'].get()),
                    parse_value(vars_dict['maxHeight'].get()),
                    parse_value(vars_dict['minArea'].get()),
                    parse_value(vars_dict['maxArea'].get()),
                    parse_value(vars_dict['minVolume'].get()),
                    parse_value(vars_dict['maxVolume'].get())
                ]

                validate_constraints(constraints)
                result[0] = tuple(constraints)
                root.destroy()

            except Exception as e:
                error_label.config(text=str(e))

        def on_cancel():
            root.destroy()

        def show_help():
            help_text = """
Particle Constraints Help:

Set bounds to filter particles by their measurements:
- Height: vertical extent above background
- Area: 2D projected area in image
- Volume: integrated intensity above background

Use "-inf" for no lower bound
Use "inf" for no upper bound
Use numbers for specific limits

Example: minHeight=0, maxHeight=5e-9 
(particles between 0 and 5 nanometers tall)
"""
            messagebox.showinfo("Constraints Help", help_text)

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=validate_and_continue,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Help", command=show_help,
                  bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]


# ========================================================================
# MAIN FUNCTIONS
# ========================================================================

def BatchHessianBlobs():
    """Detects Hessian blobs in a series of images."""

    try:
        # Get folder containing images
        ImagesDF = filedialog.askdirectory(title="Select folder containing images")
        if not ImagesDF:
            safe_print("No folder selected.")
            return ""

        CurrentDF = GetDataFolder(1)

        # Count images in folder
        image_count = 0
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_count += len([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        if image_count < 1:
            messagebox.showerror("Error", "No images found in selected folder.")
            return ""

        # Get parameters from user
        param_values = ParameterDialog.get_hessian_parameters()
        if param_values is None:
            return ""

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_values

        # Get constraints if needed
        minH = -np.inf
        maxH = np.inf
        minV = -np.inf
        maxV = np.inf
        minA = -np.inf
        maxA = np.inf

        constraints_answer = messagebox.askyesno("Constraints",
                                                 "Would you like to limit the analysis to particles of certain height, volume, or area?")
        if constraints_answer:
            constraints = ParameterDialog.get_constraints_dialog()
            if constraints is None:
                return ""
            minH, maxH, minA, maxA, minV, maxV = constraints

        # Create Series folder in current directory
        series_name = "Series"
        counter = 0
        while True:
            if counter == 0:
                series_folder_name = series_name
            else:
                series_folder_name = f"{series_name}_{counter}"

            SeriesDF = os.path.join(CurrentDF, series_folder_name)
            if not os.path.exists(SeriesDF):
                break
            counter += 1

        # Create the series folder
        os.makedirs(SeriesDF, exist_ok=True)
        safe_print(f"Created series folder: {SeriesDF}")

        # Store parameters
        Parameters = np.array([
            scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
            subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
        ])
        DataManager.save_wave_data(Parameters, os.path.join(SeriesDF, "Parameters.npy"))

        # Process images
        AllHeights = []
        AllVolumes = []
        AllAreas = []
        AllAvgHeights = []

        # Get list of image files
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_files.extend([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        image_files.sort()  # Ensure consistent ordering

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(ImagesDF, image_file)
            im = DataManager.load_image_file(image_path)

            if im is None:
                safe_print(f"Warning: Could not load {image_file}")
                continue

            safe_print("-------------------------------------------------------")
            safe_print(f"Analyzing image {i + 1} of {len(image_files)}: {image_file}")
            safe_print("-------------------------------------------------------")

            # Run analysis. Create image folder inside series folder
            imageDF = HessianBlobs_SeriesMode(im, params=Parameters, seriesFolder=SeriesDF, imageIndex=i)

            if imageDF:
                try:
                    # Load measurements
                    Heights = np.load(os.path.join(imageDF, "Heights.npy"))
                    AvgHeights = np.load(os.path.join(imageDF, "AvgHeights.npy"))
                    Areas = np.load(os.path.join(imageDF, "Areas.npy"))
                    Volumes = np.load(os.path.join(imageDF, "Volumes.npy"))

                    # Concatenate measurements
                    AllHeights.extend(Heights)
                    AllAvgHeights.extend(AvgHeights)
                    AllAreas.extend(Areas)
                    AllVolumes.extend(Volumes)

                except Exception as e:
                    safe_print(f"Warning: Could not load measurements from {imageDF}: {e}")

        # Save combined results
        DataManager.save_wave_data(np.array(AllHeights), os.path.join(SeriesDF, "AllHeights.npy"))
        DataManager.save_wave_data(np.array(AllVolumes), os.path.join(SeriesDF, "AllVolumes.npy"))
        DataManager.save_wave_data(np.array(AllAreas), os.path.join(SeriesDF, "AllAreas.npy"))
        DataManager.save_wave_data(np.array(AllAvgHeights), os.path.join(SeriesDF, "AllAvgHeights.npy"))

        # Print summary
        numParticles = len(AllHeights)
        safe_print(f"  Series complete. Total particles detected: {numParticles}")
        safe_print(f"  Results saved to: {SeriesDF}")

        # Verify the folder was created properly
        if os.path.exists(SeriesDF):
            safe_print(f"✓ Confirmed: Series folder exists at {SeriesDF}")
            files_in_series = os.listdir(SeriesDF)
            safe_print(f"✓ Files in series folder: {files_in_series}")
        else:
            safe_print(f"✗ ERROR: Series folder was not created at {SeriesDF}")

        return SeriesDF

    except Exception as e:
        error_msg = handle_error("BatchHessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""


def HessianBlobs(im, params=None):
    """Executes the Hessian blob algorithm on an image."""
    try:
        # Validate input
        if im is None:
            raise HessianBlobError("Input image is None")

        if len(im.shape) < 2:
            raise HessianBlobError("Input must be at least 2D")

        # Measurement ranges
        min_h, max_h, min_v, max_v, min_a, max_a = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf

        # Get parameters
        if params is None:
            param_values = ParameterDialog.get_hessian_parameters()
            if param_values is None:
                safe_print("Analysis cancelled by user.")
                return ""

            scale_start, layers, scale_factor, det_h_response_thresh, particle_type, subpixel_mult, allow_overlap = param_values

            constraints_answer = messagebox.askyesno("Constraints",
                                                     "Would you like to limit the analysis to particles of certain height, volume, or area?")

            if constraints_answer:
                constraints = ParameterDialog.get_constraints_dialog()
                if constraints is None:
                    safe_print("Analysis cancelled by user.")
                    return ""
                min_h, max_h, min_a, max_a, min_v, max_v = constraints
        else:
            if len(params) < 13:
                raise HessianBlobError("Provided parameter array must contain 13 parameters")

            scale_start = params[0]
            layers = int(params[1])
            scale_factor = params[2]
            det_h_response_thresh = params[3]
            particle_type = int(params[4])
            subpixel_mult = int(params[5])
            allow_overlap = int(params[6])
            min_h = params[7]
            max_h = params[8]
            min_a = params[9]
            max_a = params[10]
            min_v = params[11]
            max_v = params[12]

        # Print parameters
        if params is None:
            all_params = [scale_start, layers, scale_factor, det_h_response_thresh,
                          particle_type, subpixel_mult, allow_overlap,
                          min_h, max_h, min_a, max_a, min_v, max_v]
            print_analysis_parameters(all_params)

        # Validate and convert parameters
        converted_params = validate_and_convert_parameters([scale_start, layers, scale_factor,
                                                            det_h_response_thresh, particle_type,
                                                            subpixel_mult, allow_overlap])

        scale_start, layers, scale_factor, det_h_response_thresh, particle_type, subpixel_mult, allow_overlap = converted_params

        # Hard coded parameters
        gamma_norm = 1
        max_curvature_ratio = 10
        allow_boundary_particles = 1

        # Create particle folder
        current_df = GetDataFolder(1)
        new_df = NameOfWave(im) + "_Particles"

        full_path = os.path.join(current_df, new_df)
        if os.path.exists(full_path):
            new_df = UniqueName(new_df, 11, 2)

        new_df = os.path.join(current_df, new_df)
        DataManager.create_igor_folder_structure(new_df, "particles")

        safe_print(f"Created particle analysis folder: {new_df}")

        # Store original image
        if len(im.shape) == 3:
            original = im[:, :, 0].copy()
        else:
            original = im.copy()

        DataManager.save_wave_data(original, os.path.join(new_df, "Original.npy"))
        im = original

        # Scale-space analysis with progress reporting
        safe_print("Calculating scale-space representation..")
        L = ScaleSpaceRepresentation(im, layers, np.sqrt(scale_start) / 1.0, scale_factor)
        DataManager.save_wave_data(L, os.path.join(new_df, "ScaleSpaceRep.npy"))

        safe_print("Calculating scale-space derivatives..")
        LapG, detH = BlobDetectors(L, gamma_norm)
        DataManager.save_wave_data(LapG, os.path.join(new_df, "LapG.npy"))
        DataManager.save_wave_data(detH, os.path.join(new_df, "detH.npy"))

        # Threshold determination
        if det_h_response_thresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            det_h_response_thresh = np.sqrt(OtsuThreshold(detH, LapG, particle_type, max_curvature_ratio))
            safe_print(f"Otsu's Threshold: {det_h_response_thresh}")
        elif det_h_response_thresh == -2:
            det_h_response_thresh = InteractiveThreshold(im, detH, LapG, particle_type, max_curvature_ratio)
            safe_print(f"Chosen Det H Response Threshold: {det_h_response_thresh}")

        # Detect particles
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, det_h_response_thresh,
                                                         particle_type, max_curvature_ratio)

        num_potential_particles = len(Info) if Info is not None else 0

        if num_potential_particles == 0:
            safe_print("No particles detected.")
            # Save empty arrays
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Volumes.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Heights.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "COM.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Areas.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "AvgHeights.npy"))
            return new_df

        # Remove overlapping particles if requested
        if allow_overlap == 0:
            safe_print("Determining scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
        else:
            if Info is not None:
                for i in range(len(Info)):
                    Info[i][10] = 1

        # Initialize particle status
        if num_potential_particles > 0:
            for i in range(len(Info)):
                Info[i][13] = 0
                Info[i][14] = 0

        # Process particles with enhanced progress reporting
        safe_print("Cropping and measuring particles..")
        volumes = []
        heights = []
        com = []
        areas = []
        avg_heights = []
        count = 0

        for i in range(num_potential_particles - 1, -1, -1):
            try:
                # Skip overlapping particles if not allowed
                if allow_overlap == 0 and Info[i][10] == 0:
                    continue

                # Basic validation
                if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
                    continue

                # Boundary particles check
                if (allow_boundary_particles == 0 and
                        (Info[i][4] <= 2 or Info[i][5] >= im.shape[0] - 3 or
                         Info[i][6] <= 2 or Info[i][7] >= im.shape[1] - 3)):
                    continue

                # Extract particle region
                padding = int(np.ceil(max(Info[i][5] - Info[i][4] + 2, Info[i][7] - Info[i][6] + 2)))
                p_start = max(int(Info[i][4]) - padding, 0)
                p_end = min(int(Info[i][5]) + padding, im.shape[0] - 1)
                q_start = max(int(Info[i][6]) - padding, 0)
                q_end = min(int(Info[i][7]) + padding, im.shape[1] - 1)

                particle = im[p_start:p_end + 1, q_start:q_end + 1].copy()

                # Create mask
                mask = create_particle_mask(mapNum, i, int(Info[i][9]), p_start, p_end, q_start, q_end)

                # Create perimeter
                perim = create_perimeter_mask(mask)

                # Calculate measurements
                bg = M_MinBoundary(particle, mask)
                particle_bg_sub = particle - bg
                height = M_Height(particle_bg_sub, mask, 0)
                vol = M_Volume(particle_bg_sub, mask, 0)
                center_of_mass = M_CenterOfMass(particle_bg_sub, mask, 0)
                particle_area = M_Area(mask)
                particle_perim = M_Perimeter(mask)
                avg_height = vol / particle_area if particle_area > 0 else 0

                # Check constraints
                if not (min_h < height < max_h and min_a < particle_area < max_a and min_v < vol < max_v):
                    continue

                # Accept particle
                Info[i][14] = count

                # Create particle data
                particle_data = {
                    'parent': NameOfWave(im),
                    'height': height,
                    'avg_height': avg_height,
                    'volume': vol,
                    'area': particle_area,
                    'perimeter': particle_perim,
                    'scale': Info[i][8],
                    'com': center_of_mass,
                    'p_seed': Info[i][0],
                    'q_seed': Info[i][1],
                    'r_seed': Info[i][9]
                }

                # Save particle to Igor Pro-style folder
                save_particle_data(new_df, count, particle, mask, perim, particle_data)

                # Store measurements
                volumes.append(vol)
                heights.append(height)
                com.append(center_of_mass)
                areas.append(particle_area)
                avg_heights.append(avg_height)

                count += 1

                # Progress reporting every 10 particles
                if count % 10 == 0:
                    safe_print(f"  Processed {count} particles...")

            except Exception as e:
                handle_error("HessianBlobs", e, f"processing particle {i}")
                continue

        # Save measurement arrays
        DataManager.save_wave_data(np.array(volumes), os.path.join(new_df, "Volumes.npy"))
        DataManager.save_wave_data(np.array(heights), os.path.join(new_df, "Heights.npy"))
        DataManager.save_wave_data(np.array(com), os.path.join(new_df, "COM.npy"))
        DataManager.save_wave_data(np.array(areas), os.path.join(new_df, "Areas.npy"))
        DataManager.save_wave_data(np.array(avg_heights), os.path.join(new_df, "AvgHeights.npy"))

        # Create particle map
        particle_map = np.full_like(im, -1)
        DataManager.save_wave_data(particle_map, os.path.join(new_df, "ParticleMap.npy"))

        # Final summary
        safe_print(f"Analysis complete. {count} particles detected and measured.")
        if count > 0:
            safe_print(f"Average height: {np.mean(heights):.3e}")
            safe_print(f"Average volume: {np.mean(volumes):.3e}")
            safe_print(f"Average area: {np.mean(areas):.3e}")

        # Verify folder structure
        verify_igor_compatibility(new_df)
        SetDataFolder(current_df)
        return new_df

    except Exception as e:
        error_msg = handle_error("HessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""


def HessianBlobs_SeriesMode(im, params=None, seriesFolder=None, imageIndex=0):
    """Creates image folder inside series folder."""

    if params is None or len(params) < 13:
        raise HessianBlobError("Parameter array must contain 13 parameters")

    # Extract parameters
    scaleStart = params[0]
    layers = int(params[1])
    scaleFactor = params[2]
    detHResponseThresh = params[3]
    particleType = int(params[4])
    subPixelMult = int(params[5])
    allowOverlap = int(params[6])
    minH = params[7]
    maxH = params[8]
    minA = params[9]
    maxA = params[10]
    minV = params[11]
    maxV = params[12]

    # Parameter conversion
    dimDelta_im_0 = 1.0
    scaleStart = (scaleStart * dimDelta_im_0) ** 2 / 2
    layers = int(np.ceil(np.log((layers * dimDelta_im_0) ** 2 / (2 * scaleStart)) / np.log(scaleFactor)))
    subPixelMult = max(1, round(subPixelMult))
    scaleFactor = max(1.1, scaleFactor)

    # Hard coded parameters
    gammaNorm = 1
    maxCurvatureRatio = 10
    allowBoundaryParticles = 1

    # Create image folder INSIDE series folder
    CurrentDF = GetDataFolder(1)
    if seriesFolder:
        # Create image folder inside the series folder with a clean name
        image_folder_name = f"image_Particles"
        NewDF = os.path.join(seriesFolder, image_folder_name)

        # Make it unique if it already exists
        counter = 0
        base_path = NewDF
        while os.path.exists(NewDF):
            counter += 1
            NewDF = f"{base_path}_{counter}"
    else:
        NewDF = os.path.join(CurrentDF, f"image_Particles")

    # Create the folder
    DataManager.create_igor_folder_structure(NewDF, "particles")
    safe_print(f"Created image particle folder: {NewDF}")

    # Store original image
    if len(im.shape) == 3:
        Original = im[:, :, 0].copy()
    else:
        Original = im.copy()

    DataManager.save_wave_data(Original, os.path.join(NewDF, "Original.npy"))
    im = Original

    # Run the analysis
    numPotentialParticles = 0
    count = 0
    limP = im.shape[0]
    limQ = im.shape[1]

    # Calculate scale-space representation
    safe_print("Calculating scale-space representation..")
    L = ScaleSpaceRepresentation(im, layers, np.sqrt(scaleStart) / dimDelta_im_0, scaleFactor)
    DataManager.save_wave_data(L, os.path.join(NewDF, "ScaleSpaceRep.npy"))

    # Calculate derivatives
    safe_print("Calculating scale-space derivatives..")
    LapG, detH = BlobDetectors(L, gammaNorm)
    DataManager.save_wave_data(LapG, os.path.join(NewDF, "LapG.npy"))
    DataManager.save_wave_data(detH, os.path.join(NewDF, "detH.npy"))

    # Threshold determination
    if detHResponseThresh == -1:
        safe_print("Calculating Otsu's Threshold..")
        detHResponseThresh = np.sqrt(OtsuThreshold(detH, LapG, particleType, maxCurvatureRatio))
        safe_print(f"Otsu's Threshold: {detHResponseThresh}")
    elif detHResponseThresh == -2:
        detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
        safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")

    # Detect particles
    safe_print("Detecting Hessian blobs..")
    mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh, particleType,
                                                     maxCurvatureRatio)
    numPotentialParticles = len(Info) if Info is not None else 0

    if numPotentialParticles == 0:
        safe_print("No particles detected.")
        # Save empty arrays
        DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Volumes.npy"))
        DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Heights.npy"))
        DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "COM.npy"))
        DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Areas.npy"))
        DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "AvgHeights.npy"))
        return NewDF

    # Remove overlapping particles if requested
    if allowOverlap == 0:
        safe_print("Determining scale-maximal particles..")
        MaximalBlobs(Info, mapNum)
    else:
        for i in range(len(Info)):
            Info[i][10] = 1

    # Initialize particle status
    if numPotentialParticles > 0:
        for i in range(len(Info)):
            Info[i][13] = 0
            Info[i][14] = 0

    # Process and measure particles
    Volumes = []
    Heights = []
    COM = []
    Areas = []
    AvgHeights = []

    safe_print("Cropping and measuring particles..")
    for i in range(numPotentialParticles - 1, -1, -1):
        # Skip overlapping particles if not allowed
        if allowOverlap == 0 and Info[i][10] == 0:
            continue

        # Basic validation
        if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
            continue

        # Boundary particles check
        if (allowBoundaryParticles == 0 and
                (Info[i][4] <= 2 or Info[i][5] >= limP - 3 or Info[i][6] <= 2 or Info[i][7] >= limQ - 3)):
            continue

        # Extract and process particle
        padding = int(np.ceil(max(Info[i][5] - Info[i][4] + 2, Info[i][7] - Info[i][6] + 2)))
        p_start = max(int(Info[i][4]) - padding, 0)
        p_end = min(int(Info[i][5]) + padding, limP - 1)
        q_start = max(int(Info[i][6]) - padding, 0)
        q_end = min(int(Info[i][7]) + padding, limQ - 1)

        particle = im[p_start:p_end + 1, q_start:q_end + 1].copy()
        mask = create_particle_mask(mapNum, i, int(Info[i][9]), p_start, p_end, q_start, q_end)
        perim = create_perimeter_mask(mask)

        # Calculate measurements
        bg = M_MinBoundary(particle, mask)
        particle_bg_sub = particle - bg
        height = M_Height(particle_bg_sub, mask, 0)
        vol = M_Volume(particle_bg_sub, mask, 0)
        centerOfMass = M_CenterOfMass(particle_bg_sub, mask, 0)
        particleArea = M_Area(mask)
        particlePerim = M_Perimeter(mask)
        avgHeight = vol / particleArea if particleArea > 0 else 0

        # Check constraints
        if not (minH < height < maxH and minA < particleArea < maxA and minV < vol < maxV):
            continue

        # Accept particle
        Info[i][14] = count

        # Create particle data
        particle_data = {
            'parent': "image",  # Use simple name instead of NameOfWave(im)
            'height': height,
            'avg_height': avgHeight,
            'volume': vol,
            'area': particleArea,
            'perimeter': particlePerim,
            'scale': Info[i][8],
            'com': centerOfMass,
            'p_seed': Info[i][0],
            'q_seed': Info[i][1],
            'r_seed': Info[i][9]
        }

        # Save particle data
        save_particle_data(NewDF, count, particle, mask, perim, particle_data)

        # Store measurements
        Volumes.append(vol)
        Heights.append(height)
        COM.append(centerOfMass)
        Areas.append(particleArea)
        AvgHeights.append(avgHeight)

        count += 1

    # Save measurement arrays
    DataManager.save_wave_data(np.array(Volumes), os.path.join(NewDF, "Volumes.npy"))
    DataManager.save_wave_data(np.array(Heights), os.path.join(NewDF, "Heights.npy"))
    DataManager.save_wave_data(np.array(COM), os.path.join(NewDF, "COM.npy"))
    DataManager.save_wave_data(np.array(Areas), os.path.join(NewDF, "Areas.npy"))
    DataManager.save_wave_data(np.array(AvgHeights), os.path.join(NewDF, "AvgHeights.npy"))

    # Create particle map
    ParticleMap = np.full_like(im, -1)
    DataManager.save_wave_data(ParticleMap, os.path.join(NewDF, "ParticleMap.npy"))

    safe_print(f"Analysis complete. {count} particles detected and measured.")

    # Verify folder structure
    verify_folder_structure(NewDF)

    return NewDF


# ========================================================================
# HELPER FUNCTIONS FOR PARTICLE PROCESSING
# ========================================================================

def create_particle_mask(mapNum, particle_index, layer, p_start, p_end, q_start, q_end):
    """Create particle mask from mapNum array"""
    try:
        mask_height = p_end - p_start + 1
        mask_width = q_end - q_start + 1
        mask = np.zeros((mask_height, mask_width))

        for ii in range(mask_height):
            for jj in range(mask_width):
                global_i = p_start + ii
                global_j = q_start + jj
                if (global_i < mapNum.shape[0] and global_j < mapNum.shape[1] and
                        layer < mapNum.shape[2] and mapNum[global_i, global_j, layer] == particle_index):
                    mask[ii, jj] = 1

        return mask

    except Exception as e:
        handle_error("create_particle_mask", e)
        return np.zeros((p_end - p_start + 1, q_end - q_start + 1))


def create_perimeter_mask(mask):
    """Create perimeter mask from particle mask"""
    try:
        perim = np.zeros_like(mask)
        for ii in range(1, mask.shape[0] - 1):
            for jj in range(1, mask.shape[1] - 1):
                if mask[ii, jj] == 1:
                    neighbors = [mask[ii + 1, jj], mask[ii - 1, jj], mask[ii, jj + 1], mask[ii, jj - 1],
                                 mask[ii + 1, jj + 1], mask[ii - 1, jj + 1], mask[ii + 1, jj - 1], mask[ii - 1, jj - 1]]
                    if 0 in neighbors:
                        perim[ii, jj] = 1
        return perim

    except Exception as e:
        handle_error("create_perimeter_mask", e)
        return np.zeros_like(mask)


def save_particle_data(base_folder, particle_id, particle, mask, perim, particle_data):
    """Save particle data in Igor Pro-compatible format"""
    try:
        particle_folder = os.path.join(base_folder, f"Particle_{particle_id}")
        os.makedirs(particle_folder, exist_ok=True)
        safe_print(f"  Created Particle_{particle_id} folder")

        # Save particle arrays
        DataManager.save_wave_data(particle, os.path.join(particle_folder, f"Particle_{particle_id}.npy"))
        DataManager.save_wave_data(mask, os.path.join(particle_folder, f"Mask_{particle_id}.npy"))
        DataManager.save_wave_data(perim, os.path.join(particle_folder, f"Perimeter_{particle_id}.npy"))

        # Save particle info
        particle_info = DataManager.create_particle_info(particle_data, particle_id)

        with open(os.path.join(particle_folder, f"Particle_{particle_id}_info.json"), 'w') as f:
            json.dump(particle_info, f, indent=2)

        # Create Igor Pro-style note file
        with open(os.path.join(particle_folder, f"Particle_{particle_id}_info.txt"), 'w') as f:
            for key, value in particle_info.items():
                f.write(f"{key}:{value}\n")

        return True

    except Exception as e:
        handle_error("save_particle_data", e, f"particle {particle_id}")
        return False


# ========================================================================
# SCALE-SPACE FUNCTIONS
# ========================================================================

def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """Computes the discrete scale-space representation L of an image."""
    try:
        # Convert t0 to image units
        t0 = (t0 * 1.0) ** 2

        # Go to Fourier space
        im_fft = fft2(im)

        # Create frequency grids
        freq_x = fftfreq(im.shape[0])
        freq_y = fftfreq(im.shape[1])
        fx, fy = np.meshgrid(freq_x, freq_y, indexing='ij')

        # Make the layers of the scale-space representation
        L = np.zeros((im.shape[0], im.shape[1], layers))

        for i in range(layers):
            scale = t0 * (tFactor ** i)
            gaussian_kernel = np.exp(-(fx ** 2 + fy ** 2) * np.pi ** 2 * 2 * scale)
            Layer = im_fft * gaussian_kernel
            L[:, :, i] = np.real(ifft2(Layer))

        return L

    except Exception as e:
        handle_error("ScaleSpaceRepresentation", e)
        raise HessianBlobError(f"Failed to compute scale-space representation: {e}")


def BlobDetectors(L, gammaNorm):
    """Computes the two blob detectors, the determinant of the Hessian and the Laplacian of Gaussian."""
    try:
        # Make convolution kernels for calculating central difference derivatives
        LxxKernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        LyyKernel = np.array([
            [0, 0, -1 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -30 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -1 / 12, 0, 0]
        ])

        LxyKernel = np.array([
            [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
            [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
            [0, 0, 0, 0, 0],
            [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
            [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
        ])

        # Compute Lxx, Lyy, and Lxy
        Lxx = np.zeros_like(L)
        Lyy = np.zeros_like(L)
        Lxy = np.zeros_like(L)

        for i in range(L.shape[2]):
            Lxx[:, :, i] = ndimage.convolve(L[:, :, i], LxxKernel, mode='constant')
            Lyy[:, :, i] = ndimage.convolve(L[:, :, i], LyyKernel, mode='constant')
            Lxy[:, :, i] = ndimage.convolve(L[:, :, i], LxyKernel, mode='constant')

        # Compute the Laplacian of Gaussian
        LapG = Lxx + Lyy

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** gammaNorm / (1.0 * 1.0)
            LapG[:, :, r] *= scale_factor

        # Fix errors on the boundary of the image
        FixBoundaries(LapG)

        # Compute the determinant of the Hessian
        detH = Lxx * Lyy - Lxy ** 2

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** (2 * gammaNorm) / (1.0 * 1.0) ** 2
            detH[:, :, r] *= scale_factor

        # Fix the boundary issues again
        FixBoundaries(detH)

        return LapG, detH

    except Exception as e:
        handle_error("BlobDetectors", e)
        raise HessianBlobError(f"Failed to compute blob detectors: {e}")


def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """Uses Otsu's method to automatically define a threshold blob strength."""
    try:
        # First identify the maxes
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio)
        if len(Maxes) == 0:
            return 0.0

        # Create a histogram of the maxes
        Hist, bin_edges = np.histogram(Maxes, bins=50)

        # Search for the best threshold
        minICV = np.inf
        bestThresh = -np.inf

        for i in range(len(Hist)):
            xThresh = bin_edges[i]
            below_thresh = Maxes[Maxes < xThresh]
            above_thresh = Maxes[Maxes >= xThresh]

            if len(below_thresh) == 0 or len(above_thresh) == 0:
                continue

            w1 = len(below_thresh) / len(Maxes)
            w2 = len(above_thresh) / len(Maxes)

            if w1 > 0 and w2 > 0:
                ICV = w1 * np.var(below_thresh) + w2 * np.var(above_thresh)
                if ICV < minICV:
                    bestThresh = xThresh
                    minICV = ICV

        return bestThresh

    except Exception as e:
        handle_error("OtsuThreshold", e)
        return 0.0


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """Lets the user interactively choose a blob strength - EXACT IGOR PRO INTERFACE."""
    try:
        # First identify the maxes
        SS_MAXMAP = np.full_like(im, -1)
        SS_MAXSCALEMAP = np.zeros_like(im)
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)
        Maxes = np.sqrt(np.maximum(Maxes, 0))  # Put it into image units

        if len(Maxes) == 0:
            safe_print("No maxima found for interactive threshold selection.")
            return 0.0

        # CRITICAL: Ensure thread-safe matplotlib backend
        import matplotlib
        matplotlib.use('TkAgg')

        # Close any existing plots
        plt.close('all')

        # Create interactive plot exactly matching Igor Pro Figure 17
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.8)

        im_display = ax.imshow(im, cmap='gray', interpolation='bilinear')
        ax.set_title('Interactive Blob Strength Selection\nAdjust slider to see detected blobs',
                     fontsize=14, fontweight='bold')

        SS_THRESH = np.max(Maxes) / 2

        # Create slider panel exactly like Igor Pro
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Blob Strength', 0, np.max(Maxes) * 1.1,
                        valinit=SS_THRESH, valfmt='%.3e')

        # Create text display for current threshold value
        ax_text = plt.axes([0.82, 0.5, 0.15, 0.3])
        ax_text.axis('off')
        threshold_text = ax_text.text(0.1, 0.9, f'Blob Strength:\n{SS_THRESH:.3e}',
                                      fontsize=10, transform=ax_text.transAxes)

        circles = []

        def update_display(thresh):
            # Clear previous circles
            for circle in circles:
                try:
                    circle.remove()
                except:
                    pass
            circles.clear()

            thresh_squared = thresh ** 2
            count = 0
            for i in range(SS_MAXMAP.shape[0]):
                for j in range(SS_MAXMAP.shape[1]):
                    if SS_MAXMAP[i, j] > thresh_squared:
                        xc = j  # Column is x-coordinate
                        yc = i  # Row is y-coordinate
                        rad = max(2, np.sqrt(2 * SS_MAXSCALEMAP[i, j]))

                        # FIXED: Create RED circles exactly like Igor Pro Figure 17
                        # Using bright red color that stands out on hot colormap
                        circle = plt.Circle((xc, yc), rad, color='red', fill=False,
                                            linewidth=2.5, alpha=0.9)
                        ax.add_patch(circle)
                        circles.append(circle)
                        count += 1

            # Update title and text display exactly like Igor Pro
            ax.set_title(f'Interactive Blob Strength Selection\n'
                         f'Blob Strength: {thresh:.3e}, Particles: {count}',
                         fontsize=14, fontweight='bold')

            threshold_text.set_text(f'Blob Strength:\n{thresh:.3e}\n\nParticles: {count}')

            # Thread-safe canvas update
            try:
                fig.canvas.draw_idle()
            except:
                pass

        slider.on_changed(update_display)
        update_display(SS_THRESH)

        # Create Accept and Quit buttons exactly like Igor Pro
        ax_accept = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.81, 0.02, 0.1, 0.04])
        button_accept = Button(ax_accept, 'Accept')
        button_quit = Button(ax_quit, 'Quit')

        result = [SS_THRESH]

        def accept_threshold(event):
            result[0] = slider.val
            plt.close(fig)

        def quit_threshold(event):
            result[0] = SS_THRESH
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)
        button_quit.on_clicked(quit_threshold)

        # Add instructions exactly like Igor Pro
        instructions_text = ('Use the slider to adjust blob strength threshold.\n'
                             'Red circles show detected particles.\n'
                             'Click "Accept" when satisfied with detection.')
        ax_text.text(0.1, 0.3, instructions_text, fontsize=9,
                     transform=ax_text.transAxes, style='italic')

        safe_print("Interactive threshold selection:")
        safe_print("- Use slider to adjust blob strength threshold")
        safe_print("- Red circles show detected particles")
        safe_print("- Click 'Accept' when satisfied with detection")

        # CRITICAL: Use blocking show() to prevent threading issues
        plt.show(block=True)

        return result[0]

    except Exception as e:
        handle_error("InteractiveThreshold", e)
        return 0.0


# ========================================================================
# PARTICLE MEASUREMENTS
# ========================================================================

def M_AvgBoundary(im, mask):
    """Measure the average pixel value of the particle on the boundary of the particle."""
    try:
        limI, limJ = im.shape
        bg = 0
        cnt = 0

        for i in range(1, limI - 1):
            for j in range(1, limJ - 1):
                if (mask[i, j] == 0 and
                        (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or mask[i, j + 1] == 1 or mask[i, j - 1] == 1)):
                    bg += im[i, j]
                    cnt += 1
                elif (mask[i, j] == 1 and
                      (mask[i + 1, j] == 0 or mask[i - 1, j] == 0 or mask[i, j + 1] == 0 or mask[i, j - 1] == 0)):
                    bg += im[i, j]
                    cnt += 1

        return bg / cnt if cnt > 0 else 0

    except Exception as e:
        handle_error("M_AvgBoundary", e)
        return 0.0


def M_MinBoundary(im, mask):
    """Measure the minimum pixel value of the particle."""
    try:
        bg = np.inf
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j] == 1 and im[i, j] < bg:
                    bg = im[i, j]
        return bg if bg != np.inf else 0.0

    except Exception as e:
        handle_error("M_MinBoundary", e)
        return 0.0


def M_Height(im, mask, bg, negParticle=False):
    """Measures the maximum height of the particle above the background level."""
    try:
        masked_im = np.where(mask, im, np.nan)
        if negParticle:
            height = bg - np.nanmin(masked_im)
        else:
            height = np.nanmax(masked_im) - bg
        return height if not np.isnan(height) else 0.0

    except Exception as e:
        handle_error("M_Height", e)
        return 0.0


def M_Volume(im, mask, bg):
    """Computes the volume of the particle."""
    try:
        V = 0
        cnt = 0
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j]:
                    V += im[i, j]
                    cnt += 1

        V -= cnt * bg
        V *= 1.0 * 1.0  # DimDelta(im,0) * DimDelta(im,1)
        return V

    except Exception as e:
        handle_error("M_Volume", e)
        return 0.0


def M_CenterOfMass(im, mask, bg):
    """Computes the center of mass of the particle."""
    try:
        xsum = 0
        ysum = 0
        count = 0
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j]:
                    weight = im[i, j] - bg
                    xsum += i * weight
                    ysum += j * weight
                    count += weight

        if count > 0:
            return (xsum / count, ysum / count)
        else:
            return (0.0, 0.0)

    except Exception as e:
        handle_error("M_CenterOfMass", e)
        return (0.0, 0.0)


def M_Area(mask):
    """Computes the area of the particle using the method employed by Gwyddion."""
    try:
        a = 0
        limI, limJ = mask.shape

        for i in range(limI - 1):
            for j in range(limJ - 1):
                pixels = mask[i, j] + mask[i + 1, j] + mask[i, j + 1] + mask[i + 1, j + 1]

                if pixels == 1:
                    a += 0.125  # 1/8
                elif pixels == 2:
                    if mask[i, j] == mask[i + 1, j] or mask[i, j] == mask[i, j + 1]:
                        a += 0.5
                    else:
                        a += 0.75
                elif pixels == 3:
                    a += 0.875  # 7/8
                elif pixels == 4:
                    a += 1

        return a * 1.0 ** 2  # DimDelta(mask,0)^2

    except Exception as e:
        handle_error("M_Area", e)
        return 0.0


def M_Perimeter(mask):
    """Computes the perimeter of the particle using the method employed by Gwyddion."""
    try:
        l = 0
        limI, limJ = mask.shape

        for i in range(limI - 1):
            for j in range(limJ - 1):
                pixels = mask[i, j] + mask[i + 1, j] + mask[i, j + 1] + mask[i + 1, j + 1]

                if pixels == 1 or pixels == 3:
                    l += np.sqrt(2) / 2
                elif pixels == 2:
                    if mask[i, j] == mask[i + 1, j] or mask[i, j] == mask[i, j + 1]:
                        l += 1
                    else:
                        l += np.sqrt(2)

        return l * 1.0  # DimDelta(mask,0)

    except Exception as e:
        handle_error("M_Perimeter", e)
        return 0.0


# ========================================================================
# PREPROCESSING FUNCTIONS
# ========================================================================

def BatchPreprocess():
    """Preprocess multiple images - IGOR PRO EXACT BEHAVIOR."""
    try:
        # Get folder containing images - matching Igor Pro GetBrowserSelection
        images_df = filedialog.askdirectory(title="Select folder containing images to preprocess")
        if not images_df:
            raise HessianBlobError("No folder selected")

        current_df = GetDataFolder(1)

        # Count images - matching Igor Pro CountObjects
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            found_files = [f for f in os.listdir(images_df) if f.lower().endswith(ext.lower())]
            image_files.extend(found_files)

        if len(image_files) < 1:
            raise HessianBlobError("Selected folder contains no images")

        # Get preprocessing parameters - matching Igor Pro prompts
        params = _get_preprocess_parameters()
        if params is None:
            safe_print("Preprocessing cancelled by user.")
            return -1

        streak_removal_sdevs, flatten_order = params

        # Create preprocessed folder - matching Igor Pro DuplicateDataFolder behavior
        base_folder_name = os.path.basename(images_df.rstrip(os.sep))
        preprocessed_folder_name = f"{base_folder_name}_dup"  # Matching Igor Pro "_dup" suffix

        counter = 0
        while True:
            if counter == 0:
                test_folder_name = preprocessed_folder_name
            else:
                test_folder_name = f"{preprocessed_folder_name}_{counter}"

            preprocessed_folder_path = os.path.join(current_df, test_folder_name)
            if not os.path.exists(preprocessed_folder_path):
                break
            counter += 1

        os.makedirs(preprocessed_folder_path, exist_ok=True)
        safe_print(f"Created duplicated folder: {preprocessed_folder_path}")

        # Process each image - matching Igor Pro For loop exactly
        for i, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(images_df, image_file)
                im = DataManager.load_image_file(image_path)

                if im is None:
                    safe_print(f"Warning: Could not load {image_file}")
                    continue

                # CRITICAL FIX: Ensure image is writable
                if not im.flags.writeable:
                    im = im.copy()

                safe_print(f"Preprocessing image {i + 1} of {len(image_files)}: {image_file}")

                # Apply preprocessing in Igor Pro order
                if streak_removal_sdevs > 0:
                    safe_print(f"  RemoveStreaks with {streak_removal_sdevs} std devs")
                    RemoveStreaks(im, sigma=streak_removal_sdevs)

                if flatten_order > 0:
                    safe_print(f"  Flatten with order {flatten_order}")
                    # Use automatic threshold for batch processing to avoid threading issues
                    result = Flatten(im, flatten_order, noThresh=True)
                    if result != 0:
                        safe_print(f"  Flattening failed for {image_file}")
                        continue

                # Save to duplicated folder - matching Igor Pro behavior
                output_path = os.path.join(preprocessed_folder_path, image_file)
                if image_file.endswith('.npy'):
                    DataManager.save_wave_data(im, output_path)
                else:
                    # Save as .npy for processed data
                    output_npy = os.path.join(preprocessed_folder_path,
                                              os.path.splitext(image_file)[0] + '.npy')
                    DataManager.save_wave_data(im, output_npy)

            except Exception as e:
                handle_error("BatchPreprocess", e, f"image {image_file}")
                continue

        safe_print("Preprocessing complete.")
        return 0

    except Exception as e:
        error_msg = handle_error("BatchPreprocess", e)
        messagebox.showerror("Preprocessing Error", error_msg)
        return -1


def _get_preprocess_parameters():
    """Get preprocessing parameters from user with validation"""
    try:
        root = tk.Tk()
        root.title("Preprocessing Parameters")
        root.geometry("500x300")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (500 // 2)
        y = (root.winfo_screenheight() // 2) - (300 // 2)
        root.geometry(f"500x300+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Preprocessing Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=15)

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=30)

        # Default preprocessing values
        streak_var = tk.DoubleVar(value=3)  # 3 standard deviations
        flatten_var = tk.IntVar(value=2)  # 2nd order polynomial

        # Streak removal parameter
        frame1 = tk.Frame(main_frame, bg='#f0f0f0')
        frame1.pack(fill='x', pady=10)
        tk.Label(frame1, text="Std. Deviations for streak removal:",
                 width=35, anchor='w', font=('Arial', 11), bg='#f0f0f0').pack(side='left')
        tk.Entry(frame1, textvariable=streak_var, width=15, font=('Arial', 11)).pack(side='right')

        # Flattening parameter
        frame2 = tk.Frame(main_frame, bg='#f0f0f0')
        frame2.pack(fill='x', pady=10)
        tk.Label(frame2, text="Polynomial order for flattening:",
                 width=35, anchor='w', font=('Arial', 11), bg='#f0f0f0').pack(side='left')
        tk.Entry(frame2, textvariable=flatten_var, width=15, font=('Arial', 11)).pack(side='right')

        # Help text
        help_text = tk.Label(main_frame,
                             text="Note: Enter 0 to skip either preprocessing step\n" +
                                  "Streak removal: removes horizontal artifacts\n" +
                                  "Flattening: removes background slope/curvature",
                             font=('Arial', 9), fg='#7f8c8d', bg='#f0f0f0')
        help_text.pack(pady=15)

        result = [None]

        def on_ok():
            try:
                result[0] = (streak_var.get(), flatten_var.get())
                root.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

        def on_cancel():
            root.destroy()

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=on_ok,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]

    except Exception as e:
        handle_error("_get_preprocess_parameters", e)
        return None


def Flatten(im, order, mask=None, noThresh=False):
    """Flattens horizontal line scans - EXACT IGOR PRO ALGORITHM."""
    try:
        # CRITICAL FIX: Ensure image is writable by making a copy if needed
        if not im.flags.writeable:
            im = im.copy()

        # Interactive threshold determination - matching Igor Pro logic exactly
        if not noThresh and mask is None:
            if threading.current_thread() == threading.main_thread():
                # Interactive mode - matching Igor Pro interface
                threshold = _get_flatten_threshold_interactive(im)
                if threshold is None:
                    safe_print("Flattening cancelled by user.")
                    return -1
            else:
                # Automatic mode for threading
                threshold = np.mean(im) + 0.5 * np.std(im)
                safe_print(f"Automatic Flatten Height Threshold: {threshold}")

            mask = (im <= threshold).astype(int)
            safe_print(f"Flatten Height Threshold: {threshold}")
        elif noThresh and mask is None:
            # Automatic threshold - matching Igor Pro noThresh behavior
            threshold = np.mean(im) + 0.5 * np.std(im)
            mask = (im <= threshold).astype(int)

        if mask is None:
            mask = np.ones_like(im, dtype=int)

        # Make 1D waves for fitting - matching Igor Pro FLATTEN_SCANLINE, FLATTEN_MASK
        lines = im.shape[1]
        x = np.arange(im.shape[0])

        # Fit to each scan line - matching Igor Pro algorithm exactly
        for i in range(lines):
            try:
                scanline = im[:, i].copy()
                mask_1d = mask[:, i] if mask is not None else np.ones(im.shape[0], dtype=int)

                valid_indices = mask_1d == 1
                if np.sum(valid_indices) < order + 1:
                    continue

                x_valid = x[valid_indices]
                y_valid = scanline[valid_indices]

                # Igor Pro CurveFit behavior exactly
                if order == 1:
                    # CurveFit/W=2 /Q line
                    coefs = np.polyfit(x_valid, y_valid, 1)
                    im[:, i] -= coefs[1] + x * coefs[0]  # Coefs[0] + x*Coefs[1]
                elif order == 0:
                    # CurveFit/W=2 /Q /H="01" line
                    coef = np.mean(y_valid)
                    im[:, i] -= coef
                elif order > 1:
                    # CurveFit/W=2 /Q Poly
                    coefs = np.polyfit(x_valid, y_valid, order)
                    im[:, i] -= np.polyval(coefs, x)

            except Exception as e:
                handle_error("Flatten", e, f"scan line {i}")
                continue

        return 0

    except Exception as e:
        handle_error("Flatten", e)
        return -1


def _get_flatten_threshold_interactive(im):
    """Interactive threshold selection for flattening - EXACT IGOR PRO INTERFACE."""
    try:
        # Ensure we're in main thread for GUI operations
        if threading.current_thread() != threading.main_thread():
            safe_print("Warning: Interactive flattening requires main thread. Using automatic threshold.")
            return np.mean(im) + 0.5 * np.std(im)

        # Close any existing plots
        plt.close('all')

        import matplotlib
        matplotlib.use('TkAgg')

        # Create display matching Igor Pro flattening interface exactly
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.8)

        # Display image - matching Igor Pro display
        im_display = ax.imshow(im, cmap='hot')
        ax.set_title('Flattening User Interface\nChoose threshold to mask off particles', fontsize=14)

        # Create mask overlay - matching Igor Pro blue mask
        mask_overlay = np.zeros_like(im)
        mask_im = ax.imshow(mask_overlay, cmap='Blues', alpha=0.7, vmin=0, vmax=1)

        # Initial threshold - matching Igor Pro FLATTEN_THRESH
        flatten_thresh = np.mean(im)

        # Create slider - matching Igor Pro ThreshSlide
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Height Threshold', np.min(im), np.max(im),
                        valinit=flatten_thresh, valfmt='%.2e')

        # Create text display panel exactly like Igor Pro
        ax_text = plt.axes([0.82, 0.3, 0.15, 0.4])
        ax_text.axis('off')
        threshold_text = ax_text.text(0.1, 0.9, f'Height Threshold:\n{flatten_thresh:.3e}',
                                      fontsize=10, transform=ax_text.transAxes)

        def update_mask(thresh):
            """Update mask display - matching Igor Pro FlattenSlider"""
            # Mask = Im <= FLATTEN_THRESH - exact Igor Pro logic
            mask_overlay[:] = (im <= thresh).astype(float)
            mask_im.set_array(mask_overlay)

            # Update title and text with threshold value - matching Igor Pro display
            ax.set_title(f'Flattening User Interface\nHeight Threshold: {thresh:.3e}\nMasked pixels appear in blue',
                         fontsize=14)

            threshold_text.set_text(
                f'Height Threshold:\n{thresh:.3e}\n\nMasked Area:\n{np.sum(mask_overlay) / mask_overlay.size * 100:.1f}%')
            fig.canvas.draw_idle()

        slider.on_changed(update_mask)
        update_mask(flatten_thresh)

        # Create buttons - matching Igor Pro Accept button exactly
        ax_accept = plt.axes([0.75, 0.02, 0.1, 0.04])
        button_accept = Button(ax_accept, 'Accept')

        # Add instructions matching Igor Pro
        instructions = ('Adjust slider to set height threshold.\n'
                        'Blue areas will be masked off (ignored)\n'
                        'during polynomial fitting.\n'
                        'Click "Accept" when satisfied.')
        ax_text.text(0.1, 0.4, instructions, fontsize=9,
                     transform=ax_text.transAxes, style='italic')

        result = [None]

        def accept_threshold(event):
            """Accept threshold - matching Igor Pro FlattenButton"""
            result[0] = slider.val
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)

        # Instructions matching Igor Pro
        safe_print("Adjust slider to set height threshold for flattening.")
        safe_print("Blue areas will be masked off (ignored) during polynomial fitting.")
        safe_print("Click 'Accept' when satisfied with the threshold.")

        plt.show(block=True)

        return result[0]

    except Exception as e:
        handle_error("_get_flatten_threshold_interactive", e)
        # Fallback to automatic threshold
        return np.mean(im) + 0.5 * np.std(im)


def RemoveStreaks(image, sigma=3):
    """Removes streak artifacts from the image - EXACT IGOR PRO ALGORITHM."""
    try:
        # CRITICAL FIX: Ensure image is writable by making a copy if needed
        if not image.flags.writeable:
            image = image.copy()

        # Produce the dY map exactly like Igor Pro
        dy_map = dyMap_func(image)
        dy_map = np.abs(dy_map)

        # Calculate statistics - matching Igor Pro exactly
        max_dy = np.mean(dy_map) + np.std(dy_map) * sigma
        avg_dy = np.mean(dy_map)

        safe_print(f"Streak removal statistics:")
        safe_print(f"  Average dY: {avg_dy:.6e}")
        safe_print(f"  Threshold dY: {max_dy:.6e}")
        safe_print(f"  Sigma multiplier: {sigma}")

        # Process streaks - matching Igor Pro algorithm exactly
        lim_i, lim_j = image.shape[0], image.shape[1] - 1
        streaks_removed = 0

        for i in range(lim_i):
            for j in range(1, lim_j):
                if dy_map[i, j] > max_dy:
                    i0 = i
                    streak_length = 0

                    # Go left until the left side of the streak is gone
                    while i >= 0 and dy_map[i, j] > avg_dy:
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dy_map[i, j] = 0
                        streak_length += 1
                        i -= 1

                    i = i0

                    # Go right from the original point doing the same thing
                    while i < lim_i and dy_map[i, j] > avg_dy:
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dy_map[i, j] = 0
                        streak_length += 1
                        i += 1

                    i = i0
                    if streak_length > 0:
                        streaks_removed += 1

        safe_print(f"  Streaks removed: {streaks_removed}")
        return 0

    except Exception as e:
        handle_error("RemoveStreaks", e)
        return -1


def dyMap_func(image):
    """Create dy map for streak detection - EXACT IGOR PRO ALGORITHM."""
    try:
        dyMap = np.zeros_like(image)
        limQ = image.shape[1] - 1

        # Exact Igor Pro calculation
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                next_j = min(j + 1, limQ)
                prev_j = max(j - 1, 0)
                dyMap[i, j] = image[i, j] - (image[i, next_j] + image[i, prev_j]) / 2

        return dyMap

    except Exception as e:
        handle_error("dyMap_func", e)
        return np.zeros_like(image)


def _get_flatten_threshold_interactive(im):
    """Interactive threshold selection for flattening - IGOR PRO EXACT INTERFACE."""
    try:
        # Ensure we're in main thread for GUI operations
        if threading.current_thread() != threading.main_thread():
            safe_print("Warning: Interactive flattening requires main thread. Using automatic threshold.")
            return np.mean(im) + 0.5 * np.std(im)

        # Close any existing plots
        plt.close('all')

        import matplotlib
        matplotlib.use('TkAgg')

        # Create display matching Igor Pro flattening interface exactly
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.8)

        # Display image - matching Igor Pro display
        im_display = ax.imshow(im, cmap='hot')
        ax.set_title('Flattening User Interface\nChoose threshold to mask off particles', fontsize=14)

        # Create mask overlay - matching Igor Pro blue mask
        mask_overlay = np.zeros_like(im)
        mask_im = ax.imshow(mask_overlay, cmap='Blues', alpha=0.7, vmin=0, vmax=1)

        # Initial threshold - matching Igor Pro FLATTEN_THRESH
        flatten_thresh = np.mean(im)

        # Create slider - matching Igor Pro ThreshSlide
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Height Threshold', np.min(im), np.max(im),
                        valinit=flatten_thresh, valfmt='%.2e')

        def update_mask(thresh):
            """Update mask display - matching Igor Pro FlattenSlider"""
            # Mask = Im <= FLATTEN_THRESH - exact Igor Pro logic
            mask_overlay[:] = (im <= thresh).astype(float)
            mask_im.set_array(mask_overlay)

            # Update title with threshold value - matching Igor Pro display
            ax.set_title(f'Flattening User Interface\nHeight Threshold: {thresh:.3e}\nMasked pixels appear in blue',
                         fontsize=14)
            fig.canvas.draw_idle()

        slider.on_changed(update_mask)
        update_mask(flatten_thresh)

        # Create buttons - matching Igor Pro Accept button exactly
        ax_accept = plt.axes([0.75, 0.02, 0.1, 0.04])
        button_accept = Button(ax_accept, 'Accept')

        result = [None]

        def accept_threshold(event):
            """Accept threshold - matching Igor Pro FlattenButton"""
            result[0] = slider.val
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)

        # Instructions matching Igor Pro
        safe_print("Adjust slider to set height threshold for flattening.")
        safe_print("Blue areas will be masked off (ignored) during polynomial fitting.")
        safe_print("Click 'Accept' when satisfied with the threshold.")

        plt.show(block=True)

        return result[0]

    except Exception as e:
        handle_error("_get_flatten_threshold_interactive", e)
        # Fallback to automatic threshold
        return np.mean(im) + 0.5 * np.std(im)


def RemoveStreaks(image, sigma=3):
    """Removes streak artifacts from the image - IGOR PRO EXACT ALGORITHM."""
    try:
        # CRITICAL FIX: Ensure image is writable by making a copy if needed
        if not image.flags.writeable:
            image = image.copy()

        # Produce the dY map
        dy_map = dyMap_func(image)
        dy_map = np.abs(dy_map)

        # Calculate statistics - matching Igor Pro exactly
        max_dy = np.mean(dy_map) + np.std(dy_map) * sigma
        avg_dy = np.mean(dy_map)

        # Process streaks - matching Igor Pro algorithm exactly
        lim_i, lim_j = image.shape[0], image.shape[1] - 1

        for i in range(lim_i):
            for j in range(1, lim_j):
                if dy_map[i, j] > max_dy:
                    i0 = i

                    # Go left until the left side of the streak is gone
                    while i >= 0 and dy_map[i, j] > avg_dy:
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dy_map[i, j] = 0
                        i -= 1

                    i = i0

                    # Go right from the original point doing the same thing
                    while i < lim_i and dy_map[i, j] > avg_dy:
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dy_map[i, j] = 0
                        i += 1
                    i = i0

        return 0

    except Exception as e:
        handle_error("RemoveStreaks", e)
        return -1


# ========================================================================
# UTILITIES
# ========================================================================

def FixBoundaries(detH):
    """Fixes a boundary issue in the blob detectors."""
    try:
        limP, limQ = detH.shape[:2]
        limP -= 1
        limQ -= 1

        # Do the sides first. Corners need extra care.
        for i in range(2, limP - 1):
            detH[i, 0, :] = detH[i, 2, :] / 3
            detH[i, 1, :] = detH[i, 2, :] * 2 / 3

        for i in range(2, limP - 1):
            detH[i, limQ, :] = detH[i, limQ - 2, :] / 3
            detH[i, limQ - 1, :] = detH[i, limQ - 2, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[0, i, :] = detH[2, i, :] / 3
            detH[1, i, :] = detH[2, i, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[limP, i, :] = detH[limP - 2, i, :] / 3
            detH[limP - 1, i, :] = detH[limP - 2, i, :] * 2 / 3

        # Corners
        # Top Left Corner
        detH[1, 1, :] = (detH[1, 2, :] + detH[2, 1, :]) / 2
        detH[1, 0, :] = (detH[1, 1, :] + detH[2, 0, :]) / 2
        detH[0, 1, :] = (detH[1, 1, :] + detH[0, 2, :]) / 2
        detH[0, 0, :] = (detH[0, 1, :] + detH[1, 0, :]) / 2

        # Bottom Right Corner
        detH[limP - 1, limQ - 1, :] = (detH[limP - 1, limQ - 2, :] + detH[limP - 2, limQ - 1, :]) / 2
        detH[limP - 1, limQ, :] = (detH[limP - 1, limQ - 1, :] + detH[limP - 2, limQ, :]) / 2
        detH[limP, limQ - 1, :] = (detH[limP - 1, limQ - 1, :] + detH[limP, limQ - 2, :]) / 2
        detH[limP, limQ, :] = (detH[limP - 1, limQ, :] + detH[limP, limQ - 1, :]) / 2

        # Top Right Corner
        detH[limP - 1, 1, :] = (detH[limP - 1, 2, :] + detH[limP - 2, 1, :]) / 2
        detH[limP - 1, 0, :] = (detH[limP - 1, 1, :] + detH[limP - 2, 0, :]) / 2
        detH[limP, 1, :] = (detH[limP - 1, 1, :] + detH[limP, 2, :]) / 2
        detH[limP, 0, :] = (detH[limP - 1, 0, :] + detH[limP, 1, :]) / 2

        # Bottom Left Corner
        detH[1, limQ - 1, :] = (detH[1, limQ - 2, :] + detH[2, limQ - 1, :]) / 2
        detH[1, limQ, :] = (detH[1, limQ - 1, :] + detH[2, limQ, :]) / 2
        detH[0, limQ - 1, :] = (detH[1, limQ - 1, :] + detH[0, limQ - 2, :]) / 2
        detH[0, limQ, :] = (detH[1, limQ, :] + detH[0, limQ - 1, :]) / 2

        return 0

    except Exception as e:
        handle_error("FixBoundaries", e)
        return -1


def GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """Returns a wave with the values of the local maxes of the determinant of Hessian."""
    try:
        Maxes = []
        limI, limJ, limK = detH.shape

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum
                    is_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < limI and 0 <= nj < limJ and 0 <= nk < limK):
                                    if detH[ni, nj, nk] > detH[i, j, k]:
                                        is_max = False
                                        break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if not is_max:
                        continue

                    Maxes.append(detH[i, j, k])

                    if map_wave is not None:
                        map_wave[i, j] = max(map_wave[i, j], detH[i, j, k])

                    if scaleMap is not None:
                        scaleMap[i, j] = 1.0 * (1.5 ** k)

        return np.array(Maxes)

    except Exception as e:
        handle_error("GetMaxes", e)
        return np.array([])


def FindHessianBlobs(im, detH, LG, minResponse, particleType, maxCurvatureRatio):
    """Find Hessian blobs by detecting scale-space extrema."""
    try:
        # Square the minResponse
        minResponse = minResponse ** 2

        # mapNum: Map identifying particle numbers
        mapNum = np.full(detH.shape, -1, dtype=int)

        # mapLG: Map identifying the value of the LoG at the defined scale
        mapLG = np.zeros_like(detH)

        # mapMax: Map identifying the value of the LoG of the maximum pixel
        mapMax = np.zeros_like(detH)

        # Maintain an info list with particle boundaries and info
        Info = []

        limI, limJ, limK = detH.shape
        cnt = 0

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Does it hit the threshold?
                    if detH[i, j, k] < minResponse:
                        continue

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is there a particle there already?
                    if mapNum[i, j, k] > -1 and detH[i, j, k] <= Info[mapNum[i, j, k]][3]:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum
                    is_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < limI and 0 <= nj < limJ and 0 <= nk < limK):
                                    if detH[ni, nj, nk] > detH[i, j, k]:
                                        is_max = False
                                        break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if not is_max:
                        continue

                    # It's a local max, is it overlapped and bigger than another one already?
                    if mapNum[i, j, k] > -1:
                        Info[mapNum[i, j, k]][0] = i
                        Info[mapNum[i, j, k]][1] = j
                        Info[mapNum[i, j, k]][3] = detH[i, j, k]
                        continue

                    # It's a local max, proceed to fill out the feature
                    numPixels, boundingBox = ScanlineFill8_LG(detH, mapMax, LG, i, j, k, 0, mapNum, cnt)

                    if numPixels > 0:
                        particle_info = [0] * 15
                        particle_info[0] = i
                        particle_info[1] = j
                        particle_info[2] = numPixels
                        particle_info[3] = detH[i, j, k]
                        particle_info[4] = boundingBox[0]
                        particle_info[5] = boundingBox[1]
                        particle_info[6] = boundingBox[2]
                        particle_info[7] = boundingBox[3]
                        particle_info[8] = 1.0 * (1.5 ** k)  # scale
                        particle_info[9] = k
                        particle_info[10] = 1
                        Info.append(particle_info)
                        cnt += 1

        # Make the mapLG
        mapLG = np.where(mapNum != -1, detH, 0)

        return mapNum, mapLG, mapMax, Info

    except Exception as e:
        handle_error("FindHessianBlobs", e)
        return np.full(detH.shape, -1, dtype=int), np.zeros_like(detH), np.zeros_like(detH), []


def MaximalBlobs(info, mapNum):
    """Determine scale-maximal particles."""
    try:
        if len(info) == 0:
            return -1

        # Initialize maximality of each particle as undetermined (-1)
        for i in range(len(info)):
            info[i][10] = -1

        # Sort by blob strength
        sorted_indices = sorted(range(len(info)), key=lambda i: info[i][3], reverse=True)

        limK = mapNum.shape[2]

        for idx_pos in range(len(info)):
            blocked = False
            index = sorted_indices[idx_pos]
            k = int(info[index][9])

            for ii in range(int(info[index][4]), int(info[index][5]) + 1):
                for jj in range(int(info[index][6]), int(info[index][7]) + 1):
                    if mapNum[ii, jj, k] == index:
                        for kk in range(limK):
                            if mapNum[ii, jj, kk] != -1 and info[mapNum[ii, jj, kk]][10] == 1:
                                blocked = True
                                break
                        if blocked:
                            break
                if blocked:
                    break

            info[index][10] = 0 if blocked else 1

        return 0

    except Exception as e:
        handle_error("MaximalBlobs", e)
        return -1


def ScanlineFill8_LG(image, dest, LG, seedP, seedQ, layer, Thresh, dest2=None, fillVal2=None):
    """Scanline fill algorithm for blob detection."""
    try:
        fill = image[seedP, seedQ, layer]
        count = 0
        sgn = np.sign(LG[seedP, seedQ, layer])

        x0, xf = 0, image.shape[0] - 1
        y0, yf = 0, image.shape[1] - 1

        if seedP < x0 or seedQ < y0 or seedP > xf or seedQ > yf:
            return 0, [0, 0, 0, 0]
        elif image[seedP, seedQ, layer] <= Thresh:
            return 0, [0, 0, 0, 0]

        stack = [(seedP, seedQ)]
        visited = set()
        pixels = []

        while stack:
            i, j = stack.pop()

            if (i, j) in visited:
                continue
            if i < x0 or i > xf or j < y0 or j > yf:
                continue
            if image[i, j, layer] < Thresh:
                continue
            if np.sign(LG[i, j, layer]) != sgn:
                continue

            visited.add((i, j))
            pixels.append((i, j))
            dest[i, j, layer] = fill
            count += 1

            if dest2 is not None and fillVal2 is not None:
                dest2[i, j, layer] = fillVal2

            # Add 8-connected neighbors
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

        if len(pixels) == 0:
            return 0, [0, 0, 0, 0]

        p_coords = [p for p, q in pixels]
        q_coords = [q for p, q in pixels]

        BoundingBox = [min(p_coords), max(p_coords), min(q_coords), max(q_coords)]
        return count, BoundingBox

    except Exception as e:
        handle_error("ScanlineFill8_LG", e)
        return 0, [0, 0, 0, 0]


def dyMap_func(image):
    """Create dy map for streak detection."""
    try:
        dyMap = np.zeros_like(image)
        limQ = image.shape[1] - 1

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                next_j = min(j + 1, limQ)
                prev_j = max(j - 1, 0)
                dyMap[i, j] = image[i, j] - (image[i, next_j] + image[i, prev_j]) / 2

        return dyMap

    except Exception as e:
        handle_error("dyMap_func", e)
        return np.zeros_like(image)


# ========================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# ========================================================================

def ViewParticles():
    """View and examine individual detected particles - EXACT IGOR PRO TUTORIAL MATCH."""
    try:
        # Get the folder containing particles - matching Igor Pro GetBrowserSelection
        particles_folder = filedialog.askdirectory(title="Select particles folder (e.g., YEG_1_Particles)")
        if not particles_folder:
            safe_print("No folder selected.")
            return

        # Look for Particle_X folders - matching Igor Pro structure exactly
        particle_folders = []
        try:
            for item in os.listdir(particles_folder):
                item_path = os.path.join(particles_folder, item)
                if os.path.isdir(item_path) and item.startswith("Particle_"):
                    try:
                        # Verify it's a valid particle number
                        particle_num = int(item.split("_")[-1])
                        particle_folders.append(item_path)
                    except ValueError:
                        continue
        except Exception as e:
            safe_print(f"Error reading folder contents: {e}")
            return

        if len(particle_folders) == 0:
            safe_print("No particle folders found in selected directory.")
            safe_print("Make sure you selected a folder like 'YEG_1_Particles' that contains 'Particle_X' subfolders.")
            return

        # Sort by particle number - matching Igor Pro order
        particle_folders.sort(key=lambda x: int(x.split("_")[-1]))

        safe_print(f"Found {len(particle_folders)} particles to view.")

        # Import required for image display
        try:
            from PIL import ImageTk
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
        except ImportError:
            safe_print("PIL/Pillow or matplotlib not found. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'Pillow'])
            from PIL import ImageTk
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize

        # Create Igor Pro-style particle viewer
        class IgorProParticleViewer:
            def __init__(self, folders):
                self.folders = folders
                self.current_index = 0

                # Create main window - matching Igor Pro Particle Viewer size
                self.root = tk.Toplevel()
                self.root.title("Particle Viewer")
                self.root.geometry("1200x900")
                self.root.configure(bg='#e6e6e6')  # Igor Pro gray background

                # Set up the interface matching Figure 24 from tutorial
                self.setup_igor_interface()

                # Show first particle
                self.show_particle()

                # Bind keyboard events - matching Igor Pro shortcuts
                self.root.bind('<Key>', self.on_key_press)
                self.root.focus_set()

            def setup_igor_interface(self):
                """Set up interface matching Igor Pro tutorial Figure 24"""

                # Main content frame
                main_frame = tk.Frame(self.root, bg='#e6e6e6')
                main_frame.pack(fill='both', expand=True, padx=10, pady=10)

                # Left side - image display (larger, matching Igor Pro)
                left_frame = tk.Frame(main_frame, bg='#e6e6e6', width=800)
                left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
                left_frame.pack_propagate(False)

                # Image frame with border (matching Igor Pro image window)
                image_frame = tk.Frame(left_frame, bg='white', relief='sunken', bd=2)
                image_frame.pack(pady=10, padx=10, fill='both', expand=True)

                # Image canvas - matching Igor Pro image size
                self.canvas = tk.Canvas(image_frame, bg='black', width=700, height=700)
                self.canvas.pack(padx=5, pady=5)

                # Right side - controls panel (matching Igor Pro Controls panel)
                right_frame = tk.Frame(main_frame, bg='#e6e6e6', width=350)
                right_frame.pack(side='right', fill='y')
                right_frame.pack_propagate(False)

                # Controls title - matching Igor Pro "Controls" panel
                controls_title = tk.Label(right_frame, text="Controls",
                                          font=('Arial', 14, 'bold'), bg='#e6e6e6')
                controls_title.pack(pady=(0, 10))

                # Particle title - matching Igor Pro "Particle 21" style
                self.particle_title = tk.Label(right_frame, text="Particle 0",
                                               font=('Arial', 16, 'bold'), bg='#e6e6e6')
                self.particle_title.pack(pady=5)

                # Navigation buttons - matching Igor Pro layout
                nav_frame = tk.Frame(right_frame, bg='#e6e6e6')
                nav_frame.pack(pady=10)

                # Prev and Next buttons side by side
                btn_frame = tk.Frame(nav_frame, bg='#e6e6e6')
                btn_frame.pack()

                self.prev_btn = tk.Button(btn_frame, text="Prev",
                                          command=self.prev_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.prev_btn.pack(side='left', padx=5)

                self.next_btn = tk.Button(btn_frame, text="Next",
                                          command=self.next_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.next_btn.pack(side='left', padx=5)

                # Go To field - matching Igor Pro "Go To: 0"
                goto_frame = tk.Frame(right_frame, bg='#e6e6e6')
                goto_frame.pack(pady=10)

                tk.Label(goto_frame, text="Go To:", font=('Arial', 12), bg='#e6e6e6').pack(side='left')
                self.goto_entry = tk.Entry(goto_frame, width=10, font=('Arial', 12))
                self.goto_entry.pack(side='left', padx=5)
                self.goto_entry.bind('<Return>', self.goto_particle)

                # Counter display - matching Igor Pro "1/10" style
                self.counter_label = tk.Label(right_frame, text="1/1",
                                              font=('Arial', 12), bg='#e6e6e6')
                self.counter_label.pack(pady=5)

                # Measurements section - matching Igor Pro Figure 24
                measurements_frame = tk.LabelFrame(right_frame, text="Measurements",
                                                   font=('Arial', 12, 'bold'), bg='#e6e6e6')
                measurements_frame.pack(fill='x', pady=10, padx=10)

                # Height display - matching Igor Pro "Height (nm)"
                height_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                height_frame.pack(fill='x', pady=5)
                tk.Label(height_frame, text="Height (nm)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.height_label = tk.Label(height_frame, text="0.0000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.height_label.pack(fill='x', padx=5)

                # Volume display - matching Igor Pro "Volume (m^3 e-25)"
                volume_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                volume_frame.pack(fill='x', pady=5)
                tk.Label(volume_frame, text="Volume (m^3 e-25)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.volume_label = tk.Label(volume_frame, text="0.000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.volume_label.pack(fill='x', padx=5)

                # Area display - additional measurement
                area_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                area_frame.pack(fill='x', pady=5)
                tk.Label(area_frame, text="Area", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.area_label = tk.Label(area_frame, text="0.0",
                                           font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.area_label.pack(fill='x', padx=5)

                # DELETE button - matching Igor Pro red DELETE button
                self.delete_btn = tk.Button(right_frame, text="DELETE",
                                            command=self.delete_particle,
                                            bg='#ff6b6b', fg='white',
                                            font=('Arial', 12, 'bold'),
                                            width=20, height=2, relief='raised', bd=2)
                self.delete_btn.pack(pady=20)

            def show_particle(self):
                """Display current particle - matching Igor Pro tutorial visualization"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Load particle files
                    particle_file = os.path.join(current_folder, f"Particle_{particle_num}.npy")
                    mask_file = os.path.join(current_folder, f"Mask_{particle_num}.npy")
                    info_file = os.path.join(current_folder, f"Particle_{particle_num}_info.txt")

                    if not os.path.exists(particle_file):
                        safe_print(f"Particle file not found: {particle_file}")
                        return

                    particle = np.load(particle_file)

                    # Update labels - matching Igor Pro style
                    self.particle_title.config(text=f"Particle {particle_num}")
                    self.counter_label.config(text=f"{self.current_index + 1}/{len(self.folders)}")
                    self.goto_entry.delete(0, tk.END)
                    self.goto_entry.insert(0, str(self.current_index))

                    # Display particle image with hot colormap and green contour
                    self.display_igor_image(particle, mask_file)

                    # Update measurements - matching Igor Pro format
                    self.update_measurements(info_file)

                    # Update window title
                    self.root.title(f"Particle Viewer - Particle {particle_num}")

                except Exception as e:
                    handle_error("show_particle", e)

            def display_igor_image(self, particle, mask_file):
                """Display particle image matching Igor Pro tutorial Figure 24 exactly"""
                try:
                    # Clear canvas
                    self.canvas.delete("all")

                    # Apply hot colormap exactly like Igor Pro
                    # Normalize data
                    vmin, vmax = np.min(particle), np.max(particle)
                    norm = Normalize(vmin=vmin, vmax=vmax)

                    hot_cmap = cm.get_cmap('hot')
                    colored = hot_cmap(norm(particle))
                    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

                    # Convert to PIL Image
                    img = Image.fromarray(colored_rgb)

                    # Resize to fit canvas (matching Igor Pro size)
                    canvas_size = 650
                    img_resized = img.resize((canvas_size, canvas_size), Image.NEAREST)

                    # Convert to PhotoImage
                    self.photo = ImageTk.PhotoImage(img_resized)

                    # Display on canvas
                    self.canvas.create_image(350, 350, image=self.photo)

                    # Add GREEN contour exactly like Igor Pro Figure 24
                    if os.path.exists(mask_file):
                        try:
                            mask = np.load(mask_file)

                            # Resize mask to match image
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_resized = np.array(mask_img.resize((canvas_size, canvas_size), Image.NEAREST))

                            # Create GREEN contour - matching Igor Pro exactly
                            contour_points = []
                            for i in range(1, canvas_size - 1):
                                for j in range(1, canvas_size - 1):
                                    if mask_resized[i, j] > 127:  # Inside mask
                                        # Check if it's an edge pixel
                                        neighbors = [
                                            mask_resized[i - 1, j], mask_resized[i + 1, j],
                                            mask_resized[i, j - 1], mask_resized[i, j + 1]
                                        ]
                                        if any(n < 127 for n in neighbors):
                                            x, y = j + 25, i + 25  # Offset for centering
                                            # Draw GREEN pixels - matching Igor Pro green contour
                                            self.canvas.create_rectangle(x, y, x + 2, y + 2,
                                                                         fill='#00ff00', outline='#00ff00')

                        except Exception as e:
                            safe_print(f"Could not display mask contour: {e}")

                    # Add coordinate axes labels if needed (matching Igor Pro)
                    self.canvas.create_text(350, 680, text="Pixels", font=('Arial', 10))

                except Exception as e:
                    handle_error("display_igor_image", e)

            def update_measurements(self, info_file):
                """Update measurements display - matching Igor Pro format"""
                try:
                    height_val = "0.0000"
                    volume_val = "0.000"
                    area_val = "0.0"

                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith('Height:'):
                                    height_val = line.split(':')[1].strip()
                                elif line.startswith('Volume:'):
                                    volume_val = line.split(':')[1].strip()
                                elif line.startswith('Area:'):
                                    area_val = line.split(':')[1].strip()

                    # Update labels with proper formatting
                    self.height_label.config(text=height_val)
                    self.volume_label.config(text=volume_val)
                    self.area_label.config(text=area_val)

                except Exception as e:
                    handle_error("update_measurements", e)

            def prev_particle(self):
                """Navigate to previous particle - matching Igor Pro Prev button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index - 1) % len(self.folders)
                    self.show_particle()

            def next_particle(self):
                """Navigate to next particle - matching Igor Pro Next button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index + 1) % len(self.folders)
                    self.show_particle()

            def goto_particle(self, event=None):
                """Go to specific particle - matching Igor Pro Go To functionality"""
                try:
                    particle_index = int(self.goto_entry.get())
                    if 0 <= particle_index < len(self.folders):
                        self.current_index = particle_index
                        self.show_particle()
                except ValueError:
                    pass  # Invalid input, ignore

            def delete_particle(self):
                """Delete current particle - matching Igor Pro DELETE button"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Matching Igor Pro delete dialog exactly
                    result = messagebox.askyesno(
                        f"Deleting Particle {particle_num}..",
                        f"Are you sure you want to delete Particle {particle_num}?",
                        parent=self.root
                    )

                    if result:
                        shutil.rmtree(current_folder)
                        self.folders.pop(self.current_index)

                        if len(self.folders) == 0:
                            safe_print("No more particles to view.")
                            self.root.destroy()
                            return

                        if self.current_index >= len(self.folders):
                            self.current_index = len(self.folders) - 1

                        self.show_particle()

                except Exception as e:
                    handle_error("delete_particle", e)

            def on_key_press(self, event):
                """Handle keyboard shortcuts - matching Igor Pro tutorial"""
                if event.keysym == 'Left':
                    self.prev_particle()
                elif event.keysym == 'Right':
                    self.next_particle()
                elif event.keysym == 'space' or event.keysym == 'Down':
                    self.delete_particle()
                elif event.keysym == 'Return':
                    self.goto_particle()

        # Create and run viewer
        viewer = IgorProParticleViewer(particle_folders)

        # Print Igor Pro-style instructions
        safe_print("=" * 60)
        safe_print("PARTICLE VIEWER CONTROLS:")
        safe_print("- Left/Right arrows: Navigate between particles")
        safe_print("- Space bar or Down arrow: Delete current particle")
        safe_print("- Enter in Go To field: Jump to particle")
        safe_print("- Close window when finished")
        safe_print("=" * 60)

    except Exception as e:
        error_msg = handle_error("ViewParticles", e)
        try:
            messagebox.showerror("Viewer Error", error_msg)
        except:
            safe_print(error_msg)


def WaveStats(data_file):
    """Compute basic statistics exactly matching Igor Pro WaveStats output."""
    try:
        if isinstance(data_file, str):
            # Load from file
            if data_file.endswith('.npy'):
                data = np.load(data_file)
            else:
                raise HessianBlobError("Unsupported file format. Please use .npy files.")
            base_name = os.path.splitext(os.path.basename(data_file))[0]
        else:
            # Assume it's already a numpy array
            data = data_file
            base_name = "data"

        # Clean data (remove NaN values for statistics) exactly like Igor Pro
        clean_data = data[~np.isnan(data.flatten())]

        # Calculate statistics exactly matching Igor Pro WaveStats output
        V_npnts = len(clean_data)
        V_numNaNs = np.sum(np.isnan(data.flatten()))
        V_avg = np.mean(clean_data) if len(clean_data) > 0 else 0
        V_sum = np.sum(clean_data)
        V_sdev = np.std(clean_data, ddof=1) if len(clean_data) > 1 else 0
        V_rms = np.sqrt(np.mean(clean_data ** 2)) if len(clean_data) > 0 else 0
        V_min = np.min(clean_data) if len(clean_data) > 0 else 0
        V_max = np.max(clean_data) if len(clean_data) > 0 else 0

        # Print results in exact Igor Pro format
        safe_print(f"WaveStats {base_name}")
        safe_print(f"  V_npnts= {V_npnts}; V_numNaNs= {V_numNaNs};")
        safe_print(f"  V_avg= {V_avg:.6g}; V_sum= {V_sum:.6g};")
        safe_print(f"  V_sdev= {V_sdev:.6g}; V_rms= {V_rms:.6g};")
        safe_print(f"  V_min= {V_min:.6g}; V_max= {V_max:.6g};")

        return {
            'V_npnts': V_npnts,
            'V_numNaNs': V_numNaNs,
            'V_avg': V_avg,
            'V_sum': V_sum,
            'V_sdev': V_sdev,
            'V_rms': V_rms,
            'V_min': V_min,
            'V_max': V_max
        }

    except Exception as e:
        error_msg = handle_error("WaveStats", e)
        messagebox.showerror("Statistics Error", error_msg)
        return None


def print_series_analysis_summary(result_folder):
    """Print comprehensive series analysis summary matching Igor Pro exactly"""
    try:
        safe_print("\n" + "=" * 70)
        safe_print("SERIES ANALYSIS COMPLETE")
        safe_print("=" * 70)

        # Load all measurement arrays
        heights_file = os.path.join(result_folder, "AllHeights.npy")
        volumes_file = os.path.join(result_folder, "AllVolumes.npy")
        areas_file = os.path.join(result_folder, "AllAreas.npy")
        avg_heights_file = os.path.join(result_folder, "AllAvgHeights.npy")

        if not all(os.path.exists(f) for f in [heights_file, volumes_file, areas_file, avg_heights_file]):
            safe_print("Warning: Some measurement files not found")
            return

        heights = np.load(heights_file)
        volumes = np.load(volumes_file)
        areas = np.load(areas_file)
        avg_heights = np.load(avg_heights_file)

        # Print summary statistics exactly like Igor Pro
        safe_print(f"Total particles detected: {len(heights)}")
        safe_print(f"Results saved to: {result_folder}")
        safe_print("")

        if len(heights) > 0:
            safe_print("PARTICLE MEASUREMENT STATISTICS:")
            safe_print("-" * 50)

            # Heights statistics
            safe_print(f"Heights (n={len(heights)}):")
            safe_print(f"  Mean: {np.mean(heights):.6e} m")
            safe_print(f"  Std:  {np.std(heights, ddof=1):.6e} m")
            safe_print(f"  Min:  {np.min(heights):.6e} m")
            safe_print(f"  Max:  {np.max(heights):.6e} m")
            safe_print("")

            # Volumes statistics
            safe_print(f"Volumes (n={len(volumes)}):")
            safe_print(f"  Mean: {np.mean(volumes):.6e} m³")
            safe_print(f"  Std:  {np.std(volumes, ddof=1):.6e} m³")
            safe_print(f"  Min:  {np.min(volumes):.6e} m³")
            safe_print(f"  Max:  {np.max(volumes):.6e} m³")
            safe_print("")

            # Areas statistics
            safe_print(f"Areas (n={len(areas)}):")
            safe_print(f"  Mean: {np.mean(areas):.6e} m²")
            safe_print(f"  Std:  {np.std(areas, ddof=1):.6e} m²")
            safe_print(f"  Min:  {np.min(areas):.6e} m²")
            safe_print(f"  Max:  {np.max(areas):.6e} m²")
            safe_print("")

        # Print file structure created
        safe_print("FILES CREATED:")
        safe_print("-" * 20)
        safe_print("• AllHeights.npy    - Combined particle heights")
        safe_print("• AllVolumes.npy    - Combined particle volumes")
        safe_print("• AllAreas.npy      - Combined particle areas")
        safe_print("• AllAvgHeights.npy - Combined average heights")
        safe_print("• Parameters.npy    - Analysis parameters used")

        # Count individual image folders
        image_folders = [d for d in os.listdir(result_folder)
                         if os.path.isdir(os.path.join(result_folder, d)) and d.endswith('_Particles')]
        safe_print(f"• {len(image_folders)} individual image analysis folders")

        safe_print("\n" + "=" * 70)

    except Exception as e:
        handle_error("print_series_analysis_summary", e)


def Testing(str_input, num):
    """Testing function to demonstrate how user-defined functions work."""
    try:
        safe_print(f"You typed: {str_input}")
        safe_print(f"Your number plus two is {num + 2}")

    except Exception as e:
        handle_error("Testing", e)


# ========================================================================
# IGOR PRO COMPATIBILITY FUNCTIONS
# ========================================================================

def GetBrowserSelection(index):
    """Get folder selection (simulates Igor Pro browser selection)."""
    try:
        folder = filedialog.askdirectory(title="Select folder containing images")
        return folder if folder else ""

    except Exception as e:
        handle_error("GetBrowserSelection", e)
        return ""


# Global variable to track current data folder
current_data_folder = None


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


def NewDataFolder(folder_path):
    """Create new data folder in script directory."""
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


# ========================================================================
# MAIN GUI APPLICATION CLASS
# ========================================================================

class HessianBlobGUI:
    """Main GUI application for Hessian blob detection - matching Igor Pro tutorial style."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hessian Blob Particle Detection Suite")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Set up the GUI to match Igor Pro style
        self.setup_main_gui()

        # Output capture
        self.output_queue = queue.Queue()

        # Initialize global folder to script directory
        SetDataFolder(get_script_directory())
        safe_print(f"Working directory set to: {get_script_directory()}")

    def setup_main_gui(self):
        """Set up the main GUI similar to Igor Pro interface."""

        # Main title frame
        title_frame = tk.Frame(self.root, bg='#34495e', pady=15)
        title_frame.pack(fill='x')

        title_label = tk.Label(title_frame,
                               text="Hessian Blob Particle Detection Suite",
                               font=('Arial', 18, 'bold'),
                               fg='white', bg='#34495e')
        title_label.pack()

        subtitle_label = tk.Label(title_frame,
                                  text="G.M. King Laboratory - University of Missouri-Columbia",
                                  font=('Arial', 11),
                                  fg='#ecf0f1', bg='#34495e')
        subtitle_label.pack(pady=(5, 0))

        author_label = tk.Label(title_frame,
                                text="Python Port - Coded by: Brendan Marsh (marshbp@stanford.edu)",
                                font=('Arial', 9),
                                fg='#bdc3c7', bg='#34495e')
        author_label.pack()

        # Create main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        # Left panel for buttons (similar to Igor Pro menus)
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)

        # Right panel for output
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side='right', fill='both', expand=True)

        # Setup left panel buttons
        self.setup_button_panel(left_panel)

        # Setup right panel output
        self.setup_output_panel(right_panel)

    def setup_button_panel(self, parent):
        """Setup the button panel similar to Igor Pro menus."""

        # Section 1: Main Analysis Functions
        analysis_frame = tk.LabelFrame(parent, text="I. Main Analysis Functions",
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0', pady=10)
        analysis_frame.pack(fill='x', pady=(0, 15))

        btn_batch = tk.Button(analysis_frame,
                              text="BatchHessianBlobs()",
                              font=('Arial', 11, 'bold'),
                              width=35, height=2,
                              bg='#3498db', fg='white',
                              command=self.run_batch_hessian_blobs,
                              cursor='hand2')
        btn_batch.pack(pady=5, padx=10, fill='x')

        batch_desc = tk.Label(analysis_frame,
                              text="Detect Hessian blobs in a series of images\nin a chosen data folder.",
                              font=('Arial', 9),
                              fg='#7f8c8d', bg='#f0f0f0')
        batch_desc.pack(pady=(0, 10))

        btn_single = tk.Button(analysis_frame,
                               text="HessianBlobs(image)",
                               font=('Arial', 11, 'bold'),
                               width=35, height=2,
                               bg='#2ecc71', fg='white',
                               command=self.run_single_hessian_blobs,
                               cursor='hand2')
        btn_single.pack(pady=5, padx=10, fill='x')

        single_desc = tk.Label(analysis_frame,
                               text="Execute the Hessian blob algorithm\non a single image.",
                               font=('Arial', 9),
                               fg='#7f8c8d', bg='#f0f0f0')
        single_desc.pack(pady=(0, 5))

        # Section 2: Preprocessing Functions
        preprocess_frame = tk.LabelFrame(parent, text="II. Preprocessing Functions",
                                         font=('Arial', 12, 'bold'),
                                         bg='#f0f0f0', pady=10)
        preprocess_frame.pack(fill='x', pady=(0, 15))

        btn_preprocess = tk.Button(preprocess_frame,
                                   text="BatchPreprocess()",
                                   font=('Arial', 11, 'bold'),
                                   width=35, height=2,
                                   bg='#e67e22', fg='white',
                                   command=self.run_batch_preprocess,
                                   cursor='hand2')
        btn_preprocess.pack(pady=5, padx=10, fill='x')

        preprocess_desc = tk.Label(preprocess_frame,
                                   text="Preprocess multiple images in a data folder\nsuccessively using flattening and streak removal.",
                                   font=('Arial', 9),
                                   fg='#7f8c8d', bg='#f0f0f0')
        preprocess_desc.pack(pady=(0, 5))

        # Section 3: Data Analysis and Visualization
        analysis_viz_frame = tk.LabelFrame(parent, text="III. Data Analysis & Visualization",
                                           font=('Arial', 12, 'bold'),
                                           bg='#f0f0f0', pady=10)
        analysis_viz_frame.pack(fill='x', pady=(0, 15))

        btn_view_particles = tk.Button(analysis_viz_frame,
                                       text="ViewParticles()",
                                       font=('Arial', 11, 'bold'),
                                       width=35, height=2,
                                       bg='#9b59b6', fg='white',
                                       command=self.run_view_particles,
                                       cursor='hand2')
        btn_view_particles.pack(pady=5, padx=10, fill='x')

        view_desc = tk.Label(analysis_viz_frame,
                             text="Convenient method to view and examine\nindividual detected particles.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        view_desc.pack(pady=(0, 10))

        btn_wave_stats = tk.Button(analysis_viz_frame,
                                   text="WaveStats(data)",
                                   font=('Arial', 11, 'bold'),
                                   width=35, height=2,
                                   bg='#34495e', fg='white',
                                   command=self.run_wave_stats,
                                   cursor='hand2')
        btn_wave_stats.pack(pady=5, padx=10, fill='x')

        stats_desc = tk.Label(analysis_viz_frame,
                              text="Compute basic statistics of particle\nmeasurements (heights, areas, volumes).",
                              font=('Arial', 9),
                              fg='#7f8c8d', bg='#f0f0f0')
        stats_desc.pack(pady=(0, 10))

        btn_create_histogram = tk.Button(analysis_viz_frame,
                                         text="Create Histogram",
                                         font=('Arial', 11, 'bold'),
                                         width=35, height=2,
                                         bg='#8e44ad', fg='white',
                                         command=self.create_histogram,
                                         cursor='hand2')
        btn_create_histogram.pack(pady=5, padx=10, fill='x')

        hist_desc = tk.Label(analysis_viz_frame,
                             text="Generate histograms depicting distributions\nof particle measurements.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        hist_desc.pack(pady=(0, 5))

        # Section 4: Testing and Demo
        test_frame = tk.LabelFrame(parent, text="IV. Testing & Demo",
                                   font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', pady=10)
        test_frame.pack(fill='x', pady=(0, 15))

        btn_demo = tk.Button(test_frame,
                             text="Run Synthetic Demo",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#1abc9c', fg='white',
                             command=self.run_synthetic_demo,
                             cursor='hand2')
        btn_demo.pack(pady=5, padx=10, fill='x')

        demo_desc = tk.Label(test_frame,
                             text="Create synthetic blob image and run\nHessian blob detection demonstration.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        demo_desc.pack(pady=(0, 10))

        btn_test = tk.Button(test_frame,
                             text="Testing(string, number)",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#95a5a6', fg='white',
                             command=self.run_test_function,
                             cursor='hand2')
        btn_test.pack(pady=5, padx=10, fill='x')

        test_desc = tk.Label(test_frame,
                             text="Demonstrate how user-defined functions\nwork with input parameters.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        test_desc.pack(pady=(0, 5))

        # Exit button
        btn_exit = tk.Button(parent,
                             text="Exit Application",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#e74c3c', fg='white',
                             command=self.root.quit,
                             cursor='hand2')
        btn_exit.pack(side='bottom', pady=10, padx=10, fill='x')

    def setup_output_panel(self, parent):
        """Setup the output panel for displaying results."""

        output_frame = tk.LabelFrame(parent, text="Command History / Output",
                                     font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0')
        output_frame.pack(fill='both', expand=True)

        # Create text widget with scrollbar
        text_frame = tk.Frame(output_frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.output_text = tk.Text(text_frame,
                                   wrap='word',
                                   font=('Consolas', 10),
                                   bg='#2c3e50',
                                   fg='#ecf0f1',
                                   insertbackground='white')

        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self.output_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Clear button
        btn_clear = tk.Button(output_frame,
                              text="Clear Output",
                              command=self.clear_output,
                              bg='#7f8c8d', fg='white')
        btn_clear.pack(pady=5)

        # Initial welcome message
        self.print_to_output("=" * 80)
        self.print_to_output("Hessian Blob Particle Detection Suite - Python Port")
        self.print_to_output("=" * 80)
        self.print_to_output("Based on Igor Pro code by Brendan Marsh")
        self.print_to_output("G.M. King Laboratory, University of Missouri-Columbia")
        self.print_to_output("Python Port Email: marshbp@stanford.edu")
        self.print_to_output("=" * 80)
        self.print_to_output("")
        self.print_to_output("INSTRUCTIONS:")
        self.print_to_output("1. Select an analysis function from the left panel to begin")
        self.print_to_output("2. Follow the Igor Pro tutorial structure")
        self.print_to_output("3. Results are saved to folders matching Igor Pro format")
        self.print_to_output("")
        self.print_to_output("Current working directory: " + os.getcwd())
        self.print_to_output("")

    def print_to_output(self, text):
        """Print text to the output panel."""
        try:
            self.output_text.insert('end', text + '\n')
            self.output_text.see('end')
            self.root.update_idletasks()
        except Exception as e:
            handle_error("print_to_output", e)

    def clear_output(self):
        """Clear the output panel."""
        try:
            self.output_text.delete(1.0, 'end')
        except Exception as e:
            handle_error("clear_output", e)

    def run_in_thread(self, func, *args, **kwargs):
        """Run a function in a separate thread - FIXED VERSION."""

        def worker():
            try:
                result = func(*args, **kwargs)
                # Schedule GUI update on main thread
                self.root.after(0, lambda: self._handle_worker_success(result))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                import traceback
                full_traceback = traceback.format_exc()
                # Schedule error handling on main thread
                self.root.after(0, lambda: self._handle_worker_error(error_msg, full_traceback))

        # Only start thread if we're not already processing
        if not hasattr(self, '_is_processing') or not self._is_processing:
            self._is_processing = True
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

    def _handle_worker_success(self, result):
        """Handle successful worker completion on main thread"""
        self._is_processing = False
        # Result handling is done by individual methods

    def _handle_worker_error(self, error_msg, full_traceback):
        """Handle worker error on main thread"""
        self._is_processing = False
        self.print_to_output(error_msg)
        self.print_to_output(full_traceback)

    def run_batch_hessian_blobs(self):
        """Run batch Hessian blob analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: BatchHessianBlobs()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select folder containing images...")

        def batch_analysis():
            try:
                result_folder = BatchHessianBlobs()

                if result_folder:
                    # Use root.after to safely update GUI from worker thread
                    self.root.after(0, lambda: self._show_batch_results(result_folder))
                else:
                    self.root.after(0, lambda: self.print_to_output("Analysis cancelled or failed."))

            except Exception as e:
                raise e  # Let the thread handler deal with it

        self.run_in_thread(batch_analysis)

    def _show_batch_results(self, result_folder):
        """Show batch analysis results - called from main thread"""
        self.print_to_output(f"\n✓ Batch analysis complete!")
        self.print_to_output(f"Results saved to: {result_folder}")
        self.print_to_output("Series data folder created with:")
        self.print_to_output("  - AllHeights.npy")
        self.print_to_output("  - AllVolumes.npy")
        self.print_to_output("  - AllAreas.npy")
        self.print_to_output("  - AllAvgHeights.npy")
        self.print_to_output("  - Parameters.npy")
        self.print_to_output("  - Individual image particle folders")

        # Show summary
        try:
            heights = np.load(os.path.join(result_folder, "AllHeights.npy"))
            volumes = np.load(os.path.join(result_folder, "AllVolumes.npy"))
            areas = np.load(os.path.join(result_folder, "AllAreas.npy"))

            self.print_to_output(f"\nSERIES ANALYSIS SUMMARY:")
            self.print_to_output(f"  Series complete. Total particles detected: {len(heights)}")
            if len(heights) > 0:
                self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                self.print_to_output(f"  Average area: {np.mean(areas):.3e}")
        except Exception as e:
            self.print_to_output(f"Could not load summary data: {e}")

    def run_single_hessian_blobs(self):
        """Run single image Hessian blob analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: HessianBlobs(image)")
        self.print_to_output("=" * 60)
        self.print_to_output("Select an image file...")

        def single_analysis():
            try:
                # File dialog
                image_path = filedialog.askopenfilename(
                    title="Select Image File",
                    filetypes=[
                        ("Igor Binary Wave", "*.ibw"),
                        ("TIFF files", "*.tiff *.tif"),
                        ("PNG files", "*.png"),
                        ("JPEG files", "*.jpg *.jpeg"),
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not image_path:
                    self.print_to_output("No file selected.")
                    return

                self.print_to_output(f"Loading: {os.path.basename(image_path)}")
                image = DataManager.load_image_file(image_path)

                if image is None:
                    self.print_to_output("Failed to load image.")
                    return

                self.print_to_output(f"Image loaded. Shape: {image.shape}")

                # Run analysis
                result_folder = HessianBlobs(image)

                if result_folder:
                    self.print_to_output(f"\n✓ Analysis complete!")
                    self.print_to_output(f"Results saved to: {result_folder}")
                    self.print_to_output("Image_Particles folder created with:")
                    self.print_to_output("  - Heights.npy")
                    self.print_to_output("  - Volumes.npy")
                    self.print_to_output("  - Areas.npy")
                    self.print_to_output("  - AvgHeights.npy")
                    self.print_to_output("  - COM.npy")
                    self.print_to_output("  - Original.npy")
                    self.print_to_output("  - Individual Particle_X folders")

                    # Show summary
                    try:
                        heights = np.load(os.path.join(result_folder, "Heights.npy"))
                        volumes = np.load(os.path.join(result_folder, "Volumes.npy"))
                        areas = np.load(os.path.join(result_folder, "Areas.npy"))

                        self.print_to_output(f"\nSINGLE IMAGE ANALYSIS SUMMARY:")
                        self.print_to_output(f"  Particles detected: {len(heights)}")
                        if len(heights) > 0:
                            self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                            self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                            self.print_to_output(f"  Average area: {np.mean(areas):.3e}")
                    except Exception as e:
                        self.print_to_output(f"Could not load summary data: {e}")
                else:
                    self.print_to_output("Analysis cancelled or failed.")

            except Exception as e:
                self.print_to_output(f"Error during analysis: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(single_analysis)

    def run_batch_preprocess(self):
        """Run batch preprocessing - IGOR PRO STYLE."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: BatchPreprocess()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select folder containing images to preprocess...")
        self.print_to_output("Note: Preprocessed images will be saved to a new folder")

        def preprocess():
            try:
                result = BatchPreprocess()
                if result == 0:
                    self.print_to_output("✓ Preprocessing completed successfully.")
                    self.print_to_output("✓ Preprocessed images saved to new folder.")
                else:
                    self.print_to_output("Preprocessing failed or was cancelled.")
            except Exception as e:
                self.print_to_output(f"Error during preprocessing: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(preprocess)

    def run_view_particles(self):
        """Run particle viewer - FIXED THREADING VERSION."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: ViewParticles()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select particles folder...")

        # CRITICAL: Run ViewParticles in main thread, not worker thread
        try:
            ViewParticles()
        except Exception as e:
            error_msg = handle_error("ViewParticles", e)
            self.print_to_output(error_msg)
            messagebox.showerror("Viewer Error", error_msg)

    def run_wave_stats(self):
        """Run WaveStats analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: WaveStats(data)")
        self.print_to_output("=" * 60)
        self.print_to_output("Select data file (.npy)...")

        def wave_stats_analysis():
            try:
                data_file = filedialog.askopenfilename(
                    title="Select Data File",
                    filetypes=[
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not data_file:
                    self.print_to_output("No file selected.")
                    return

                self.print_to_output(f"WaveStats {os.path.basename(data_file)}")
                stats = WaveStats(data_file)

                if stats:
                    self.print_to_output("Statistics calculated successfully.")

            except Exception as e:
                self.print_to_output(f"Error running WaveStats: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(wave_stats_analysis)

    def create_histogram(self):
        """Create histogram from data."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("CREATING HISTOGRAM")
        self.print_to_output("=" * 60)
        self.print_to_output("Select data file (.npy)...")

        def make_histogram():
            try:
                data_file = filedialog.askopenfilename(
                    title="Select Data File",
                    filetypes=[
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not data_file:
                    self.print_to_output("No file selected.")
                    return

                data = np.load(data_file)
                base_name = os.path.splitext(os.path.basename(data_file))[0]

                self.print_to_output(f"Creating histogram for {base_name}...")

                # Create histogram
                plt.figure(figsize=(12, 8))
                clean_data = data[~np.isnan(data)]
                counts, bins, patches = plt.hist(clean_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f"Histogram of {base_name}", fontsize=14, fontweight='bold')
                plt.xlabel("Value", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.grid(True, alpha=0.3)

                # Add statistics text box
                stats_text = f"Count: {len(clean_data)}\n"
                stats_text += f"Mean: {np.mean(clean_data):.3e}\n"
                stats_text += f"Std Dev: {np.std(clean_data):.3e}\n"
                stats_text += f"Min: {np.min(clean_data):.3e}\n"
                stats_text += f"Max: {np.max(clean_data):.3e}"

                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                plt.tight_layout()
                plt.show()

                # Print statistics Igor Pro style
                self.print_to_output(f"\nHistogram created for {base_name}")
                self.print_to_output(f"Statistics:")
                self.print_to_output(f"  V_npnts= {len(clean_data)}; V_numNaNs= {np.sum(np.isnan(data))};")
                self.print_to_output(f"  V_avg= {np.mean(clean_data):.6g}; V_sum= {np.sum(clean_data):.6g};")
                self.print_to_output(
                    f"  V_sdev= {np.std(clean_data, ddof=1):.6g}; V_rms= {np.sqrt(np.mean(clean_data ** 2)):.6g};")
                self.print_to_output(f"  V_min= {np.min(clean_data):.6g}; V_max= {np.max(clean_data):.6g};")

            except Exception as e:
                self.print_to_output(f"Error creating histogram: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(make_histogram)

    def run_synthetic_demo(self):
        """Run synthetic data demo."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("RUNNING SYNTHETIC DEMO")
        self.print_to_output("=" * 60)
        self.print_to_output("Creating synthetic blob image...")

        def demo():
            try:
                # Create synthetic image with blob-like features
                size = 100
                x, y = np.meshgrid(np.arange(size), np.arange(size))

                # Create multiple Gaussian blobs
                image = np.zeros((size, size))

                # Add some blobs of different sizes
                blob_params = [
                    (25, 25, 5, 1.0),  # x, y, sigma, amplitude
                    (75, 30, 8, 0.8),
                    (40, 70, 6, 1.2),
                    (80, 80, 4, 0.9),
                    (15, 60, 3, 0.7),
                    (60, 15, 4, 1.1),
                ]

                for bx, by, sigma, amp in blob_params:
                    blob = amp * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * sigma ** 2))
                    image += blob

                # Add some noise
                image += np.random.normal(0, 0.05, (size, size))

                self.print_to_output(f"Synthetic image created with {len(blob_params)} blobs.")
                self.print_to_output("Running Hessian blob detection...")

                # Run analysis
                result_folder = HessianBlobs(image)

                if result_folder:
                    self.print_to_output(f"\n✓ Demo analysis complete!")
                    self.print_to_output(f"Results saved to: {result_folder}")

                    # Load and show results
                    heights = np.load(os.path.join(result_folder, "Heights.npy"))
                    volumes = np.load(os.path.join(result_folder, "Volumes.npy"))
                    areas = np.load(os.path.join(result_folder, "Areas.npy"))

                    self.print_to_output(f"\nDEMO ANALYSIS SUMMARY:")
                    self.print_to_output(f"  Expected blobs: {len(blob_params)}")
                    self.print_to_output(f"  Detected particles: {len(heights)}")
                    if len(heights) > 0:
                        self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                        self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                        self.print_to_output(f"  Average area: {np.mean(areas):.3e}")

                    # Show the synthetic image
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='hot', interpolation='bilinear')
                    plt.title('Synthetic Blob Image')
                    plt.colorbar()

                    # Show detection results if available
                    if len(heights) > 0:
                        plt.subplot(1, 2, 2)
                        plt.hist(heights, bins=20, alpha=0.7, color='green', edgecolor='black')
                        plt.title('Detected Height Distribution')
                        plt.xlabel('Height')
                        plt.ylabel('Count')
                        plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                else:
                    self.print_to_output("Demo analysis failed.")

            except Exception as e:
                self.print_to_output(f"Error during demo: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(demo)

    def run_test_function(self):
        """Run test function."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: Testing(string, number)")
        self.print_to_output("=" * 60)

        def test_function():
            try:
                # Get input from user
                test_string = simpledialog.askstring("Testing Function", "Enter a test string:")
                if test_string is None:
                    self.print_to_output("Test cancelled.")
                    return

                test_number = simpledialog.askfloat("Testing Function", "Enter a test number:")
                if test_number is None:
                    self.print_to_output("Test cancelled.")
                    return

                # Run the testing function
                Testing(test_string, test_number)

            except Exception as e:
                self.print_to_output(f"Error running test function: {e}")

        self.run_in_thread(test_function)

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except Exception as e:
            handle_error("GUI.run", e)

    def _get_image_file_threadsafe(self):
        """Get image file using thread-safe file dialog"""
        return filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Igor Binary Wave", "*.ibw"),
                ("TIFF files", "*.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )


# ========================================================================
# MAIN EXECUTION
# ========================================================================

def main():
    """Main function to run the Hessian Blob Detection Suite - FIXED VERSION."""
    try:
        print("Starting Hessian Blob Detection Suite...")

        # CRITICAL: Set matplotlib to use thread-safe backend before any plotting
        import matplotlib
        matplotlib.use('TkAgg')

        # Configure matplotlib for thread safety
        import matplotlib.pyplot as plt
        plt.rcParams['figure.raise_window'] = False
        plt.ioff()  # Turn off interactive mode

        # Create and run application
        app = HessianBlobGUI()
        app.run()

    except Exception as e:
        error_msg = handle_error("main", e)
        try:
            messagebox.showerror("Application Error", error_msg)
        except:
            print(error_msg)


if __name__ == "__main__":
    main()