"""
Main Functions Module
Contains the primary analysis functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
FIXED: Interactive threshold with red circle display like Igor Pro Figure 17
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from igor_compatibility import *
from file_io import *
from utilities import *
from scale_space import *

# Additional imports for missing functionality
try:
    from skimage.filters import threshold_otsu

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def Duplicate(source_wave, new_name):
    """
    Create a duplicate of a wave - matches Igor Pro Duplicate function
    """
    new_data = source_wave.data.copy()
    new_wave = Wave(new_data, new_name, source_wave.note)

    # Copy scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


def ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, min_response, subPixelMult=1, allowOverlap=0):
    """
    Extract blob information from maxima maps
    FIXED: Better blob extraction with proper filtering
    """
    print("Extracting blob information...")

    # Find pixels above threshold
    valid_pixels = SS_MAXMAP.data > min_response

    if not np.any(valid_pixels):
        print("No blobs found above threshold")
        return Wave(np.zeros((0, 13)), "BlobInfo")

    # Get coordinates of valid pixels
    coords = np.where(valid_pixels)
    num_blobs = len(coords[0])

    print(f"Found {num_blobs} blob candidates above threshold")

    # Initialize blob info array
    blob_info = np.zeros((num_blobs, 13))

    for i, (row, col) in enumerate(zip(coords[0], coords[1])):
        # Convert pixel coordinates to physical coordinates
        x_coord = DimOffset(SS_MAXMAP, 1) + col * DimDelta(SS_MAXMAP, 1)
        y_coord = DimOffset(SS_MAXMAP, 0) + row * DimDelta(SS_MAXMAP, 0)

        # Get detector response and scale
        response = SS_MAXMAP.data[row, col]
        scale = SS_MAXSCALEMAP.data[row, col]

        # Store basic blob information
        blob_info[i, 0] = x_coord  # X coordinate
        blob_info[i, 1] = y_coord  # Y coordinate
        blob_info[i, 2] = scale  # Radius (scale)
        blob_info[i, 3] = response  # Detector response
        blob_info[i, 4] = scale  # Scale
        blob_info[i, 5] = row  # Original row index
        blob_info[i, 6] = col  # Original col index
        blob_info[i, 7] = i  # Blob number

    print(f"Extracted {num_blobs} blob positions")
    return Wave(blob_info, "BlobInfo")


def remove_overlapping_blobs(blobs, overlap_threshold=0.5):
    """Remove overlapping blobs, keeping stronger ones"""
    if len(blobs) <= 1:
        return blobs

    # Sort by detector response (column 3) in descending order
    sorted_indices = np.argsort(blobs[:, 3])[::-1]
    sorted_blobs = blobs[sorted_indices]

    keep_mask = np.ones(len(sorted_blobs), dtype=bool)

    for i in range(len(sorted_blobs)):
        if not keep_mask[i]:
            continue

        x1, y1, r1 = sorted_blobs[i, 0], sorted_blobs[i, 1], sorted_blobs[i, 2]

        for j in range(i + 1, len(sorted_blobs)):
            if not keep_mask[j]:
                continue

            x2, y2, r2 = sorted_blobs[j, 0], sorted_blobs[j, 1], sorted_blobs[j, 2]

            # Check for overlap
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < (r1 + r2) / 2:  # Overlapping
                keep_mask[j] = False

    return sorted_blobs[keep_mask]


def GetBlobDetectionParams():
    """
    Get blob detection parameters from user
    FIXED: Ensures particle type is properly set to avoid negative blobs
    """
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window

    dialog = tk.Toplevel()
    dialog.title("Hessian Blob Parameters")
    dialog.geometry("700x750")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Hessian Blob Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Scale parameters - Exact Igor Pro defaults
    scale_frame = ttk.LabelFrame(main_frame, text="Scale-Space Parameters", padding="10")
    scale_frame.pack(fill=tk.X, pady=5)

    ttk.Label(scale_frame, text="Minimum Size in Pixels").grid(row=0, column=0, sticky=tk.W)
    scale_start_var = tk.DoubleVar(value=1)
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(scale_frame, text="Maximum Size in Pixels").grid(row=1, column=0, sticky=tk.W)
    scale_max_var = tk.IntVar(value=120)
    ttk.Entry(scale_frame, textvariable=scale_max_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scaling Factor").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.5)
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=15).grid(row=2, column=1, padx=5)

    # Detection parameters
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=5)

    ttk.Label(detect_frame, text="Minimum Blob Strength (-2 for interactive, -1 for Otsu)").grid(row=0, column=0,
                                                                                                 sticky=tk.W)
    min_response_var = tk.DoubleVar(value=-2)
    ttk.Entry(detect_frame, textvariable=min_response_var, width=15).grid(row=0, column=1, padx=5)

    # FIXED: Particle type - default to positive to avoid negative blob issues
    ttk.Label(detect_frame, text="Particle Type (-1 for neg only, +1 for pos only, 0 for both)").grid(row=1, column=0,
                                                                                                      sticky=tk.W)
    particle_type_var = tk.IntVar(value=1)  # Default to positive
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(detect_frame, text="Maximum Curvature Ratio").grid(row=2, column=0, sticky=tk.W)
    max_curv_var = tk.DoubleVar(value=10)
    ttk.Entry(detect_frame, textvariable=max_curv_var, width=15).grid(row=2, column=1, padx=5)

    ttk.Label(detect_frame, text="Subpixel Multiplier").grid(row=3, column=0, sticky=tk.W)
    subpixel_var = tk.DoubleVar(value=1)
    ttk.Entry(detect_frame, textvariable=subpixel_var, width=15).grid(row=3, column=1, padx=5)

    ttk.Label(detect_frame, text="Allow Overlap (0 = no, 1 = yes)").grid(row=4, column=0, sticky=tk.W)
    overlap_var = tk.IntVar(value=0)
    ttk.Entry(detect_frame, textvariable=overlap_var, width=15).grid(row=4, column=1, padx=5)

    def ok_clicked():
        result[0] = {
            'scaleStart': scale_start_var.get(),
            'scaleMax': scale_max_var.get(),
            'scaleFactor': scale_factor_var.get(),
            'minResponse': min_response_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': max_curv_var.get(),
            'subPixelMult': subpixel_var.get(),
            'allowOverlap': overlap_var.get()
        }
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="Continue", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Help", command=lambda: messagebox.showinfo("Help",
                                                                              "Positive blobs = bright spots (particles on bright background)\nNegative blobs = dark spots (particles on dark background)")).pack(
        side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    # Wait for user input
    dialog.wait_window()

    return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Interactive threshold selection window
    FIXED: Complete implementation with red circles like Igor Pro Figure 17
    """
    print(f"Starting Interactive Threshold Selection for particle type: {particleType}")

    # First identify the maxes - exactly like Igor Pro
    SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
    SS_MAXMAP.data = np.full(im.data.shape, -1.0)

    SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

    # FIXED: Pass particleType parameter correctly
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                       map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

    # Maxes = Sqrt(Maxes) // Put it into image units
    maxes_data = np.sqrt(np.maximum(maxes_wave.data, 0))

    max_value = np.max(maxes_data)
    if max_value == 0:
        messagebox.showwarning("No Blobs", "No suitable blob candidates found in image.")
        return None

    print(f"Max blob strength: {max_value}")

    # Create interactive threshold window
    class InteractiveThresholdWindow:
        """FIXED: Interactive threshold window that displays red circles like Igor Pro Figure 17"""

        def __init__(self, im, maxes_data, SS_MAXMAP, SS_MAXSCALEMAP, max_value):
            self.im = im
            self.maxes_data = maxes_data
            self.SS_MAXMAP = SS_MAXMAP
            self.SS_MAXSCALEMAP = SS_MAXSCALEMAP
            self.max_value = max_value
            self.threshold = max_value / 2
            self.result = None
            self.root = None

            self.create_window()

        def create_window(self):
            """Create the interactive threshold window"""
            self.root = tk.Toplevel()
            self.root.title("IMAGE:Original")
            self.root.geometry("1000x600")
            self.root.transient()
            self.root.grab_set()

            # Create main layout
            main_container = ttk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True)

            # Left side - Image display
            image_frame = ttk.Frame(main_container)
            image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Right side - Controls (matches Igor Pro layout)
            control_frame = ttk.Frame(main_container, width=200)
            control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
            control_frame.pack_propagate(False)

            # Setup matplotlib figure
            self.figure = plt.Figure(figsize=(8, 6))
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, image_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Control panel (matches Igor Pro "Continue Button" panel)
            control_label = ttk.Label(control_frame, text="Continue Button", font=("TkDefaultFont", 12, "bold"))
            control_label.pack(pady=(0, 10))

            # Accept and Quit buttons
            button_frame = ttk.Frame(control_frame)
            button_frame.pack(fill=tk.X, pady=(0, 20))

            ttk.Button(button_frame, text="Accept", command=self.on_accept, width=12).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Quit", command=self.on_quit, width=12).pack(side=tk.LEFT, padx=2)

            # Blob strength entry (matches Igor Pro SetVariable)
            strength_frame = ttk.Frame(control_frame)
            strength_frame.pack(fill=tk.X, pady=(0, 20))

            ttk.Label(strength_frame, text="Blob Strength:").pack()
            self.threshold_var = tk.DoubleVar(value=self.threshold)
            threshold_entry = ttk.Entry(strength_frame, textvariable=self.threshold_var, width=15)
            threshold_entry.pack(pady=2)
            threshold_entry.bind('<Return>', self.on_entry_change)

            # Threshold slider (matches Igor Pro Slider)
            slider_frame = ttk.Frame(control_frame)
            slider_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            self.slider_var = tk.DoubleVar(value=self.threshold)
            slider = tk.Scale(slider_frame,
                              from_=0,
                              to=max_value * 1.1,
                              resolution=max_value * 1.1 / 200,
                              orient=tk.VERTICAL,
                              variable=self.slider_var,
                              command=self.on_slider_change,
                              length=400)
            slider.pack(fill=tk.BOTH, expand=True)

            # Initial display
            self.update_display()

            # Protocol for window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

        def update_display(self):
            """Update the image display with current threshold - FIXED: Red circles like Igor Pro Figure 17"""
            self.ax.clear()

            # Display the original image
            height, width = self.im.data.shape
            x_min = DimOffset(self.im, 1)
            x_max = x_min + width * DimDelta(self.im, 1)
            y_min = DimOffset(self.im, 0)
            y_max = y_min + height * DimDelta(self.im, 0)

            self.ax.imshow(self.im.data, extent=[x_min, x_max, y_max, y_min],
                           cmap='gray', aspect='equal', origin='upper')

            # FIXED: Add blob overlays - red circles like Igor Pro Figure 17
            current_threshold_squared = self.threshold ** 2

            # Draw circles for blobs above threshold
            limI = DimSize(self.SS_MAXMAP, 0)
            limJ = DimSize(self.SS_MAXMAP, 1)

            circle_count = 0
            for i in range(limI):
                for j in range(limJ):
                    if self.SS_MAXMAP.data[i, j] > current_threshold_squared:
                        # Convert indices to coordinates
                        xc = DimOffset(self.SS_MAXMAP, 1) + j * DimDelta(self.SS_MAXMAP, 1)
                        yc = DimOffset(self.SS_MAXMAP, 0) + i * DimDelta(self.SS_MAXMAP, 0)
                        rad = self.SS_MAXSCALEMAP.data[i, j]

                        # FIXED: Draw red circle - matches Igor Pro exactly
                        circle = Circle((xc, yc), rad, fill=False, edgecolor='red', linewidth=1.5)
                        self.ax.add_patch(circle)
                        circle_count += 1

            self.ax.set_xlabel(f"X ({DimUnits(self.im, 1)})")
            self.ax.set_ylabel(f"Y ({DimUnits(self.im, 0)})")
            self.ax.set_title("IMAGE:Original")

            self.figure.tight_layout()
            self.canvas.draw()

            print(f"Threshold: {self.threshold:.6f}, Circles displayed: {circle_count}")

        def on_slider_change(self, value):
            """Handle slider change"""
            self.threshold = float(value)
            self.threshold_var.set(self.threshold)
            self.update_display()

        def on_entry_change(self, event):
            """Handle threshold entry change"""
            try:
                new_threshold = self.threshold_var.get()
                if 0 <= new_threshold <= self.max_value * 1.1:
                    self.threshold = new_threshold
                    self.slider_var.set(self.threshold)
                    self.update_display()
            except tk.TclError:
                pass

        def on_accept(self):
            """Accept current threshold"""
            self.result = self.threshold
            self.root.destroy()

        def on_quit(self):
            """Quit without saving"""
            self.result = None
            self.root.destroy()

        def run(self):
            """Run the interactive threshold selection"""
            self.root.wait_window()
            return self.result

    # Create and run the interactive window
    window = InteractiveThresholdWindow(im, maxes_data, SS_MAXMAP, SS_MAXSCALEMAP, max_value)
    result = window.run()

    print(f"Interactive threshold selection completed. Selected threshold: {result}")
    return result


def FindHessianBlobs(im, params=None):
    """
    Main function to find Hessian blobs in an image
    FIXED: Proper parameter passing to ensure particle type is respected
    """
    print(f"Starting Hessian blob detection on {im.name}...")

    # Get parameters if not provided
    if params is None:
        params = GetBlobDetectionParams()
        if params is None:
            return None  # User cancelled

    # FIXED: Log the particle type being used
    particle_type = params['particleType']
    particle_type_str = {1: "positive (bright)", -1: "negative (dark)", 0: "both positive and negative"}
    print(f"Detecting {particle_type_str.get(particle_type, 'unknown')} blobs")

    try:
        # Calculate scale layers from max size
        scale_start = params['scaleStart']
        scale_max = params['scaleMax']
        scale_factor = params['scaleFactor']

        # Calculate number of layers needed to reach max size
        scale_layers = int(np.log(scale_max / scale_start) / np.log(scale_factor)) + 1

        print(f"Scale parameters: start={scale_start}, max={scale_max}, factor={scale_factor}")
        print(f"Calculated {scale_layers} scale layers")

        # Compute scale-space representation
        L = ScaleSpaceRepresentation(im, scale_layers, scale_start, scale_factor)
        print("Scale-space representation computed")

        # Compute Hessian determinant and Laplacian
        detH, LG = BlobDetectors(L, gammaNorm=1.0)
        print("Hessian determinant and Laplacian computed")

        # Interactive threshold selection if requested
        min_response = params['minResponse']
        if min_response == -2:  # Interactive mode
            # FIXED: Pass particleType parameter correctly
            threshold = InteractiveThreshold(im, detH, LG, particle_type, params['maxCurvatureRatio'])
            if threshold is None:
                return None  # User cancelled
            min_response = threshold ** 2  # Convert back to detH units
        elif min_response == -1:  # Otsu's method
            # Implement Otsu's thresholding
            if SKIMAGE_AVAILABLE:
                flat_data = detH.data.flatten()
                valid_data = flat_data[flat_data > 0]
                if len(valid_data) > 0:
                    min_response = threshold_otsu(valid_data)
                else:
                    min_response = 0
            else:
                print("Otsu's method requires scikit-image. Using manual threshold.")
                min_response = np.percentile(detH.data[detH.data > 0], 75) if np.any(detH.data > 0) else 0

        print(f"Using threshold: {min_response}")

        # Find local maxima
        SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        # FIXED: Pass particleType parameter correctly to Maxes function
        maxes_wave = Maxes(detH, LG, particle_type, params['maxCurvatureRatio'],
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Extract blob information
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, min_response,
                               params['subPixelMult'], params['allowOverlap'])

        if info is None or info.data.shape[0] == 0:
            print("No blobs found")
            return {'num_particles': 0, 'info': None}

        # Import particle measurements function
        from particle_measurements import MeasureParticles

        # Measure particle properties
        success = MeasureParticles(im, info)
        if not success:
            print("Failed to measure particles")
            return {'num_particles': 0, 'info': None}

        # Apply size constraints if specified
        if 'constraints' in params:
            info = apply_size_constraints(info, params['constraints'])

        num_particles = info.data.shape[0]
        print(f"Final result: {num_particles} particles detected")

        return {
            'num_particles': num_particles,
            'info': info,
            'detH': detH,
            'LG': LG,
            'params': params
        }

    except Exception as e:
        print(f"Error in blob detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def apply_size_constraints(info, constraints):
    """Apply size constraints to filter particles"""
    if info is None or info.data.shape[0] == 0:
        return info

    # Get constraint values
    min_height = constraints.get('min_height', 0)
    max_height = constraints.get('max_height', np.inf)
    min_area = constraints.get('min_area', -np.inf)
    max_area = constraints.get('max_area', np.inf)
    min_volume = constraints.get('min_volume', -np.inf)
    max_volume = constraints.get('max_volume', np.inf)

    # Create mask for particles that meet constraints
    valid_mask = np.ones(info.data.shape[0], dtype=bool)

    # Apply height constraints (column 10 is height)
    if min_height > 0 or max_height < np.inf:
        heights = info.data[:, 10]
        valid_mask &= (heights >= min_height) & (heights <= max_height)

    # Apply area constraints (column 8 is area)
    if min_area > -np.inf or max_area < np.inf:
        areas = info.data[:, 8]
        valid_mask &= (areas >= min_area) & (areas <= max_area)

    # Apply volume constraints (column 9 is volume)
    if min_volume > -np.inf or max_volume < np.inf:
        volumes = info.data[:, 9]
        valid_mask &= (volumes >= min_volume) & (volumes <= max_volume)

    # Filter particles
    if np.any(valid_mask):
        filtered_data = info.data[valid_mask]
        filtered_info = Wave(filtered_data, info.name + "_filtered", info.note)
        print(f"Applied size constraints: {np.sum(valid_mask)} of {len(valid_mask)} particles retained")
        return filtered_info
    else:
        print("No particles meet the size constraints")
        empty_info = Wave(np.zeros((0, info.data.shape[1])), info.name + "_filtered", info.note)
        return empty_info


def BatchHessianBlobs(images_dict, params=None):
    """Run Hessian blob detection on multiple images"""
    if not images_dict:
        return {}

    # Get parameters once for all images
    if params is None:
        params = GetBlobDetectionParams()
        if params is None:
            return {}

    results = {}
    total_images = len(images_dict)

    print(f"Starting batch analysis of {total_images} images...")

    for i, (image_name, image_wave) in enumerate(images_dict.items()):
        print(f"Processing image {i + 1}/{total_images}: {image_name}")

        # Run detection with same parameters
        result = FindHessianBlobs(image_wave, params)
        if result:
            results[image_name] = result
            print(f"  Found {result['num_particles']} particles")
        else:
            print(f"  Failed to process {image_name}")

    print(f"Batch analysis complete. Processed {len(results)} images successfully.")
    return results


def Testing(string_input, number_input):
    """Testing function for main functions"""
    print(f"Main functions testing: {string_input}, {number_input}")
    return len(string_input) + number_input