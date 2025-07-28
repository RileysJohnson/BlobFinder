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
    y_coords, x_coords = np.where(valid_pixels)

    blobs = []
    for i in range(len(y_coords)):
        y_idx, x_idx = y_coords[i], x_coords[i]

        # Convert indices to real coordinates
        x_coord = DimOffset(SS_MAXMAP, 1) + x_idx * DimDelta(SS_MAXMAP, 1)
        y_coord = DimOffset(SS_MAXMAP, 0) + y_idx * DimDelta(SS_MAXMAP, 0)
        radius = SS_MAXSCALEMAP.data[y_idx, x_idx]
        response = SS_MAXMAP.data[y_idx, x_idx]

        # Store blob info [x, y, radius, response, ...]
        blob_info = [x_coord, y_coord, radius, response, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        blobs.append(blob_info)

    if not blobs:
        return Wave(np.zeros((0, 13)), "BlobInfo")

    blobs = np.array(blobs)

    # Remove overlapping blobs if requested
    if allowOverlap == 0:
        blobs = remove_overlapping_blobs(blobs)

    print(f"Extracted {len(blobs)} blobs")
    return Wave(blobs, "BlobInfo")


def remove_overlapping_blobs(blobs):
    """Remove overlapping blobs, keeping the one with highest response"""
    if len(blobs) <= 1:
        return blobs

    # Sort by response strength (column 3)
    sorted_indices = np.argsort(-blobs[:, 3])  # Descending order
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
    FIXED: Added size constraints dialog matching Igor Pro
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

    ttk.Label(detect_frame, text="Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)").grid(row=0,
                                                                                                          column=0,
                                                                                                          sticky=tk.W)
    threshold_var = tk.DoubleVar(value=-2)
    ttk.Entry(detect_frame, textvariable=threshold_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(detect_frame, text="Particle Type (-1 for negative, +1 for positive, 0 for both)").grid(row=1,
                                                                                                      column=0,
                                                                                                      sticky=tk.W)
    particle_type_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(detect_frame, text="Subpixel Ratio").grid(row=2, column=0, sticky=tk.W)
    subpixel_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=subpixel_var, width=15).grid(row=2, column=1, padx=5)

    ttk.Label(detect_frame, text="Allow Hessian Blobs to Overlap? (1=yes 0=no)").grid(row=3, column=0, sticky=tk.W)
    overlap_var = tk.IntVar(value=0)
    ttk.Entry(detect_frame, textvariable=overlap_var, width=15).grid(row=3, column=1, padx=5)

    # Additional parameters
    max_curv_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
    max_curv_frame.pack(fill=tk.X, pady=5)

    ttk.Label(max_curv_frame, text="Maximum Curvature Ratio").grid(row=0, column=0, sticky=tk.W)
    max_curv_var = tk.DoubleVar(value=10.0)
    ttk.Entry(max_curv_frame, textvariable=max_curv_var, width=15).grid(row=0, column=1, padx=5)

    def ok_clicked():
        # ADDED: Size constraints dialog like Igor Pro Figure 15
        use_constraints = messagebox.askyesno("Igor Pro wants to know...",
                                              "Would you like to limit the analysis to particles of certain height, volume, or area?")

        constraints = None
        if use_constraints:
            constraints = get_size_constraints()
            if constraints is None:  # User cancelled constraints
                return

        result[0] = {
            'scaleStart': scale_start_var.get(),
            'layers': scale_max_var.get(),
            'scaleFactor': scale_factor_var.get(),
            'detHResponseThresh': threshold_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': max_curv_var.get(),
            'subPixelMult': subpixel_var.get(),
            'allowOverlap': overlap_var.get(),
            'constraints': constraints  # ADDED: This line
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

            # FIXED: Controls section - matches Igor Pro exactly
            ttk.Label(control_frame, text="Continue Button", font=('TkDefaultFont', 12, 'bold')).pack(pady=10)

            button_frame = ttk.Frame(control_frame)
            button_frame.pack(fill=tk.X, pady=10)

            ttk.Button(button_frame, text="Accept", command=self.accept_clicked).pack(fill=tk.X, pady=2)
            ttk.Button(button_frame, text="Quit", command=self.quit_clicked).pack(fill=tk.X, pady=2)

            # Blob strength controls
            ttk.Label(control_frame, text="Blob Strength", font=('TkDefaultFont', 10)).pack(pady=(20, 5))

            # Text entry for manual threshold
            self.threshold_var = tk.StringVar(value=f"{self.threshold:.3e}")
            threshold_entry = ttk.Entry(control_frame, textvariable=self.threshold_var, width=15)
            threshold_entry.pack(pady=5)
            threshold_entry.bind('<Return>', self.on_threshold_entry)

            # FIXED: Vertical slider like Igor Pro
            slider_frame = ttk.Frame(control_frame)
            slider_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            # Create scale widget for threshold (vertical)
            self.threshold_scale = tk.Scale(slider_frame, from_=0, to=self.max_value,
                                            orient=tk.VERTICAL, resolution=self.max_value / 1000,
                                            command=self.on_threshold_change,
                                            length=400)
            self.threshold_scale.set(self.threshold)
            self.threshold_scale.pack(fill=tk.BOTH, expand=True)

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
                        circle = Circle((xc, yc), rad, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
                        self.ax.add_patch(circle)
                        circle_count += 1

            print(f"Displaying {circle_count} blobs above threshold {self.threshold:.3e}")

            self.ax.set_title("Interactive Blob Threshold Selection")
            self.canvas.draw()

        def on_threshold_change(self, value):
            """Handle threshold slider change"""
            self.threshold = float(value)
            self.threshold_var.set(f"{self.threshold:.3e}")
            self.update_display()

        def on_threshold_entry(self, event):
            """Handle manual threshold entry"""
            try:
                new_threshold = float(self.threshold_var.get())
                if 0 <= new_threshold <= self.max_value:
                    self.threshold = new_threshold
                    self.threshold_scale.set(self.threshold)
                    self.update_display()
                else:
                    messagebox.showwarning("Invalid Threshold", f"Threshold must be between 0 and {self.max_value}")
                    self.threshold_var.set(f"{self.threshold:.3e}")
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter a valid number")
                self.threshold_var.set(f"{self.threshold:.3e}")

        def accept_clicked(self):
            """Accept current threshold"""
            self.result = self.threshold
            self.root.destroy()

        def quit_clicked(self):
            """Cancel threshold selection"""
            self.result = None
            self.root.destroy()

        def on_quit(self):
            """Handle window close"""
            self.result = None
            self.root.destroy()

    # Create and show the interactive window
    threshold_window = InteractiveThresholdWindow(im, maxes_data, SS_MAXMAP, SS_MAXSCALEMAP, max_value)
    threshold_window.root.wait_window()

    print(f"Selected threshold: {threshold_window.result}")
    return threshold_window.result


def FindHessianBlobs(im, params=None):
    """
    Main function to find Hessian blobs in an image
    FIXED: Complete implementation with size constraints
    """
    try:
        # Get parameters if not provided
        if params is None:
            params = GetBlobDetectionParams()
            if params is None:
                return None

        print("Starting Hessian blob detection...")
        print(f"Parameters: {params}")

        # Extract parameters
        scaleStart = params['scaleStart']
        layers = params['layers']
        scaleFactor = params['scaleFactor']
        detHResponseThresh = params['detHResponseThresh']
        particleType = params['particleType']
        maxCurvatureRatio = params['maxCurvatureRatio']
        subPixelMult = params['subPixelMult']
        allowOverlap = params['allowOverlap']

        # Compute scale-space representation
        print("Computing scale-space representation...")
        detH, LG = compute_scale_space(im, scaleStart, layers, scaleFactor)

        if detH is None or LG is None:
            print("Failed to compute scale-space representation")
            return None

        # Handle threshold selection
        if detHResponseThresh == -2:  # Interactive threshold
            threshold = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
            if threshold is None:
                print("Threshold selection cancelled")
                return None
        elif detHResponseThresh == -1:  # Otsu's method
            if SKIMAGE_AVAILABLE:
                threshold = threshold_otsu(detH.data)
            else:
                threshold = np.mean(detH.data) + np.std(detH.data)
            print(f"Using Otsu threshold: {threshold}")
        else:  # Fixed threshold
            threshold = detHResponseThresh
            print(f"Using fixed threshold: {threshold}")

        # Extract blob information
        print(f"Extracting blobs with threshold: {threshold}")

        # First get the maxes
        SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Extract blob info using the threshold
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, threshold ** 2, subPixelMult, allowOverlap)

        # ADDED: Apply size constraints if specified
        if params.get('constraints') is not None:
            print("Applying size constraints...")
            info = apply_size_constraints(info, params['constraints'])

        num_particles = info.data.shape[0] if info is not None else 0
        print(f"Found {num_particles} particles")

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


# ADDED: Size constraints functions
def get_size_constraints():
    """
    Get size constraints dialog - matches Igor Pro Figure 16 exactly
    """
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel()
    dialog.title("Constraints")
    dialog.geometry("600x400")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Height constraints - matches Igor Pro defaults
    height_frame = ttk.Frame(main_frame)
    height_frame.pack(fill=tk.X, pady=10)

    ttk.Label(height_frame, text="Minimum height in m").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
    min_height_var = tk.StringVar(value="-inf")  # Igor Pro default
    ttk.Entry(height_frame, textvariable=min_height_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(height_frame, text="Maximum height in m").grid(row=0, column=2, sticky=tk.W, padx=(20, 20))
    max_height_var = tk.StringVar(value="inf")  # Igor Pro default
    ttk.Entry(height_frame, textvariable=max_height_var, width=15).grid(row=0, column=3, padx=5)

    # Area constraints - matches Igor Pro defaults
    area_frame = ttk.Frame(main_frame)
    area_frame.pack(fill=tk.X, pady=10)

    ttk.Label(area_frame, text="Minimum area in m^2").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
    min_area_var = tk.StringVar(value="-inf")  # Igor Pro default
    ttk.Entry(area_frame, textvariable=min_area_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(area_frame, text="Maximum area in m^2").grid(row=0, column=2, sticky=tk.W, padx=(20, 20))
    max_area_var = tk.StringVar(value="inf")  # Igor Pro default
    ttk.Entry(area_frame, textvariable=max_area_var, width=15).grid(row=0, column=3, padx=5)

    # Volume constraints - matches Igor Pro defaults
    volume_frame = ttk.Frame(main_frame)
    volume_frame.pack(fill=tk.X, pady=10)

    ttk.Label(volume_frame, text="Minimum volume in m^3").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
    min_volume_var = tk.StringVar(value="-inf")  # Igor Pro default
    ttk.Entry(volume_frame, textvariable=min_volume_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(volume_frame, text="Maximum volume in m^3").grid(row=0, column=2, sticky=tk.W, padx=(20, 20))
    max_volume_var = tk.StringVar(value="inf")  # Igor Pro default
    ttk.Entry(volume_frame, textvariable=max_volume_var, width=15).grid(row=0, column=3, padx=5)

    def parse_value(value_str):
        """Parse constraint value, handling inf and -inf"""
        if value_str.lower() == "inf":
            return np.inf
        elif value_str.lower() == "-inf":
            return -np.inf
        else:
            try:
                return float(value_str)
            except ValueError:
                return np.inf if "inf" in value_str.lower() else 0

    def ok_clicked():
        result[0] = {
            'min_height': parse_value(min_height_var.get()),
            'max_height': parse_value(max_height_var.get()),
            'min_area': parse_value(min_area_var.get()),
            'max_area': parse_value(max_area_var.get()),
            'min_volume': parse_value(min_volume_var.get()),
            'max_volume': parse_value(max_volume_var.get())
        }
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=30)

    ttk.Button(button_frame, text="Continue", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Help", command=lambda: messagebox.showinfo("Help",
                                                                              "The default values, minimum values as -inf (minus infinity) or maximum values as inf (positive infinity), puts no constraint on the particles.")).pack(
        side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    dialog.wait_window()
    return result[0]


def apply_size_constraints(info, constraints):
    """Apply size constraints to filter particles"""
    if info is None or info.data.shape[0] == 0:
        return info

    # Get constraint values
    min_height = constraints.get('min_height', -np.inf)
    max_height = constraints.get('max_height', np.inf)
    min_area = constraints.get('min_area', -np.inf)
    max_area = constraints.get('max_area', np.inf)
    min_volume = constraints.get('min_volume', -np.inf)
    max_volume = constraints.get('max_volume', np.inf)

    # Create mask for particles that meet constraints
    valid_mask = np.ones(info.data.shape[0], dtype=bool)

    # Apply height constraints (column 10 is height)
    if min_height > -np.inf or max_height < np.inf:
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