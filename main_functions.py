"""
Main Functions Module
Contains the primary analysis functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
COMPLETE FIX: Proper blob visualization, manual threshold support, enhanced UI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage

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
    valid_pixels = np.where(SS_MAXMAP.data >= min_response)

    if len(valid_pixels[0]) == 0:
        print("No blobs found above threshold")
        empty_info = Wave(np.zeros((0, 13)), "info")
        return empty_info

    num_blobs = len(valid_pixels[0])
    blob_info = np.zeros((num_blobs, 13))

    print(f"Found {num_blobs} candidate blobs")

    for idx, (i, j) in enumerate(zip(valid_pixels[0], valid_pixels[1])):
        # Get blob information
        x_coord = j  # Column index -> x coordinate
        y_coord = i  # Row index -> y coordinate
        response = SS_MAXMAP.data[i, j]
        scale = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP is not None else 1.0

        # Calculate radius from scale (matching Igor Pro)
        radius = scale * np.sqrt(2)  # σ_scaled * sqrt(2) for Hessian blobs

        # Store blob information (matching Igor Pro format)
        blob_info[idx, 0] = x_coord  # X position
        blob_info[idx, 1] = y_coord  # Y position
        blob_info[idx, 2] = radius  # Radius
        blob_info[idx, 3] = response  # Response strength
        blob_info[idx, 4] = scale  # Scale
        # Other columns can be filled with additional measurements

    # Filter overlapping blobs if not allowed
    if allowOverlap == 0:
        blob_info = filter_overlapping_blobs(blob_info)

    print(f"Final blob count after filtering: {blob_info.shape[0]}")

    # Create output wave
    info_wave = Wave(blob_info, "info")
    return info_wave


def filter_overlapping_blobs(blob_info):
    """Remove overlapping blobs, keeping stronger ones"""
    if blob_info.shape[0] <= 1:
        return blob_info

    # Sort by response strength (descending)
    sorted_indices = np.argsort(-blob_info[:, 3])
    sorted_blobs = blob_info[sorted_indices]

    # Keep track of which blobs to keep
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
    FIXED: Enhanced parameter dialog matching Igor Pro exactly
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
    scale_max_var = tk.DoubleVar(value=25)
    ttk.Entry(scale_frame, textvariable=scale_max_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scale Factor").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.25)
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=15).grid(row=2, column=1, padx=5)

    # Detection parameters
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=10)

    ttk.Label(detect_frame, text="Blob Strength Threshold (-2=interactive, -1=auto)").grid(row=0, column=0, sticky=tk.W)
    thresh_var = tk.DoubleVar(value=-2)  # Default to interactive
    ttk.Entry(detect_frame, textvariable=thresh_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(detect_frame, text="Particle Type (1=positive, -1=negative, 0=both)").grid(row=1, column=0, sticky=tk.W)
    particle_type_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(detect_frame, text="Subpixel Ratio (1=pixel precision, >1=subpixel)").grid(row=2, column=0, sticky=tk.W)
    subpixel_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=subpixel_var, width=15).grid(row=2, column=1, padx=5)

    ttk.Label(detect_frame, text="Allow Overlap (1=yes 0=no)").grid(row=3, column=0, sticky=tk.W)
    overlap_var = tk.IntVar(value=0)
    ttk.Entry(detect_frame, textvariable=overlap_var, width=15).grid(row=3, column=1, padx=5)

    def ok_clicked():
        # Calculate layers from scale parameters
        scale_start = scale_start_var.get()
        scale_max = scale_max_var.get()
        scale_factor = scale_factor_var.get()

        # Calculate number of layers needed
        layers = int(np.log(scale_max / scale_start) / np.log(scale_factor)) + 1

        result[0] = {
            'scaleStart': scale_start,
            'layers': layers,
            'scaleFactor': scale_factor,
            'detHResponseThresh': thresh_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': 10,  # Igor Pro default
            'subPixelMult': subpixel_var.get(),
            'allowOverlap': overlap_var.get()
        }
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    ttk.Button(button_frame, text="Continue", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    dialog.wait_window()
    return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Interactive threshold selection matching Igor Pro behavior
    FIXED: Proper slider precision, better bounds, and blob region visualization
    """
    print("Opening interactive threshold window...")

    # Create the threshold selection window
    threshold_window = ThresholdSelectionWindow(im, detH, LG, particleType, maxCurvatureRatio)
    threshold_window.run()

    print(f"Interactive threshold selected: {threshold_window.result}")
    return threshold_window.result


class ThresholdSelectionWindow:
    """Interactive threshold selection window with proper blob visualization"""

    def __init__(self, im, detH, LG, particleType, maxCurvatureRatio):
        self.im = im
        self.detH = detH
        self.LG = LG
        self.particleType = particleType
        self.maxCurvatureRatio = maxCurvatureRatio
        self.result = None

        # FIXED: Find range where particles actually exist
        self.particle_min, self.particle_max = self.find_particle_range()

        # FIXED: Center the default threshold like Igor Pro (WaveMax(Maxes)/2)
        self.current_thresh = (self.particle_min + self.particle_max) / 2.0

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Selection")
        self.root.geometry("1000x700")

        self.setup_gui()

    def find_particle_range(self):
        """Find the actual range where particles are detected"""
        # Get maxes at very low threshold to find all possible particles
        SS_MAXMAP_temp = Duplicate(self.im, "SS_MAXMAP_temp")
        SS_MAXMAP_temp.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP_temp = Duplicate(SS_MAXMAP_temp, "SS_MAXSCALEMAP_temp")

        # Run maxes with very permissive settings
        maxes_wave = Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                           map_wave=SS_MAXMAP_temp, scaleMap=SS_MAXSCALEMAP_temp)

        # Find actual range of particle responses like Igor Pro
        particle_responses = SS_MAXMAP_temp.data[SS_MAXMAP_temp.data > 0]

        if len(particle_responses) > 0:
            # Take sqrt to match Igor Pro units (detector response is squared)
            particle_responses = np.sqrt(particle_responses)
            min_val = np.min(particle_responses)
            max_val = np.max(particle_responses)
            # Set default to middle like Igor Pro: WaveMax(Maxes)/2
            return min_val, max_val
        else:
            # Fallback to detector range
            detH_positive = self.detH.data[self.detH.data > 0]
            if len(detH_positive) > 0:
                return np.min(detH_positive), np.max(detH_positive)
            else:
                return 0.0, 1.0

    def format_scientific(self, value):
        """Format number using scientific notation for small values like Igor Pro"""
        if abs(value) < 1e-3 or abs(value) > 1e6:
            return f"{value:.3e}"
        else:
            return f"{value:.6f}"

    def setup_gui(self):
        """Setup the GUI components"""
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # Create canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # FIXED: Find actual particle range for slider bounds
        self.particle_min, self.particle_max = self.find_particle_range()

        # FIXED: Slider starts in middle like Igor Pro (WaveMax(Maxes)/2)
        self.current_thresh = (self.particle_min + self.particle_max) / 2.0

        ttk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT)
        self.thresh_var = tk.DoubleVar(value=self.current_thresh)

        self.thresh_scale = ttk.Scale(control_frame, from_=self.particle_min, to=self.particle_max,
                                      variable=self.thresh_var, orient=tk.HORIZONTAL, length=400,
                                      command=self.on_threshold_change)
        self.thresh_scale.pack(side=tk.LEFT, padx=10)

        # FIXED: Scientific notation for small numbers like Igor Pro
        self.thresh_label = ttk.Label(control_frame, text=self.format_scientific(self.current_thresh))
        self.thresh_label.pack(side=tk.LEFT, padx=5)

        # Manual entry
        self.thresh_entry = ttk.Entry(control_frame, textvariable=self.thresh_var, width=15)
        self.thresh_entry.pack(side=tk.LEFT, padx=5)
        self.thresh_entry.bind('<Return>', self.on_manual_entry)

        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="Accept", command=self.accept_threshold).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_threshold).pack(side=tk.LEFT, padx=5)

        # Display options - FIXED: Show regions toggle works with manual mode
        self.show_regions_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Show Blob Regions",
                        variable=self.show_regions_var,
                        command=self.update_display).pack(side=tk.LEFT, padx=10)

        # Initial display
        self.update_display()

    def on_threshold_change(self, value):
        """Handle threshold slider change"""
        self.current_thresh = float(value)
        self.thresh_label.config(text=self.format_scientific(self.current_thresh))
        self.update_display()

    def on_manual_entry(self, event):
        """Handle manual threshold entry"""
        try:
            value = float(self.thresh_entry.get())
            if self.particle_min <= value <= self.particle_max:
                self.current_thresh = value
                self.thresh_scale.set(value)
                self.thresh_label.config(text=self.format_scientific(self.current_thresh))
                self.update_display()
        except ValueError:
            pass

    def update_display(self):
        """FIXED: Update display with blob regions and red tinting like Igor Pro"""
        self.ax.clear()

        # Display the original image
        self.ax.imshow(self.im.data, cmap='gray', aspect='equal')
        self.ax.set_title(f"Threshold: {self.format_scientific(self.current_thresh)}")

        # Get maxes with current threshold
        SS_MAXMAP = Duplicate(self.im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        maxes_wave = Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Extract blobs above threshold
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, self.current_thresh)

        if info.data.shape[0] > 0 and self.show_regions_var.get():
            # FIXED: Show actual blob regions with red tinting like Igor Pro
            self.draw_blob_regions(info)

        self.ax.set_xlim(0, self.im.data.shape[1])
        self.ax.set_ylim(self.im.data.shape[0], 0)  # Flip y axis for image coordinates

        blob_count = info.data.shape[0] if info.data.shape[0] > 0 else 0
        self.ax.text(10, 30, f"Blobs found: {blob_count}", color='yellow', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        self.canvas.draw()

    def draw_blob_regions(self, info):
        """FIXED: Draw blob regions with red tinting like Igor Pro"""
        # Create mask for all blob regions
        blob_mask = np.zeros(self.im.data.shape, dtype=bool)

        for i in range(info.data.shape[0]):
            x, y, radius = info.data[i, 0], info.data[i, 1], info.data[i, 2]

            # Create circular mask for this blob
            y_coords, x_coords = np.ogrid[:self.im.data.shape[0], :self.im.data.shape[1]]
            distance = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
            blob_region = distance <= radius

            blob_mask |= blob_region

            # Draw perimeter circle (green like Igor Pro)
            circle = Circle((x, y), radius, fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
            self.ax.add_patch(circle)

        # Create red tinted overlay for blob regions like Igor Pro
        red_overlay = np.zeros((*self.im.data.shape, 4))
        red_overlay[blob_mask] = [1, 0, 0, 0.3]  # Red with transparency

        # Apply the overlay
        self.ax.imshow(red_overlay, aspect='equal', alpha=0.5)

    def accept_threshold(self):
        """Accept the current threshold and close"""
        self.result = self.current_thresh
        self.root.destroy()

    def cancel_threshold(self):
        """Cancel threshold selection"""
        self.result = None
        self.root.destroy()

    def run(self):
        """Run the threshold selection dialog"""
        self.root.mainloop()


def HessianBlobs(im, scaleStart=1, layers=20, scaleFactor=1.25,
                 detHResponseThresh=-2, particleType=1, maxCurvatureRatio=10,
                 subPixelMult=1, allowOverlap=0):
    """
    Main Hessian blob detection function
    Direct port from Igor Pro HessianBlobs function
    FIXED: Integer conversion for layers calculation
    """
    print("Starting Hessian Blob Detection...")
    print(f"Parameters: scaleStart={scaleStart}, layers={layers}, scaleFactor={scaleFactor}")
    print(f"Threshold mode: {detHResponseThresh} (-2=interactive, -1=auto)")

    # STEP 1: Create scale-space representation (matches Igor Pro exactly)
    print("Creating scale-space representation...")

    # FIXED: The key fix - ensure layers is always an integer
    # Igor Pro: layers = ceil( log( (layers*DimDelta(im,0))^2/(2*scaleStart))/log(scaleFactor) )
    scaleStart_converted = (scaleStart * DimDelta(im, 0)) ** 2 / 2
    layers_calculated = np.log((layers * DimDelta(im, 0)) ** 2 / (2 * scaleStart_converted)) / np.log(scaleFactor)
    layers = int(np.ceil(layers_calculated))  # FIXED: Convert to int

    print(f"Calculated layers: {layers} (was {layers_calculated})")

    # Ensure minimum values
    layers = max(1, layers)  # At least 1 layer
    scaleFactor = max(1.1, scaleFactor)  # Minimum scale factor
    subPixelMult = max(1, int(np.round(subPixelMult)))  # FIXED: Ensure integer

    L = ScaleSpaceRepresentation(im, layers, scaleStart, scaleFactor)

    if L is None:
        print("Failed to create scale-space representation")
        return None

    # STEP 2: Compute blob detectors (like Igor Pro)
    print("Computing blob detectors...")
    BlobDetectors(L, 1)  # gammaNorm = 1 as per Igor Pro default

    # Get the computed detector waves
    detH = GetWave("detH")
    LG = GetWave("LapG")

    if detH is None or LG is None:
        print("Failed to compute blob detectors")
        return None

    # STEP 3: Handle threshold selection (matching Igor Pro behavior)
    minResponse = detHResponseThresh

    if detHResponseThresh == -2:  # Interactive threshold
        threshold = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
        if threshold is None:
            print("Threshold selection cancelled")
            return None
        minResponse = threshold
    elif detHResponseThresh == -1:  # Otsu's method
        if SKIMAGE_AVAILABLE:
            threshold = threshold_otsu(detH.data)
        else:
            threshold = np.mean(detH.data) + np.std(detH.data)
        print(f"Using Otsu threshold: {threshold}")
        minResponse = threshold
    else:  # Fixed threshold
        minResponse = detHResponseThresh
        print(f"Using fixed threshold: {minResponse}")

    # Square the minResponse like Igor Pro does
    minResponse_squared = minResponse * minResponse
    print(f"Squared minimum response: {minResponse_squared}")

    # STEP 4: Create output waves (matching Igor Pro)
    mapNum = Duplicate(im, "mapNum")
    mapNum.data = np.zeros(im.data.shape)

    mapLG = Duplicate(im, "mapLG")
    mapLG.data = np.zeros(im.data.shape)

    mapMax = Duplicate(im, "mapMax")
    mapMax.data = np.zeros(im.data.shape)

    # Initialize info wave for particle information
    info = Wave(np.zeros((1000, 13)), "info")  # Pre-allocate like Igor Pro

    # STEP 5: Find blobs
    print("Finding blobs with computed detectors...")

    # Find local maxima
    SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
    SS_MAXMAP.data = np.full(im.data.shape, -1.0)
    SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                       map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

    # Extract blob information
    info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, minResponse_squared, subPixelMult, allowOverlap)

    # Store results in global variables for access
    RegisterWave(SS_MAXMAP, "SS_MAXMAP")
    RegisterWave(SS_MAXSCALEMAP, "SS_MAXSCALEMAP")
    RegisterWave(info, "info")

    print(f"Hessian blob detection complete. Found {info.data.shape[0]} blobs.")

    return {
        'info': info,
        'SS_MAXMAP': SS_MAXMAP,
        'SS_MAXSCALEMAP': SS_MAXSCALEMAP,
        'detH': detH,
        'LG': LG,
        'threshold': minResponse
    }


def TestingMainFunctions(string_input, number_input):
    """Testing function for main functions module"""
    print(f"Main functions testing: {string_input}, {number_input}")
    return f"Main: {string_input}_{number_input}"


# Alias for Igor Pro compatibility
Testing = TestingMainFunctions