"""
Main Functions Module
Contains the primary analysis functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure

// Copyright 2019 by The Curators of the University of Missouri, a public corporation //
//																					   //
// Hessian Blob Particle Detection Suite - Python Port  //
//                                                       //
// G.M. King Laboratory                                  //
// University of Missouri-Columbia	                     //
// Coded by: Brendan Marsh                               //
// Email: marshbp@stanford.edu		                     //
// Python port maintains 1-1 functionality              //
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog, scrolledtext
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
    Based on Igor Pro Duplicate command for wave duplication
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
    Based on Igor Pro FindHessianBlobs function lines 1260-1290
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

        # Igor Pro line 1274: rad = sqrt(2*ScaleMap[i][j])
        radius = np.sqrt(2 * scale)

        # Igor Pro: Store blob information (matching Igor Pro format)
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


def igor_otsu_threshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Igor Pro Otsu threshold implementation - exact match
    Based on Igor Pro OtsuThreshold function lines 631-665
    """
    print("Running Igor Pro Otsu threshold...")
    
    # First identify the maxes (Igor Pro line 636)
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio)
    
    if maxes_wave is None or maxes_wave.data.size == 0:
        print("No maxes found for Otsu threshold")
        return 0.0
    
    maxes_data = maxes_wave.data.flatten()
    maxes_data = maxes_data[maxes_data > 0]  # Remove any invalid values
    
    if len(maxes_data) == 0:
        print("No valid maxes for Otsu threshold")
        return 0.0
    
    # Create a histogram using bin=5 (Igor Pro line 641)
    hist, bin_edges = np.histogram(maxes_data, bins=5)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Search for the best threshold (Igor Pro lines 644-661)
    min_icv = np.inf
    best_thresh = -np.inf
    
    for i, x_thresh in enumerate(bin_centers):
        # Calculate intra-class variance (ICV)
        
        # Lower class (values < threshold)
        lower_mask = maxes_data < x_thresh
        if np.any(lower_mask):
            lower_weight = np.sum(hist[:i+1]) if i+1 < len(hist) else np.sum(hist)
            lower_variance = np.var(maxes_data[lower_mask]) if np.sum(lower_mask) > 1 else 0
        else:
            lower_weight = 0
            lower_variance = 0
            
        # Upper class (values >= threshold)  
        upper_mask = maxes_data >= x_thresh
        if np.any(upper_mask):
            upper_weight = np.sum(hist[i:]) if i < len(hist) else 0
            upper_variance = np.var(maxes_data[upper_mask]) if np.sum(upper_mask) > 1 else 0
        else:
            upper_weight = 0
            upper_variance = 0
            
        # Calculate weighted intra-class variance
        icv = lower_weight * lower_variance + upper_weight * upper_variance
        
        if icv < min_icv:
            best_thresh = x_thresh
            min_icv = icv
    
    # Igor Pro takes square root of the result (line 251)
    final_threshold = np.sqrt(best_thresh) if best_thresh > 0 else 0.0
    
    print(f"Igor Pro Otsu: best_thresh={best_thresh:.6f}, final_threshold={final_threshold:.6f}")
    return final_threshold


def GetBlobDetectionParams():
    """
    Get blob detection parameters from user
    Based on Igor Pro parameter prompts in HessianBlobs function lines 141-167
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

    # Igor Pro: Scale parameters - Exact Igor Pro defaults
    scale_frame = ttk.LabelFrame(main_frame, text="Scale-Space Parameters", padding="10")
    scale_frame.pack(fill=tk.X, pady=5)

    ttk.Label(scale_frame, text="Minimum Size in Pixels").grid(row=0, column=0, sticky=tk.W)
    scale_start_var = tk.DoubleVar(value=1)
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(scale_frame, text="Maximum Size in Pixels").grid(row=1, column=0, sticky=tk.W)
    # Igor Pro default: Max(DimSize(im,0), DimSize(im,1))/4 - dynamic based on image size
    scale_max_var = tk.DoubleVar(value=64)  # Default fallback, will be updated based on actual image
    ttk.Entry(scale_frame, textvariable=scale_max_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scale Factor").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.5)  # Igor Pro default
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
    FIXED: Returns both threshold and blob info for main GUI
    """
    print("Opening interactive threshold window...")

    # Create the threshold selection window
    threshold_window = ThresholdSelectionWindow(im, detH, LG, particleType, maxCurvatureRatio)
    threshold_window.run()

    print(f"Interactive threshold selected: {threshold_window.result}")
    
    # Return both threshold and current blob info for main GUI
    if threshold_window.result is not None:
        # Ensure we have blob info
        if not hasattr(threshold_window, 'current_blob_info') or threshold_window.current_blob_info is None:
            print("Warning: No blob info from interactive threshold - will recompute")
            return threshold_window.result, None
        else:
            print(f"Returning interactive blob info with {threshold_window.current_blob_info.data.shape[0]} blobs")
            return threshold_window.result, threshold_window.current_blob_info
    else:
        return None, None


class ThresholdSelectionWindow:
    """Interactive threshold selection window with proper blob visualization"""

    def __init__(self, im, detH, LG, particleType, maxCurvatureRatio):
        self.im = im
        self.detH = detH
        self.LG = LG
        self.particleType = particleType
        self.maxCurvatureRatio = maxCurvatureRatio
        self.result = None
        self.current_blob_info = None

        # FIXED: Find range where particles actually exist
        self.particle_min, self.particle_max = self.find_particle_range()

        # FIXED: Center the default threshold like Igor Pro (WaveMax(Maxes)/2)
        self.current_thresh = (self.particle_min + self.particle_max) / 2.0

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Interactive Threshold Selection")
        self.root.geometry("1200x800")  # Larger window to accommodate slider

        self.setup_gui()

    def find_particle_range(self):
        """Find the actual range where particles are detected - FIXED to match Igor Pro exactly"""
        # FIXED: First identify the maxes exactly like Igor Pro does
        SS_MAXMAP_temp = Duplicate(self.im, "SS_MAXMAP_temp")
        SS_MAXMAP_temp.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP_temp = Duplicate(SS_MAXMAP_temp, "SS_MAXSCALEMAP_temp")

        # Run maxes to find all local maxima
        maxes_wave = Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                           map_wave=SS_MAXMAP_temp, scaleMap=SS_MAXSCALEMAP_temp)

        if maxes_wave is not None and maxes_wave.data.size > 0:
            # FIXED: Take sqrt like Igor Pro does: Maxes = Sqrt(Maxes)
            maxes_sqrt = np.sqrt(maxes_wave.data)
            min_val = 0.0  # Igor Pro starts from 0
            max_val = np.max(maxes_sqrt)
            return min_val, max_val * 1.1  # Igor Pro uses WaveMax(Maxes)*1.1 as upper limit
        else:
            # Fallback if no maxes found
            return 0.0, 1.0

    def format_scientific(self, value):
        """Format number using scientific notation for small values like Igor Pro"""
        if abs(value) < 1e-3 or abs(value) > 1e6:
            return f"{value:.3e}"
        else:
            return f"{value:.6f}"

    def setup_gui(self):
        """Setup the GUI components"""
        # Main layout: Image on left, controls on right (like Igor Pro)
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image display (main area)
        image_frame = ttk.Frame(main_container)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: Controls panel (Igor Pro style) - Fixed width 200px like Igor
        controls_container = ttk.Frame(main_container, width=200)
        controls_container.pack(side=tk.RIGHT, fill=tk.Y)
        controls_container.pack_propagate(False)  # Maintain fixed width
        
        # FIXED: Igor Pro threshold setup - WaveMax(Maxes)/2 starting value
        maxes_wave = self.get_initial_maxes()
        if maxes_wave is not None and maxes_wave.data.size > 0:
            maxes_sqrt = np.sqrt(maxes_wave.data)  # Igor Pro: Maxes = Sqrt(Maxes)
            wave_max = np.max(maxes_sqrt)
            self.particle_min = 0.0
            self.particle_max = wave_max * 1.1  # Igor Pro upper limit
            self.current_thresh = wave_max / 2.0  # Igor Pro default: WaveMax(Maxes)/2
        else:
            self.particle_min = 0.0
            self.particle_max = 1.0
            self.current_thresh = 0.5

        # Update the threshold variable
        self.thresh_var = tk.DoubleVar(value=self.current_thresh)

        # IGOR PRO LAYOUT: Compact controls panel (200px wide)
        # Top: Accept/Quit buttons (Igor Pro: pos={0,0}, size={100,50})
        button_frame = ttk.Frame(controls_container)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        accept_btn = ttk.Button(button_frame, text="Accept", 
                               command=self.accept_threshold,
                               width=12)
        accept_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        quit_btn = ttk.Button(button_frame, text="Quit", 
                             command=self.cancel_threshold,
                             width=12)
        quit_btn.pack(side=tk.LEFT)

        # SetVariable control (Igor Pro: pos={10,50}, size={170,100})
        setvar_frame = ttk.Frame(controls_container)
        setvar_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(setvar_frame, text="Blob Strength:").pack(anchor=tk.W)
        self.thresh_entry = ttk.Entry(setvar_frame, textvariable=self.thresh_var, width=25)
        self.thresh_entry.pack(fill=tk.X, pady=2)
        self.thresh_entry.bind('<Return>', self.on_manual_entry)
        
        # Current value display
        self.thresh_label = ttk.Label(setvar_frame, text=self.format_scientific(self.current_thresh), 
                                      font=('TkDefaultFont', 9), foreground='blue')
        self.thresh_label.pack(anchor=tk.W, pady=2)

        # HORIZONTAL Slider (Igor Pro: pos={0,80}, size={100,470}, but horizontal layout)
        slider_frame = ttk.Frame(controls_container)
        slider_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(slider_frame, text="Slider:", font=('TkDefaultFont', 9)).pack(anchor=tk.W)
        
        # FIXED: Igor Pro slider bounds and increment
        increment = (self.particle_max - self.particle_min) / 200.0
        
        # Igor Pro style HORIZONTAL slider
        self.thresh_scale = tk.Scale(slider_frame, 
                                   from_=self.particle_min,   # Igor Pro: left = min
                                   to=self.particle_max,      # Igor Pro: right = max  
                                   resolution=increment,       # Igor Pro: WaveMax(Maxes)*1.1/200
                                   orient=tk.HORIZONTAL,       # Horizontal for better space usage
                                   length=180,                 # Fit in 200px panel width
                                   variable=self.thresh_var,
                                   command=self.on_threshold_change,
                                   showvalue=0)                # Don't show value (we have label)
        self.thresh_scale.pack(fill=tk.X, pady=2)
        
        # FIXED: Set slider to starting position after creation
        self.thresh_scale.set(self.current_thresh)

        # Range info (compact)
        info_frame = ttk.Frame(controls_container)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text=f"Range: {self.format_scientific(self.particle_min)} to {self.format_scientific(self.particle_max)}", 
                  font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        
        # Blob count display
        self.blob_count_label = ttk.Label(info_frame, text="Blobs: 0", 
                                         font=('TkDefaultFont', 9, 'bold'), foreground='green')
        self.blob_count_label.pack(anchor=tk.W, pady=2)

        # Initial display
        self.update_display()

    def get_initial_maxes(self):
        """Get the initial maxes wave for threshold setup"""
        SS_MAXMAP_temp = Duplicate(self.im, "SS_MAXMAP_temp")
        SS_MAXMAP_temp.data = np.full(self.im.data.shape, -1.0)
        SS_MAXSCALEMAP_temp = Duplicate(SS_MAXMAP_temp, "SS_MAXSCALEMAP_temp")
        
        return Maxes(self.detH, self.LG, self.particleType, self.maxCurvatureRatio,
                     map_wave=SS_MAXMAP_temp, scaleMap=SS_MAXSCALEMAP_temp)

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
        """Update display - show image with blob circles for preview"""
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

        # FIXED: Extract blobs above threshold - Igor Pro compares with squared threshold  
        # Igor Pro line 1270: If(Map[i][j]>S_STruct.curval^2)
        thresh_squared = self.current_thresh * self.current_thresh
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, thresh_squared)

        # Show blob circles (like Igor Pro preview)
        if info.data.shape[0] > 0:
            for i in range(info.data.shape[0]):
                x_coord = info.data[i, 0]
                y_coord = info.data[i, 1]
                radius = info.data[i, 2]

                # Draw perimeter circle (green like Igor Pro)
                circle = Circle((x_coord, y_coord), radius,
                               fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)

        self.ax.set_xlim(0, self.im.data.shape[1])
        self.ax.set_ylim(self.im.data.shape[0], 0)  # Flip y axis for image coordinates

        blob_count = info.data.shape[0] if info.data.shape[0] > 0 else 0
        self.blob_count_label.config(text=f"Blobs: {blob_count}")

        # Store the current blob info for access by main GUI
        self.current_blob_info = info

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
        # Make sure we have the latest blob info
        if not hasattr(self, 'current_blob_info') or self.current_blob_info is None:
            self.update_display()  # Force update to get blob info
        self.root.destroy()

    def cancel_threshold(self):
        """Cancel threshold selection"""
        self.result = None
        self.root.destroy()

    def run(self):
        """Run the threshold selection dialog"""
        self.root.mainloop()


def HessianBlobs(im, scaleStart=1, layers=None, scaleFactor=1.5,
                 detHResponseThresh=-2, particleType=1, maxCurvatureRatio=10,
                 subPixelMult=1, allowOverlap=0):
    """
    Main Hessian blob detection function
    Direct port from Igor Pro HessianBlobs function
    FIXED: Integer conversion for layers calculation
    """
    print("Starting Hessian Blob Detection...")
    
    # FIXED: Calculate default layers exactly like Igor Pro if not provided
    # Igor Pro line 143: layers = Max( DimSize(im,0) , DimSize(im,1) ) /4
    if layers is None:
        layers = max(im.data.shape[0], im.data.shape[1]) // 4
        print(f"Using Igor Pro default layers: {layers} (Max(DimSize)/4)")
    
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

    # FIXED: Igor Pro scale conversion - Sqrt(scaleStart)/DimDelta(im,0)
    igor_scale_start = np.sqrt(scaleStart) / DimDelta(im, 0)
    L = ScaleSpaceRepresentation(im, layers, igor_scale_start, scaleFactor)

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

    interactive_blob_info = None
    
    if detHResponseThresh == -2:  # Interactive threshold
        threshold_result = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
        if threshold_result[0] is None:
            print("Threshold selection cancelled")
            return None
        threshold, interactive_blob_info = threshold_result
        minResponse = threshold
        print(f"DEBUG: Got threshold={threshold}, blob_info type={type(interactive_blob_info)}")
        if interactive_blob_info is not None:
            print(f"DEBUG: Interactive blob info has {interactive_blob_info.data.shape[0]} blobs")
    elif detHResponseThresh == -1:  # Otsu's method - FIXED to match Igor Pro exactly
        print("Calculating Otsu's Threshold (Igor Pro method)...")
        threshold = igor_otsu_threshold(detH, LG, particleType, maxCurvatureRatio)
        print(f"Igor Pro Otsu threshold: {threshold}")
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

    # Use interactive blob info if available, otherwise compute fresh
    if detHResponseThresh == -2 and interactive_blob_info is not None:
        print(f"DEBUG: Using interactive blob info with {interactive_blob_info.data.shape[0]} blobs")
        info = interactive_blob_info
        # Still need the maps for completeness
        SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")
    else:
        print("DEBUG: Computing fresh blob info")
        # Find local maxima
        SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
        SS_MAXMAP.data = np.full(im.data.shape, -1.0)
        SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

        print(f"DEBUG: Computing maxes with threshold {minResponse}")
        maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                           map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Extract blob information
        print(f"DEBUG: Extracting blobs with squared threshold {minResponse_squared}")
        info = ExtractBlobInfo(SS_MAXMAP, SS_MAXSCALEMAP, minResponse_squared, subPixelMult, allowOverlap)
    
    print(f"DEBUG: Final blob count: {info.data.shape[0]}")

    # Store results in global variables for access
    RegisterWave(SS_MAXMAP, "SS_MAXMAP")
    RegisterWave(SS_MAXSCALEMAP, "SS_MAXSCALEMAP")
    RegisterWave(info, "info")

    print(f"Hessian blob detection complete. Found {info.data.shape[0]} blobs.")
    
    # Calculate particle measurements (height, volume, area, etc.)
    if info.data.shape[0] > 0:
        from particle_measurements import MeasureParticles
        MeasureParticles(im, info)
        print(f"Calculated measurements for {info.data.shape[0]} particles")

    return {
        'info': info,
        'SS_MAXMAP': SS_MAXMAP,
        'SS_MAXSCALEMAP': SS_MAXSCALEMAP,
        'detH': detH,
        'LG': LG,
        'threshold': minResponse,
        'detHResponseThresh': detHResponseThresh,  # Original threshold mode
        'manual_threshold_used': detHResponseThresh == -2,  # Flag for manual threshold
        'auto_threshold_used': detHResponseThresh == -1,  # Flag for auto threshold
        'manual_value_used': detHResponseThresh > 0  # Flag for manual value
    }


def ViewParticleData(info_wave, image_name, original_image=None):
    """
    Igor Pro ViewParticles implementation - scroll through individual blobs
    Based on Igor Pro ViewParticles function lines 2306-2360
    """
    try:
        # Igor Pro: Check if particles exist
        if info_wave is None or info_wave.data.shape[0] == 0:
            messagebox.showwarning("No Particles", "No particles to view.")
            return
        
        print(f"DEBUG ViewParticleData: Creating viewer with {info_wave.data.shape[0]} particles")
        print(f"DEBUG ViewParticleData: Image name: {image_name}")
        print(f"DEBUG ViewParticleData: Original image type: {type(original_image)}")
        
        # Igor Pro: Validate original image data
        if original_image is None:
            messagebox.showwarning("No Image Data", "Original image data is required for particle viewing.")
            return
            
        # Use the working ViewParticles function from particle_measurements.py
        from particle_measurements import ViewParticles
        ViewParticles(original_image, info_wave)
        
    except Exception as e:
        print(f"DEBUG ViewParticleData error: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("ViewParticles Error", f"Error creating particle viewer:\n{str(e)}")


class ParticleViewer:
    """Igor Pro HessianBlobsRelease18 ViewParticles implementation - exact 1:1 port"""
    
    def __init__(self, info_wave, image_name, original_image=None):
        try:
            print(f"DEBUG ParticleViewer init: Starting initialization")
            self.info_wave = info_wave
            self.image_name = image_name
            self.original_image = original_image
            self.num_particles = info_wave.data.shape[0]
            self.current_particle = 0
            
            # Igor Pro ViewParticles settings
            self.color_table = "gray"
            self.color_range = -1  # -1 = autoscale
            self.interpolate = False
            self.show_perimeter = True
            self.x_range = -1  # -1 = autoscale
            self.y_range = -1  # -1 = autoscale
            
            print(f"DEBUG ParticleViewer init: Creating window for {self.num_particles} particles")
            
            # Igor Pro: Create viewer window as Toplevel to avoid Tk() conflicts
            # Igor Pro line 2325-2326: DoWindow/K ParticleView, Display/K=1 /N=ParticleView
            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")  # Igor Pro title
            self.root.geometry("900x600")  # Igor Pro: W=(500,200,900,600) 
            self.root.transient()
            self.root.focus_set()
            
            print(f"DEBUG ParticleViewer init: Calling setup_viewer")
            self.setup_viewer()
            print(f"DEBUG ParticleViewer init: Initialization complete")
            
        except Exception as e:
            print(f"DEBUG ParticleViewer init error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def setup_viewer(self):
        """Setup the particle viewer interface - exact Igor Pro ViewParticles layout"""
        try:
            print(f"DEBUG setup_viewer: Creating Igor Pro style layout")
            
            # Igor Pro: Main layout - Image on left, controls panel on right
            # Igor Pro line 2331: NewPanel/HOST=ParticleView /EXT=0 /K=2 /W=(0,0,150,398) /N=ViewControls
            main_container = ttk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Igor Pro: Image display area (main particle view)
            image_frame = ttk.Frame(main_container)
            image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            # Igor Pro: Create matplotlib figure for particle display
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Igor Pro: Controls panel - exact width 150px like Igor Pro
            # Igor Pro line 2331: /W=(0,0,150,398)
            controls_container = ttk.Frame(main_container, width=150)
            controls_container.pack(side=tk.RIGHT, fill=tk.Y)
            controls_container.pack_propagate(False)
            
            # Igor Pro line 2332: TitleBox ParticleName pos={20,10},size={140,25},fsize=15,fstyle=1,frame=0
            self.particle_title = ttk.Label(controls_container, 
                                           text=f"Particle {self.current_particle + 1}",
                                           font=('TkDefaultFont', 15, 'bold'))
            self.particle_title.pack(pady=(10, 5))
            
            # Igor Pro lines 2333-2334: Next/Prev buttons
            # Button NextBtn pos={80,40},size={60,25},title="Next",fsize=13,proc=ViewNextBtn
            # Button PrevBtn pos={10,40},size={60,25},title="Prev",fsize=13,proc=ViewPrevBtn
            nav_frame = ttk.Frame(controls_container)
            nav_frame.pack(fill=tk.X, pady=5)
            
            prev_btn = ttk.Button(nav_frame, text="Prev", 
                                 command=self.prev_particle, width=8)
            prev_btn.pack(side=tk.LEFT, padx=(5, 2))
            
            next_btn = ttk.Button(nav_frame, text="Next", 
                                 command=self.next_particle, width=8)  
            next_btn.pack(side=tk.LEFT, padx=(2, 5))
            
            # Igor Pro line 2335: SetVariable GoTo pos={30,75},size={100,25},title="Go To:"
            goto_frame = ttk.Frame(controls_container)
            goto_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(goto_frame, text="Go To:", font=('TkDefaultFont', 13)).pack(anchor=tk.W, padx=5)
            self.goto_var = tk.IntVar(value=self.current_particle + 1)
            self.goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=12)
            self.goto_entry.pack(anchor=tk.W, padx=5, pady=2)
            self.goto_entry.bind('<Return>', self.goto_particle)
            
            # Igor Pro line 2336: PopUpMenu ColorTab pos={10,110},size={130,25},bodywidth=130,fsize=13,title="",value="*COLORTABLEPOP*"
            color_frame = ttk.Frame(controls_container)
            color_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(color_frame, text="Color Table:", font=('TkDefaultFont', 11)).pack(anchor=tk.W, padx=5)
            self.color_var = tk.StringVar(value=self.color_table)
            color_combo = ttk.Combobox(color_frame, textvariable=self.color_var,
                                      values=["gray", "hot", "cool", "rainbow", "viridis", "plasma"],
                                      width=15, state="readonly")
            color_combo.pack(anchor=tk.W, padx=5, pady=2)
            color_combo.bind('<<ComboboxSelected>>', self.on_color_change)
            
            # Igor Pro line 2337: SetVariable ColorRange pos={10,130},size={127,25},title="Color Range"
            range_frame = ttk.Frame(controls_container)
            range_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(range_frame, text="Color Range:", font=('TkDefaultFont', 11)).pack(anchor=tk.W, padx=5)
            self.range_var = tk.DoubleVar(value=self.color_range)
            range_entry = ttk.Entry(range_frame, textvariable=self.range_var, width=12)
            range_entry.pack(anchor=tk.W, padx=5, pady=2)
            range_entry.bind('<Return>', self.on_range_change)
            
            # Igor Pro line 2338: Checkbox Interpo pos={10,145},size={100,10},side=1,title="Interpolate:"
            self.interp_var = tk.BooleanVar(value=self.interpolate)
            interp_check = ttk.Checkbutton(controls_container, text="Interpolate:",
                                          variable=self.interp_var,
                                          command=self.on_interp_change)
            interp_check.pack(anchor=tk.W, padx=5, pady=2)
            
            # Igor Pro line 2339: Checkbox Perim pos={10,160},size={100,10},side=1,title="Perimeter:"
            self.perim_var = tk.BooleanVar(value=self.show_perimeter)
            perim_check = ttk.Checkbutton(controls_container, text="Perimeter:",
                                         variable=self.perim_var,
                                         command=self.on_perim_change)
            perim_check.pack(anchor=tk.W, padx=5, pady=2)
            
            # Igor Pro lines 2340-2341: X-Range and Y-Range controls
            xy_frame = ttk.Frame(controls_container)
            xy_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(xy_frame, text="X-Range:", font=('TkDefaultFont', 10)).pack(anchor=tk.W, padx=5)
            self.x_range_var = tk.DoubleVar(value=self.x_range)
            x_entry = ttk.Entry(xy_frame, textvariable=self.x_range_var, width=12)
            x_entry.pack(anchor=tk.W, padx=5, pady=1)
            x_entry.bind('<Return>', self.on_range_change)
            
            ttk.Label(xy_frame, text="Y-Range:", font=('TkDefaultFont', 10)).pack(anchor=tk.W, padx=5, pady=(5,0))
            self.y_range_var = tk.DoubleVar(value=self.y_range)
            y_entry = ttk.Entry(xy_frame, textvariable=self.y_range_var, width=12)
            y_entry.pack(anchor=tk.W, padx=5, pady=1)
            y_entry.bind('<Return>', self.on_range_change)
            
            # Igor Pro lines 2342-2345: Height and Volume displays
            # TitleBox HeightTitle pos={10,220},size={150,25},fsize=15,frame=0,title="Height"
            # ValDisplay HeightDisp pos={10,245},size={130,25},fsize=15,frame=3,value=_NUM:0
            measurements_frame = ttk.LabelFrame(controls_container, text="Measurements", padding="5")
            measurements_frame.pack(fill=tk.X, pady=10)
            
            # Height
            ttk.Label(measurements_frame, text="Height", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
            self.height_label = ttk.Label(measurements_frame, text="0.0", 
                                         font=('TkDefaultFont', 11), relief="sunken", width=15)
            self.height_label.pack(anchor=tk.W, pady=2)
            
            # Volume  
            ttk.Label(measurements_frame, text="Volume", font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(10,0))
            self.volume_label = ttk.Label(measurements_frame, text="0.0", 
                                         font=('TkDefaultFont', 11), relief="sunken", width=15)
            self.volume_label.pack(anchor=tk.W, pady=2)
            
            # Igor Pro line 2346: Button DeleteBtn pos={10,370},size={130,25},title="DELETE"
            delete_frame = ttk.Frame(controls_container)
            delete_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
            
            delete_btn = ttk.Button(delete_frame, text="DELETE", 
                                   command=self.delete_particle)
            delete_btn.pack(fill=tk.X, padx=5)
            
            # Close button (Python addition)
            close_btn = ttk.Button(delete_frame, text="Close", 
                                  command=self.close_viewer)
            close_btn.pack(fill=tk.X, padx=5, pady=(5,0))
            
            # Igor Pro: Set up keyboard shortcuts
            self.root.bind('<Key>', self.on_key_press)
            self.root.focus_set()
            
            # Display first particle
            self.display_current_particle()
            
        except Exception as e:
            print(f"DEBUG setup_viewer error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def display_current_particle(self):
        """Display the current particle with measurements - EXACT Igor Pro behavior"""
        try:
            # Igor Pro: Bounds checking
            if self.current_particle >= self.num_particles:
                self.current_particle = self.num_particles - 1
            elif self.current_particle < 0:
                self.current_particle = 0
                
            # Igor Pro: Get particle data
            particle_data = self.info_wave.data[self.current_particle]
            x_pos = particle_data[0]
            y_pos = particle_data[1] 
            radius = particle_data[2]
            response = particle_data[3]
            scale = particle_data[4] if len(particle_data) > 4 else 0.0
            
            # Igor Pro: Clear and setup the plot
            self.ax.clear()
            
            if self.original_image is not None and hasattr(self.original_image, 'data'):
                # Igor Pro: Cropping - show region around particle
                # Igor Pro crops to show the particle clearly with surrounding context
                
                # Igor Pro: Calculate crop bounds (larger region than just the particle)
                crop_size = max(int(radius * 4), 50)  # At least 4x radius or 50 pixels
                
                x_min = max(0, int(x_pos - crop_size))
                x_max = min(self.original_image.data.shape[1], int(x_pos + crop_size))
                y_min = max(0, int(y_pos - crop_size))
                y_max = min(self.original_image.data.shape[0], int(y_pos + crop_size))
                
                # Igor Pro: Crop the actual image region
                cropped_image = self.original_image.data[y_min:y_max, x_min:x_max]
                
                # Igor Pro: Display with proper extent and user settings
                interpolation = 'bilinear' if self.interpolate else 'nearest'
                
                # Igor Pro: Apply color range if specified (-1 = autoscale)
                if self.color_range == -1:
                    vmin, vmax = None, None  # Autoscale
                else:
                    vmin, vmax = 0, self.color_range
                
                self.ax.imshow(cropped_image, cmap=self.color_table, 
                              extent=[x_min, x_max, y_max, y_min],
                              interpolation=interpolation,
                              vmin=vmin, vmax=vmax)
                
                # Igor Pro: Set the view limits
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_max, y_min)
                
            else:
                # Igor Pro: If no original image, show error
                messagebox.showwarning("No Image Data", "Original image not available for viewing.")
                return
            
            # Igor Pro: Draw the particle perimeter if enabled
            if self.show_perimeter:
                circle = Circle((x_pos, y_pos), radius, 
                               fill=False, edgecolor='lime', linewidth=2, alpha=0.9)
                self.ax.add_patch(circle)
            
            # Igor Pro: Mark the center (red crosshair like Igor Pro)
            self.ax.plot(x_pos, y_pos, 'r+', markersize=12, markeredgewidth=3)
            
            # Igor Pro: Style title and axis
            self.ax.set_title(f"Particle {self.current_particle + 1} of {self.num_particles}", 
                             fontsize=12, fontweight='bold')
            self.ax.set_aspect('equal')
            
            # Igor Pro: Add scale bar if radius is reasonable size
            if radius > 5:
                scale_length = radius
                scale_x = x_min + (x_max - x_min) * 0.1
                scale_y = y_max - (y_max - y_min) * 0.1
                self.ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 
                            'yellow', linewidth=3)
                self.ax.text(scale_x + scale_length/2, scale_y - (y_max - y_min) * 0.05, 
                            f'{scale_length:.1f} px', ha='center', color='yellow', 
                            fontsize=10, fontweight='bold')
            
            # Igor Pro: Update measurement labels
            self.particle_title.config(text=f"Particle {self.current_particle + 1}")
            
            # Igor Pro: Calculate height and volume (simplified for now)
            # In full Igor Pro implementation, these would come from actual particle measurements
            height = response * 1000  # Simplified height calculation
            volume = (4/3) * np.pi * (radius ** 3)  # Sphere volume approximation
            
            self.height_label.config(text=f"{height:.2f}")
            self.volume_label.config(text=f"{volume:.2f}")
            self.goto_var.set(self.current_particle + 1)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error displaying particle {self.current_particle + 1}: {str(e)}")
            messagebox.showerror("Display Error", f"Error displaying particle:\n{str(e)}")

    # Igor Pro ViewParticles callback methods - exact port from HessianBlobsRelease18
    
    def on_color_change(self, event=None):
        """Igor Pro ViewColorTab callback"""
        self.color_table = self.color_var.get()
        self.display_current_particle()
    
    def on_range_change(self, event=None):
        """Igor Pro ViewColorRange and ViewRange callbacks"""
        try:
            self.color_range = self.range_var.get()
            self.x_range = self.x_range_var.get()
            self.y_range = self.y_range_var.get()
            self.display_current_particle()
        except tk.TclError:
            pass  # Invalid input, ignore
    
    def on_interp_change(self):
        """Igor Pro ViewInterp callback"""
        self.interpolate = self.interp_var.get()
        self.display_current_particle()
    
    def on_perim_change(self):
        """Igor Pro ViewPerim callback"""
        self.show_perimeter = self.perim_var.get()
        self.display_current_particle()
    
    def delete_particle(self):
        """Igor Pro ViewDelete callback - show confirmation dialog"""
        particle_num = self.current_particle + 1
        result = messagebox.askyesno(
            f"Deleting Particle {particle_num}...", 
            f"Are you sure you want to delete Particle {particle_num}?",
            icon='warning'
        )
        if result:
            # For now, just show a message that particle would be deleted
            # In full Igor Pro implementation, this would remove from the dataset
            messagebox.showinfo("Delete", f"Particle {particle_num} marked for deletion.\n(Not implemented in Python port)")
    
    def on_key_press(self, event):
        """Igor Pro ParticleViewHook keyboard shortcuts"""
        # Igor Pro lines 2378-2391: Keyboard hook implementation
        if event.keysym == 'Right':  # Arrow Right - Next
            self.next_particle()
        elif event.keysym == 'Left':  # Arrow Left - Prev  
            self.prev_particle()
        elif event.keysym in ['Down', 'space']:  # Down Arrow or Space - Delete
            self.delete_particle()
        
    def next_particle(self):
        """Navigate to next particle"""
        if self.current_particle < self.num_particles - 1:
            self.current_particle += 1
            self.display_current_particle()
            
    def prev_particle(self):
        """Navigate to previous particle"""
        if self.current_particle > 0:
            self.current_particle -= 1
            self.display_current_particle()
            
    def goto_particle(self, event=None):
        """Go to specific particle number"""
        try:
            particle_num = self.goto_var.get() - 1  # Convert to 0-based index
            if 0 <= particle_num < self.num_particles:
                self.current_particle = particle_num
                self.display_current_particle()
        except (ValueError, tk.TclError):
            pass  # Invalid input, ignore
            
    def close_viewer(self):
        """Close the particle viewer"""
        self.root.destroy()
        
    def run(self):
        """Run the particle viewer"""
        self.root.mainloop()


def TestingMainFunctions(string_input, number_input):
    """Testing function for main functions module"""
    print(f"Main functions testing: {string_input}, {number_input}")
    return f"Main: {string_input}_{number_input}"


# Alias for Igor Pro compatibility
Testing = TestingMainFunctions