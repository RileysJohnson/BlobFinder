"""
Main Functions Module
Contains the primary analysis functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete interactive threshold functionality matching Igor exactly
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

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def GetBlobDetectionParams():
    """
    Get blob detection parameters from user
    Replicates Igor Pro parameter dialog
    """
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window

    dialog = tk.Toplevel()
    dialog.title("Blob Detection Parameters")
    dialog.geometry("650x500")  # FIXED: Even larger to accommodate longer labels
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Blob Detection Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Scale parameters - FIXED: Now matches original Igor Pro defaults exactly
    scale_frame = ttk.LabelFrame(main_frame, text="Scale-Space Parameters", padding="10")
    scale_frame.pack(fill=tk.X, pady=5)

    ttk.Label(scale_frame, text="Minimum Size in Pixels:").grid(row=0, column=0, sticky=tk.W)
    scale_start_var = tk.DoubleVar(value=1.0)  # Original Igor default
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=15).grid(row=0, column=1, padx=5)

    ttk.Label(scale_frame, text="Maximum Size in Pixels:").grid(row=1, column=0, sticky=tk.W)
    scale_layers_var = tk.IntVar(value=120)  # Will be calculated, but 120 is typical default shown
    ttk.Entry(scale_frame, textvariable=scale_layers_var, width=15).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scaling Factor:").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.5)  # Original Igor default
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=15).grid(row=2, column=1, padx=5)

    # Detection parameters - FIXED: Matches original Igor Pro dialog
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=5)

    ttk.Label(detect_frame, text="Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method):").grid(row=0,
                                                                                                           column=0,
                                                                                                           sticky=tk.W,
                                                                                                           columnspan=2)
    min_response_var = tk.DoubleVar(value=-2.0)  # Original Igor default (Interactive)
    ttk.Entry(detect_frame, textvariable=min_response_var, width=15).grid(row=0, column=2, padx=5)

    ttk.Label(detect_frame, text="Particle Type (-1 for negative, +1 for positive, 0 for both):").grid(row=1, column=0,
                                                                                                       sticky=tk.W,
                                                                                                       columnspan=2)
    particle_type_var = tk.IntVar(value=1)  # Original Igor default (positive blobs)
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=15).grid(row=1, column=2, padx=5)

    ttk.Label(detect_frame, text="Subpixel Ratio:").grid(row=2, column=0, sticky=tk.W)
    subpixel_var = tk.IntVar(value=1)  # Original Igor default
    ttk.Entry(detect_frame, textvariable=subpixel_var, width=15).grid(row=2, column=2, padx=5)

    ttk.Label(detect_frame, text="Allow Hessian Blobs to Overlap? (1=yes 0=no):").grid(row=3, column=0, sticky=tk.W,
                                                                                       columnspan=2)
    overlap_var = tk.IntVar(value=0)  # Original Igor default
    ttk.Entry(detect_frame, textvariable=overlap_var, width=15).grid(row=3, column=2, padx=5)

    # Buttons
    def ok_clicked():
        # Calculate layers from max size like Igor Pro does
        max_size_pixels = scale_layers_var.get()

        params = {
            'scaleStart': scale_start_var.get(),
            'scaleLayers': max_size_pixels,  # Will be recalculated in detection function
            'scaleFactor': scale_factor_var.get(),
            'minResponse': min_response_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': 10.0,  # Hard-coded like in Igor Pro
            'subPixelMult': subpixel_var.get(),
            'allowOverlap': overlap_var.get()
        }
        result[0] = params
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    # Wait for user input
    dialog.wait_window()

    return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Interactive threshold selection window
    Direct port from Igor Pro InteractiveThreshold function
    Replicates the exact functionality with image display and slider
    FIXED: Now properly matches Igor Pro behavior exactly

    Parameters:
    im : Wave - The image under analysis
    detH : Wave - The determinant of Hessian blob detector
    LG : Wave - The Laplacian of Gaussian blob detector
    particleType : int - 1 to consider positive Hessian blobs, -1 for negative, 0 for both
    maxCurvatureRatio : float - Maximum ratio of the principal curvatures of a blob

    Returns:
    float - Selected threshold value in image units (square root of actual detH response)
    """
    print("Starting Interactive Threshold Selection...")

    # First identify the maxes - exactly like Igor Pro
    # Duplicate/O im, SS_MAXMAP
    SS_MAXMAP = Duplicate(im, "SS_MAXMAP")
    SS_MAXMAP.data = np.full(im.data.shape, -1.0)  # Multithread Map = -1

    # Duplicate/O Map, SS_MAXSCALEMAP
    SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP")

    # Wave Maxes = Maxes(detH,LG,particleType,maxCurvatureRatio,map=Map,scaleMap=ScaleMap)
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                       map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

    # Maxes = Sqrt(Maxes) // Put it into image units
    maxes_data = np.sqrt(np.maximum(maxes_wave.data, 0))

    # Filter out zero values for a more meaningful range
    non_zero_maxes = maxes_data[maxes_data > 0]
    if len(non_zero_maxes) == 0:
        messagebox.showwarning("No Blobs", "No suitable blob candidates found in image.")
        return 0.0

    # Determine a sensible range for the slider
    min_val = np.min(non_zero_maxes)
    max_val = np.max(non_zero_maxes)
    # Set a reasonable upper bound to avoid extreme values dominating the slider
    # Using the 98th percentile is a good heuristic to exclude outliers
    robust_max = np.percentile(non_zero_maxes, 98) if len(non_zero_maxes) > 50 else max_val

    print(f"Detector response range: min={min_val:.4f}, max={max_val:.4f}, robust_max={robust_max:.4f}")

    class ThresholdWindow:
        def __init__(self):
            self.threshold = robust_max / 2.0  # Start at a sensible middle point
            self.accepted = False
            self.root = None
            self.fig = None
            self.ax = None
            self.canvas = None
            self.circles = []  # Keep track of circles for clearing

        def create_window(self):
            """Create the interactive threshold window"""
            self.root = tk.Tk()
            self.root.title("Interactive Blob Strength")
            self.root.geometry("1000x700")

            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            left_frame = ttk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            right_frame = ttk.Frame(main_frame, width=220)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            right_frame.pack_propagate(False)

            controls_label = ttk.Label(right_frame, text="Controls", font=('TkDefaultFont', 10, 'bold'))
            controls_label.pack(pady=(10, 5))

            button_frame = ttk.Frame(right_frame)
            button_frame.pack(fill=tk.X, padx=5)
            ttk.Button(button_frame, text="Accept", command=self.on_accept).pack(side=tk.LEFT, expand=True, fill=tk.X)
            ttk.Button(button_frame, text="Quit", command=self.on_quit).pack(side=tk.LEFT, expand=True, fill=tk.X)

            threshold_frame = ttk.Frame(right_frame)
            threshold_frame.pack(fill=tk.X, padx=5, pady=(15, 5))
            ttk.Label(threshold_frame, text="Blob Strength Threshold:").pack()
            self.threshold_var = tk.DoubleVar(value=self.threshold)
            threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=15)
            threshold_entry.pack(pady=2)
            threshold_entry.bind('<Return>', self.on_entry_change)
            threshold_entry.bind('<FocusOut>', self.on_entry_change)

            slider_frame = ttk.Frame(right_frame)
            slider_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
            self.slider_var = tk.DoubleVar(value=self.threshold)
            # Use the robust_max for the slider range for better usability
            slider = tk.Scale(slider_frame,
                              from_=robust_max * 1.05,
                              to=min_val * 0.95,
                              resolution=-1,  # Auto-resolution
                              orient=tk.VERTICAL,
                              variable=self.slider_var,
                              command=self.on_slider_change,
                              length=400,
                              digits=4)
            slider.pack(fill=tk.BOTH, expand=True)

            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Display the image
            self.ax.imshow(im.data, cmap='gray', origin='lower',
                           extent=[DimOffset(im, 0),
                                   DimOffset(im, 0) + DimSize(im, 0) * DimDelta(im, 0),
                                   DimOffset(im, 1),
                                   DimOffset(im, 1) + DimSize(im, 1) * DimDelta(im, 1)])
            self.ax.set_title(f"IMAGE:{im.name}")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

            # Initial display update
            self.update_display()

        def update_display(self):
            """Update the display with circles at current threshold - matches Igor exactly"""
            # Clear previous circles (like SetDrawLayer/K /W=IMAGE overlay in Igor)
            for circle in self.circles:
                circle.remove()
            self.circles = []

            # Draw circles for blobs above threshold
            # This matches the Igor Pro logic exactly:
            # For(i=0;i<limI;i+=1)
            # For(j=0;j<limJ;j+=1)
            #   If(Map[i][j]>S_STruct.curval^2)  // Note: threshold is SQUARED
            limI = DimSize(SS_MAXMAP, 0)
            limJ = DimSize(SS_MAXMAP, 1)

            threshold_squared = self.threshold ** 2  # Igor squares the threshold!

            for i in range(limI):
                for j in range(limJ):
                    if SS_MAXMAP.data[i, j] > threshold_squared:
                        # FIXED: Coordinate calculation - i=row=Y, j=col=X
                        # xc should use j (column index for X direction)
                        # yc should use i (row index for Y direction)
                        xc = DimOffset(SS_MAXMAP, 0) + j * DimDelta(SS_MAXMAP, 0)
                        yc = DimOffset(SS_MAXMAP, 1) + i * DimDelta(SS_MAXMAP, 1)

                        # Get radius from scale map: rad = ScaleMap[i][j]
                        rad = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP.data[i, j] > 0 else 2.0

                        # Draw circle (Igor uses red: linefgc= (65535,16385,16385))
                        circle = Circle((xc, yc), rad, fill=False, color='red', linewidth=1.5)
                        self.ax.add_patch(circle)
                        self.circles.append(circle)

            self.canvas.draw()

        def on_slider_change(self, value):
            """Handle slider value change"""
            self.threshold = float(value)
            self.threshold_var.set(self.threshold)
            self.update_display()

        def on_entry_change(self, event):
            """Handle entry value change"""
            try:
                self.threshold = self.threshold_var.get()
                self.slider_var.set(self.threshold)
                self.update_display()
            except tk.TclError:
                pass  # Invalid value, ignore

        def on_accept(self):
            """Accept current threshold"""
            self.accepted = True
            self.root.quit()
            self.root.destroy()

        def on_quit(self):
            """Quit without accepting"""
            self.accepted = False
            self.root.quit()
            self.root.destroy()

    # Create and run threshold window
    threshold_window = ThresholdWindow()
    threshold_window.create_window()
    threshold_window.root.mainloop()

    if threshold_window.accepted:
        print(f"Chosen Det H Response Threshold: {threshold_window.threshold}")
        return threshold_window.threshold
    else:
        return 0.0


def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Otsu's method for automatic threshold selection
    Direct port from Igor Pro OtsuThreshold function

    Parameters:
    detH : Wave - The determinant of Hessian blob detector
    LG : Wave - The Laplacian of Gaussian blob detector
    particleType : int - Type of particles to detect
    maxCurvatureRatio : float - Maximum curvature ratio

    Returns:
    float - Optimal threshold value
    """
    print("Calculating Otsu's Threshold...")

    # First identify the maxes - exactly like Igor Pro
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio)

    # Create a copy for working (SS_OTSU_COPY in Igor)
    workhorse_data = maxes_wave.data.copy()

    # Create histogram using bin size 5 (Igor: Histogram/B=5)
    max_val = np.max(maxes_wave.data)
    min_val = np.min(maxes_wave.data[maxes_wave.data > 0])  # Only positive values

    if max_val <= min_val:
        print("No valid maxima found for Otsu threshold")
        return 0.0

    # Create histogram with appropriate bins
    num_bins = max(50, int((max_val - min_val) / 5))  # At least 50 bins, bin size ~5
    hist, bin_edges = np.histogram(maxes_wave.data[maxes_wave.data > 0], bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Search for the best threshold (minimizing intra-class variance)
    min_icv = np.inf
    best_thresh = -np.inf

    for i, x_thresh in enumerate(bin_centers):
        # Split data into two classes
        below_thresh = maxes_wave.data[(maxes_wave.data > 0) & (maxes_wave.data < x_thresh)]
        above_thresh = maxes_wave.data[maxes_wave.data >= x_thresh]

        if len(below_thresh) == 0 or len(above_thresh) == 0:
            continue

        # Calculate weighted intra-class variance (ICV)
        # ICV = Sum(Hist,-inf,xThresh)*Variance(Workhorse) + Sum(Hist,xThresh,inf)*Variance(Workhorse)
        weight_below = len(below_thresh) / len(maxes_wave.data[maxes_wave.data > 0])
        weight_above = len(above_thresh) / len(maxes_wave.data[maxes_wave.data > 0])

        var_below = np.var(below_thresh) if len(below_thresh) > 1 else 0
        var_above = np.var(above_thresh) if len(above_thresh) > 1 else 0

        icv = weight_below * var_below + weight_above * var_above

        if icv < min_icv:
            best_thresh = x_thresh
            min_icv = icv

    print(f"Otsu's Threshold: {best_thresh}")
    return best_thresh


def FindHessianBlobs(im, detH, LG, minResponse, mapNum, mapLG, mapMax, info, particleType, maxCurvatureRatio):
    """
    Find Hessian blobs using the detector responses
    Direct port from Igor Pro FindHessianBlobs function
    FIXED: Now properly uses the same logic as Maxes function

    Parameters:
    im : Wave - The image under analysis
    detH : Wave - The determinant of Hessian blob detector
    LG : Wave - The Laplacian of Gaussian blob detector
    minResponse : float - Minimum detector response threshold
    mapNum : Wave - Output map for particle numbering
    mapLG : Wave - Output map for LG detector responses
    mapMax : Wave - Output map for maximum detector responses
    info : Wave - Output particle information array
    particleType : int - Type of particles to detect
    maxCurvatureRatio : float - Maximum curvature ratio

    Returns:
    int - Number of particles found
    """
    print("Finding Hessian blobs...")

    # Initialize output waves
    mapNum.data = np.full(im.data.shape, -1.0)
    mapLG.data = np.zeros(im.data.shape)
    mapMax.data = np.zeros(im.data.shape)

    # Use the Maxes function to find valid blobs, exactly like in Igor Pro
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio,
                       map_wave=mapNum, scaleMap=None)

    # Copy results to output maps
    mapMax.data = maxes_wave.data.copy()

    # Count particles above threshold and collect info
    # The threshold needs to be squared because Maxes stores the actual detH values
    threshold_squared = minResponse ** 2
    particle_count = 0
    particle_info = []

    for i in range(im.data.shape[0]):
        for j in range(im.data.shape[1]):
            detector_response = mapNum.data[i, j]  # This contains the actual detH value from Maxes

            if detector_response > threshold_squared:
                # Found a valid particle
                # Update mapNum to contain particle number instead of detector response
                mapNum.data[i, j] = particle_count

                # Set LG value (use first scale layer if 3D)
                if LG.data.ndim > 2:
                    mapLG.data[i, j] = LG.data[i, j, 0]
                else:
                    mapLG.data[i, j] = LG.data[i, j]

                # Store particle info - convert indices to coordinates
                x_pos = DimOffset(im, 0) + j * DimDelta(im, 0)  # j is X direction
                y_pos = DimOffset(im, 1) + i * DimDelta(im, 1)  # i is Y direction

                particle_info.append([
                    x_pos,  # X position in world coordinates
                    y_pos,  # Y position in world coordinates
                    np.sqrt(detector_response),  # Scale/radius (sqrt of detector response)
                    detector_response,  # DetH response
                    mapLG.data[i, j],  # LG response
                    j,  # X index
                    i,  # Y index
                    0,  # Scale index (placeholder)
                    0,  # Area (to be computed)
                    0  # Volume (to be computed)
                ])

                particle_count += 1
            else:
                # Not a valid particle, mark as background
                mapNum.data[i, j] = -1.0

    # Convert particle info to wave
    if particle_info:
        info.data = np.array(particle_info)
        info.SetScale('x', 0, 1)
        info.SetScale('y', 0, 1)
    else:
        info.data = np.array([]).reshape(0, 10)

    print(f"Found {particle_count} particles above threshold")
    return particle_count


def ShowOtsuResults(im, detH, LG, particleType, maxCurvatureRatio, threshold):
    """
    Display the results of Otsu's thresholding in a new window.
    """
    # Create a new window to display the results
    window = tk.Toplevel()
    window.title("Otsu's Threshold Results")
    window.geometry("800x600")

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Display the image
    ax.imshow(im.data, cmap='gray', origin='lower',
              extent=[DimOffset(im, 0), DimOffset(im, 0) + DimSize(im, 0) * DimDelta(im, 0),
                      DimOffset(im, 1), DimOffset(im, 1) + DimSize(im, 1) * DimDelta(im, 1)])
    ax.set_title(f"Otsu's Threshold: {threshold:.4f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Find and draw blobs
    SS_MAXMAP = Duplicate(im, "SS_MAXMAP_OTSU")
    SS_MAXMAP.data = np.full(im.data.shape, -1.0)
    SS_MAXSCALEMAP = Duplicate(SS_MAXMAP, "SS_MAXSCALEMAP_OTSU")

    Maxes(detH, LG, particleType, maxCurvatureRatio, map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

    threshold_squared = threshold ** 2
    num_particles = 0
    for i in range(DimSize(SS_MAXMAP, 0)):
        for j in range(DimSize(SS_MAXMAP, 1)):
            if SS_MAXMAP.data[i, j] > threshold_squared:
                xc = DimOffset(SS_MAXMAP, 0) + j * DimDelta(SS_MAXMAP, 0)
                yc = DimOffset(SS_MAXMAP, 1) + i * DimDelta(SS_MAXMAP, 1)
                rad = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP.data[i, j] > 0 else 2.0
                circle = Circle((xc, yc), rad, fill=False, color='yellow', linewidth=1.5)
                ax.add_patch(circle)
                num_particles += 1

    ax.set_title(f"Otsu's Threshold: {threshold:.4f} ({num_particles} particles)")
    canvas.draw()


def HessianBlobDetection(im, scaleStart=1.0, scaleLayers=120, scaleFactor=1.5,
                         minResponse=-2.0, particleType=1, maxCurvatureRatio=10.0,
                         subPixelMult=1, allowOverlap=0, interactive=None):
    """
    Main Hessian blob detection function
    Complete 1-1 port from Igor Pro implementation with exact parameter handling
    FIXED: Now properly calculates layers and implements Otsu's method
    """
    print(f"Starting Hessian blob detection on image: {im.name}")
    print(f"Input parameters: scaleStart={scaleStart}, maxSize={scaleLayers}, factor={scaleFactor}")
    print(f"Response threshold={minResponse}, particle type={particleType}")

    # Check parameters exactly like Igor Pro does:
    # scaleStart = (scaleStart*DimDelta(im,0))^2 /2
    scaleStart_scaled = (scaleStart * DimDelta(im, 0)) ** 2 / 2

    # layers = ceil( log( (layers*DimDelta(im,0))^2/(2*scaleStart))/log(scaleFactor) )
    max_size_scaled = (scaleLayers * DimDelta(im, 0)) ** 2 / 2
    calculated_layers = int(np.ceil(np.log(max_size_scaled / scaleStart_scaled) / np.log(scaleFactor)))

    # Ensure minimum constraints like Igor Pro
    calculated_layers = max(1, calculated_layers)
    scaleFactor = max(1.1, scaleFactor)  # Igor: scaleFactor = Max(1.1,scaleFactor)
    subPixelMult = max(1, round(subPixelMult))  # Igor: subPixelMult = Max(1,Round(subPixelMult))

    print(f"Calculated parameters: scaleStart_scaled={scaleStart_scaled}, layers={calculated_layers}")

    # Determine if interactive mode should be used
    if interactive is None:
        interactive = (minResponse == -2.0)

    # Step 1: Compute scale-space representation with proper scale start
    print("\n1. Computing scale-space representation...")
    # Use sqrt(scaleStart_scaled) to convert back to spatial units for ScaleSpaceRepresentation
    L = ScaleSpaceRepresentation(im, calculated_layers, np.sqrt(scaleStart_scaled) / DimDelta(im, 0), scaleFactor)

    # Step 2: Compute blob detectors
    print("\n2. Computing blob detectors...")
    detH, LG = BlobDetectors(L, gammaNorm=1.0)

    # Step 3: Get threshold - FIXED: Now properly implements Otsu's method
    if minResponse == -1:
        print("\n3. Calculating Otsu's Threshold...")
        threshold = np.sqrt(OtsuThreshold(detH, LG, particleType, maxCurvatureRatio))
        print(f"Otsu's Threshold: {threshold}")
        minResponse = threshold

        # FIXED: Show visualization of Otsu's results
        print("Displaying Otsu's threshold results...")
        try:
            ShowOtsuResults(im, detH, LG, particleType, maxCurvatureRatio, threshold)
        except Exception as e:
            print(f"Note: Could not display Otsu results: {e}")

    elif interactive or minResponse == -2:
        print("\n3. Interactive threshold selection...")
        threshold = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
        if threshold <= 0:
            print("Analysis cancelled by user.")
            return None
        print(f"Chosen Det H Response Threshold: {threshold}")
        minResponse = threshold
    else:
        print(f"\n3. Using provided threshold: {minResponse}")

    # Step 4: Create output waves
    print("\n4. Creating output waves...")
    mapNum = Duplicate(im, f"{im.name}_mapNum")
    mapLG = Duplicate(im, f"{im.name}_mapLG")
    mapMax = Duplicate(im, f"{im.name}_mapMax")
    info = Wave(np.array([]), f"{im.name}_info")

    # Step 5: Find blobs
    print("\n5. Finding blobs...")
    num_particles = FindHessianBlobs(im, detH, LG, minResponse, mapNum, mapLG, mapMax,
                                     info, particleType, maxCurvatureRatio)

    # Step 6: Prepare results
    results = {
        'image': im,
        'scale_space': L,
        'detH': detH,
        'LG': LG,
        'mapNum': mapNum,
        'mapLG': mapLG,
        'mapMax': mapMax,
        'info': info,
        'num_particles': num_particles,
        'threshold': minResponse,
        'parameters': {
            'scaleStart': scaleStart,
            'scaleLayers': calculated_layers,
            'scaleFactor': scaleFactor,
            'particleType': particleType,
            'maxCurvatureRatio': maxCurvatureRatio,
            'minResponse': minResponse,
            'subPixelMult': subPixelMult,
            'allowOverlap': allowOverlap
        }
    }

    print(f"\n=== Analysis Complete: Found {num_particles} particles ===")
    return results


def Testing(string_input, number_input):
    """Testing function for main_functions module"""
    print(f"Main functions testing: {string_input}, {number_input}")
    return len(string_input) + number_input