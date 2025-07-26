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
    dialog.geometry("400x350")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Blob Detection Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Scale parameters
    scale_frame = ttk.LabelFrame(main_frame, text="Scale-Space Parameters", padding="10")
    scale_frame.pack(fill=tk.X, pady=5)

    ttk.Label(scale_frame, text="Scale Start (pixels):").grid(row=0, column=0, sticky=tk.W)
    scale_start_var = tk.DoubleVar(value=1.0)
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=10).grid(row=0, column=1, padx=5)

    ttk.Label(scale_frame, text="Scale Layers:").grid(row=1, column=0, sticky=tk.W)
    scale_layers_var = tk.IntVar(value=20)
    ttk.Entry(scale_frame, textvariable=scale_layers_var, width=10).grid(row=1, column=1, padx=5)

    ttk.Label(scale_frame, text="Scale Factor:").grid(row=2, column=0, sticky=tk.W)
    scale_factor_var = tk.DoubleVar(value=1.2)
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=10).grid(row=2, column=1, padx=5)

    # Detection parameters
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=5)

    ttk.Label(detect_frame, text="Particle Type (-1, 0, 1):").grid(row=0, column=0, sticky=tk.W)
    particle_type_var = tk.IntVar(value=1)
    ttk.Entry(detect_frame, textvariable=particle_type_var, width=10).grid(row=0, column=1, padx=5)

    ttk.Label(detect_frame, text="Max Curvature Ratio:").grid(row=1, column=0, sticky=tk.W)
    max_curv_var = tk.DoubleVar(value=10.0)
    ttk.Entry(detect_frame, textvariable=max_curv_var, width=10).grid(row=1, column=1, padx=5)

    ttk.Label(detect_frame, text="Min Response (auto=-1):").grid(row=2, column=0, sticky=tk.W)
    min_response_var = tk.DoubleVar(value=-1.0)
    ttk.Entry(detect_frame, textvariable=min_response_var, width=10).grid(row=2, column=1, padx=5)

    def ok_clicked():
        result[0] = {
            'scaleStart': scale_start_var.get(),
            'scaleLayers': scale_layers_var.get(),
            'scaleFactor': scale_factor_var.get(),
            'particleType': particle_type_var.get(),
            'maxCurvatureRatio': max_curv_var.get(),
            'minResponse': min_response_var.get()
        }
        dialog.destroy()
        root.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()
        root.destroy()

    # Buttons
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

    # First identify the maxes
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

    max_value = np.max(maxes_data)
    if max_value == 0:
        messagebox.showwarning("No Blobs", "No suitable blob candidates found in image.")
        return 0.0

    # Create interactive threshold window
    class ThresholdWindow:
        def __init__(self):
            self.threshold = max_value / 2.0
            self.accepted = False
            self.root = None
            self.fig = None
            self.ax = None
            self.canvas = None
            self.slider = None
            self.circles = []

        def create_window(self):
            # Create new tkinter window
            self.root = tk.Toplevel()
            self.root.title("Interactive Blob Strength Selection")
            self.root.geometry("800x600")
            self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

            # Create main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Control frame
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, pady=5)

            # Threshold controls
            ttk.Label(control_frame, text="Blob Strength Threshold:").pack(side=tk.LEFT)

            self.threshold_var = tk.DoubleVar(value=self.threshold)
            self.threshold_entry = ttk.Entry(control_frame, textvariable=self.threshold_var, width=15)
            self.threshold_entry.pack(side=tk.LEFT, padx=5)
            self.threshold_entry.bind('<Return>', self.on_entry_change)

            # Slider
            slider_frame = ttk.Frame(main_frame)
            slider_frame.pack(fill=tk.X, pady=5)

            ttk.Label(slider_frame, text="0.0").pack(side=tk.LEFT)
            self.slider_var = tk.DoubleVar(value=self.threshold)
            self.slider = ttk.Scale(slider_frame, from_=0.0, to=max_value * 1.1,
                                    variable=self.slider_var, orient=tk.HORIZONTAL,
                                    command=self.on_slider_change)
            self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            ttk.Label(slider_frame, text=f"{max_value * 1.1:.3f}").pack(side=tk.LEFT)

            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=10)

            ttk.Button(button_frame, text="Accept", command=self.on_accept).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Quit", command=self.on_quit).pack(side=tk.LEFT, padx=5)

            # Initial plot
            self.update_display()

        def update_display(self):
            """Update the display with current threshold"""
            self.ax.clear()

            # Display image
            extent = [DimOffset(im, 0),
                      DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
                      DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1),
                      DimOffset(im, 1)]

            self.ax.imshow(im.data, extent=extent, cmap='gray', origin='lower')
            self.ax.set_title(f"Interactive Blob Strength Selection\nThreshold: {self.threshold:.6f}")
            self.ax.set_xlabel("X (pixels)")
            self.ax.set_ylabel("Y (pixels)")

            # Draw circles for blobs above threshold
            threshold_squared = self.threshold ** 2

            for i in range(SS_MAXMAP.data.shape[0]):
                for j in range(SS_MAXMAP.data.shape[1]):
                    if SS_MAXMAP.data[i, j] > threshold_squared:
                        # Calculate position in real coordinates
                        xc = DimOffset(SS_MAXMAP, 0) + j * DimDelta(SS_MAXMAP, 0)
                        yc = DimOffset(SS_MAXMAP, 1) + i * DimDelta(SS_MAXMAP, 1)

                        # Get radius from scale map
                        rad = SS_MAXSCALEMAP.data[i, j] if SS_MAXSCALEMAP.data[i, j] > 0 else 2.0

                        # Draw circle
                        circle = Circle((xc, yc), rad, fill=False, color='red', linewidth=1.5)
                        self.ax.add_patch(circle)

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
        return threshold_window.threshold
    else:
        return 0.0


def FindHessianBlobs(im, detH, LG, minResponse, mapNum, mapLG, mapMax, info, particleType, maxCurvatureRatio):
    """
    Find Hessian blobs using the detector responses
    Direct port from Igor Pro FindHessianBlobs function

    Parameters:
    im : Wave - The image under analysis
    detH : Wave - The determinant of Hessian blob detector
    LG : Wave - The Laplacian of Gaussian blob detector
    minResponse : float - Minimum detector response threshold
    mapNum : Wave - Output particle number map
    mapLG : Wave - Output Laplacian of Gaussian map
    mapMax : Wave - Output maximum detector response map
    info : Wave - Output particle information array
    particleType : int - Type of particles to detect (-1, 0, 1)
    maxCurvatureRatio : float - Maximum ratio of principal curvatures

    Returns:
    int - Number of particles found
    """
    print(f"Finding Hessian blobs with minResponse={minResponse}")

    # Square the minResponse since it's provided as square root
    minResponse_squared = minResponse ** 2

    # Initialize maps
    mapNum.data = np.full(im.data.shape, -1.0)
    mapLG.data = np.zeros(im.data.shape)
    mapMax.data = np.zeros(im.data.shape)

    particle_count = 0
    particle_list = []

    # Search through all positions and scales
    for k in range(1, detH.data.shape[2] - 1):  # Scale dimension
        for i in range(1, detH.data.shape[0] - 1):  # Y dimension
            for j in range(1, detH.data.shape[1] - 1):  # X dimension

                current_detH = detH.data[i, j, k]
                current_LG = LG.data[i, j, k]

                # Check minimum response
                if current_detH < minResponse_squared:
                    continue

                # Check curvature ratio
                if current_LG != 0 and current_detH != 0:
                    curvature_ratio = (current_LG ** 2) / current_detH
                    max_allowed = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio
                    if curvature_ratio >= max_allowed:
                        continue

                # Check particle type
                if ((particleType == -1 and current_LG >= 0) or
                        (particleType == 1 and current_LG <= 0)):
                    continue

                # Check if it's a local maximum in 3D (26 neighbors)
                is_maximum = True

                # Check strictly greater neighbors (to avoid ties)
                neighbors_strict = [
                    detH.data[i - 1, j - 1, k - 1], detH.data[i - 1, j - 1, k], detH.data[i - 1, j, k - 1],
                    detH.data[i, j - 1, k - 1], detH.data[i, j, k - 1], detH.data[i, j - 1, k], detH.data[i - 1, j, k]
                ]

                for neighbor in neighbors_strict:
                    if current_detH <= neighbor:
                        is_maximum = False
                        break

                if not is_maximum:
                    continue

                # Check greater or equal neighbors
                neighbors_equal = [
                    detH.data[i - 1, j - 1, k + 1], detH.data[i - 1, j, k + 1], detH.data[i - 1, j + 1, k - 1],
                    detH.data[i - 1, j + 1, k], detH.data[i - 1, j + 1, k + 1], detH.data[i, j - 1, k + 1],
                    detH.data[i, j, k + 1], detH.data[i, j + 1, k - 1], detH.data[i, j + 1, k],
                    detH.data[i, j + 1, k + 1], detH.data[i + 1, j - 1, k - 1], detH.data[i + 1, j - 1, k],
                    detH.data[i + 1, j - 1, k + 1], detH.data[i + 1, j, k - 1], detH.data[i + 1, j, k],
                    detH.data[i + 1, j, k + 1], detH.data[i + 1, j + 1, k - 1], detH.data[i + 1, j + 1, k],
                    detH.data[i + 1, j + 1, k + 1]
                ]

                for neighbor in neighbors_equal:
                    if current_detH < neighbor:
                        is_maximum = False
                        break

                if not is_maximum:
                    continue

                # Found a valid blob!
                # Store particle information
                particle_info = {
                    'x': j, 'y': i, 'scale': k,
                    'detH': current_detH,
                    'LG': current_LG,
                    'x_coord': DimOffset(im, 0) + j * DimDelta(im, 0),
                    'y_coord': DimOffset(im, 1) + i * DimDelta(im, 1),
                    'scale_coord': DimOffset(detH, 2) + k * DimDelta(detH, 2)
                }
                particle_list.append(particle_info)

                # Update maps
                mapNum.data[i, j] = particle_count
                mapLG.data[i, j] = current_LG
                mapMax.data[i, j] = max(mapMax.data[i, j], current_detH)

                particle_count += 1

    # Update info wave with particle data
    if particle_count > 0:
        info.data = np.zeros((particle_count, 10))  # 10 columns for particle properties
        for idx, particle in enumerate(particle_list):
            info.data[idx, 0] = particle['x_coord']  # X position
            info.data[idx, 1] = particle['y_coord']  # Y position
            info.data[idx, 2] = particle['scale_coord']  # Scale
            info.data[idx, 3] = particle['detH']  # DetH response
            info.data[idx, 4] = particle['LG']  # LG response
            info.data[idx, 5] = particle['x']  # X index
            info.data[idx, 6] = particle['y']  # Y index
            info.data[idx, 7] = particle['scale']  # Scale index
            info.data[idx, 8] = 0  # Area (to be computed)
            info.data[idx, 9] = 0  # Volume (to be computed)

    print(f"Found {particle_count} Hessian blobs")
    return particle_count


def RunHessianBlobs(im, scaleStart=1.0, scaleLayers=20, scaleFactor=1.2,
                    particleType=1, maxCurvatureRatio=10.0, minResponse=-1.0,
                    interactive=True):
    """
    Main function to run complete Hessian blob analysis
    Direct port from Igor Pro RunHessianBlobs function

    Parameters:
    im : Wave - Input image
    scaleStart : float - Starting scale in pixels
    scaleLayers : int - Number of scale layers
    scaleFactor : float - Factor between scale layers
    particleType : int - Type of particles to detect
    maxCurvatureRatio : float - Maximum curvature ratio
    minResponse : float - Minimum response (-1 for interactive)
    interactive : bool - Use interactive threshold selection

    Returns:
    dict - Results containing all output waves and statistics
    """
    print("=== Running Hessian Blob Analysis ===")
    print(f"Image: {im.name}, Shape: {im.data.shape}")
    print(f"Parameters: scaleStart={scaleStart}, layers={scaleLayers}, factor={scaleFactor}")
    print(f"Particle type={particleType}, maxCurvRatio={maxCurvatureRatio}")

    # Step 1: Compute scale-space representation
    print("\n1. Computing scale-space representation...")
    L = ScaleSpaceRepresentation(im, scaleLayers, scaleStart, scaleFactor)

    # Step 2: Compute blob detectors
    print("\n2. Computing blob detectors...")
    detH, LG = BlobDetectors(L, gammaNorm=1.0)

    # Step 3: Get threshold (interactive or provided)
    if interactive or minResponse < 0:
        print("\n3. Interactive threshold selection...")
        threshold = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
        if threshold <= 0:
            print("Analysis cancelled by user.")
            return None
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
            'scaleLayers': scaleLayers,
            'scaleFactor': scaleFactor,
            'particleType': particleType,
            'maxCurvatureRatio': maxCurvatureRatio,
            'minResponse': minResponse
        }
    }

    print(f"\n=== Analysis Complete: Found {num_particles} particles ===")
    return results


def Testing(string_input, number_input):
    """Testing function for main_functions module"""
    print(f"Main functions testing: {string_input}, {number_input}")
    return len(string_input) + number_input