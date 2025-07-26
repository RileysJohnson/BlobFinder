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

    ttk.Label(scale_frame, text="Scale Start (pixels):").grid(row=0, column=0, sticky="w", padx=5, pady=3)
    scale_start_var = tk.StringVar(value="1")
    ttk.Entry(scale_frame, textvariable=scale_start_var, width=15).grid(row=0, column=1, padx=5, pady=3)

    ttk.Label(scale_frame, text="Layers:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
    layers_var = tk.StringVar(value="256")
    ttk.Entry(scale_frame, textvariable=layers_var, width=15).grid(row=1, column=1, padx=5, pady=3)

    ttk.Label(scale_frame, text="Scale Factor:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
    scale_factor_var = tk.StringVar(value="1.5")
    ttk.Entry(scale_frame, textvariable=scale_factor_var, width=15).grid(row=2, column=1, padx=5, pady=3)

    # Detection parameters
    detect_frame = ttk.LabelFrame(main_frame, text="Detection Parameters", padding="10")
    detect_frame.pack(fill=tk.X, pady=5)

    ttk.Label(detect_frame, text="Blob Strength:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
    thresh_var = tk.StringVar(value="-2")
    ttk.Entry(detect_frame, textvariable=thresh_var, width=15).grid(row=0, column=1, padx=5, pady=3)
    ttk.Label(detect_frame, text="(-2 for interactive, -1 for Otsu)",
              font=('TkDefaultFont', 8)).grid(row=0, column=2, padx=5)

    ttk.Label(detect_frame, text="Particle Type:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
    particle_var = tk.StringVar(value="1")
    particle_combo = ttk.Combobox(detect_frame, textvariable=particle_var, width=12)
    particle_combo['values'] = ('1', '-1', '0')
    particle_combo.grid(row=1, column=1, padx=5, pady=3)
    ttk.Label(detect_frame, text="(1=positive, -1=negative, 0=both)",
              font=('TkDefaultFont', 8)).grid(row=1, column=2, padx=5)

    ttk.Label(detect_frame, text="Max Curvature Ratio:").grid(row=2, column=0, sticky="w", padx=5, pady=3)
    curvature_var = tk.StringVar(value="4")
    ttk.Entry(detect_frame, textvariable=curvature_var, width=15).grid(row=2, column=1, padx=5, pady=3)

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    def ok_clicked():
        try:
            result[0] = {
                'scale_start': float(scale_start_var.get()),
                'layers': int(layers_var.get()),
                'scale_factor': float(scale_factor_var.get()),
                'threshold': float(thresh_var.get()),
                'particle_type': int(particle_var.get()),
                'max_curvature_ratio': float(curvature_var.get())
            }
            dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    ttk.Button(button_frame, text="Continue", command=ok_clicked, width=15).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked, width=15).pack(side=tk.LEFT, padx=10)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")

    # Wait for dialog to complete
    dialog.mainloop()

    # Clean up
    try:
        dialog.destroy()
        root.destroy()
    except:
        pass

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
        return None

    # Create the interactive threshold window
    # This replicates: NewImage/N=IMAGE im
    threshold_window = tk.Toplevel()
    threshold_window.title("Interactive Blob Strength Threshold")
    threshold_window.geometry("1000x700")
    threshold_window.transient()
    threshold_window.grab_set()

    # Global variable for threshold (like Igor's Variable/G SS_THRESH)
    SS_THRESH = max_value / 2
    threshold_result = [SS_THRESH]  # Use list to allow modification in nested functions
    window_closed = [False]

    # Create matplotlib figure for image display
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Display the original image
    im_display = ax.imshow(im.data, cmap='gray', origin='lower',
                           extent=[DimOffset(im, 0),
                                   DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
                                   DimOffset(im, 1),
                                   DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)])
    ax.set_title("Original Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Store circle patches for updating
    circle_patches = []

    def update_threshold_display(thresh_val):
        """Update the display with circles for blobs above threshold"""
        # Clear existing circles (like Igor's SetDrawLayer/K /W=IMAGE overlay)
        for patch in circle_patches:
            patch.remove()
        circle_patches.clear()

        # Draw circles for blobs above threshold
        # This replicates the Igor Pro InteractiveSlider function logic
        thresh_squared = thresh_val ** 2

        for i in range(SS_MAXMAP.data.shape[0]):
            for j in range(SS_MAXMAP.data.shape[1]):
                # If(Map[i][j]>S_STruct.curval^2)
                if SS_MAXMAP.data[i, j] > thresh_squared:
                    # Calculate radius from scale (matches Igor Pro logic)
                    scale_idx = int(SS_MAXSCALEMAP.data[i, j])
                    if 0 <= scale_idx < detH.data.shape[2]:
                        # rad = Sqrt(2*Exp(DimOffset(detH,2)+ScaleMap[i][j]*DimDelta(detH,2)))
                        scale_value = np.exp(DimOffset(detH, 2) + scale_idx * DimDelta(detH, 2))
                        rad = np.sqrt(2 * scale_value)

                        # Convert to real coordinates
                        # IMPORTANT: Igor Pro uses (i,j) as (row,col) but displays as (x,y)
                        # xc = DimOffset(map,0)+i*DimDelta(map,0)
                        # yc = DimOffset(map,1)+j*DimDelta(map,1)
                        xc = DimOffset(SS_MAXMAP, 0) + j * DimDelta(SS_MAXMAP, 0)  # j for x
                        yc = DimOffset(SS_MAXMAP, 1) + i * DimDelta(SS_MAXMAP, 1)  # i for y

                        # SetDrawEnv xcoord= bottom,ycoord= left,linefgc= (65535,0,0)
                        # DrawOval xc-rad,yc-rad,xc+rad,yc+rad
                        circle = Circle((xc, yc), rad, fill=False, color='red', linewidth=2)
                        ax.add_patch(circle)
                        circle_patches.append(circle)

        # Update title with current threshold
        ax.set_title(f"Blobs Above Threshold (Strength: {thresh_val:.6f})")
        fig.canvas.draw()

    # Create tkinter frame for matplotlib
    canvas = FigureCanvasTkAgg(fig, threshold_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Control panel (replicates Igor's NewPanel/EXT=0 /HOST=IMAGE /N=SubControl)
    control_frame = tk.Frame(threshold_window, bg='lightgray')
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Accept and Quit buttons (matches Igor Pro layout)
    def InteractiveContinue():
        """Replicates Igor Pro InteractiveContinue function"""
        window_closed[0] = True
        threshold_window.destroy()

    def InteractiveQuit():
        """Replicates Igor Pro InteractiveQuit function"""
        threshold_result[0] = None
        window_closed[0] = True
        threshold_window.destroy()

    # Button layout matches Igor Pro: Accept at (0,0), Quit at (100,0)
    btn_frame = tk.Frame(control_frame)
    btn_frame.pack(side=tk.LEFT)

    accept_btn = tk.Button(btn_frame, text="Accept", command=InteractiveContinue,
                           width=12, height=2, bg='lightblue')
    accept_btn.pack(side=tk.LEFT, padx=2)

    quit_btn = tk.Button(btn_frame, text="Quit", command=InteractiveQuit,
                         width=12, height=2, bg='lightcoral')
    quit_btn.pack(side=tk.LEFT, padx=2)

    # Threshold display and controls
    thresh_frame = tk.Frame(control_frame)
    thresh_frame.pack(side=tk.LEFT, padx=20)

    tk.Label(thresh_frame, text="Blob Strength:", font=('TkDefaultFont', 10, 'bold')).pack()

    # SetVariable ThreshSetVar (matches Igor Pro)
    thresh_var = tk.StringVar(value=f"{SS_THRESH:.6e}")
    thresh_entry = tk.Entry(thresh_frame, textvariable=thresh_var, width=15)
    thresh_entry.pack(pady=2)

    def update_from_entry():
        """Update threshold from text entry"""
        try:
            new_thresh = float(thresh_var.get())
            if 0 <= new_thresh <= max_value * 1.1:
                threshold_result[0] = new_thresh
                slider.set(new_thresh)
                update_threshold_display(new_thresh)
        except ValueError:
            thresh_var.set(f"{threshold_result[0]:.6e}")

    thresh_entry.bind('<Return>', lambda e: update_from_entry())

    # Slider ThreshSlide (matches Igor Pro slider configuration)
    slider_frame = tk.Frame(control_frame)
    slider_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

    tk.Label(slider_frame, text="Threshold Slider:", font=('TkDefaultFont', 10, 'bold')).pack()

    # Slider limits={0,WaveMax(Maxes)*1.1,WaveMax(Maxes)*1.1/200}
    slider = tk.Scale(slider_frame, from_=0, to=max_value * 1.1,
                      resolution=max_value * 1.1 / 200,
                      orient=tk.HORIZONTAL, length=300,
                      command=lambda val: InteractiveSlider(float(val)))
    slider.set(SS_THRESH)
    slider.pack(pady=5)

    def InteractiveSlider(val):
        """
        Replicates Igor Pro InteractiveSlider function
        Updates display when slider moves
        """
        threshold_result[0] = val
        thresh_var.set(f"{val:.6e}")
        update_threshold_display(val)

    # Initial display update
    update_threshold_display(SS_THRESH)

    # Let the user pick the appropriate threshold (replicates PauseForUser IMAGE)
    threshold_window.wait_window()

    # Return the selected threshold value
    returnVal = threshold_result[0] if not window_closed[0] or threshold_result[0] is not None else None

    # Clean up (replicates KillVariables/Z SS_THRESH; KillWaves/Z Map)
    try:
        plt.close(fig)
    except:
        pass

    print(f"Interactive threshold selection completed. Selected threshold: {returnVal}")
    return returnVal


def BatchHessianBlobs(images_folder=None):
    """
    Detects Hessian blobs in a series of images in a chosen data folder.
    Direct port from Igor Pro BatchHessianBlobs function
    """
    if images_folder is None:
        ImagesDF = GetBrowserSelection(0)
        CurrentDF = GetDataFolder(1)

        if not DataFolderExists(ImagesDF) or CountObjects(ImagesDF, 1) < 1:
            messagebox.showerror("Error",
                                 "Select the folder with your images in it in the data browser, then try again.")
            return ""
    else:
        ImagesDF = images_folder
        CurrentDF = "root:"

    # Get parameters from user
    params = GetBlobDetectionParams()
    if params is None:
        return ""  # User cancelled

    scaleStart = params['scale_start']  # In pixel units
    layers = params['layers']
    scaleFactor = params['scale_factor']
    detHResponseThresh = params['threshold']  # Use -1 for Otsu's method, -2 for interactive
    particleType = params['particle_type']  # -1 for neg only, 1 for pos only, 0 for both
    maxCurvatureRatio = params['max_curvature_ratio']

    # Additional parameters (fixed for now)
    subPixelMult = 1  # 1 or more, should be integer
    allowOverlap = 0

    # Get constraints if desired
    minH = -np.inf
    maxH = np.inf
    minV = -np.inf
    maxV = np.inf
    minA = -np.inf
    maxA = np.inf

    response = messagebox.askyesnocancel("Constraints",
                                         "Would you like to limit the analysis to particles of certain height, volume, or area?")

    if response is None:  # Cancel
        return ""
    elif response:  # Yes
        constraints = GetConstraints()
        if constraints:
            minH, maxH, minV, maxV, minA, maxA = constraints
        else:
            return ""  # User cancelled

    # Process images
    print("Starting batch Hessian blob detection...")

    # Create results folder
    results_folder = NewDataFolder("HessianBlobResults")

    # Get list of images to process
    images = []
    if hasattr(images_folder, 'waves'):
        images = list(images_folder.waves.values())
    else:
        # Get from data browser
        folder = data_browser.get_folder(ImagesDF.rstrip(':'))
        images = list(folder.waves.values())

    num_images = len(images)
    print(f"Processing {num_images} images...")

    # Process each image
    for idx, im in enumerate(images):
        print(f"\nProcessing image {idx + 1}/{num_images}: {im.name}")

        try:
            # Create scale-space representation
            t0 = scaleStart
            L = ScaleSpaceRepresentation(im, layers, t0, scaleFactor)

            # Compute blob detectors
            detH, LG = BlobDetectors(L, True)

            # Get threshold
            if detHResponseThresh == -2:
                # Interactive threshold
                threshold = InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio)
                if threshold is None:
                    print(f"Skipping {im.name} - threshold selection cancelled")
                    continue
            elif detHResponseThresh == -1:
                # Otsu's method
                threshold = OtsuThreshold(detH, particleType)
            else:
                threshold = detHResponseThresh

            # Find blobs
            mapNum = Duplicate(im, f"{im.name}_mapNum")
            mapNum.data = np.full(im.data.shape, -1)

            mapDetH = Duplicate(im, f"{im.name}_mapDetH")
            mapDetH.data = np.zeros(im.data.shape)

            mapMax = Duplicate(im, f"{im.name}_mapMax")
            mapMax.data = np.zeros(im.data.shape)

            info = Wave(np.zeros((0, 20)), f"{im.name}_info")

            numParticles = FindHessianBlobs(im, detH, LG, threshold, mapNum, mapDetH,
                                            mapMax, info, particleType, maxCurvatureRatio)

            print(f"Found {numParticles} particles in {im.name}")

            # Apply constraints if any
            if minH != -np.inf or maxH != np.inf or minV != -np.inf or maxV != np.inf or minA != -np.inf or maxA != np.inf:
                # Filter particles based on constraints
                # This would be implemented in particle filtering function
                pass

            # Store results
            results_folder.add_wave(mapNum)
            results_folder.add_wave(mapDetH)
            results_folder.add_wave(mapMax)
            results_folder.add_wave(info)

        except Exception as e:
            print(f"Error processing {im.name}: {str(e)}")
            continue

    print("\nBatch processing completed!")
    return results_folder.name


def GetConstraints():
    """
    Get particle constraints from user
    Replicates Igor Pro constraint dialog
    """
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel()
    dialog.title("Particle Constraints")
    dialog.geometry("450x300")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Particle Constraints",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Create constraint entry fields
    constraints_frame = ttk.Frame(main_frame)
    constraints_frame.pack(fill=tk.X)

    # Headers
    ttk.Label(constraints_frame, text="Parameter", font=('TkDefaultFont', 10, 'bold')).grid(
        row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Label(constraints_frame, text="Min", font=('TkDefaultFont', 10, 'bold')).grid(
        row=0, column=1, padx=5, pady=5)
    ttk.Label(constraints_frame, text="Max", font=('TkDefaultFont', 10, 'bold')).grid(
        row=0, column=2, padx=5, pady=5)

    row = 1

    # Height constraints
    ttk.Label(constraints_frame, text="Height:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    min_h_var = tk.StringVar(value="-inf")
    max_h_var = tk.StringVar(value="inf")
    ttk.Entry(constraints_frame, textvariable=min_h_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Entry(constraints_frame, textvariable=max_h_var, width=12).grid(row=row, column=2, padx=5, pady=3)
    row += 1

    # Volume constraints
    ttk.Label(constraints_frame, text="Volume:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    min_v_var = tk.StringVar(value="-inf")
    max_v_var = tk.StringVar(value="inf")
    ttk.Entry(constraints_frame, textvariable=min_v_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Entry(constraints_frame, textvariable=max_v_var, width=12).grid(row=row, column=2, padx=5, pady=3)
    row += 1

    # Area constraints
    ttk.Label(constraints_frame, text="Area:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    min_a_var = tk.StringVar(value="-inf")
    max_a_var = tk.StringVar(value="inf")
    ttk.Entry(constraints_frame, textvariable=min_a_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Entry(constraints_frame, textvariable=max_a_var, width=12).grid(row=row, column=2, padx=5, pady=3)

    def parse_constraint(value):
        """Parse constraint value, handling 'inf' and '-inf'"""
        val = value.strip().lower()
        if val == 'inf':
            return np.inf
        elif val == '-inf':
            return -np.inf
        else:
            return float(val)

    def ok_clicked():
        try:
            result[0] = (
                parse_constraint(min_h_var.get()),
                parse_constraint(max_h_var.get()),
                parse_constraint(min_v_var.get()),
                parse_constraint(max_v_var.get()),
                parse_constraint(min_a_var.get()),
                parse_constraint(max_a_var.get())
            )
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid constraint values: {str(e)}")

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="Continue", command=ok_clicked, width=15).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked, width=15).pack(side=tk.LEFT, padx=10)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")

    # Wait for dialog to complete
    dialog.mainloop()

    # Clean up
    try:
        dialog.destroy()
        root.destroy()
    except:
        pass

    return result[0]


def OtsuThreshold(detH, particleType):
    """
    Calculate Otsu's threshold for blob detection
    Direct port from Igor Pro implementation

    Parameters:
    detH : Wave - Determinant of Hessian detector response
    particleType : int - Type of particles to consider

    Returns:
    float - Otsu threshold in image units
    """
    # Flatten the 3D detector response
    data = detH.data.flatten()

    # Filter based on particle type
    if particleType == 1:
        data = data[data > 0]
    elif particleType == -1:
        data = -data[data < 0]

    if len(data) == 0:
        return 0

    # Calculate histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate Otsu's threshold
    total = np.sum(hist)
    sum_total = np.sum(hist * bin_centers)

    sum_bg = 0
    weight_bg = 0
    max_variance = 0
    threshold = 0

    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        # Calculate between-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    # Convert to image units (square root)
    return np.sqrt(threshold)


def Testing(input_string, input_number):
    """
    Testing function - replicates Igor Pro Testing function
    """
    print(f"Testing function called with string: '{input_string}' and number: {input_number}")

    # Create a simple test result
    result = len(input_string) + input_number

    messagebox.showinfo("Testing",
                        f"Function executed successfully!\n"
                        f"String: {input_string}\n"
                        f"Number: {input_number}\n"
                        f"Result: {result}")

    return result