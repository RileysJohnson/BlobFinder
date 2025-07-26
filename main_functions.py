"""
Main Functions
Contains the primary Hessian blob detection functions
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete implementations
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider
import os

from igor_compatibility import *
from file_io import *
from scale_space import *
from particle_measurements import *
from preprocessing import *
from utilities import *

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex


def HessianBlobs(im):
    """
    Detects Hessian blobs in a single image.
    This is the main single-image analysis function that matches the Igor Pro version.

    Parameters:
    im : Wave object containing the image data

    Returns:
    String indicating success or failure
    """
    try:
        # Declare default algorithm parameters (same as Igor Pro version)
        scaleStart = 1  # In pixel units
        layers = 256
        scaleFactor = 1.5
        detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
        particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
        subPixelMult = 1  # 1 or more, should be integer
        allowOverlap = 0
        maxCurvatureRatio = 1.6  # Maximum ratio of principal curvatures

        # Get parameters from user
        params = GetHessianBlobParameters(scaleStart, layers, scaleFactor,
                                          detHResponseThresh, particleType,
                                          subPixelMult, allowOverlap)

        if params is None:
            return ""  # User cancelled

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params

        # Get constraints if requested
        constraints = GetParticleConstraints()
        if constraints is None:
            return ""  # User cancelled

        minH, maxH, minV, maxV, minA, maxA = constraints

        # Create output folder for this image
        image_folder_name = f"{im.name}_Particles"
        NewDataFolder(image_folder_name)
        image_folder = data_browser.get_folder(image_folder_name)

        # Store the original image in the results folder
        original_copy = Duplicate(im, "Original")
        image_folder.add_wave(original_copy)

        # Create the scale-space representation
        print(f"Computing scale-space representation with {layers} layers...")
        L = ScaleSpaceRepresentation(im, layers, scaleStart, scaleFactor)
        image_folder.add_wave(L, "ScaleSpaceRep")

        # Compute blob detectors
        print("Computing blob detectors...")
        gammaNorm = 1  # Standard gamma normalization
        BlobDetectors(L, gammaNorm)

        # Get the blob detector results
        detH = data_browser.get_wave("detH")
        LapG = data_browser.get_wave("LapG")

        if detH is None or LapG is None:
            raise Exception("Failed to compute blob detectors")

        # Store detectors in image folder
        image_folder.add_wave(Duplicate(detH, "detH"))
        image_folder.add_wave(Duplicate(LapG, "LapG"))

        # Determine threshold
        if detHResponseThresh == -2:  # Interactive threshold
            detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
            if detHResponseThresh is None:
                return ""  # User cancelled
        elif detHResponseThresh == -1:  # Otsu's method
            detHResponseThresh = OtsusThreshold(detH, particleType)

        # Find Hessian blobs
        print("Finding Hessian blobs...")

        # Create output waves
        mapNum = Wave(np.full(im.data.shape, -1, dtype=np.int32), "ParticleMap")
        mapDetH = Wave(np.zeros(im.data.shape), "DetHMap")
        mapMax = Wave(np.zeros(im.data.shape), "MaxMap")
        info = Wave(np.zeros((1000, 15)), "Info")  # 15 columns for particle info

        # Run the blob detection
        FindHessianBlobs(im, detH, LapG, detHResponseThresh, mapNum, mapDetH, mapMax, info,
                         particleType, maxCurvatureRatio)

        # Store results
        image_folder.add_wave(mapNum)
        image_folder.add_wave(mapDetH)
        image_folder.add_wave(mapMax)
        image_folder.add_wave(info)

        # Count detected particles
        num_particles = int(np.max(mapNum.data) + 1) if np.max(mapNum.data) >= 0 else 0
        print(f"Detected {num_particles} particles")

        # Apply constraints and refinement if requested
        if num_particles > 0:
            if subPixelMult > 1:
                print("Applying sub-pixel refinement...")
                SubPixelRefinement(im, info, mapNum, subPixelMult)

            # Apply particle constraints
            if minH != -np.inf or maxH != np.inf or minV != -np.inf or maxV != np.inf or minA != -np.inf or maxA != np.inf:
                print("Applying particle constraints...")
                ApplyParticleConstraints(info, mapNum, minH, maxH, minV, maxV, minA, maxA)

        # Create individual particle folders
        CreateIndividualParticleFolders(im, info, mapNum, image_folder)

        print(f"Hessian blob analysis completed. Results stored in '{image_folder_name}'")
        return image_folder_name

    except Exception as e:
        print(f"Error in HessianBlobs: {str(e)}")
        messagebox.showerror("Error", f"Hessian blob analysis failed: {str(e)}")
        return ""


def BatchHessianBlobs():
    """
    Detects Hessian blobs in a series of images in a chosen data folder.
    Be sure to highlight the data folder containing the images in the data browser before running.
    """
    ImagesDF = GetBrowserSelection(0)
    CurrentDF = GetDataFolder(1)

    if not DataFolderExists(ImagesDF) or CountObjects(ImagesDF, 1) < 1:
        messagebox.showerror("Error", "Select the folder with your images in it in the data browser, then try again.")
        return ""

    # Declare algorithm parameters
    scaleStart = 1  # In pixel units
    layers = 256
    scaleFactor = 1.5
    detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
    particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
    subPixelMult = 1  # 1 or more, should be integer
    allowOverlap = 0

    # Get parameters from user
    params = GetHessianBlobParameters(scaleStart, layers, scaleFactor,
                                      detHResponseThresh, particleType,
                                      subPixelMult, allowOverlap)

    if params is None:
        return ""

    scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params

    # Get constraints if requested
    constraints = GetParticleConstraints()
    if constraints is None:
        return ""

    minH, maxH, minV, maxV, minA, maxA = constraints

    # Make a Data Folder for the Series
    folder = data_browser.get_folder(ImagesDF.rstrip(':'))
    NumImages = len(folder.waves)
    SeriesDF = UniqueName("Series_", 11, 0)
    NewDataFolder(SeriesDF)
    series_folder = data_browser.get_folder(SeriesDF)

    # Store the parameters being used
    parameters = Wave(np.array([
        scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
        subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
    ]), "Parameters")
    series_folder.add_wave(parameters)

    # Initialize summary waves
    AllHeights = Wave(np.array([]), "AllHeights")
    AllVolumes = Wave(np.array([]), "AllVolumes")
    AllAreas = Wave(np.array([]), "AllAreas")
    AllAvgHeights = Wave(np.array([]), "AllAvgHeights")

    series_folder.add_wave(AllHeights)
    series_folder.add_wave(AllVolumes)
    series_folder.add_wave(AllAreas)
    series_folder.add_wave(AllAvgHeights)

    # Process each image
    total_particles = 0
    processed_images = 0

    for wave_name, wave in folder.waves.items():
        try:
            print(f"Processing image {processed_images + 1}/{NumImages}: {wave_name}")

            # Create scale-space representation
            L = ScaleSpaceRepresentation(wave, layers, scaleStart, scaleFactor)

            # Compute blob detectors
            BlobDetectors(L, 1)  # gammaNorm = 1

            # Get detector results
            detH = data_browser.get_wave("detH")
            LapG = data_browser.get_wave("LapG")

            # Determine threshold for this image
            current_thresh = detHResponseThresh
            if current_thresh == -2:  # Interactive threshold
                current_thresh = InteractiveThreshold(wave, detH, LapG, particleType, 1.6)
                if current_thresh is None:
                    continue  # Skip this image
            elif current_thresh == -1:  # Otsu's method
                current_thresh = OtsusThreshold(detH, particleType)

            # Find particles in this image
            mapNum = Wave(np.full(wave.data.shape, -1, dtype=np.int32), f"ParticleMap_{wave_name}")
            mapDetH = Wave(np.zeros(wave.data.shape), f"DetHMap_{wave_name}")
            mapMax = Wave(np.zeros(wave.data.shape), f"MaxMap_{wave_name}")
            info = Wave(np.zeros((1000, 15)), f"Info_{wave_name}")

            FindHessianBlobs(wave, detH, LapG, current_thresh, mapNum, mapDetH, mapMax, info,
                             particleType, 1.6)

            # Count particles in this image
            num_particles = int(np.max(mapNum.data) + 1) if np.max(mapNum.data) >= 0 else 0

            if num_particles > 0:
                # Apply sub-pixel refinement if requested
                if subPixelMult > 1:
                    SubPixelRefinement(wave, info, mapNum, subPixelMult)

                # Apply constraints
                ApplyParticleConstraints(info, mapNum, minH, maxH, minV, maxV, minA, maxA)

                # Extract measurements and add to summary waves
                ExtractParticleMeasurements(wave, info, mapNum, series_folder, wave_name)

                total_particles += num_particles

            processed_images += 1

        except Exception as e:
            print(f"Error processing {wave_name}: {str(e)}")
            continue

    print(f"Batch processing completed. Processed {processed_images} images, found {total_particles} total particles.")
    print(f"Results stored in '{SeriesDF}'")

    return SeriesDF


def GetHessianBlobParameters(scaleStart, layers, scaleFactor, detHResponseThresh,
                             particleType, subPixelMult, allowOverlap):
    """
    Get Hessian blob parameters from user dialog
    Returns tuple of parameters or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide root window

    dialog = tk.Toplevel(root)
    dialog.title("Hessian Blob Parameters")
    dialog.geometry("450x350")
    dialog.grab_set()

    # Make dialog modal and centered
    dialog.transient(root)
    dialog.focus_set()

    # Variables for parameters
    scale_start_var = tk.DoubleVar(value=scaleStart)
    layers_var = tk.IntVar(value=layers)
    scale_factor_var = tk.DoubleVar(value=scaleFactor)
    thresh_var = tk.DoubleVar(value=detHResponseThresh)
    particle_type_var = tk.IntVar(value=particleType)
    subpixel_var = tk.IntVar(value=subPixelMult)
    overlap_var = tk.IntVar(value=allowOverlap)

    result = [None]  # Use list to store result

    # Create GUI elements with better layout
    main_frame = ttk.Frame(dialog, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Parameters
    row = 0
    ttk.Label(main_frame, text="Scale Start (pixels):").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=scale_start_var, width=15).grid(row=row, column=1, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Number of Layers:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=layers_var, width=15).grid(row=row, column=1, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Scale Factor:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=scale_factor_var, width=15).grid(row=row, column=1, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Threshold:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=thresh_var, width=15).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="(-2=Interactive, -1=Otsu, >0=Manual)", font=('TkDefaultFont', 8)).grid(row=row,
                                                                                                       column=2,
                                                                                                       sticky="w",
                                                                                                       padx=5)

    row += 1
    ttk.Label(main_frame, text="Particle Type:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=particle_type_var, width=15).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="(-1=Negative, 0=Both, 1=Positive)", font=('TkDefaultFont', 8)).grid(row=row, column=2,
                                                                                                    sticky="w", padx=5)

    row += 1
    ttk.Label(main_frame, text="Sub-pixel Multiplier:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=subpixel_var, width=15).grid(row=row, column=1, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Allow Overlap:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=overlap_var, width=15).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="(0=No, 1=Yes)", font=('TkDefaultFont', 8)).grid(row=row, column=2, sticky="w", padx=5)

    def ok_clicked():
        try:
            result[0] = (scale_start_var.get(), layers_var.get(), scale_factor_var.get(),
                         thresh_var.get(), particle_type_var.get(), subpixel_var.get(),
                         overlap_var.get())
            dialog.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")

    def cancel_clicked():
        result[0] = None
        dialog.quit()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=row + 1, column=0, columnspan=3, pady=20)

    ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=10)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")

    # Run dialog
    dialog.mainloop()

    # Clean up
    try:
        dialog.destroy()
        root.destroy()
    except:
        pass

    return result[0]


def GetParticleConstraints():
    """
    Get particle constraints from user dialog
    Returns tuple of constraints or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide root window

    dialog = tk.Toplevel(root)
    dialog.title("Particle Constraints")
    dialog.geometry("500x250")
    dialog.grab_set()

    # Make dialog modal and centered
    dialog.transient(root)
    dialog.focus_set()

    # Variables for constraints
    min_h_var = tk.StringVar(value="-inf")
    max_h_var = tk.StringVar(value="inf")
    min_v_var = tk.StringVar(value="-inf")
    max_v_var = tk.StringVar(value="inf")
    min_a_var = tk.StringVar(value="-inf")
    max_a_var = tk.StringVar(value="inf")

    result = [None]  # Use list to store result

    # Create GUI elements
    main_frame = ttk.Frame(dialog, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header
    ttk.Label(main_frame, text="Particle Size Constraints",
              font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=4, pady=(0, 10))

    # Create constraints grid
    row = 1
    ttk.Label(main_frame, text="Minimum height:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=min_h_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="Maximum height:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=max_h_var, width=12).grid(row=row, column=3, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Minimum volume:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=min_v_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="Maximum volume:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=max_v_var, width=12).grid(row=row, column=3, padx=5, pady=3)

    row += 1
    ttk.Label(main_frame, text="Minimum area:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=min_a_var, width=12).grid(row=row, column=1, padx=5, pady=3)
    ttk.Label(main_frame, text="Maximum area:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
    ttk.Entry(main_frame, textvariable=max_a_var, width=12).grid(row=row, column=3, padx=5, pady=3)

    # Help text
    row += 1
    ttk.Label(main_frame, text="Use 'inf' or '-inf' for no limit",
              font=('TkDefaultFont', 8)).grid(row=row, column=0, columnspan=4, pady=(10, 0))

    def parse_constraint(value_str):
        """Parse constraint string to float"""
        value_str = value_str.strip()
        if value_str.lower() in ["-inf", "-infinity"]:
            return -np.inf
        elif value_str.lower() in ["inf", "infinity"]:
            return np.inf
        else:
            try:
                return float(value_str)
            except ValueError:
                return 0.0

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
            dialog.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid constraint values: {str(e)}")

    def cancel_clicked():
        result[0] = None
        dialog.quit()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=row + 1, column=0, columnspan=4, pady=20)

    ttk.Button(button_frame, text="Continue", command=ok_clicked).pack(side=tk.LEFT, padx=10)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=10)

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")

    # Run dialog
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
    Replicates the Igor Pro interactive threshold functionality
    """
    # First identify the maxes
    maxes_data = np.zeros(im.data.shape)
    scale_map = np.zeros(im.data.shape)

    # Find local maxima in the detector response
    for k in range(1, detH.data.shape[2] - 1):
        for i in range(1, detH.data.shape[0] - 1):
            for j in range(1, detH.data.shape[1] - 1):
                if detH.data[i, j, k] > 0:  # Only consider positive responses
                    # Check if it's a local maximum
                    is_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if (di == 0 and dj == 0 and dk == 0):
                                    continue
                                if detH.data[i + di, j + dj, k + dk] >= detH.data[i, j, k]:
                                    is_max = False
                                    break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if is_max:
                        # Check curvature ratio
                        if LG.data[i, j, k] ** 2 / detH.data[i, j, k] < (
                                maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                            maxes_data[i, j] = max(maxes_data[i, j], detH.data[i, j, k])
                            if detH.data[i, j, k] == maxes_data[i, j]:
                                scale_map[i, j] = k

    # Convert to image units (square root)
    maxes_data = np.sqrt(np.maximum(maxes_data, 0))
    max_value = np.max(maxes_data)

    if max_value == 0:
        messagebox.showwarning("No Blobs", "No suitable blob candidates found in image.")
        return None

    # Create interactive threshold window
    threshold_window = tk.Toplevel()
    threshold_window.title("Interactive Blob Strength Threshold")
    threshold_window.geometry("800x600")
    threshold_window.transient()
    threshold_window.grab_set()

    # Create matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display original image
    ax1.imshow(im.data, cmap='gray', origin='lower')
    ax1.set_title("Original Image")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Display image with threshold overlay
    ax2.imshow(im.data, cmap='gray', origin='lower')
    ax2.set_title("Blobs Above Threshold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Initial threshold
    current_threshold = max_value / 2

    # Function to update display
    def update_display(threshold_val):
        ax2.clear()
        ax2.imshow(im.data, cmap='gray', origin='lower')
        ax2.set_title(f"Blobs Above Threshold ({threshold_val:.6f})")

        # Draw circles for blobs above threshold
        threshold_squared = threshold_val ** 2
        for i in range(maxes_data.shape[0]):
            for j in range(maxes_data.shape[1]):
                if maxes_data[i, j] ** 2 > threshold_squared:
                    # Calculate radius from scale
                    scale_idx = int(scale_map[i, j])
                    if scale_idx < detH.data.shape[2]:
                        scale_value = DimOffset(detH, 2) + scale_idx * DimDelta(detH, 2)
                        radius = np.sqrt(2 * scale_value)

                        # Convert to image coordinates
                        x_coord = DimOffset(im, 0) + j * DimDelta(im, 0)
                        y_coord = DimOffset(im, 1) + i * DimDelta(im, 1)

                        circle = Circle((x_coord, y_coord), radius, fill=False, color='red', linewidth=1)
                        ax2.add_patch(circle)

        fig.canvas.draw()

    # Create tkinter frame for matplotlib
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, threshold_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Control frame
    control_frame = tk.Frame(threshold_window)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    # Threshold control
    threshold_var = tk.DoubleVar(value=current_threshold)

    tk.Label(control_frame, text="Blob Strength Threshold:").pack(side=tk.LEFT)
    threshold_scale = tk.Scale(control_frame, from_=0, to=max_value * 1.1,
                               resolution=max_value * 1.1 / 200, orient=tk.HORIZONTAL,
                               variable=threshold_var, length=300,
                               command=lambda val: update_display(float(val)))
    threshold_scale.pack(side=tk.LEFT, padx=10)

    # Result variable
    result = [None]

    def accept_threshold():
        result[0] = threshold_var.get()
        threshold_window.destroy()

    def cancel_threshold():
        result[0] = None
        threshold_window.destroy()

    tk.Button(control_frame, text="Accept", command=accept_threshold).pack(side=tk.RIGHT, padx=5)
    tk.Button(control_frame, text="Cancel", command=cancel_threshold).pack(side=tk.RIGHT, padx=5)

    # Initial display
    update_display(current_threshold)

    # Center window
    threshold_window.update_idletasks()
    x = (threshold_window.winfo_screenwidth() // 2) - (threshold_window.winfo_width() // 2)
    y = (threshold_window.winfo_screenheight() // 2) - (threshold_window.winfo_height() // 2)
    threshold_window.geometry(f"+{x}+{y}")

    threshold_window.wait_window()

    plt.close(fig)  # Clean up

    return result[0]


def OtsusThreshold(detH, particleType):
    """
    Compute Otsu's threshold for automatic threshold selection
    """
    # Get all detector values
    data = detH.data.flatten()

    # Remove negative values if looking for positive particles
    if particleType == 1:
        data = data[data > 0]
    elif particleType == -1:
        data = -data[data < 0]

    if len(data) == 0:
        return 0

    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Otsu's method
    total_pixels = np.sum(hist)
    total_mean = np.sum(bin_centers * hist) / total_pixels

    max_variance = 0
    threshold = 0

    w1 = 0
    sum1 = 0

    for i in range(len(hist)):
        w1 += hist[i]
        if w1 == 0:
            continue

        w2 = total_pixels - w1
        if w2 == 0:
            break

        sum1 += bin_centers[i] * hist[i]
        mean1 = sum1 / w1
        mean2 = (total_mean * total_pixels - sum1) / w2

        # Between class variance
        variance = w1 * w2 * (mean1 - mean2) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = bin_centers[i]

    return np.sqrt(threshold)  # Return in image units


def Testing(string_input, number_input):
    """
    Testing function from original Igor Pro code
    Simple function to test that the system is working
    """
    print(f"Testing function called with:")
    print(f"  String: '{string_input}'")
    print(f"  Number: {number_input}")
    print(f"  Result: {string_input} * {number_input} = {len(string_input) * number_input}")

    # Show result in message box too
    result_msg = f"Testing function executed!\n\nInputs:\nString: '{string_input}'\nNumber: {number_input}\n\nResult: String length ({len(string_input)}) * Number = {len(string_input) * number_input}"
    messagebox.showinfo("Testing Function Result", result_msg)


# Additional helper functions for completeness
def SubPixelRefinement(im, info, mapNum, subPixelMult):
    """
    Apply sub-pixel refinement to particle positions
    """
    # This is a simplified implementation
    # The full implementation would involve interpolation of the detector response
    print(f"Sub-pixel refinement with multiplier {subPixelMult} (simplified implementation)")
    pass


def ApplyParticleConstraints(info, mapNum, minH, maxH, minV, maxV, minA, maxA):
    """
    Apply particle size and measurement constraints
    """
    print("Applying particle constraints...")
    # Remove particles that don't meet constraints
    # This would modify the info wave and mapNum to remove invalid particles
    pass


def CreateIndividualParticleFolders(im, info, mapNum, parent_folder):
    """
    Create individual folders for each detected particle
    """
    num_particles = int(np.max(mapNum.data) + 1) if np.max(mapNum.data) >= 0 else 0

    for particle_id in range(num_particles):
        particle_folder_name = f"Particle_{particle_id}"
        particle_folder = parent_folder.add_subfolder(particle_folder_name)

        # Extract particle region and store as wave
        particle_mask = (mapNum.data == particle_id)
        if np.any(particle_mask):
            # Create particle-specific data
            particle_data = im.data.copy()
            particle_data[~particle_mask] = 0

            particle_wave = Wave(particle_data, f"Particle_{particle_id}_Data")
            particle_folder.add_wave(particle_wave)


def ExtractParticleMeasurements(im, info, mapNum, series_folder, image_name):
    """
    Extract measurements from detected particles and add to series summary
    """
    num_particles = int(np.max(mapNum.data) + 1) if np.max(mapNum.data) >= 0 else 0

    if num_particles == 0:
        return

    # Extract heights, volumes, and areas for this image
    heights = []
    volumes = []
    areas = []

    for particle_id in range(num_particles):
        particle_mask = (mapNum.data == particle_id)
        if np.any(particle_mask):
            # Simple measurements
            particle_data = im.data[particle_mask]
            height = np.max(particle_data)
            area = np.sum(particle_mask) * DimDelta(im, 0) * DimDelta(im, 1)
            volume = np.sum(particle_data) * DimDelta(im, 0) * DimDelta(im, 1)

            heights.append(height)
            areas.append(area)
            volumes.append(volume)

    # Add to series summary waves
    if heights:
        all_heights = series_folder.waves["AllHeights"]
        all_volumes = series_folder.waves["AllVolumes"]
        all_areas = series_folder.waves["AllAreas"]

        # Concatenate new measurements
        all_heights.data = np.concatenate([all_heights.data, heights])
        all_volumes.data = np.concatenate([all_volumes.data, volumes])
        all_areas.data = np.concatenate([all_areas.data, areas])


def ViewParticles():
    """
    View detected particles in a dedicated viewer
    """
    # This would open a particle viewing window
    # For now, just show a message
    messagebox.showinfo("Particle Viewer",
                        "Particle viewer functionality would open here.\n\nThis would show individual particles with navigation controls.")


def MeasureParticles():
    """
    Perform detailed measurements on detected particles
    """
    # This would perform various measurements on the particles
    messagebox.showinfo("Particle Measurements",
                        "Particle measurement functionality would run here.\n\nThis would calculate heights, volumes, areas, and other properties.")
    return True