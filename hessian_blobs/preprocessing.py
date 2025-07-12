"""
Hessian Blob Particle Detection Suite - Preprocessing Functions

Contains image preprocessing functions:
- BatchPreprocess(): Preprocess multiple images in a data folder
- Flatten(): Flattens horizontal line scans by subtracting polynomials
- RemoveStreaks(): Removes streak artifacts from images

Corresponds to Section IV. Preprocessing Functions in the original Igor Pro code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import threading

from igor_compatibility.data_management import DataManager
from igor_compatibility.wave_operations import GetBrowserSelection, GetDataFolder
from core.error_handling import handle_error, safe_print


# ========================================================================
# PREPROCESSING FUNCTIONS
# ========================================================================

def BatchPreprocess():
    """
    Allows you to preprocess multiple images in a data folder successively. Make sure to highlight the
    folder containing the images in the data browser before executing.

    Exact translation of Igor Pro BatchPreprocess() function.
    """
    try:
        # Get folder containing images - matching Igor Pro GetBrowserSelection
        ImagesDF = filedialog.askdirectory(title="Select folder containing images to preprocess")
        if not ImagesDF:
            raise ValueError("No folder selected")

        CurrentDF = GetDataFolder(1)

        # Count images - matching Igor Pro CountObjects
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            found_files = [f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())]
            image_files.extend(found_files)

        if len(image_files) < 1:
            raise ValueError("Selected folder contains no images")

        # Declare algorithm parameters
        flattenOrder = 2
        streakRemovalSDevs = 3

        # Retrieve parameters from user
        params = _get_preprocess_parameters()
        if params is None:
            safe_print("Preprocessing cancelled by user.")
            return -1

        streakRemovalSDevs, flattenOrder = params

        # Create preprocessed folder - matching Igor Pro DuplicateDataFolder behavior
        base_folder_name = os.path.basename(ImagesDF.rstrip(os.sep))
        preprocessed_folder_name = f"{base_folder_name}_dup"  # Matching Igor Pro "_dup" suffix

        counter = 0
        while True:
            if counter == 0:
                test_folder_name = preprocessed_folder_name
            else:
                test_folder_name = f"{preprocessed_folder_name}_{counter}"

            preprocessed_folder_path = os.path.join(CurrentDF, test_folder_name)
            if not os.path.exists(preprocessed_folder_path):
                break
            counter += 1

        os.makedirs(preprocessed_folder_path, exist_ok=True)
        safe_print(f"Created duplicated folder: {preprocessed_folder_path}")

        # Preprocess the images
        NumImages = len(image_files)
        for i, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(ImagesDF, image_file)
                Im = DataManager.load_image_file(image_path)

                if Im is None:
                    safe_print(f"Warning: Could not load {image_file}")
                    continue

                # CRITICAL FIX: Ensure image is writable
                if not Im.flags.writeable:
                    Im = Im.copy()

                safe_print(f"Preprocessing image {i + 1} of {NumImages}: {image_file}")

                # Apply preprocessing in Igor Pro order
                if streakRemovalSDevs > 0:
                    safe_print(f"  RemoveStreaks with {streakRemovalSDevs} std devs")
                    RemoveStreaks(Im, sigma=streakRemovalSDevs)

                if flattenOrder > 0:
                    safe_print(f"  Flatten with order {flattenOrder}")
                    # Use automatic threshold for batch processing to avoid threading issues
                    result = Flatten(Im, flattenOrder, noThresh=True)
                    if result != 0:
                        safe_print(f"  Flattening failed for {image_file}")
                        continue

                # Save to duplicated folder - matching Igor Pro behavior
                output_path = os.path.join(preprocessed_folder_path, image_file)
                if image_file.endswith('.npy'):
                    DataManager.save_wave_data(Im, output_path)
                else:
                    # Save as .npy for processed data
                    output_npy = os.path.join(preprocessed_folder_path,
                                              os.path.splitext(image_file)[0] + '.npy')
                    DataManager.save_wave_data(Im, output_npy)

            except Exception as e:
                handle_error("BatchPreprocess", e, f"image {image_file}")
                continue

        safe_print("Preprocessing complete.")
        return 0

    except Exception as e:
        error_msg = handle_error("BatchPreprocess", e)
        messagebox.showerror("Preprocessing Error", error_msg)
        return -1


def Flatten(im, order, mask=None, noThresh=False):
    """
    Flattens ever horizontal line scan of the image by subtracting off a least-squares
    fitted polynomial of a given order.

    Args:
        im: The image to be flattened
        order: The order of the polynomial to be subtracted off
        mask: An optional mask identifying pixels to fit the polynomial to.
             In the mask, 1 corresponds to pixels which will be used for fitting, 0 for pixels to be ignored.
        noThresh: An optional parameter, if given any value will not prompt the user to set the threshold level.

    Returns:
        0 on success, -1 on failure

    Exact translation of Igor Pro Flatten() function.
    """
    try:
        # CRITICAL FIX: Ensure image is writable by making a copy if needed
        if not im.flags.writeable:
            im = im.copy()

        # Want to interactively determine a threshold?
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

            # Make the mask wave
            mask = (im <= threshold).astype(int)
            safe_print(f"Flatten Height Threshold: {threshold}")
        elif noThresh and mask is None:
            # Automatic threshold - matching Igor Pro noThresh behavior
            threshold = np.mean(im) + 0.5 * np.std(im)
            mask = (im <= threshold).astype(int)

        if mask is None:
            mask = np.ones_like(im, dtype=int)

        # Make a 1D wave for fitting and masking
        # Make/N=(DimSize(im,0)) /O FLATTEN_SCANLINE, FLATTEN_MASK
        lines = im.shape[1]
        x = np.arange(im.shape[0])

        # Make the coefficient wave
        # Make/N=(max(2,order+1)) /O FLATTEN_COEFS=0

        # Fit to each scan line
        for i in range(lines):
            try:
                # Multithread Scanline = im[p][i]
                scanline = im[:, i].copy()
                mask_1d = mask[:, i] if mask is not None else np.ones(im.shape[0], dtype=int)

                valid_indices = mask_1d == 1
                if np.sum(valid_indices) < order + 1:
                    continue

                x_valid = x[valid_indices]
                y_valid = scanline[valid_indices]

                # Do a fit to the scan line
                if order == 1:
                    # CurveFit/W=2 /Q line, kwCWave=Coefs, Scanline /M=Mask1D
                    coefs = np.polyfit(x_valid, y_valid, 1)
                    # im[][i] -= Coefs[0] + x*Coefs[1]
                    im[:, i] -= coefs[1] + x * coefs[0]  # Note: numpy polyfit returns coeffs in reverse order
                elif order == 0:
                    # CurveFit/W=2 /Q /H="01" line, kwCWave=Coefs, ScanLine /M=Mask1D
                    coef = np.mean(y_valid)
                    # im[][i] -= Coefs[0]
                    im[:, i] -= coef
                elif order > 1:
                    # CurveFit/W=2 /Q Poly Order+1, kwCWave=Coefs, Scanline /M=Mask1D
                    coefs = np.polyfit(x_valid, y_valid, order)
                    # im[][i] -= Poly(Coefs,x)
                    im[:, i] -= np.polyval(coefs, x)

            except Exception as e:
                handle_error("Flatten", e, f"scan line {i}")
                continue

        return 0

    except Exception as e:
        handle_error("Flatten", e)
        return -1


def RemoveStreaks(image, sigma=3):
    """
    Removes streak artifacts from the image.

    Args:
        im: The image from which streaks will be removed
        sigma: The number of standard deviations away from mean streak level to smooth a streak

    Returns:
        0 on success, -1 on failure

    Exact translation of Igor Pro RemoveStreaks() function.
    """
    try:
        # CRITICAL FIX: Ensure image is writable by making a copy if needed
        if not image.flags.writeable:
            image = image.copy()

        # Produce the dY map
        dyMap = dyMap_func(image)
        dyMap = np.abs(dyMap)

        # WaveStats/Q dyMap
        MaxDY = np.mean(dyMap) + np.std(dyMap) * sigma
        AvgDY = np.mean(dyMap)

        safe_print(f"Streak removal statistics:")
        safe_print(f"  Average dY: {AvgDY:.6e}")
        safe_print(f"  Threshold dY: {MaxDY:.6e}")
        safe_print(f"  Sigma multiplier: {sigma}")

        limI, limJ = image.shape[0], image.shape[1] - 1
        streaks_removed = 0

        for i in range(limI):
            for j in range(1, limJ):
                if dyMap[i, j] > MaxDY:
                    i0 = i
                    streak_length = 0

                    # Go left until the left side of the streak is gone
                    while i >= 0 and dyMap[i, j] > AvgDY:
                        # image[i][j]=(image[i][j+1]+image[i][j-1])/2
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dyMap[i, j] = 0
                        streak_length += 1
                        i -= 1

                    i = i0

                    # Go right from the original point doing the same thing
                    while i < limI and dyMap[i, j] > AvgDY:
                        # image[i][j]=(image[i][j+1]+image[i][j-1])/2
                        image[i, j] = (image[i, j + 1] + image[i, j - 1]) / 2
                        dyMap[i, j] = 0
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


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

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


def dyMap_func(image):
    """
    Returns a wave with the dY map for streak detection.

    Exact translation of Igor Pro dyMap() function.
    """
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