"""
Preprocessing Module
Contains image preprocessing functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from scipy import ndimage
from scipy.optimize import minimize_scalar

from igor_compatibility import *
from file_io import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def BatchPreprocess():
    """
    Preprocess a series of images in a chosen data folder.
    Be sure to highlight the data folder containing the images in the data browser before running.
    """
    ImagesDF = GetBrowserSelection(0)
    CurrentDF = GetDataFolder(1)

    if not DataFolderExists(ImagesDF) or CountObjects(ImagesDF, 1) < 1:
        messagebox.showerror("Error", "Select the folder with your images in it in the data browser, then try again.")
        return False

    # Get preprocessing parameters from user
    params = GetPreprocessingParameters()
    if params is None:
        return False  # User cancelled

    streak_removal, polynomial_order, flatten_order = params

    # Get source folder
    folder = data_browser.get_folder(ImagesDF.rstrip(':'))
    NumImages = len(folder.waves)

    # Create destination folder
    PreprocessedDF = UniqueName("Preprocessed_", 11, 0)
    NewDataFolder(PreprocessedDF)
    preprocessed_folder = data_browser.get_folder(PreprocessedDF)

    # Store the parameters being used
    parameters = Wave(np.array([streak_removal, polynomial_order, flatten_order]), "PreprocessingParameters")
    preprocessed_folder.add_wave(parameters)

    print(f"Preprocessing {NumImages} images...")
    processed_count = 0

    # Process each image
    for wave_name, wave in folder.waves.items():
        try:
            print(f"Processing image {processed_count + 1}/{NumImages}: {wave_name}")

            # Create a copy for preprocessing
            processed_wave = Duplicate(wave, f"{wave_name}_processed")

            # Apply preprocessing steps
            if streak_removal > 0:
                RemoveStreaks(processed_wave, streak_removal)

            if polynomial_order > 0:
                FlattenImage(processed_wave, polynomial_order)

            # Store processed image
            preprocessed_folder.add_wave(processed_wave)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {wave_name}: {str(e)}")
            continue

    print(f"Batch preprocessing completed. Processed {processed_count} images.")
    print(f"Results stored in '{PreprocessedDF}'")

    return True


def GetPreprocessingParameters():
    """
    Get preprocessing parameters from user dialog
    Returns tuple of parameters or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide root window

    dialog = tk.Toplevel(root)
    dialog.title("Preprocessing Parameters")
    dialog.geometry("400x200")
    dialog.grab_set()

    # Make dialog modal and centered
    dialog.transient(root)
    dialog.focus_set()

    # Variables for parameters
    streak_var = tk.DoubleVar(value=3.0)  # Standard deviations for streak detection
    poly_var = tk.IntVar(value=2)  # Polynomial order for flattening
    flatten_var = tk.IntVar(value=0)  # Additional flattening order

    result = [None]  # Use list to store result

    # Create GUI elements
    main_frame = ttk.Frame(dialog, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Streak Removal (std devs, 0=off):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=streak_var, width=15).grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(main_frame, text="Polynomial Flattening Order (0=off):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=poly_var, width=15).grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(main_frame, text="Additional Flattening Order (0=off):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=flatten_var, width=15).grid(row=2, column=1, padx=5, pady=5)

    def ok_clicked():
        try:
            result[0] = (streak_var.get(), poly_var.get(), flatten_var.get())
            dialog.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")

    def cancel_clicked():
        result[0] = None
        dialog.quit()

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)

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


def RemoveStreaks(wave, threshold_stddevs=3.0):
    """
    Remove streaks from an image using statistical analysis

    Parameters:
    wave : Wave - The image to process
    threshold_stddevs : float - Number of standard deviations for streak detection

    Returns:
    bool - Success flag
    """
    try:
        print(f"Removing streaks with threshold {threshold_stddevs} standard deviations...")

        data = wave.data.copy()
        height, width = data.shape

        # Analyze row-wise streaks (horizontal streaks)
        row_means = np.mean(data, axis=1)
        row_mean = np.mean(row_means)
        row_std = np.std(row_means)

        if row_std > 0:
            # Identify problematic rows
            row_threshold = row_mean + threshold_stddevs * row_std
            bad_rows = row_means > row_threshold

            # Correct bad rows
            for i in np.where(bad_rows)[0]:
                # Find nearby good rows for interpolation
                good_rows = []
                for offset in range(1, min(10, height)):
                    if i - offset >= 0 and not bad_rows[i - offset]:
                        good_rows.append(i - offset)
                        break
                    if i + offset < height and not bad_rows[i + offset]:
                        good_rows.append(i + offset)
                        break

                if good_rows:
                    # Replace with average of nearby good rows
                    replacement_data = np.mean([data[r, :] for r in good_rows], axis=0)
                    data[i, :] = replacement_data

        # Analyze column-wise streaks (vertical streaks)
        col_means = np.mean(data, axis=0)
        col_mean = np.mean(col_means)
        col_std = np.std(col_means)

        if col_std > 0:
            # Identify problematic columns
            col_threshold = col_mean + threshold_stddevs * col_std
            bad_cols = col_means > col_threshold

            # Correct bad columns
            for j in np.where(bad_cols)[0]:
                # Find nearby good columns for interpolation
                good_cols = []
                for offset in range(1, min(10, width)):
                    if j - offset >= 0 and not bad_cols[j - offset]:
                        good_cols.append(j - offset)
                        break
                    if j + offset < width and not bad_cols[j + offset]:
                        good_cols.append(j + offset)
                        break

                if good_cols:
                    # Replace with average of nearby good columns
                    replacement_data = np.mean([data[:, c] for c in good_cols], axis=0)
                    data[:, j] = replacement_data

        # Update wave data
        wave.data = data

        print("Streak removal completed")
        return True

    except Exception as e:
        print(f"Error in streak removal: {e}")
        return False


def FlattenImage(wave, polynomial_order=2):
    """
    Flatten an image by removing polynomial background

    Parameters:
    wave : Wave - The image to flatten
    polynomial_order : int - Order of polynomial to fit and subtract

    Returns:
    bool - Success flag
    """
    try:
        print(f"Flattening image with polynomial order {polynomial_order}...")

        if polynomial_order <= 0:
            return True  # No flattening requested

        data = wave.data.copy()
        height, width = data.shape

        # Create coordinate grids
        y_coords = np.arange(height)
        x_coords = np.arange(width)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Flatten coordinates for fitting
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = data.flatten()

        # Remove NaN values
        valid_mask = ~np.isnan(z_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]

        if len(z_valid) < (polynomial_order + 1) ** 2:
            print("Warning: Not enough valid data points for polynomial fitting")
            return False

        # Build polynomial basis functions
        basis_functions = []
        for i in range(polynomial_order + 1):
            for j in range(polynomial_order + 1 - i):
                basis_functions.append((x_valid ** i) * (y_valid ** j))

        # Create design matrix
        A = np.column_stack(basis_functions)

        # Solve least squares problem
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, z_valid, rcond=None)
        except np.linalg.LinAlgError:
            print("Error: Singular matrix in polynomial fitting")
            return False

        # Evaluate polynomial on full grid
        background = np.zeros_like(data)
        coeff_idx = 0
        for i in range(polynomial_order + 1):
            for j in range(polynomial_order + 1 - i):
                background += coeffs[coeff_idx] * (X ** i) * (Y ** j)
                coeff_idx += 1

        # Subtract background
        wave.data = data - background

        print("Image flattening completed")
        return True

    except Exception as e:
        print(f"Error in image flattening: {e}")
        return False


def SubtractBackground(wave, method='plane'):
    """
    Subtract background from an image using various methods

    Parameters:
    wave : Wave - The image to process
    method : str - Background subtraction method ('plane', 'polynomial', 'median')

    Returns:
    bool - Success flag
    """
    try:
        print(f"Subtracting background using {method} method...")

        data = wave.data.copy()
        height, width = data.shape

        if method == 'plane':
            # Fit a plane to the image edges
            edge_points = []
            edge_values = []

            # Top and bottom edges
            for j in range(width):
                edge_points.extend([(0, j), (height - 1, j)])
                edge_values.extend([data[0, j], data[height - 1, j]])

            # Left and right edges (excluding corners already added)
            for i in range(1, height - 1):
                edge_points.extend([(i, 0), (i, width - 1)])
                edge_values.extend([data[i, 0], data[i, width - 1]])

            # Fit plane: z = a + b*x + c*y
            if len(edge_points) >= 3:
                edge_points = np.array(edge_points)
                edge_values = np.array(edge_values)

                A = np.column_stack([
                    np.ones(len(edge_points)),
                    edge_points[:, 1],  # x coordinates
                    edge_points[:, 0]  # y coordinates
                ])

                coeffs, _, _, _ = np.linalg.lstsq(A, edge_values, rcond=None)

                # Create coordinate grids
                Y, X = np.mgrid[0:height, 0:width]
                background = coeffs[0] + coeffs[1] * X + coeffs[2] * Y

                wave.data = data - background

        elif method == 'polynomial':
            # Use existing polynomial flattening
            return FlattenImage(wave, polynomial_order=2)

        elif method == 'median':
            # Subtract median value
            median_val = np.nanmedian(data)
            wave.data = data - median_val

        else:
            print(f"Unknown background subtraction method: {method}")
            return False

        print("Background subtraction completed")
        return True

    except Exception as e:
        print(f"Error in background subtraction: {e}")
        return False


def SmoothImage(wave, method='gaussian', kernel_size=3, sigma=1.0):
    """
    Smooth an image using various filtering methods

    Parameters:
    wave : Wave - The image to smooth
    method : str - Smoothing method ('gaussian', 'mean', 'median')
    kernel_size : int - Size of the smoothing kernel
    sigma : float - Standard deviation for Gaussian smoothing

    Returns:
    bool - Success flag
    """
    try:
        print(f"Smoothing image using {method} filter...")

        data = wave.data.copy()

        if method == 'gaussian':
            smoothed_data = ndimage.gaussian_filter(data, sigma=sigma)

        elif method == 'mean':
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            smoothed_data = ndimage.convolve(data, kernel, mode='reflect')

        elif method == 'median':
            smoothed_data = ndimage.median_filter(data, size=kernel_size)

        else:
            print(f"Unknown smoothing method: {method}")
            return False

        wave.data = smoothed_data

        print("Image smoothing completed")
        return True

    except Exception as e:
        print(f"Error in image smoothing: {e}")
        return False


def EnhanceContrast(wave, method='histogram_equalization', clip_limit=0.02):
    """
    Enhance image contrast using various methods

    Parameters:
    wave : Wave - The image to enhance
    method : str - Enhancement method ('histogram_equalization', 'adaptive', 'stretch')
    clip_limit : float - Clipping limit for adaptive methods

    Returns:
    bool - Success flag
    """
    try:
        print(f"Enhancing contrast using {method} method...")

        data = wave.data.copy()

        if method == 'histogram_equalization':
            # Simple histogram equalization
            hist, bins = np.histogram(data.flatten(), bins=256, density=True)
            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())

            # Interpolate to get new values
            data_flat = data.flatten()
            data_eq = np.interp(data_flat, bins[:-1], cdf)
            wave.data = data_eq.reshape(data.shape)

        elif method == 'stretch':
            # Linear contrast stretching
            min_val = np.nanpercentile(data, 1)
            max_val = np.nanpercentile(data, 99)

            if max_val > min_val:
                wave.data = (data - min_val) / (max_val - min_val)
                wave.data = np.clip(wave.data, 0, 1)

        elif method == 'adaptive':
            # Simple adaptive contrast enhancement
            # Divide image into blocks and enhance each separately
            block_size = 64
            height, width = data.shape

            enhanced_data = data.copy()

            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    i_end = min(i + block_size, height)
                    j_end = min(j + block_size, width)

                    block = data[i:i_end, j:j_end]
                    block_min = np.nanpercentile(block, 5)
                    block_max = np.nanpercentile(block, 95)

                    if block_max > block_min:
                        enhanced_block = (block - block_min) / (block_max - block_min)
                        enhanced_data[i:i_end, j:j_end] = enhanced_block

            wave.data = enhanced_data

        else:
            print(f"Unknown contrast enhancement method: {method}")
            return False

        print("Contrast enhancement completed")
        return True

    except Exception as e:
        print(f"Error in contrast enhancement: {e}")
        return False


def NormalizeImage(wave, method='minmax', target_range=(0, 1)):
    """
    Normalize image intensity values

    Parameters:
    wave : Wave - The image to normalize
    method : str - Normalization method ('minmax', 'zscore', 'robust')
    target_range : tuple - Target range for minmax normalization

    Returns:
    bool - Success flag
    """
    try:
        print(f"Normalizing image using {method} method...")

        data = wave.data.copy()

        if method == 'minmax':
            # Min-max normalization
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)

            if max_val > min_val:
                normalized = (data - min_val) / (max_val - min_val)
                normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]
                wave.data = normalized

        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)

            if std_val > 0:
                wave.data = (data - mean_val) / std_val

        elif method == 'robust':
            # Robust normalization using percentiles
            q25 = np.nanpercentile(data, 25)
            q75 = np.nanpercentile(data, 75)
            median_val = np.nanmedian(data)

            if q75 > q25:
                wave.data = (data - median_val) / (q75 - q25)

        else:
            print(f"Unknown normalization method: {method}")
            return False

        print("Image normalization completed")
        return True

    except Exception as e:
        print(f"Error in image normalization: {e}")
        return False


def RemoveOutliers(wave, method='iqr', factor=1.5):
    """
    Remove outlier pixels from an image

    Parameters:
    wave : Wave - The image to process
    method : str - Outlier detection method ('iqr', 'zscore', 'percentile')
    factor : float - Factor for outlier detection threshold

    Returns:
    bool - Success flag
    """
    try:
        print(f"Removing outliers using {method} method...")

        data = wave.data.copy()

        if method == 'iqr':
            # Interquartile range method
            q25 = np.nanpercentile(data, 25)
            q75 = np.nanpercentile(data, 75)
            iqr = q75 - q25

            lower_bound = q25 - factor * iqr
            upper_bound = q75 + factor * iqr

            # Replace outliers with median
            median_val = np.nanmedian(data)
            outliers = (data < lower_bound) | (data > upper_bound)
            data[outliers] = median_val

        elif method == 'zscore':
            # Z-score method
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)

            if std_val > 0:
                z_scores = np.abs((data - mean_val) / std_val)
                outliers = z_scores > factor
                data[outliers] = mean_val

        elif method == 'percentile':
            # Percentile method
            lower_percentile = factor
            upper_percentile = 100 - factor

            lower_bound = np.nanpercentile(data, lower_percentile)
            upper_bound = np.nanpercentile(data, upper_percentile)

            median_val = np.nanmedian(data)
            outliers = (data < lower_bound) | (data > upper_bound)
            data[outliers] = median_val

        else:
            print(f"Unknown outlier removal method: {method}")
            return False

        wave.data = data

        print("Outlier removal completed")
        return True

    except Exception as e:
        print(f"Error in outlier removal: {e}")
        return False


def TestPreprocessing():
    """Test function for preprocessing module"""
    print("Testing preprocessing module...")

    # Create test data with streaks
    test_data = np.random.rand(100, 100) * 100

    # Add artificial streaks
    test_data[25, :] += 50  # Horizontal streak
    test_data[:, 75] += 30  # Vertical streak

    test_image = Wave(test_data, "TestImage")
    test_image.SetScale('x', 0, 1)
    test_image.SetScale('y', 0, 1)

    # Test streak removal
    original_max = np.max(test_image.data)
    RemoveStreaks(test_image, 2.0)
    processed_max = np.max(test_image.data)

    if processed_max < original_max:
        print("✓ Streak removal working (reduced maximum value)")
    else:
        print("? Streak removal completed (no change detected)")

    # Test flattening
    FlattenImage(test_image, 2)
    print("✓ Image flattening completed")

    # Test smoothing
    SmoothImage(test_image, 'gaussian', sigma=1.0)
    print("✓ Image smoothing completed")

    print("Preprocessing test completed")
    return True


if __name__ == "__main__":
    TestPreprocessing()