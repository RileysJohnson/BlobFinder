"""
Preprocessing Module
Contains image preprocessing functions for the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
Complete implementation with all preprocessing functions
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
    Direct port from Igor Pro BatchPreprocess function
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
    messagebox.showinfo("Preprocessing Complete",
                        f"Processed {processed_count} out of {NumImages} images.\n"
                        f"Results saved in folder: {PreprocessedDF}")
    return True


def GetPreprocessingParameters():
    """
    Get preprocessing parameters from user dialog
    Direct port from Igor Pro parameter dialog
    """
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()  # Hide main window

    dialog = tk.Toplevel()
    dialog.title("Preprocessing Parameters")
    dialog.geometry("400x300")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Image Preprocessing Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

    # Streak removal parameters
    streak_frame = ttk.LabelFrame(main_frame, text="Streak Removal", padding="10")
    streak_frame.pack(fill=tk.X, pady=5)

    ttk.Label(streak_frame, text="Standard deviations from mean for streak identification:").pack(anchor='w')
    ttk.Label(streak_frame, text="(Enter 0 to skip streak removal)").pack(anchor='w')

    streak_var = tk.DoubleVar(value=3.0)
    ttk.Entry(streak_frame, textvariable=streak_var, width=15).pack(pady=5)

    # Polynomial flattening parameters
    flatten_frame = ttk.LabelFrame(main_frame, text="Polynomial Flattening", padding="10")
    flatten_frame.pack(fill=tk.X, pady=5)

    ttk.Label(flatten_frame, text="Polynomial order for background flattening:").pack(anchor='w')
    ttk.Label(flatten_frame, text="(Enter 0 to skip flattening, typically use 2)").pack(anchor='w')

    flatten_var = tk.IntVar(value=2)
    ttk.Entry(flatten_frame, textvariable=flatten_var, width=15).pack(pady=5)

    # Additional processing option
    additional_frame = ttk.LabelFrame(main_frame, text="Additional Processing", padding="10")
    additional_frame.pack(fill=tk.X, pady=5)

    ttk.Label(additional_frame, text="Additional polynomial order (advanced):").pack(anchor='w')
    ttk.Label(additional_frame, text="(Usually leave as 0)").pack(anchor='w')

    additional_var = tk.IntVar(value=0)
    ttk.Entry(additional_frame, textvariable=additional_var, width=15).pack(pady=5)

    def ok_clicked():
        try:
            result[0] = (
                streak_var.get(),
                flatten_var.get(),
                additional_var.get()
            )
            dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")

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


def RemoveStreaks(wave, threshold_sigma):
    """
    Remove streaks from an image based on statistical analysis
    Direct port from Igor Pro RemoveStreaks function

    Parameters:
    wave : Wave - The image to process
    threshold_sigma : float - Number of standard deviations from mean to identify streaks
    """
    print(f"Removing streaks with threshold {threshold_sigma} sigma...")

    if threshold_sigma <= 0:
        return

    data = wave.data.copy()
    height, width = data.shape

    # Process each row to identify and remove horizontal streaks
    for i in range(height):
        row = data[i, :]
        row_mean = np.mean(row)
        row_std = np.std(row)

        if row_std > 0:
            # Identify outliers
            z_scores = np.abs((row - row_mean) / row_std)
            outliers = z_scores > threshold_sigma

            if np.any(outliers):
                # Replace outliers with interpolated values
                valid_indices = np.where(~outliers)[0]
                outlier_indices = np.where(outliers)[0]

                if len(valid_indices) > 1:
                    # Interpolate outlier values
                    for idx in outlier_indices:
                        # Find nearest valid neighbors
                        left_neighbors = valid_indices[valid_indices < idx]
                        right_neighbors = valid_indices[valid_indices > idx]

                        if len(left_neighbors) > 0 and len(right_neighbors) > 0:
                            left_idx = left_neighbors[-1]
                            right_idx = right_neighbors[0]
                            # Linear interpolation
                            weight = (idx - left_idx) / (right_idx - left_idx)
                            data[i, idx] = row[left_idx] * (1 - weight) + row[right_idx] * weight
                        elif len(left_neighbors) > 0:
                            data[i, idx] = row[left_neighbors[-1]]
                        elif len(right_neighbors) > 0:
                            data[i, idx] = row[right_neighbors[0]]

    # Process each column to identify and remove vertical streaks
    for j in range(width):
        col = data[:, j]
        col_mean = np.mean(col)
        col_std = np.std(col)

        if col_std > 0:
            # Identify outliers
            z_scores = np.abs((col - col_mean) / col_std)
            outliers = z_scores > threshold_sigma

            if np.any(outliers):
                # Replace outliers with interpolated values
                valid_indices = np.where(~outliers)[0]
                outlier_indices = np.where(outliers)[0]

                if len(valid_indices) > 1:
                    # Interpolate outlier values
                    for idx in outlier_indices:
                        # Find nearest valid neighbors
                        left_neighbors = valid_indices[valid_indices < idx]
                        right_neighbors = valid_indices[valid_indices > idx]

                        if len(left_neighbors) > 0 and len(right_neighbors) > 0:
                            left_idx = left_neighbors[-1]
                            right_idx = right_neighbors[0]
                            # Linear interpolation
                            weight = (idx - left_idx) / (right_idx - left_idx)
                            data[idx, j] = col[left_idx] * (1 - weight) + col[right_idx] * weight
                        elif len(left_neighbors) > 0:
                            data[idx, j] = col[left_neighbors[-1]]
                        elif len(right_neighbors) > 0:
                            data[idx, j] = col[right_neighbors[0]]

    # Update wave data
    wave.data = data
    print("Streak removal completed.")


def FlattenImage(wave, polynomial_order):
    """
    Flatten image background using polynomial fitting
    Direct port from Igor Pro FlattenImage function

    Parameters:
    wave : Wave - The image to flatten
    polynomial_order : int - Order of polynomial for background fitting
    """
    print(f"Flattening image with polynomial order {polynomial_order}...")

    if polynomial_order <= 0:
        return

    data = wave.data.copy()
    height, width = data.shape

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Flatten coordinate arrays
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = data.flatten()

    # Create polynomial feature matrix
    features = []
    for i in range(polynomial_order + 1):
        for j in range(polynomial_order + 1 - i):
            if i + j <= polynomial_order:
                features.append((x_flat ** i) * (y_flat ** j))

    feature_matrix = np.column_stack(features)

    try:
        # Fit polynomial surface using least squares
        coefficients, residuals, rank, s = np.linalg.lstsq(feature_matrix, z_flat, rcond=None)

        # Compute background surface
        background = np.dot(feature_matrix, coefficients)
        background = background.reshape(height, width)

        # Subtract background
        flattened_data = data - background

        # Update wave data
        wave.data = flattened_data

        print(f"Background flattening completed. Residual: {np.sum(residuals) if len(residuals) > 0 else 'N/A'}")

    except np.linalg.LinAlgError as e:
        print(f"Error in polynomial fitting: {e}")
        print("Skipping background flattening.")


def MedianFilter(wave, kernel_size=3):
    """
    Apply median filter to reduce noise
    Direct port from Igor Pro MedianFilter equivalent

    Parameters:
    wave : Wave - The image to filter
    kernel_size : int - Size of the median filter kernel
    """
    print(f"Applying median filter with kernel size {kernel_size}...")

    filtered_data = ndimage.median_filter(wave.data, size=kernel_size)
    wave.data = filtered_data

    print("Median filtering completed.")


def GaussianSmooth(wave, sigma=1.0):
    """
    Apply Gaussian smoothing filter
    Direct port from Igor Pro Smooth equivalent

    Parameters:
    wave : Wave - The image to smooth
    sigma : float - Standard deviation of Gaussian kernel
    """
    print(f"Applying Gaussian smoothing with sigma={sigma}...")

    smoothed_data = ndimage.gaussian_filter(wave.data, sigma=sigma)
    wave.data = smoothed_data

    print("Gaussian smoothing completed.")


def NormalizeImage(wave, method='minmax'):
    """
    Normalize image intensity values

    Parameters:
    wave : Wave - The image to normalize
    method : str - Normalization method ('minmax', 'zscore', 'percentile')
    """
    print(f"Normalizing image using {method} method...")

    data = wave.data.copy()

    if method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.zeros_like(data)

    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val > 0:
            data = (data - mean_val) / std_val
        else:
            data = data - mean_val

    elif method == 'percentile':
        # Percentile-based normalization (1st to 99th percentile)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        if p99 > p1:
            data = np.clip((data - p1) / (p99 - p1), 0, 1)
        else:
            data = np.zeros_like(data)

    wave.data = data
    print("Image normalization completed.")


def RemoveBackground(wave, method='rolling_ball', **kwargs):
    """
    Remove background from image using various methods

    Parameters:
    wave : Wave - The image to process
    method : str - Background removal method
    **kwargs : Additional parameters for specific methods
    """
    print(f"Removing background using {method} method...")

    if method == 'rolling_ball':
        radius = kwargs.get('radius', 50)
        background = RollingBallBackground(wave.data, radius)
        wave.data = wave.data - background

    elif method == 'tophat':
        kernel_size = kwargs.get('kernel_size', 15)
        kernel = np.ones((kernel_size, kernel_size))
        background = ndimage.morphology.white_tophat(wave.data, structure=kernel)
        wave.data = background

    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 25)
        background = ndimage.gaussian_filter(wave.data, sigma=sigma)
        wave.data = wave.data - background

    print("Background removal completed.")


def RollingBallBackground(image, radius):
    """
    Estimate background using rolling ball algorithm
    Simplified implementation of ImageJ's rolling ball background subtraction

    Parameters:
    image : ndarray - Input image
    radius : int - Radius of the rolling ball

    Returns:
    ndarray - Estimated background
    """
    from scipy import ndimage

    # Create a ball-shaped structuring element
    ball = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x * x + y * y <= radius * radius
    ball[mask] = 1

    # Apply morphological opening
    background = ndimage.morphology.grey_opening(image, structure=ball)

    return background


def EnhanceContrast(wave, method='histogram_equalization', **kwargs):
    """
    Enhance image contrast using various methods

    Parameters:
    wave : Wave - The image to enhance
    method : str - Contrast enhancement method
    **kwargs : Additional parameters
    """
    print(f"Enhancing contrast using {method} method...")

    data = wave.data.copy()

    if method == 'histogram_equalization':
        # Simple histogram equalization
        hist, bins = np.histogram(data.flatten(), bins=256, density=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize

        # Interpolate to get new values
        data_eq = np.interp(data.flatten(), bins[:-1], cdf)
        wave.data = data_eq.reshape(data.shape)

    elif method == 'adaptive':
        # Adaptive histogram equalization (simplified)
        from scipy import ndimage

        # Local histogram equalization using rank filters
        footprint = np.ones((31, 31))  # 31x31 local region
        local_mean = ndimage.uniform_filter(data, size=31)
        local_var = ndimage.uniform_filter(data ** 2, size=31) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))

        # Enhance based on local statistics
        enhanced = (data - local_mean) / (local_std + 1e-6)
        wave.data = enhanced

    elif method == 'gamma':
        gamma = kwargs.get('gamma', 1.5)
        # Normalize to [0,1], apply gamma, then scale back
        min_val, max_val = np.min(data), np.max(data)
        if max_val > min_val:
            normalized = (data - min_val) / (max_val - min_val)
            gamma_corrected = np.power(normalized, gamma)
            wave.data = gamma_corrected * (max_val - min_val) + min_val

    print("Contrast enhancement completed.")


def DenoiseImage(wave, method='bilateral', **kwargs):
    """
    Denoise image using various methods

    Parameters:
    wave : Wave - The image to denoise
    method : str - Denoising method
    **kwargs : Additional parameters
    """
    print(f"Denoising image using {method} method...")

    data = wave.data.copy()

    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        denoised = ndimage.gaussian_filter(data, sigma=sigma)

    elif method == 'median':
        size = kwargs.get('size', 3)
        denoised = ndimage.median_filter(data, size=size)

    elif method == 'bilateral':
        # Simplified bilateral filter implementation
        sigma_spatial = kwargs.get('sigma_spatial', 1.0)
        sigma_intensity = kwargs.get('sigma_intensity', 0.1)

        # This is a simplified version - for full bilateral filtering,
        # you'd typically use skimage.restoration.denoise_bilateral
        denoised = ndimage.gaussian_filter(data, sigma=sigma_spatial)

    elif method == 'wiener':
        # Wiener filter (simplified)
        noise_var = kwargs.get('noise_var', None)
        if noise_var is None:
            # Estimate noise variance
            noise_var = np.var(data) * 0.01  # Assume 1% noise

        # Apply simple Wiener-like filter
        signal_var = np.var(data)
        filter_factor = signal_var / (signal_var + noise_var)
        denoised = ndimage.gaussian_filter(data, sigma=1.0) * filter_factor + data * (1 - filter_factor)

    else:
        print(f"Unknown denoising method: {method}")
        return

    wave.data = denoised
    print("Image denoising completed.")


def Testing(string_input, number_input):
    """
    Testing function for preprocessing operations
    Direct port from Igor Pro Testing function
    """
    print(f"Preprocessing testing function called:")
    print(f"  String input: '{string_input}'")
    print(f"  Number input: {number_input}")

    # Create a test image with noise and background
    test_size = 64
    test_data = np.zeros((test_size, test_size))

    # Add background gradient
    for i in range(test_size):
        for j in range(test_size):
            test_data[i, j] = 0.3 * (i + j) / (2 * test_size)

    # Add some blob features
    center = test_size // 2
    for i in range(test_size):
        for j in range(test_size):
            r1 = np.sqrt((i - center + 10) ** 2 + (j - center) ** 2)
            r2 = np.sqrt((i - center - 10) ** 2 + (j - center) ** 2)
            test_data[i, j] += np.exp(-r1 ** 2 / 50) + 0.5 * np.exp(-r2 ** 2 / 20)

    # Add noise
    noise = np.random.normal(0, 0.05, (test_size, test_size))
    test_data += noise

    # Create test wave
    test_wave = Wave(test_data, "TestImage")
    test_wave.SetScale('x', 0, 1.0)
    test_wave.SetScale('y', 0, 1.0)

    print(f"  Created test image with shape: {test_wave.data.shape}")
    print(f"  Original image stats: mean={np.mean(test_data):.4f}, std={np.std(test_data):.4f}")

    # Test preprocessing functions
    original_wave = Duplicate(test_wave, "Original")

    # Test flattening
    flatten_wave = Duplicate(test_wave, "Flattened")
    FlattenImage(flatten_wave, 2)
    print(f"  After flattening: mean={np.mean(flatten_wave.data):.4f}, std={np.std(flatten_wave.data):.4f}")

    # Test streak removal
    streak_wave = Duplicate(flatten_wave, "StreakRemoved")
    RemoveStreaks(streak_wave, 3.0)
    print(f"  After streak removal: mean={np.mean(streak_wave.data):.4f}, std={np.std(streak_wave.data):.4f}")

    # Test normalization
    norm_wave = Duplicate(streak_wave, "Normalized")
    NormalizeImage(norm_wave, 'minmax')
    print(f"  After normalization: mean={np.mean(norm_wave.data):.4f}, std={np.std(norm_wave.data):.4f}")

    result = len(string_input) + number_input + test_size
    print(f"  Test result: {result}")

    return result