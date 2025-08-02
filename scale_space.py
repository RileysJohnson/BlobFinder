"""
Scale-Space Module
Contains functions for scale-space representation and blob detection
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from igor_compatibility import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex

# Global wave registry for Igor Pro compatibility
_wave_registry = {}


def RegisterWave(wave, name):
    """Register a wave in the global registry"""
    global _wave_registry
    _wave_registry[name] = wave


def GetWave(name):
    """Get a wave from the global registry"""
    global _wave_registry
    return _wave_registry.get(name, None)


def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """
    Calculate the discrete scale-space representation
    Direct port from Igor Pro ScaleSpaceRepresentation function

    Parameters:
    im : Wave - Input image
    layers : int - Number of scale layers
    t0 : float - Starting scale value in pixel units
    tFactor : float - Factor between consecutive scales

    Returns:
    Wave - 3D scale-space representation (y, x, scale)
    """
    print(f"Computing scale-space representation with {layers} layers...")
    print(f"Scale parameters: t0={t0}, tFactor={tFactor}")

    height, width = im.data.shape

    # Convert t0 to image units (like Igor Pro)
    t0_converted = (t0 * DimDelta(im, 0)) ** 2

    print(f"Converted t0: {t0_converted}")

    # Create 3D array for scale-space representation
    L_data = np.zeros((height, width, layers))

    # Get original image data
    original_data = im.data.copy()

    # Go to Fourier space (like Igor Pro)
    print("Computing FFT...")
    fft_data = np.fft.fft2(original_data)

    # Create frequency coordinate arrays
    freq_y = np.fft.fftfreq(height, d=DimDelta(im, 1))
    freq_x = np.fft.fftfreq(width, d=DimDelta(im, 0))

    # Create 2D frequency grids
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

    # Compute each scale layer
    for k in range(layers):
        # Calculate current scale: t0_converted * (tFactor^k)
        current_scale = t0_converted * (tFactor ** k)

        print(f"Layer {k}: scale = {current_scale}")

        # Apply Gaussian filter in Fourier domain (like Igor Pro)
        # Gaussian kernel in frequency domain: exp(-(fx^2 + fy^2) * pi^2 * 2 * scale)
        gaussian_kernel = np.exp(-(freq_x_grid ** 2 + freq_y_grid ** 2) * np.pi ** 2 * 2 * current_scale)

        # Apply filter
        filtered_fft = fft_data * gaussian_kernel

        # Transform back to spatial domain
        filtered_layer = np.fft.ifft2(filtered_fft).real

        # Store in scale-space representation
        L_data[:, :, k] = filtered_layer

    # Create output wave (matching Igor Pro structure)
    L_wave = Wave(L_data, f"{im.name}_L")

    # Set scaling information (matching Igor Pro)
    L_wave.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    L_wave.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))
    L_wave.SetScale('z', t0_converted, tFactor)

    print("Scale-space representation complete.")

    # Register the wave globally
    RegisterWave(L_wave, "L")

    return L_wave


def BlobDetectors(L, gammaNorm):
    """
    Computes the two blob detectors: determinant of Hessian and Laplacian of Gaussian
    Direct port from Igor Pro BlobDetectors function

    Parameters:
    L : Wave - The scale-space representation
    gammaNorm : float - Gamma normalization factor (should be 1 for blob detection)

    Returns:
    tuple - (detH_wave, LapG_wave) or int 0 for success when used in Igor Pro mode
    """
    print(f"Computing blob detectors with gammaNorm = {gammaNorm}")

    # Get dimensions
    height, width, layers = L.data.shape

    # Create convolution kernels for calculating central difference derivatives (like Igor Pro)
    # These are the exact kernels from Igor Pro code

    # Kernel for Lxx (5x1)
    LxxKernel = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Kernel for Lyy (1x5) - transpose of above
    LyyKernel = np.array([
        [0, 0, -1 / 12, 0, 0],
        [0, 0, 16 / 12, 0, 0],
        [0, 0, -30 / 12, 0, 0],
        [0, 0, 16 / 12, 0, 0],
        [0, 0, -1 / 12, 0, 0]
    ])

    # Kernel for Lxy (5x5)
    LxyKernel = np.array([
        [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
        [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
        [0, 0, 0, 0, 0],
        [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
        [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
    ])

    # Initialize output arrays
    Lxx_data = np.zeros_like(L.data)
    Lyy_data = np.zeros_like(L.data)
    Lxy_data = np.zeros_like(L.data)

    print("Computing second derivatives...")

    # Compute derivatives for each scale layer
    for k in range(layers):
        current_layer = L.data[:, :, k]

        # Compute Lxx using convolution
        Lxx_data[:, :, k] = ndimage.convolve(current_layer, LxxKernel, mode='constant')

        # Compute Lyy using convolution
        Lyy_data[:, :, k] = ndimage.convolve(current_layer, LyyKernel, mode='constant')

        # Compute Lxy using convolution
        Lxy_data[:, :, k] = ndimage.convolve(current_layer, LxyKernel, mode='constant')

    # Compute Laplacian of Gaussian (LapG = Lxx + Lyy)
    print("Computing Laplacian of Gaussian...")
    LapG_data = Lxx_data + Lyy_data

    # Apply gamma normalization and account for pixel spacing (like Igor Pro)
    for k in range(layers):
        # Calculate scale for this layer
        scale_k = DimOffset(L, 2) * (DimDelta(L, 2) ** k)

        # Gamma normalization factor
        gamma_factor = (scale_k ** gammaNorm) / (DimDelta(L, 0) * DimDelta(L, 1))

        # Apply normalization
        LapG_data[:, :, k] *= gamma_factor

    # Create LapG wave
    LapG_wave = Wave(LapG_data, "LapG")
    LapG_wave.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    LapG_wave.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    LapG_wave.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    # Fix boundaries (like Igor Pro)
    FixBoundaries(LapG_wave)

    # Compute determinant of Hessian (detH = Lxx * Lyy - Lxy^2)
    print("Computing determinant of Hessian...")
    detH_data = Lxx_data * Lyy_data - Lxy_data * Lxy_data

    # Apply gamma normalization for detH (squared normalization)
    for k in range(layers):
        # Calculate scale for this layer
        scale_k = DimOffset(L, 2) * (DimDelta(L, 2) ** k)

        # Gamma normalization factor (squared for detH)
        gamma_factor = (scale_k ** (2 * gammaNorm)) / ((DimDelta(L, 0) * DimDelta(L, 1)) ** 2)

        # Apply normalization
        detH_data[:, :, k] *= gamma_factor

    # Create detH wave
    detH_wave = Wave(detH_data, "detH")
    detH_wave.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    detH_wave.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    detH_wave.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    # Fix boundaries again (like Igor Pro)
    FixBoundaries(detH_wave)

    # Register waves globally for Igor Pro compatibility
    RegisterWave(detH_wave, "detH")
    RegisterWave(LapG_wave, "LapG")

    print("Blob detectors computation complete.")
    print(f"detH range: [{np.min(detH_data):.6f}, {np.max(detH_data):.6f}]")
    print(f"LapG range: [{np.min(LapG_data):.6f}, {np.max(LapG_data):.6f}]")

    # Return the detector waves directly for Python usage
    return detH_wave, LapG_wave


def FixBoundaries(wave):
    """
    Fix boundary artifacts in derivative computations
    Direct port from Igor Pro FixBoundaries function
    """
    # Get the data
    data = wave.data

    if len(data.shape) == 3:
        # 3D case
        height, width, layers = data.shape

        for k in range(layers):
            # Set boundary pixels to zero or use nearest neighbor
            # Top and bottom rows
            data[0, :, k] = data[1, :, k]
            data[-1, :, k] = data[-2, :, k]

            # Left and right columns
            data[:, 0, k] = data[:, 1, k]
            data[:, -1, k] = data[:, -2, k]

    elif len(data.shape) == 2:
        # 2D case
        # Top and bottom rows
        data[0, :] = data[1, :]
        data[-1, :] = data[-2, :]

        # Left and right columns
        data[:, 0] = data[:, 1]
        data[:, -1] = data[:, -2]


def GaussianKernel(sigma, truncate=4.0):
    """
    Create a Gaussian kernel for convolution

    Parameters:
    sigma : float - Standard deviation of Gaussian
    truncate : float - Truncate the kernel at this many standard deviations

    Returns:
    numpy.ndarray - 2D Gaussian kernel
    """
    # Calculate kernel size
    kernel_size = int(2 * truncate * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create coordinate arrays
    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center,
                       np.arange(kernel_size) - center)

    # Calculate Gaussian values
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    # Normalize
    kernel = kernel / np.sum(kernel)

    return kernel


def DerivativeKernels(sigma):
    """
    Create derivative kernels for Gaussian derivatives

    Parameters:
    sigma : float - Standard deviation of Gaussian

    Returns:
    tuple - (Lxx_kernel, Lyy_kernel, Lxy_kernel) derivative kernels
    """
    # Create base Gaussian kernel
    truncate = 4.0
    kernel_size = int(2 * truncate * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    center = kernel_size // 2
    x, y = np.meshgrid(np.arange(kernel_size) - center,
                       np.arange(kernel_size) - center)

    # Base Gaussian
    G = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    sigma2 = sigma * sigma
    sigma4 = sigma2 * sigma2

    # Second derivatives of Gaussian
    Lxx_kernel = G * (x * x / sigma4 - 1.0 / sigma2)
    Lyy_kernel = G * (y * y / sigma4 - 1.0 / sigma2)
    Lxy_kernel = G * (x * y / sigma4)

    return Lxx_kernel, Lyy_kernel, Lxy_kernel


def ApplyScaleNormalization(data, scale, gamma):
    """
    Apply scale normalization to derivative data

    Parameters:
    data : numpy.ndarray - Input data
    scale : float - Current scale value
    gamma : float - Normalization parameter

    Returns:
    numpy.ndarray - Normalized data
    """
    normalization_factor = scale ** gamma
    return normalization_factor * data


def OtsuThreshold(detH, L, doHoles):
    """
    Use Otsu's method to automatically define a threshold blob strength
    Direct port from Igor Pro OtsuThreshold function

    Parameters:
    detH : Wave - The determinant of Hessian blob detector
    L : Wave - The scale-space representation
    doHoles : int - 0 for maximal responses only, 1 for positive and negative extrema

    Returns:
    float - Computed threshold value
    """
    print("Computing Otsu threshold...")

    # Get the data
    detH_data = detH.data

    # For determinant of Hessian, both positive and negative blobs produce maxima
    # So we only consider positive values
    positive_values = detH_data[detH_data > 0]

    if len(positive_values) == 0:
        print("Warning: No positive detector values found")
        return 0.0

    # Simple Otsu implementation
    try:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(positive_values)
    except ImportError:
        # Fallback implementation
        threshold = np.mean(positive_values) + np.std(positive_values)

    print(f"Otsu threshold: {threshold}")
    return threshold


def Testing(string_input, number_input):
    """Testing function for scale_space module"""
    print(f"Scale-space testing: {string_input}, {number_input}")
    return len(string_input) + number_input


# Additional utility functions to match Igor Pro behavior

def ConvolveWithKernel(data, kernel, mode='constant'):
    """
    Convolve data with kernel using specified boundary conditions
    """
    return ndimage.convolve(data, kernel, mode=mode)


def NormalizeScaleSpace(data, scale_values, gamma_norm):
    """
    Apply scale normalization across all layers of scale-space
    """
    normalized_data = np.zeros_like(data)

    for k in range(data.shape[2]):
        scale_k = scale_values[k]
        normalization_factor = scale_k ** gamma_norm
        normalized_data[:, :, k] = data[:, :, k] * normalization_factor

    return normalized_data


def ComputeScaleValues(t0, t_factor, num_layers):
    """
    Compute scale values for each layer
    """
    return [t0 * (t_factor ** k) for k in range(num_layers)]