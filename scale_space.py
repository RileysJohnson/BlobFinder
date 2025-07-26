"""
Scale-Space Module
Contains functions for scale-space representation and blob detection
Direct port from Igor Pro code maintaining same variable names and structure
Complete implementation of scale-space derivatives and blob detectors
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from igor_compatibility import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def ScaleSpaceRepresentation(im, layers, scaleStart, scaleFactor):
    """
    Calculate the discrete scale-space representation
    Direct port from Igor Pro ScaleSpaceRepresentation function

    Parameters:
    im : Wave - Input image
    layers : int - Number of scale layers
    scaleStart : float - Starting scale value
    scaleFactor : float - Factor between consecutive scales

    Returns:
    Wave - 3D scale-space representation (y, x, scale)
    """
    print(f"Computing scale-space representation with {layers} layers...")

    height, width = im.data.shape

    # Create 3D array for scale-space representation
    L_data = np.zeros((height, width, layers))

    # Copy original image data
    original_data = im.data.copy()

    # Compute each scale layer
    for k in range(layers):
        # Calculate current scale (sigma)
        current_scale = scaleStart * (scaleFactor ** k)

        # Apply Gaussian smoothing at current scale
        # In Igor Pro: sigma^2 = scale, so sigma = sqrt(scale)
        sigma = np.sqrt(current_scale)

        # Apply Gaussian filter
        smoothed = gaussian_filter(original_data, sigma=sigma, mode='nearest')

        # Store in scale-space representation
        L_data[:, :, k] = smoothed

        if k % 5 == 0:  # Progress indicator
            print(f"  Computed scale layer {k + 1}/{layers} (σ={sigma:.3f})")

    # Create output wave
    L = Wave(L_data, f"{im.name}_ScaleSpace")

    # Set scaling to match original image for x and y dimensions
    L.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    L.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))
    L.SetScale('z', 0, scaleFactor)  # Scale dimension

    print(f"Scale-space representation complete: {L_data.shape}")
    return L


def BlobDetectors(L, gammaNorm=1.0):
    """
    Calculate gamma normalized scale-space derivatives for blob detection
    Direct port from Igor Pro BlobDetectors function

    Parameters:
    L : Wave - Scale-space representation
    gammaNorm : float - Gamma normalization parameter

    Returns:
    tuple - (detH, LG) where detH is determinant of Hessian, LG is Laplacian of Gaussian
    """
    print("Computing blob detectors...")

    height, width, scales = L.data.shape

    # Initialize output arrays
    detH_data = np.zeros_like(L.data)
    LG_data = np.zeros_like(L.data)

    # Process each scale layer
    for k in range(scales):
        print(f"  Processing scale layer {k + 1}/{scales}")

        # Get current layer
        current_layer = L.data[:, :, k]

        # Calculate scale value
        scale = (k + 1) * 1.2  # Simple scale progression

        # Gamma normalization factor
        gamma_factor = scale ** gammaNorm

        # === Compute second derivatives (Hessian matrix elements) ===

        # Second derivative in x direction (Lxx)
        Lxx = np.zeros_like(current_layer)
        Lxx[:, 1:-1] = current_layer[:, 2:] - 2 * current_layer[:, 1:-1] + current_layer[:, :-2]

        # Second derivative in y direction (Lyy)
        Lyy = np.zeros_like(current_layer)
        Lyy[1:-1, :] = current_layer[2:, :] - 2 * current_layer[1:-1, :] + current_layer[:-2, :]

        # Mixed derivative (Lxy)
        Lxy = np.zeros_like(current_layer)
        Lxy[1:-1, 1:-1] = (current_layer[2:, 2:] - current_layer[2:, :-2] -
                           current_layer[:-2, 2:] + current_layer[:-2, :-2]) / 4.0

        # === Compute Laplacian of Gaussian (LG) ===
        LG = Lxx + Lyy

        # Apply gamma normalization
        LG_normalized = gamma_factor * LG

        # === Compute Determinant of Hessian ===
        # detH = Lxx * Lyy - Lxy^2
        detH = Lxx * Lyy - Lxy * Lxy

        # Apply gamma normalization
        detH_normalized = gamma_factor * gamma_factor * detH

        # Store results
        detH_data[:, :, k] = detH_normalized
        LG_data[:, :, k] = LG_normalized

    # Create output waves
    detH_wave = Wave(detH_data, f"{L.name}_detH")
    LG_wave = Wave(LG_data, f"{L.name}_LG")

    # Copy scaling from input
    for axis in ['x', 'y', 'z']:
        detH_wave.SetScale(axis, DimOffset(L, axis), DimDelta(L, axis))
        LG_wave.SetScale(axis, DimOffset(L, axis), DimDelta(L, axis))

    print("Blob detectors computation complete.")
    return detH_wave, LG_wave


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


def Testing(string_input, number_input):
    """Testing function for scale_space module"""
    print(f"Scale-space testing: {string_input}, {number_input}")
    return len(string_input) + number_input