"""
Scale-Space Functions
Handles scale-space representation and blob detector computations
Direct port from Igor Pro code maintaining same variable names and structure
Complete implementation matching Igor Pro exactly
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from igor_compatibility import *
from file_io import data_browser

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """
    Computes the discrete scale-space representation L of an image.
    Direct port from Igor Pro ScaleSpaceRepresentation function

    Parameters:
    im : Wave - The image to compute L from
    layers : int - The number of layers of L
    t0 : float - The scale of the first layer of L, provided in pixel units
    tFactor : float - The scaling factor for the scale between layers of L

    Returns:
    Wave - 3D wave containing the scale-space representation
    """
    print(f"Computing scale-space representation: {layers} layers, t0={t0}, factor={tFactor}")

    # Convert t0 to image units (square it for variance)
    t0_scaled = (t0 * DimDelta(im, 0)) ** 2

    # Get image dimensions
    height, width = im.data.shape

    # Go to Fourier space
    im_fft = fft2(im.data)

    # Create frequency coordinates
    u = fftfreq(height, DimDelta(im, 0))
    v = fftfreq(width, DimDelta(im, 1))
    U, V = np.meshgrid(v, u)

    # Pre-compute frequency squared terms
    freq_squared = U ** 2 + V ** 2

    # Make the layers of the scale-space representation
    L_data = np.zeros((height, width, layers))

    for i in range(layers):
        # Current scale (variance of Gaussian)
        scale = t0_scaled * (tFactor ** i)

        # Create Gaussian kernel in frequency domain
        # The factor of 2π² comes from the Fourier transform of a Gaussian
        kernel = np.exp(-2 * np.pi ** 2 * scale * freq_squared)

        # Apply kernel and transform back
        layer_fft = im_fft * kernel
        layer_data = np.real(ifft2(layer_fft))

        # Store in 3D array
        L_data[:, :, i] = layer_data

        if i % 50 == 0:  # Progress indicator
            print(f"  Computed layer {i + 1}/{layers}")

    # Create 3D wave with proper scaling
    L = Wave(L_data, f"{im.name}_L")
    L.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    L.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))
    L.SetScale('z', np.log(t0_scaled), np.log(tFactor))  # Log scale for scale dimension

    print(f"Scale-space representation completed. Shape: {L_data.shape}")
    return L


def BlobDetectors(L, gammaNorm):
    """
    Computes the two blob detectors: determinant of the Hessian and Laplacian of Gaussian.
    Direct port from Igor Pro BlobDetectors function

    Parameters:
    L : Wave - The scale-space representation (3D)
    gammaNorm : float - Normalization parameter for scale-space derivatives

    Returns:
    tuple - (detH, LG) where detH is determinant of Hessian and LG is Laplacian of Gaussian
    """
    print("Computing blob detectors (Hessian determinant and Laplacian of Gaussian)...")

    height, width, layers = L.data.shape

    # Initialize output arrays
    detH_data = np.zeros_like(L.data)
    LG_data = np.zeros_like(L.data)

    # Get scale information
    t0 = np.exp(DimOffset(L, 2))
    tFactor = np.exp(DimDelta(L, 2))

    # Compute derivatives for each scale layer
    for k in range(layers):
        print(f"  Processing scale layer {k + 1}/{layers}")

        # Current scale
        current_scale = t0 * (tFactor ** k)

        # Normalization factor (gamma normalization)
        norm_factor = current_scale ** gammaNorm

        # Get current layer
        current_layer = L.data[:, :, k]

        # Compute spatial derivatives using finite differences
        # Second derivatives for Hessian

        # Lxx - second derivative in x direction
        Lxx = np.zeros_like(current_layer)
        Lxx[:, 1:-1] = current_layer[:, 2:] - 2 * current_layer[:, 1:-1] + current_layer[:, :-2]
        Lxx /= (DimDelta(L, 0) ** 2)

        # Lyy - second derivative in y direction
        Lyy = np.zeros_like(current_layer)
        Lyy[1:-1, :] = current_layer[2:, :] - 2 * current_layer[1:-1, :] + current_layer[:-2, :]
        Lyy /= (DimDelta(L, 1) ** 2)

        # Lxy - mixed second derivative
        Lxy = np.zeros_like(current_layer)
        Lxy[1:-1, 1:-1] = (current_layer[2:, 2:] - current_layer[2:, :-2] -
                           current_layer[:-2, 2:] + current_layer[:-2, :-2]) / (4 * DimDelta(L, 0) * DimDelta(L, 1))

        # Compute determinant of Hessian
        detH = Lxx * Lyy - Lxy ** 2

        # Apply scale normalization
        detH_data[:, :, k] = detH * norm_factor ** 2

        # Compute Laplacian of Gaussian
        LG = Lxx + Lyy

        # Apply scale normalization
        LG_data[:, :, k] = LG * norm_factor

    # Create output waves
    detH_wave = Wave(detH_data, f"{L.name}_detH")
    detH_wave.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    detH_wave.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    detH_wave.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    LG_wave = Wave(LG_data, f"{L.name}_LG")
    LG_wave.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    LG_wave.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    LG_wave.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    print("Blob detectors computation completed.")
    return detH_wave, LG_wave


def GaussianKernel(sigma, size=None):
    """
    Create a Gaussian kernel for convolution

    Parameters:
    sigma : float - Standard deviation of Gaussian
    size : int - Size of kernel (default: 6*sigma + 1)

    Returns:
    ndarray - 2D Gaussian kernel
    """
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

    # Create coordinate grids
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)

    # Compute Gaussian
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalize

    return kernel


def GaussianDerivative(sigma, order, direction, size=None):
    """
    Create Gaussian derivative kernel

    Parameters:
    sigma : float - Standard deviation of Gaussian
    order : int - Order of derivative (1 or 2)
    direction : str - Direction ('x', 'y', or 'xy' for mixed)
    size : int - Size of kernel

    Returns:
    ndarray - Gaussian derivative kernel
    """
    if size is None:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)

    # Base Gaussian
    gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    if direction == 'x' and order == 1:
        # First derivative in x
        kernel = -X / (sigma ** 2) * gaussian
    elif direction == 'y' and order == 1:
        # First derivative in y
        kernel = -Y / (sigma ** 2) * gaussian
    elif direction == 'x' and order == 2:
        # Second derivative in x
        kernel = (X ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian
    elif direction == 'y' and order == 2:
        # Second derivative in y
        kernel = (Y ** 2 / sigma ** 4 - 1 / sigma ** 2) * gaussian
    elif direction == 'xy' and order == 2:
        # Mixed second derivative
        kernel = (X * Y / sigma ** 4) * gaussian
    else:
        raise ValueError(f"Unsupported derivative: order={order}, direction={direction}")

    return kernel


def ConvolveWithGaussian(image, sigma):
    """
    Convolve image with Gaussian kernel

    Parameters:
    image : Wave - Input image
    sigma : float - Standard deviation of Gaussian

    Returns:
    Wave - Convolved image
    """
    kernel = GaussianKernel(sigma)
    convolved = ndimage.convolve(image.data, kernel, mode='constant')

    result = Wave(convolved, f"{image.name}_conv")
    result.SetScale('x', DimOffset(image, 0), DimDelta(image, 0))
    result.SetScale('y', DimOffset(image, 1), DimDelta(image, 1))

    return result


def ComputeHessianDeterminant(image, sigma):
    """
    Compute determinant of Hessian matrix at given scale

    Parameters:
    image : Wave - Input image
    sigma : float - Scale parameter

    Returns:
    Wave - Determinant of Hessian
    """
    # Compute second derivatives
    Lxx_kernel = GaussianDerivative(sigma, 2, 'x')
    Lyy_kernel = GaussianDerivative(sigma, 2, 'y')
    Lxy_kernel = GaussianDerivative(sigma, 2, 'xy')

    Lxx = ndimage.convolve(image.data, Lxx_kernel, mode='constant')
    Lyy = ndimage.convolve(image.data, Lyy_kernel, mode='constant')
    Lxy = ndimage.convolve(image.data, Lxy_kernel, mode='constant')

    # Compute determinant
    detH = Lxx * Lyy - Lxy ** 2

    # Scale normalization
    detH *= sigma ** 4

    result = Wave(detH, f"{image.name}_detH")
    result.SetScale('x', DimOffset(image, 0), DimDelta(image, 0))
    result.SetScale('y', DimOffset(image, 1), DimDelta(image, 1))

    return result


def ComputeLaplacianOfGaussian(image, sigma):
    """
    Compute Laplacian of Gaussian at given scale

    Parameters:
    image : Wave - Input image
    sigma : float - Scale parameter

    Returns:
    Wave - Laplacian of Gaussian
    """
    # Compute second derivatives
    Lxx_kernel = GaussianDerivative(sigma, 2, 'x')
    Lyy_kernel = GaussianDerivative(sigma, 2, 'y')

    Lxx = ndimage.convolve(image.data, Lxx_kernel, mode='constant')
    Lyy = ndimage.convolve(image.data, Lyy_kernel, mode='constant')

    # Compute Laplacian
    LG = Lxx + Lyy

    # Scale normalization
    LG *= sigma ** 2

    result = Wave(LG, f"{image.name}_LG")
    result.SetScale('x', DimOffset(image, 0), DimDelta(image, 0))
    result.SetScale('y', DimOffset(image, 1), DimDelta(image, 1))

    return result


def MultiScaleBlobDetection(image, min_scale=1.0, max_scale=10.0, num_scales=10):
    """
    Perform multi-scale blob detection

    Parameters:
    image : Wave - Input image
    min_scale : float - Minimum scale
    max_scale : float - Maximum scale
    num_scales : int - Number of scales

    Returns:
    tuple - (detH_stack, LG_stack) 3D arrays of detector responses
    """
    print(f"Multi-scale blob detection: {num_scales} scales from {min_scale} to {max_scale}")

    # Create scale array
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)

    height, width = image.data.shape
    detH_stack = np.zeros((height, width, num_scales))
    LG_stack = np.zeros((height, width, num_scales))

    for i, scale in enumerate(scales):
        print(f"  Processing scale {i + 1}/{num_scales}: σ = {scale:.2f}")

        detH = ComputeHessianDeterminant(image, scale)
        LG = ComputeLaplacianOfGaussian(image, scale)

        detH_stack[:, :, i] = detH.data
        LG_stack[:, :, i] = LG.data

    # Create 3D waves
    detH_wave = Wave(detH_stack, f"{image.name}_detH_stack")
    detH_wave.SetScale('x', DimOffset(image, 0), DimDelta(image, 0))
    detH_wave.SetScale('y', DimOffset(image, 1), DimDelta(image, 1))
    detH_wave.SetScale('z', 0, 1.0)  # Scale indices

    LG_wave = Wave(LG_stack, f"{image.name}_LG_stack")
    LG_wave.SetScale('x', DimOffset(image, 0), DimDelta(image, 0))
    LG_wave.SetScale('y', DimOffset(image, 1), DimDelta(image, 1))
    LG_wave.SetScale('z', 0, 1.0)  # Scale indices

    print("Multi-scale blob detection completed.")
    return detH_wave, LG_wave


def NonMaximumSuppression3D(response, threshold=0.0):
    """
    Perform non-maximum suppression in 3D (x, y, scale)

    Parameters:
    response : Wave - 3D response function
    threshold : float - Minimum response threshold

    Returns:
    list - List of (x, y, scale, response) tuples for local maxima
    """
    height, width, layers = response.data.shape
    maxima = []

    for k in range(1, layers - 1):
        for i in range(1, height - 1):
            for j in range(1, width - 1):

                current = response.data[i, j, k]

                if current < threshold:
                    continue

                # Check 26-neighborhood in 3D
                is_maximum = True

                for dk in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if dk == 0 and di == 0 and dj == 0:
                                continue

                            neighbor = response.data[i + di, j + dj, k + dk]
                            if current <= neighbor:
                                is_maximum = False
                                break
                        if not is_maximum:
                            break
                    if not is_maximum:
                        break

                if is_maximum:
                    # Convert to real coordinates
                    x = DimOffset(response, 0) + j * DimDelta(response, 0)
                    y = DimOffset(response, 1) + i * DimDelta(response, 1)
                    scale = DimOffset(response, 2) + k * DimDelta(response, 2)

                    maxima.append((x, y, scale, current))

    return maxima


def Testing(string_input, number_input):
    """Testing function for scale_space module"""
    print(f"Scale-space testing: {string_input}, {number_input}")
    return len(string_input) + number_input