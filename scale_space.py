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
    gammaNorm : bool - Whether to apply gamma normalization

    Returns:
    tuple - (detH, LG) where detH is determinant of Hessian and LG is Laplacian of Gaussian
    """
    print("Computing blob detectors (Hessian determinant and Laplacian of Gaussian)...")

    height, width, layers = L.data.shape

    # Initialize output arrays
    detH_data = np.zeros_like(L.data)
    LG_data = np.zeros_like(L.data)

    # Process each scale layer
    for k in range(layers):
        current_layer = L.data[:, :, k]

        # Calculate current scale value from the z-scaling
        scale_value = np.exp(DimOffset(L, 2) + k * DimDelta(L, 2))

        # Compute spatial derivatives using convolution with derivative of Gaussian
        # First derivatives
        sigma = np.sqrt(scale_value)

        # Create derivative kernels
        # For efficiency, we'll use scipy's gaussian_filter with different orders

        # First derivatives
        Lx = ndimage.gaussian_filter(current_layer, sigma, order=[0, 1])
        Ly = ndimage.gaussian_filter(current_layer, sigma, order=[1, 0])

        # Second derivatives
        Lxx = ndimage.gaussian_filter(current_layer, sigma, order=[0, 2])
        Lyy = ndimage.gaussian_filter(current_layer, sigma, order=[2, 0])
        Lxy = ndimage.gaussian_filter(current_layer, sigma, order=[1, 1])

        # Compute Hessian determinant
        # detH = Lxx * Lyy - Lxy²
        detH_layer = Lxx * Lyy - Lxy ** 2

        # Compute Laplacian of Gaussian
        # LG = Lxx + Lyy
        LG_layer = Lxx + Lyy

        # Apply gamma normalization if requested
        if gammaNorm:
            # Normalize by scale^gamma where gamma=2 for detH and gamma=1 for LG
            gamma_detH = 2.0
            gamma_LG = 1.0

            norm_factor_detH = scale_value ** gamma_detH
            norm_factor_LG = scale_value ** gamma_LG

            detH_layer *= norm_factor_detH
            LG_layer *= norm_factor_LG

        # Store results
        detH_data[:, :, k] = detH_layer
        LG_data[:, :, k] = LG_layer

        if k % 50 == 0:  # Progress indicator
            print(f"  Processed scale layer {k + 1}/{layers}")

    # Create output waves with proper scaling
    detH = Wave(detH_data, f"{L.name}_detH")
    detH.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    detH.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    detH.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    LG = Wave(LG_data, f"{L.name}_LG")
    LG.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    LG.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    LG.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    print("Blob detectors computation completed.")
    return detH, LG


def ComputeHessianMatrix(layer, x_idx, y_idx, sigma):
    """
    Compute the Hessian matrix at a specific point
    Direct port from Igor Pro implementation

    Parameters:
    layer : ndarray - 2D array representing a scale layer
    x_idx, y_idx : int - Pixel coordinates
    sigma : float - Scale parameter

    Returns:
    ndarray - 2x2 Hessian matrix
    """
    # Get neighborhood around the point
    if (x_idx < 1 or x_idx >= layer.shape[1] - 1 or
            y_idx < 1 or y_idx >= layer.shape[0] - 1):
        return np.zeros((2, 2))

    # Compute second derivatives using finite differences
    # Second derivative in x direction
    Lxx = layer[y_idx, x_idx + 1] - 2 * layer[y_idx, x_idx] + layer[y_idx, x_idx - 1]

    # Second derivative in y direction
    Lyy = layer[y_idx + 1, x_idx] - 2 * layer[y_idx, x_idx] + layer[y_idx - 1, x_idx]

    # Mixed derivative
    Lxy = (layer[y_idx + 1, x_idx + 1] - layer[y_idx + 1, x_idx - 1] -
           layer[y_idx - 1, x_idx + 1] + layer[y_idx - 1, x_idx - 1]) / 4

    # Construct Hessian matrix
    H = np.array([[Lxx, Lxy],
                  [Lxy, Lyy]])

    return H


def ComputeGradient(layer, x_idx, y_idx):
    """
    Compute the gradient at a specific point
    Direct port from Igor Pro implementation

    Parameters:
    layer : ndarray - 2D array representing a scale layer
    x_idx, y_idx : int - Pixel coordinates

    Returns:
    ndarray - 2D gradient vector
    """
    if (x_idx < 1 or x_idx >= layer.shape[1] - 1 or
            y_idx < 1 or y_idx >= layer.shape[0] - 1):
        return np.zeros(2)

    # Compute gradients using central differences
    Lx = (layer[y_idx, x_idx + 1] - layer[y_idx, x_idx - 1]) / 2
    Ly = (layer[y_idx + 1, x_idx] - layer[y_idx - 1, x_idx]) / 2

    return np.array([Lx, Ly])


def LocalMaxima3D(detH, threshold=0):
    """
    Find local maxima in 3D detector response
    Direct port from Igor Pro LocalMaxima3D function

    Parameters:
    detH : Wave - 3D determinant of Hessian detector
    threshold : float - Minimum threshold for detection

    Returns:
    list - List of (i, j, k) coordinates of local maxima
    """
    print("Finding local maxima in 3D detector response...")

    maxima = []
    height, width, layers = detH.data.shape

    # Check each point for local maximum
    for k in range(1, layers - 1):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                center_val = detH.data[i, j, k]

                # Skip if below threshold
                if center_val <= threshold:
                    continue

                # Check if it's a local maximum in 3x3x3 neighborhood
                is_maximum = True
                for dk in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if dk == 0 and di == 0 and dj == 0:
                                continue
                            if detH.data[i + di, j + dj, k + dk] >= center_val:
                                is_maximum = False
                                break
                        if not is_maximum:
                            break
                    if not is_maximum:
                        break

                if is_maximum:
                    maxima.append((i, j, k))

    print(f"Found {len(maxima)} local maxima")
    return maxima


def NonMaximumSuppression(detH, LG, maxima, maxCurvatureRatio):
    """
    Apply non-maximum suppression and curvature ratio test
    Direct port from Igor Pro NonMaximumSuppression function

    Parameters:
    detH : Wave - Determinant of Hessian detector
    LG : Wave - Laplacian of Gaussian detector
    maxima : list - List of maxima coordinates
    maxCurvatureRatio : float - Maximum allowed curvature ratio

    Returns:
    list - Filtered list of maxima
    """
    print("Applying non-maximum suppression and curvature test...")

    filtered_maxima = []

    for i, j, k in maxima:
        # Get detector values
        detH_val = detH.data[i, j, k]
        LG_val = LG.data[i, j, k]

        # Apply curvature ratio test
        # The test is: LG² / detH < (r+1)² / r where r = maxCurvatureRatio
        if abs(detH_val) > 1e-10:  # Avoid division by zero
            curvature_ratio = (LG_val ** 2) / abs(detH_val)
            max_allowed_ratio = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio

            if curvature_ratio < max_allowed_ratio:
                filtered_maxima.append((i, j, k))

    print(f"Retained {len(filtered_maxima)} maxima after filtering")
    return filtered_maxima


def RefineMaxima(detH, maxima):
    """
    Refine maxima positions to sub-pixel accuracy
    Direct port from Igor Pro RefineMaxima function

    Parameters:
    detH : Wave - Determinant of Hessian detector
    maxima : list - List of maxima coordinates

    Returns:
    list - List of refined maxima with sub-pixel coordinates
    """
    print("Refining maxima to sub-pixel accuracy...")

    refined_maxima = []

    for i, j, k in maxima:
        try:
            # Use 3D quadratic interpolation to refine position
            # Get 3x3x3 neighborhood
            if (i >= 1 and i < detH.data.shape[0] - 1 and
                    j >= 1 and j < detH.data.shape[1] - 1 and
                    k >= 1 and k < detH.data.shape[2] - 1):

                neighborhood = detH.data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]

                # Compute gradients and Hessian for refinement
                # First derivatives (gradients)
                dx = (neighborhood[1, 2, 1] - neighborhood[1, 0, 1]) / 2
                dy = (neighborhood[2, 1, 1] - neighborhood[0, 1, 1]) / 2
                dz = (neighborhood[1, 1, 2] - neighborhood[1, 1, 0]) / 2

                # Second derivatives (Hessian)
                dxx = neighborhood[1, 2, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 0, 1]
                dyy = neighborhood[2, 1, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[0, 1, 1]
                dzz = neighborhood[1, 1, 2] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 1, 0]

                # Mixed derivatives
                dxy = (neighborhood[2, 2, 1] - neighborhood[2, 0, 1] -
                       neighborhood[0, 2, 1] + neighborhood[0, 0, 1]) / 4
                dxz = (neighborhood[1, 2, 2] - neighborhood[1, 2, 0] -
                       neighborhood[1, 0, 2] + neighborhood[1, 0, 0]) / 4
                dyz = (neighborhood[2, 1, 2] - neighborhood[2, 1, 0] -
                       neighborhood[0, 1, 2] + neighborhood[0, 1, 0]) / 4

                # Construct Hessian matrix
                H = np.array([[dxx, dxy, dxz],
                              [dxy, dyy, dyz],
                              [dxz, dyz, dzz]])

                gradient = np.array([dx, dy, dz])

                # Solve for offset: H * offset = -gradient
                try:
                    offset = np.linalg.solve(H, -gradient)

                    # Limit offset to reasonable range
                    offset = np.clip(offset, -0.5, 0.5)

                    # Calculate refined position
                    refined_j = j + offset[0]  # x direction
                    refined_i = i + offset[1]  # y direction
                    refined_k = k + offset[2]  # z direction

                    # Calculate refined value
                    refined_value = (neighborhood[1, 1, 1] +
                                     0.5 * np.dot(gradient, offset))

                    refined_maxima.append((refined_i, refined_j, refined_k, refined_value))

                except np.linalg.LinAlgError:
                    # If Hessian is singular, use original position
                    refined_maxima.append((i, j, k, detH.data[i, j, k]))
            else:
                # Edge case - use original position
                refined_maxima.append((i, j, k, detH.data[i, j, k]))

        except Exception as e:
            # Fallback to original position
            refined_maxima.append((i, j, k, detH.data[i, j, k]))

    print(f"Refined {len(refined_maxima)} maxima")
    return refined_maxima


def ConvertToRealCoordinates(refined_maxima, im, detH):
    """
    Convert pixel coordinates to real-world coordinates
    Direct port from Igor Pro coordinate conversion

    Parameters:
    refined_maxima : list - List of refined maxima
    im : Wave - Original image for coordinate system
    detH : Wave - Detector for scale coordinate system

    Returns:
    list - List of maxima with real-world coordinates
    """
    real_coord_maxima = []

    for i, j, k, value in refined_maxima:
        # Convert to real coordinates
        x_real = DimOffset(im, 0) + j * DimDelta(im, 0)
        y_real = DimOffset(im, 1) + i * DimDelta(im, 1)
        scale_real = np.exp(DimOffset(detH, 2) + k * DimDelta(detH, 2))

        real_coord_maxima.append((x_real, y_real, scale_real, value, i, j, k))

    return real_coord_maxima


def Testing(string_input, number_input):
    """
    Testing function for scale-space operations
    Direct port from Igor Pro Testing function
    """
    print(f"Scale-space testing function called:")
    print(f"  String input: '{string_input}'")
    print(f"  Number input: {number_input}")

    # Create a simple test image
    test_size = 64
    test_data = np.zeros((test_size, test_size))

    # Add some Gaussian blobs for testing
    center = test_size // 2
    for i in range(test_size):
        for j in range(test_size):
            r1 = np.sqrt((i - center + 10) ** 2 + (j - center) ** 2)
            r2 = np.sqrt((i - center - 10) ** 2 + (j - center) ** 2)
            test_data[i, j] = np.exp(-r1 ** 2 / 50) + 0.5 * np.exp(-r2 ** 2 / 20)

    # Create test wave
    test_wave = Wave(test_data, "TestImage")
    test_wave.SetScale('x', 0, 1.0)
    test_wave.SetScale('y', 0, 1.0)

    print(f"  Created test image with shape: {test_wave.data.shape}")

    # Test scale-space representation
    L = ScaleSpaceRepresentation(test_wave, 5, 1.0, 1.5)
    print(f"  Scale-space shape: {L.data.shape}")

    # Test blob detectors
    detH, LG = BlobDetectors(L, True)
    print(f"  Detector shapes: detH={detH.data.shape}, LG={LG.data.shape}")

    # Find maxima
    maxima = LocalMaxima3D(detH, threshold=0.001)
    print(f"  Found {len(maxima)} local maxima")

    result = len(string_input) + number_input + len(maxima)
    print(f"  Test result: {result}")

    return result