"""
Scale-Space Functions
Handles scale-space representation and blob detector computations
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete implementations
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

    Parameters:
    L : Wave - The scale-space representation of the image
    gammaNorm : float - The gamma normalization factor (should be 1 for most cases)

    Returns:
    int - 0 for success (results stored in global data_browser)
    """
    print("Computing blob detectors...")

    # Get dimensions
    height, width, layers = L.data.shape

    # Make convolution kernels for calculating central difference derivatives
    # Second derivatives in x and y directions
    d2dx2_kernel = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    d2dy2_kernel = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    d2dxdy_kernel = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4

    # Initialize output arrays
    detH_data = np.zeros_like(L.data)
    LapG_data = np.zeros_like(L.data)

    # Process each layer
    for k in range(layers):
        layer = L.data[:, :, k]

        # Compute second derivatives using convolution
        Lxx = ndimage.convolve(layer, d2dx2_kernel, mode='constant', cval=0.0)
        Lyy = ndimage.convolve(layer, d2dy2_kernel, mode='constant', cval=0.0)
        Lxy = ndimage.convolve(layer, d2dxdy_kernel, mode='constant', cval=0.0)

        # Compute determinant of Hessian
        detH_data[:, :, k] = Lxx * Lyy - Lxy ** 2

        # Compute Laplacian of Gaussian (trace of Hessian)
        LapG_data[:, :, k] = Lxx + Lyy

        if k % 50 == 0:  # Progress indicator
            print(f"  Processed layer {k + 1}/{layers}")

    # Create Wave objects
    detH = Wave(detH_data, "detH")
    LapG = Wave(LapG_data, "LapG")

    # Set scaling to match input
    detH.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    detH.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    detH.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    LapG.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    LapG.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    LapG.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    # Apply gamma normalization and pixel spacing correction
    print("Applying gamma normalization...")

    for k in range(layers):
        # Get the scale for this layer
        scale_value = DimOffset(L, 2) + k * DimDelta(L, 2)
        if DimDelta(L, 2) != 0:
            # Convert from log scale back to linear scale
            actual_scale = np.exp(scale_value)
        else:
            actual_scale = scale_value

        # Apply gamma normalization
        # The scale factor is t^(2*gamma) where gamma is the normalization parameter
        scale_factor = actual_scale ** (2 * gammaNorm)

        # Account for pixel spacing (converts derivatives to image units)
        pixel_factor = (DimDelta(L, 0) * DimDelta(L, 1)) ** 2

        # Apply normalization
        normalization = scale_factor / pixel_factor
        detH_data[:, :, k] *= normalization
        LapG_data[:, :, k] *= (actual_scale ** gammaNorm) / (DimDelta(L, 0) * DimDelta(L, 1))

    # Fix boundary issues
    print("Fixing boundary conditions...")
    FixBoundaries(detH)
    FixBoundaries(LapG)

    # Store results in global data browser (simulating Igor's global storage)
    global data_browser
    data_browser.add_wave(detH, "detH")
    data_browser.add_wave(LapG, "LapG")

    print("Blob detectors computation completed")
    return 0


def FixBoundaries(wave):
    """
    Fixes boundary issues in the blob detectors.
    Arises from trying to measure derivatives on the boundary.

    Parameters:
    wave : Wave - The detector wave to fix (detH or LapG)
    """
    data = wave.data
    height, width = data.shape[:2]

    # Handle 2D or 3D data
    if len(data.shape) == 3:
        layers = data.shape[2]

        for k in range(layers):
            layer = data[:, :, k]

            # Make edges fade to zero
            fade_width = 2

            # Top and bottom edges
            for i in range(fade_width):
                factor = i / fade_width
                layer[i, :] *= factor
                layer[-(i + 1), :] *= factor

            # Left and right edges
            for j in range(fade_width):
                factor = j / fade_width
                layer[:, j] *= factor
                layer[:, -(j + 1)] *= factor

            data[:, :, k] = layer
    else:
        # 2D case
        fade_width = 2

        # Top and bottom edges
        for i in range(fade_width):
            factor = i / fade_width
            data[i, :] *= factor
            data[-(i + 1), :] *= factor

        # Left and right edges
        for j in range(fade_width):
            factor = j / fade_width
            data[:, j] *= factor
            data[:, -(j + 1)] *= factor


def ComputeMaxes(detH, LG, particleType, maxCurvatureRatio, Map=None, ScaleMap=None):
    """
    Compute local maxima in the blob detector response.

    Parameters:
    detH : Wave - Determinant of Hessian detector
    LG : Wave - Laplacian of Gaussian detector
    particleType : int - Type of particles to detect (-1, 0, or 1)
    maxCurvatureRatio : float - Maximum curvature ratio allowed
    Map : Wave - Output map of detector values at maxima (optional)
    ScaleMap : Wave - Output map of scales at maxima (optional)

    Returns:
    Wave - Map of maximum detector values
    """
    height, width, layers = detH.data.shape

    if Map is None:
        Map = Wave(np.full((height, width), -1.0), "MaxMap")
    else:
        Map.data = np.full((height, width), -1.0)

    if ScaleMap is None:
        ScaleMap = Wave(np.zeros((height, width)), "ScaleMap")
    else:
        ScaleMap.data = np.zeros((height, width))

    # Set scaling
    Map.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    Map.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))
    ScaleMap.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    ScaleMap.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))

    # Find local maxima
    for k in range(1, layers - 1):  # Skip first and last layers
        for i in range(1, height - 1):  # Skip edges
            for j in range(1, width - 1):

                current_val = detH.data[i, j, k]

                # Check particle type
                if particleType == 1 and current_val <= 0:
                    continue
                elif particleType == -1 and current_val >= 0:
                    continue

                # Skip if below current maximum at this location
                if current_val <= Map.data[i, j]:
                    continue

                # Check curvature ratio constraint
                if abs(current_val) > 1e-10:  # Avoid division by zero
                    curvature_ratio = LG.data[i, j, k] ** 2 / abs(current_val)
                    max_allowed = ((maxCurvatureRatio + 1) / maxCurvatureRatio) ** 2
                    if curvature_ratio >= max_allowed:
                        continue

                # Check if it's a local maximum in 3D neighborhood
                is_maximum = True
                for dk in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if dk == 0 and di == 0 and dj == 0:
                                continue

                            neighbor_val = detH.data[i + di, j + dj, k + dk]

                            if particleType == 1:
                                if neighbor_val >= current_val:
                                    is_maximum = False
                                    break
                            elif particleType == -1:
                                if neighbor_val <= current_val:
                                    is_maximum = False
                                    break
                            else:  # particleType == 0
                                if abs(neighbor_val) >= abs(current_val):
                                    is_maximum = False
                                    break

                        if not is_maximum:
                            break
                    if not is_maximum:
                        break

                if is_maximum:
                    Map.data[i, j] = current_val
                    ScaleMap.data[i, j] = k

    return Map


def GetScaleFromLayer(L, layer_index):
    """
    Get the actual scale value from a layer index

    Parameters:
    L : Wave - Scale-space representation
    layer_index : int - Index of the layer

    Returns:
    float - The scale value
    """
    if layer_index < 0 or layer_index >= L.data.shape[2]:
        return 0.0

    scale_offset = DimOffset(L, 2)
    scale_delta = DimDelta(L, 2)

    if scale_delta != 0:
        # Log scale
        log_scale = scale_offset + layer_index * scale_delta
        return np.exp(log_scale)
    else:
        # Linear scale
        return scale_offset + layer_index * scale_delta


def GetBlobRadius(scale):
    """
    Calculate the characteristic radius of a blob at a given scale

    Parameters:
    scale : float - The scale value

    Returns:
    float - The blob radius
    """
    # For a Gaussian blob, the characteristic radius is sqrt(2*scale)
    return np.sqrt(2 * scale)


def NormalizeDetectorResponse(detH, L, gammaNorm=1):
    """
    Apply proper normalization to detector response

    Parameters:
    detH : Wave - Determinant of Hessian detector
    L : Wave - Scale-space representation
    gammaNorm : float - Gamma normalization parameter
    """
    layers = detH.data.shape[2]

    for k in range(layers):
        scale = GetScaleFromLayer(L, k)

        # Apply gamma normalization
        scale_factor = scale ** (2 * gammaNorm)

        # Account for pixel spacing
        pixel_factor = (DimDelta(L, 0) * DimDelta(L, 1)) ** 2

        # Apply normalization
        normalization = scale_factor / pixel_factor
        detH.data[:, :, k] *= normalization


def ScaleSpaceFilter(im, scale, filter_type='gaussian'):
    """
    Apply a scale-space filter to an image

    Parameters:
    im : Wave - Input image
    scale : float - Scale parameter
    filter_type : str - Type of filter ('gaussian', 'laplacian')

    Returns:
    Wave - Filtered image
    """
    # Convert scale to image units
    scale_squared = (scale * DimDelta(im, 0)) ** 2

    # Go to Fourier space
    im_fft = fft2(im.data)

    # Create frequency coordinates
    height, width = im.data.shape
    u = fftfreq(height, DimDelta(im, 0))
    v = fftfreq(width, DimDelta(im, 1))
    U, V = np.meshgrid(v, u)
    freq_squared = U ** 2 + V ** 2

    if filter_type == 'gaussian':
        # Gaussian filter
        kernel = np.exp(-2 * np.pi ** 2 * scale_squared * freq_squared)
    elif filter_type == 'laplacian':
        # Laplacian of Gaussian filter
        kernel = -4 * np.pi ** 2 * freq_squared * np.exp(-2 * np.pi ** 2 * scale_squared * freq_squared)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Apply filter and transform back
    filtered_fft = im_fft * kernel
    filtered_data = np.real(ifft2(filtered_fft))

    # Create output wave
    result = Wave(filtered_data, f"{im.name}_filtered")
    result.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    result.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    return result


def TestScaleSpace():
    """
    Test function for scale-space computation
    """
    print("Testing scale-space computation...")

    # Create a test image with a Gaussian blob
    size = 64
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)

    # Gaussian blob
    sigma = 5
    test_data = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    test_image = Wave(test_data, "TestImage")
    test_image.SetScale('x', -size // 2, 1)
    test_image.SetScale('y', -size // 2, 1)

    # Compute scale-space representation
    L = ScaleSpaceRepresentation(test_image, 10, 1.0, 1.5)

    # Compute blob detectors
    BlobDetectors(L, 1)

    print("Scale-space test completed successfully!")
    return True


if __name__ == "__main__":
    # Run test if executed directly
    TestScaleSpace()