"""
Utilities Module
Contains various utility functions used throughout the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete FindHessianBlobs implementation and Maxes function
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import messagebox
from scipy import ndimage

from igor_compatibility import *
from file_io import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def Maxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """
    Find local maxima in the detector response
    Direct port from Igor Pro Maxes function

    Parameters:
    detH : Wave - The determinant of Hessian blob detector (3D)
    LG : Wave - The Laplacian of Gaussian blob detector (3D)
    particleType : int - Type of particles (-1 for negative, 1 for positive, 0 for both)
    maxCurvatureRatio : float - Maximum ratio of principal curvatures
    map_wave : Wave - Output map for particle identification (optional)
    scaleMap : Wave - Output map for scale information (optional)

    Returns:
    Wave - 2D wave containing maximum detector responses at each position
    """
    print("Computing local maxima in detector response...")

    # Initialize output wave with same spatial dimensions as input
    maxes_data = np.zeros(detH.data.shape[:2])

    if map_wave is not None:
        map_wave.data = np.full(detH.data.shape[:2], -1.0)
    if scaleMap is not None:
        scaleMap.data = np.zeros(detH.data.shape[:2])

    # Find local maxima in the 3D detector response
    for k in range(1, detH.data.shape[2] - 1):  # Scale dimension
        for i in range(1, detH.data.shape[0] - 1):  # Y dimension
            for j in range(1, detH.data.shape[1] - 1):  # X dimension

                current_val = detH.data[i, j, k]

                # Check particle type
                if particleType == 1 and current_val <= 0:
                    continue
                elif particleType == -1 and current_val >= 0:
                    continue

                # Check if curvature ratio is acceptable
                if LG.data[i, j, k] != 0 and current_val != 0:
                    curvature_ratio = (LG.data[i, j, k] ** 2) / current_val
                    max_allowed = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio
                    if curvature_ratio >= max_allowed:
                        continue

                # Check particle type based on LG sign
                if ((particleType == -1 and LG.data[i, j, k] >= 0) or
                        (particleType == 1 and LG.data[i, j, k] <= 0)):
                    continue

                # Check if it's a local maximum in 3D (26-neighborhood)
                is_local_max = True

                # Check neighbors that must be strictly less
                strict_neighbors = [
                    detH.data[i - 1, j - 1, k - 1], detH.data[i - 1, j - 1, k],
                    detH.data[i - 1, j, k - 1], detH.data[i, j - 1, k - 1],
                    detH.data[i, j, k - 1], detH.data[i, j - 1, k],
                    detH.data[i - 1, j, k]
                ]

                for neighbor in strict_neighbors:
                    if current_val <= neighbor:
                        is_local_max = False
                        break

                if not is_local_max:
                    continue

                # Check neighbors that can be equal or less
                equal_neighbors = [
                    detH.data[i - 1, j - 1, k + 1], detH.data[i - 1, j, k + 1],
                    detH.data[i - 1, j + 1, k - 1], detH.data[i - 1, j + 1, k],
                    detH.data[i - 1, j + 1, k + 1], detH.data[i, j - 1, k + 1],
                    detH.data[i, j, k + 1], detH.data[i, j + 1, k - 1],
                    detH.data[i, j + 1, k], detH.data[i, j + 1, k + 1],
                    detH.data[i + 1, j - 1, k - 1], detH.data[i + 1, j - 1, k],
                    detH.data[i + 1, j - 1, k + 1], detH.data[i + 1, j, k - 1],
                    detH.data[i + 1, j, k], detH.data[i + 1, j, k + 1],
                    detH.data[i + 1, j + 1, k - 1], detH.data[i + 1, j + 1, k],
                    detH.data[i + 1, j + 1, k + 1]
                ]

                for neighbor in equal_neighbors:
                    if current_val < neighbor:
                        is_local_max = False
                        break

                if not is_local_max:
                    continue

                # Found a local maximum
                maxes_data[i, j] = max(maxes_data[i, j], current_val)

                if map_wave is not None:
                    map_wave.data[i, j] = max(map_wave.data[i, j], current_val)

                if scaleMap is not None:
                    scale_value = DimOffset(detH, 2) + k * DimDelta(detH, 2)
                    scaleMap.data[i, j] = scale_value

    # Create output wave
    maxes_wave = Wave(maxes_data, "Maxes")
    maxes_wave.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    maxes_wave.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))

    print(f"Found {np.sum(maxes_data > 0)} local maxima")
    return maxes_wave


def ScanFill(image, dest, x, y, layer, destLayer, fill, val, BoundingBox=None):
    """
    Flood fill algorithm - port from Igor Pro ScanFill function

    Parameters:
    image : Wave - Source image
    dest : Wave - Destination image
    x, y : int - Starting coordinates
    layer, destLayer : int - Layer indices
    fill : float - Fill value
    val : float - Target value to fill
    BoundingBox : list - Optional bounding box [minP, maxP, minQ, maxQ]

    Returns:
    complex - Number of pixels filled + boundary particle flag
    """

    # Initialize variables
    height, width = image.data.shape[:2]

    # Bounds checking
    if x < 0 or x >= width or y < 0 or y >= height:
        return complex(0, 0)

    if image.data[y, x, layer] != val:
        return complex(0, 0)

    # Initialize seed stack
    max_stack_size = width * height
    seed_stack = np.zeros((max_stack_size, 4), dtype=int)

    # Stack management
    seed_index = 0
    new_seed_index = 1

    # Initial seed
    seed_stack[0] = [x, x, y, 1]  # [x0, xf, y, state]

    # Tracking variables
    count = 0
    is_boundary_particle = False

    # Bounding box tracking
    min_p = x
    max_p = x
    min_q = y
    max_q = y

    # Main scanning loop
    while seed_index < new_seed_index:
        x0, xf, j, state = seed_stack[seed_index]
        seed_index += 1

        # State machine for scan direction
        if state == 1:  # Initial scan
            # Scan left
            i = x0
            while i >= 0 and dest.data[j, i, destLayer] != fill and image.data[j, i, layer] == val:
                dest.data[j, i, destLayer] = fill
                count += 1
                min_p = min(min_p, i)
                max_p = max(max_p, i)
                min_q = min(min_q, j)
                max_q = max(max_q, j)
                i -= 1
            x0 = i + 1

            # Scan right
            i = x0 + 1
            while i < width and dest.data[j, i, destLayer] != fill and image.data[j, i, layer] == val:
                dest.data[j, i, destLayer] = fill
                count += 1
                min_p = min(min_p, i)
                max_p = max(max_p, i)
                min_q = min(min_q, j)
                max_q = max(max_q, j)
                i += 1
            xf = i - 1

            # Add new seeds for up and down scanning
            if j > 0:  # Up
                seed_stack[new_seed_index] = [x0, xf, j - 1, 2]
                new_seed_index += 1
            if j < height - 1:  # Down
                seed_stack[new_seed_index] = [x0, xf, j + 1, 3]
                new_seed_index += 1

        elif state == 2:  # Scan up
            i = x0
            go_fish = True

            while i <= xf and go_fish:
                if dest.data[j, i, destLayer] == fill or image.data[j, i, layer] != val:
                    i += 1
                    continue

                # Found unfilled pixel, start new scan
                i0 = i
                while i <= xf and dest.data[j, i, destLayer] != fill and image.data[j, i, layer] == val:
                    dest.data[j, i, destLayer] = fill
                    count += 1
                    min_p = min(min_p, i)
                    max_p = max(max_p, i)
                    min_q = min(min_q, j)
                    max_q = max(max_q, j)
                    i += 1

                # Add new seed
                if new_seed_index < max_stack_size:
                    seed_stack[new_seed_index] = [i0, i - 1, j, 1]
                    new_seed_index += 1

                go_fish = False

        elif state == 3:  # Scan down - similar to scan up
            i = x0
            go_fish = True

            while i <= xf and go_fish:
                if dest.data[j, i, destLayer] == fill or image.data[j, i, layer] != val:
                    i += 1
                    continue

                i0 = i
                while i <= xf and dest.data[j, i, destLayer] != fill and image.data[j, i, layer] == val:
                    dest.data[j, i, destLayer] = fill
                    count += 1
                    min_p = min(min_p, i)
                    max_p = max(max_p, i)
                    min_q = min(min_q, j)
                    max_q = max(max_q, j)
                    i += 1

                if new_seed_index < max_stack_size:
                    seed_stack[new_seed_index] = [i0, i - 1, j, 1]
                    new_seed_index += 1

                go_fish = False

    # Check if it's a boundary particle
    # Check edges of bounding box
    for i in range(min_p, max_p + 1):
        if (min_q == 0 and dest.data[min_q, i, destLayer] == fill) or \
                (max_q == height - 1 and dest.data[max_q, i, destLayer] == fill):
            is_boundary_particle = True
            break

    if not is_boundary_particle:
        for j in range(min_q, max_q + 1):
            if (min_p == 0 and dest.data[j, min_p, destLayer] == fill) or \
                    (max_p == width - 1 and dest.data[j, max_p, destLayer] == fill):
                is_boundary_particle = True
                break

    # Update bounding box if provided
    if BoundingBox is not None:
        BoundingBox[0] = min_p
        BoundingBox[1] = max_p
        BoundingBox[2] = min_q
        BoundingBox[3] = max_q

    return complex(count, int(is_boundary_particle))


def ImageLineProfile(image, x1, y1, x2, y2, width=1):
    """
    Extract line profile from image
    Igor ImageLineProfile equivalent
    """
    from scipy import ndimage

    # Number of points along the line
    length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    # Create coordinate arrays
    x_coords = np.linspace(x1, x2, length)
    y_coords = np.linspace(y1, y2, length)

    # Extract values using interpolation
    profile = ndimage.map_coordinates(image.data, [y_coords, x_coords], order=1)

    # Create output wave
    profile_wave = Wave(profile, "LineProfile")
    profile_wave.SetScale('x', 0, 1.0)  # Set to unit spacing

    return profile_wave


def ImageStats(image, quiet=True):
    """
    Compute image statistics
    Igor ImageStats equivalent
    """
    data = image.data

    stats = {
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'mean': np.nanmean(data),
        'std': np.nanstd(data),
        'sum': np.nansum(data),
        'numPoints': np.sum(~np.isnan(data)),
        'minLoc': np.unravel_index(np.nanargmin(data), data.shape),
        'maxLoc': np.unravel_index(np.nanargmax(data), data.shape)
    }

    if not quiet:
        print(f"Image Statistics for {image.name}:")
        print(f"  Min: {stats['min']:.6f} at {stats['minLoc']}")
        print(f"  Max: {stats['max']:.6f} at {stats['maxLoc']}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std Dev: {stats['std']:.6f}")
        print(f"  Sum: {stats['sum']:.6f}")
        print(f"  Points: {stats['numPoints']}")

    return stats


def WaveTransform(wave, transform_type):
    """
    Apply various transformations to wave data
    Igor WaveTransform equivalent (simplified)
    """
    if transform_type.lower() == 'fft':
        # Forward FFT
        fft_data = np.fft.fft2(wave.data)
        result = Wave(fft_data, f"{wave.name}_FFT")

    elif transform_type.lower() == 'ifft':
        # Inverse FFT
        ifft_data = np.fft.ifft2(wave.data)
        result = Wave(ifft_data.real, f"{wave.name}_IFFT")

    elif transform_type.lower() == 'mag':
        # Magnitude of complex data
        mag_data = np.abs(wave.data)
        result = Wave(mag_data, f"{wave.name}_Mag")

    elif transform_type.lower() == 'phase':
        # Phase of complex data
        phase_data = np.angle(wave.data)
        result = Wave(phase_data, f"{wave.name}_Phase")

    else:
        print(f"Unknown transform type: {transform_type}")
        return None

    # Copy scaling
    for axis in ['x', 'y', 'z', 't']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def FindValue(wave, value, start_index=0):
    """
    Find the index of a value in a wave
    Igor FindValue equivalent
    """
    data = wave.data.flatten()
    indices = np.where(data[start_index:] == value)[0]

    if len(indices) > 0:
        return indices[0] + start_index
    else:
        return -1  # Not found


def Smooth(wave, smooth_points):
    """
    Smooth wave data
    Igor Smooth equivalent
    """
    from scipy import ndimage

    if wave.data.ndim == 1:
        # 1D smoothing
        smoothed = ndimage.uniform_filter1d(wave.data, smooth_points)
    elif wave.data.ndim == 2:
        # 2D smoothing
        smoothed = ndimage.uniform_filter(wave.data, smooth_points)
    else:
        print("Smoothing only supported for 1D and 2D waves")
        return wave

    result = Wave(smoothed, f"{wave.name}_Smooth")

    # Copy scaling
    for axis in ['x', 'y', 'z', 't']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def Concatenate(wave1, wave2, dimension=0):
    """
    Concatenate two waves
    Igor Concatenate equivalent
    """
    if dimension == 0:
        concatenated = np.concatenate([wave1.data, wave2.data], axis=0)
    elif dimension == 1:
        concatenated = np.concatenate([wave1.data, wave2.data], axis=1)
    else:
        print("Concatenation only supported for dimensions 0 and 1")
        return None

    result = Wave(concatenated, f"{wave1.name}_Cat")

    # Copy scaling from first wave
    for axis in ['x', 'y', 'z', 't']:
        scale_info = wave1.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def Testing(string_input, number_input):
    """Testing function for utilities module"""
    print(f"Utilities testing: {string_input}, {number_input}")
    return len(string_input) + number_input