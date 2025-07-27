"""
Utilities Module
Contains various utility functions used throughout the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
FIXED: Corrected particle type logic in Maxes function
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
    FIXED: Corrected particle type logic to properly filter positive/negative blobs

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
    print(f"Computing local maxima for particle type: {particleType}")
    print("  particleType = 1: positive blobs (bright spots)")
    print("  particleType = -1: negative blobs (dark spots)")
    print("  particleType = 0: both types")

    # Initialize output wave with same spatial dimensions as input
    maxes_data = np.zeros(detH.data.shape[:2])

    if map_wave is not None:
        map_wave.data = np.full(detH.data.shape[:2], -1.0)
    if scaleMap is not None:
        scaleMap.data = np.zeros(detH.data.shape[:2])

    # Get dimensions
    limI = DimSize(detH, 0)  # Height (Y dimension)
    limJ = DimSize(detH, 1)  # Width (X dimension)
    limK = DimSize(detH, 2)  # Scale dimension

    blob_count = 0
    processed_count = 0

    # Find local maxima in the 3D detector response
    # This matches the Igor Pro nested loop structure exactly
    for k in range(1, limK - 1):  # Skip boundary layers
        for i in range(1, limI - 1):  # Skip boundary pixels
            for j in range(1, limJ - 1):  # Skip boundary pixels

                processed_count += 1

                # Get current detector response values
                detH_val = detH.data[i, j, k]
                LG_val = LG.data[i, j, k]

                # Skip if detector response is too small
                if detH_val <= 0:
                    continue

                # Check curvature ratio constraint
                # If( LG[i][j][k]^2/detH[i][j][k] >= (maxCurvatureRatio+1)^2/maxCurvatureRatio )
                curvature_ratio = (LG_val ** 2) / detH_val
                max_allowed_ratio = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio
                if curvature_ratio >= max_allowed_ratio:
                    continue

                # FIXED: Correct particle type logic
                # Original Igor logic was inverted - this is the corrected version
                if particleType == 1:  # Want positive blobs (bright spots)
                    if LG_val < 0:  # But this is a negative blob, skip it
                        continue
                elif particleType == -1:  # Want negative blobs (dark spots)
                    if LG_val > 0:  # But this is a positive blob, skip it
                        continue
                # If particleType == 0, we want both types, so don't skip

                # Check if this is a local maximum in the 3D neighborhood
                # First check strict inequality neighbors (26 total neighbors)
                is_local_max = True

                # Check the 6 immediate neighbors that must be strictly less
                neighbors_strict = [
                    detH.data[i - 1, j - 1, k - 1], detH.data[i - 1, j - 1, k], detH.data[i - 1, j, k - 1],
                    detH.data[i, j - 1, k - 1], detH.data[i, j, k - 1], detH.data[i, j - 1, k],
                    detH.data[i - 1, j, k]
                ]

                max_strict = np.max(neighbors_strict)
                if not (detH_val > max_strict):
                    continue

                # Check remaining neighbors that can be equal or less
                neighbors_equal = [
                    detH.data[i - 1, j - 1, k + 1], detH.data[i - 1, j, k + 1],
                    detH.data[i - 1, j + 1, k - 1], detH.data[i - 1, j + 1, k], detH.data[i - 1, j + 1, k + 1],
                    detH.data[i, j - 1, k + 1], detH.data[i, j, k + 1],
                    detH.data[i, j + 1, k - 1], detH.data[i, j + 1, k], detH.data[i, j + 1, k + 1],
                    detH.data[i + 1, j - 1, k - 1], detH.data[i + 1, j - 1, k], detH.data[i + 1, j - 1, k + 1],
                    detH.data[i + 1, j, k - 1], detH.data[i + 1, j, k], detH.data[i + 1, j, k + 1],
                    detH.data[i + 1, j + 1, k - 1], detH.data[i + 1, j + 1, k], detH.data[i + 1, j + 1, k + 1]
                ]

                max_equal = np.max(neighbors_equal)
                if not (detH_val >= max_equal):
                    continue

                # This is a valid local maximum
                blob_count += 1

                # Store the maximum response at this spatial location
                maxes_data[i, j] = max(maxes_data[i, j], detH_val)

                # Update output maps if provided
                if map_wave is not None:
                    map_wave.data[i, j] = max(map_wave.data[i, j], detH_val)

                if scaleMap is not None:
                    # Calculate scale value: DimOffset(detH,2)*(DimDelta(detH,2)^k)
                    scale_value = DimOffset(detH, 2) * (DimDelta(detH, 2) ** k)
                    scaleMap.data[i, j] = scale_value

    print(f"Processed {processed_count} candidate locations")
    print(f"Found {blob_count} valid local maxima")

    # Create output wave
    maxes_wave = Wave(maxes_data, "Maxes")

    # Copy scaling from input
    maxes_wave.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0), DimUnits(detH, 0))
    maxes_wave.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1), DimUnits(detH, 1))

    return maxes_wave


def ImageStats(wave, quiet=True):
    """
    Calculate image statistics
    Direct port from Igor Pro ImageStats function

    Parameters:
    wave : Wave - Input image
    quiet : bool - If True, suppress output

    Returns:
    dict - Statistics dictionary
    """
    data = wave.data

    stats = {
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'mean': np.nanmean(data),
        'std': np.nanstd(data),
        'sum': np.nansum(data),
        'numPoints': data.size
    }

    # Find min/max locations
    min_idx = np.unravel_index(np.nanargmin(data), data.shape)
    max_idx = np.unravel_index(np.nanargmax(data), data.shape)

    stats['minLoc'] = min_idx
    stats['maxLoc'] = max_idx

    if not quiet:
        print(f"Image Statistics for {wave.name}:")
        print(f"  Min: {stats['min']:.6f} at {min_idx}")
        print(f"  Max: {stats['max']:.6f} at {max_idx}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std Dev: {stats['std']:.6f}")
        print(f"  Sum: {stats['sum']:.6f}")
        print(f"  Points: {stats['numPoints']}")

    return stats


def FixBoundaries(wave):
    """
    Fix boundary artifacts in scale-space derivatives
    Direct port from Igor Pro FixBoundaries function
    """
    if wave.data.ndim < 3:
        return

    # Fix boundaries by setting edge values to nearest interior values
    height, width, layers = wave.data.shape

    # Fix edges of each layer
    for k in range(layers):
        layer = wave.data[:, :, k]

        # Top and bottom edges
        layer[0, :] = layer[1, :]
        layer[-1, :] = layer[-2, :]

        # Left and right edges
        layer[:, 0] = layer[:, 1]
        layer[:, -1] = layer[:, -2]


def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Calculate Otsu threshold for blob detection
    Direct port from Igor Pro OtsuThreshold function
    """
    print("Computing Otsu threshold...")

    # First identify the maxes
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio)

    # Get valid data (non-zero values)
    valid_data = maxes_wave.data[maxes_wave.data > 0]

    if len(valid_data) == 0:
        print("No valid data for Otsu threshold")
        return 0

    # Create histogram
    hist, bin_edges = np.histogram(valid_data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Search for best threshold using Otsu's method
    min_icv = np.inf
    best_thresh = -np.inf

    for i, x_thresh in enumerate(bin_centers):
        # Split data at threshold
        below_thresh = valid_data[valid_data < x_thresh]
        above_thresh = valid_data[valid_data >= x_thresh]

        if len(below_thresh) == 0 or len(above_thresh) == 0:
            continue

        # Calculate weighted intra-class variance (ICV)
        weight_below = len(below_thresh) / len(valid_data)
        weight_above = len(above_thresh) / len(valid_data)

        var_below = np.var(below_thresh) if len(below_thresh) > 1 else 0
        var_above = np.var(above_thresh) if len(above_thresh) > 1 else 0

        icv = weight_below * var_below + weight_above * var_above

        if icv < min_icv:
            best_thresh = x_thresh
            min_icv = icv

    print(f"Otsu threshold: {best_thresh}")
    return best_thresh


def Testing(string_input, number_input):
    """Testing function for utilities module"""
    print(f"Utilities testing: {string_input}, {number_input}")
    return len(string_input) + number_input