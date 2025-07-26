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

                current_response = detH.data[i, j, k]

                # Check particle type constraint
                if particleType == 1 and current_response <= 0:
                    continue
                elif particleType == -1 and current_response >= 0:
                    continue
                elif current_response == 0:
                    continue

                # Check if it's a local maximum in 3D neighborhood
                is_max = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            if detH.data[i + di, j + dj, k + dk] >= current_response:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if not is_max:
                        break

                if is_max:
                    # Check curvature ratio constraint
                    # The constraint is: LG^2 / detH < (maxCurvatureRatio + 1)^2 / maxCurvatureRatio
                    lg_val = LG.data[i, j, k]
                    if abs(current_response) > 1e-10:  # Avoid division by zero
                        curvature_ratio = (lg_val ** 2) / abs(current_response)
                        max_allowed = ((maxCurvatureRatio + 1) ** 2) / maxCurvatureRatio

                        if curvature_ratio < max_allowed:
                            # Update if this is the strongest response at this position
                            if abs(current_response) > abs(maxes_data[i, j]):
                                maxes_data[i, j] = current_response
                                if map_wave is not None:
                                    map_wave.data[i, j] = current_response
                                if scaleMap is not None:
                                    scaleMap.data[i, j] = k

    # Create output wave
    maxes_wave = Wave(maxes_data, "Maxes")
    maxes_wave.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    maxes_wave.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))

    num_maxes = np.sum(maxes_data != 0)
    print(f"Found {num_maxes} local maxima")

    return maxes_wave


def FindHessianBlobs(im, detH, LG, detHResponseThresh, mapNum, mapDetH, mapMax, info,
                     particleType, maxCurvatureRatio):
    """
    The core blob detection function - direct port from Igor Pro

    Parameters:
    im : Wave - The original image
    detH : Wave - The determinant of Hessian blob detector (3D)
    LG : Wave - The Laplacian of Gaussian blob detector (3D)
    detHResponseThresh : float - Minimum detector response threshold
    mapNum : Wave - Map identifying particle numbers (output)
    mapDetH : Wave - Map of detector values at blob locations (output)
    mapMax : Wave - Map of maximum pixel values in each blob (output)
    info : Wave - Information about each detected blob (output)
    particleType : int - Type of particles (-1 for negative, 1 for positive, 0 for both)
    maxCurvatureRatio : float - Maximum ratio of principal curvatures

    Returns:
    int - Number of particles found
    """
    print(f"Finding Hessian blobs with threshold {detHResponseThresh}")

    # Square the minResponse, since the parameter is provided as the square root
    # of the actual minimum detH response so that it is in normal image units
    minResponse = detHResponseThresh ** 2

    # Initialize output maps
    mapNum.data = np.full(im.data.shape, -1, dtype=np.int32)
    mapNum.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapNum.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    mapDetH.data = np.zeros(im.data.shape)
    mapDetH.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapDetH.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    mapMax.data = np.zeros(im.data.shape)
    mapMax.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapMax.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    # Find local maxima using the Maxes function
    map_wave = Wave(np.full(im.data.shape, -1.0), "TempMap")
    scale_map = Wave(np.zeros(im.data.shape), "TempScaleMap")
    maxes_wave = Maxes(detH, LG, particleType, maxCurvatureRatio, map_wave, scale_map)

    # Convert maxes to image units (square root)
    maxes_image_units = np.sqrt(np.maximum(np.abs(maxes_wave.data), 0))

    # Find blobs above threshold
    blob_candidates = []
    for i in range(maxes_wave.data.shape[0]):
        for j in range(maxes_wave.data.shape[1]):
            if maxes_image_units[i, j] >= detHResponseThresh and map_wave.data[i, j] > minResponse:
                scale_idx = int(scale_map.data[i, j])

                # Calculate real-world coordinates
                x_coord = DimOffset(im, 0) + j * DimDelta(im, 0)
                y_coord = DimOffset(im, 1) + i * DimDelta(im, 1)

                # Calculate scale value
                if scale_idx < detH.data.shape[2]:
                    scale_value = np.exp(DimOffset(detH, 2) + scale_idx * DimDelta(detH, 2))
                else:
                    scale_value = 1.0

                blob_candidates.append({
                    'i': i, 'j': j,
                    'x': x_coord, 'y': y_coord,
                    'scale_idx': scale_idx,
                    'scale_value': scale_value,
                    'response': map_wave.data[i, j],
                    'strength': maxes_image_units[i, j],
                    'max_value': im.data[i, j]
                })

    # Sort by detector response strength (strongest first)
    blob_candidates.sort(key=lambda x: x['response'], reverse=True)

    # Process blobs and handle overlaps
    accepted_blobs = []
    particle_number = 0

    for candidate in blob_candidates:
        i, j = candidate['i'], candidate['j']

        # Check if this position is already occupied
        if mapNum.data[i, j] >= 0:
            continue

        # Mark this blob
        mapNum.data[i, j] = particle_number
        mapDetH.data[i, j] = candidate['response']
        mapMax.data[i, j] = candidate['max_value']

        # Calculate blob radius for overlap checking
        radius = np.sqrt(2 * candidate['scale_value'])
        radius_pixels = radius / DimDelta(im, 0)

        # Mark nearby pixels as occupied (simple circular region)
        y_center, x_center = i, j
        for y in range(max(0, int(y_center - radius_pixels)),
                       min(im.data.shape[0], int(y_center + radius_pixels + 1))):
            for x in range(max(0, int(x_center - radius_pixels)),
                           min(im.data.shape[1], int(x_center + radius_pixels + 1))):
                dist = np.sqrt((y - y_center) ** 2 + (x - x_center) ** 2)
                if dist <= radius_pixels and mapNum.data[y, x] < 0:
                    mapNum.data[y, x] = particle_number

        accepted_blobs.append(candidate)
        particle_number += 1

    # Create info wave with blob information
    num_blobs = len(accepted_blobs)
    if num_blobs > 0:
        # Info wave columns: x, y, scale, strength, max_value, area, volume, etc.
        info_data = np.zeros((num_blobs, 10))  # 10 columns for various parameters

        for idx, blob in enumerate(accepted_blobs):
            info_data[idx, 0] = blob['x']  # X coordinate
            info_data[idx, 1] = blob['y']  # Y coordinate
            info_data[idx, 2] = blob['scale_idx']  # Scale index
            info_data[idx, 3] = blob['scale_value']  # Scale value
            info_data[idx, 4] = blob['strength']  # Blob strength
            info_data[idx, 5] = blob['max_value']  # Maximum pixel value
            info_data[idx, 6] = blob['response']  # Detector response

            # Calculate area and volume (simplified)
            radius = np.sqrt(2 * blob['scale_value'])
            area = np.pi * radius ** 2
            volume = area * blob['max_value']

            info_data[idx, 7] = area  # Area
            info_data[idx, 8] = volume  # Volume
            info_data[idx, 9] = blob['scale_idx']  # Store scale index again for easy access

        info.data = info_data
    else:
        info.data = np.zeros((0, 10))

    print(f"Found {num_blobs} Hessian blobs above threshold")
    return num_blobs


def SubPixelMaxima3D(detH, LG, i, j, k, maxCurvatureRatio):
    """
    Refine blob position to sub-pixel accuracy using 3D interpolation
    Direct port from Igor Pro SubPixelMaxima3D function
    """
    # Get 3x3x3 neighborhood around the maximum
    if (i < 1 or i >= detH.data.shape[0] - 1 or
            j < 1 or j >= detH.data.shape[1] - 1 or
            k < 1 or k >= detH.data.shape[2] - 1):
        return i, j, k, detH.data[i, j, k]

    # Extract 3x3x3 neighborhood
    neighborhood = detH.data[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]

    # Calculate gradients and Hessian for sub-pixel refinement
    # First derivatives
    dx = (neighborhood[1, 2, 1] - neighborhood[1, 0, 1]) / 2
    dy = (neighborhood[2, 1, 1] - neighborhood[0, 1, 1]) / 2
    dz = (neighborhood[1, 1, 2] - neighborhood[1, 1, 0]) / 2

    # Second derivatives (Hessian)
    dxx = neighborhood[1, 2, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 0, 1]
    dyy = neighborhood[2, 1, 1] - 2 * neighborhood[1, 1, 1] + neighborhood[0, 1, 1]
    dzz = neighborhood[1, 1, 2] - 2 * neighborhood[1, 1, 1] + neighborhood[1, 1, 0]

    # Mixed derivatives
    dxy = (neighborhood[2, 2, 1] - neighborhood[2, 0, 1] - neighborhood[0, 2, 1] + neighborhood[0, 0, 1]) / 4
    dxz = (neighborhood[1, 2, 2] - neighborhood[1, 2, 0] - neighborhood[1, 0, 2] + neighborhood[1, 0, 0]) / 4
    dyz = (neighborhood[2, 1, 2] - neighborhood[2, 1, 0] - neighborhood[0, 1, 2] + neighborhood[0, 1, 0]) / 4

    # Construct Hessian matrix
    H = np.array([[dxx, dxy, dxz],
                  [dxy, dyy, dyz],
                  [dxz, dyz, dzz]])

    gradient = np.array([dx, dy, dz])

    # Solve for sub-pixel offset: H * offset = -gradient
    try:
        offset = np.linalg.solve(H, -gradient)

        # Limit offset to reasonable range
        offset = np.clip(offset, -0.5, 0.5)

        # Calculate refined position
        refined_i = i + offset[1]  # Note: offset[1] is for y direction
        refined_j = j + offset[0]  # Note: offset[0] is for x direction
        refined_k = k + offset[2]

        # Calculate refined value
        refined_value = (neighborhood[1, 1, 1] +
                         0.5 * np.dot(gradient, offset))

        return refined_i, refined_j, refined_k, refined_value

    except np.linalg.LinAlgError:
        # If Hessian is singular, return original position
        return i, j, k, detH.data[i, j, k]


def MaxOccupancy(mapNum, particleNum, radius):
    """
    Check maximum occupancy in a circular region
    Direct port from Igor Pro MaxOccupancy function
    """
    if particleNum < 0:
        return 0

    # Find center of particle
    indices = np.where(mapNum.data == particleNum)
    if len(indices[0]) == 0:
        return 0

    center_i = np.mean(indices[0])
    center_j = np.mean(indices[1])

    # Count pixels in circular region around center
    count = 0
    radius_pixels = radius / DimDelta(mapNum, 0)  # Convert to pixels

    for i in range(max(0, int(center_i - radius_pixels)),
                   min(mapNum.data.shape[0], int(center_i + radius_pixels + 1))):
        for j in range(max(0, int(center_j - radius_pixels)),
                       min(mapNum.data.shape[1], int(center_j + radius_pixels + 1))):
            dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
            if dist <= radius_pixels:
                if mapNum.data[i, j] == particleNum:
                    count += 1

    return count


def DistanceBetweenParticles(info, p1_idx, p2_idx):
    """
    Calculate distance between two particles
    Direct port from Igor Pro function
    """
    if (p1_idx >= info.data.shape[0] or p2_idx >= info.data.shape[0] or
            p1_idx < 0 or p2_idx < 0):
        return np.inf

    x1, y1 = info.data[p1_idx, 0], info.data[p1_idx, 1]
    x2, y2 = info.data[p2_idx, 0], info.data[p2_idx, 1]

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def RemoveParticle(mapNum, mapDetH, mapMax, particleNum):
    """
    Remove a particle from all maps
    Direct port from Igor Pro function
    """
    mask = mapNum.data == particleNum
    mapNum.data[mask] = -1
    mapDetH.data[mask] = 0
    mapMax.data[mask] = 0


def Testing(input_string, input_number):
    """
    Testing function for the algorithm
    Direct port from Igor Pro Testing function
    """
    print(f"Testing function called:")
    print(f"  Input string: '{input_string}'")
    print(f"  Input number: {input_number}")

    # Perform some basic calculations to test functionality
    result = input_number * 2 + len(input_string)
    print(f"  Calculated result: {result}")

    return result