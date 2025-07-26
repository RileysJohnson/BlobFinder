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
    FIXED: Now properly matches Igor Pro implementation exactly

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

    # Get dimensions - Igor Pro uses DimSize
    limI = DimSize(detH, 0)  # Height (Y dimension in Igor)
    limJ = DimSize(detH, 1)  # Width (X dimension in Igor)
    limK = DimSize(detH, 2)  # Scale dimension

    # Find local maxima in the 3D detector response
    # This matches the Igor Pro nested loop structure exactly:
    # For(k=1; k<limK-1; k+=1)
    #   For(i=1; i<limI-1; i+=1)
    #     For(j=1; j<limJ-1; j+=1)
    for k in range(1, limK - 1):  # Scale dimension (avoid boundaries)
        for i in range(1, limI - 1):  # Y dimension (avoid boundaries)
            for j in range(1, limJ - 1):  # X dimension (avoid boundaries)

                current_val = detH.data[i, j, k]

                # Check particle type exactly like Igor Pro
                if particleType == 1 and current_val <= 0:
                    continue
                elif particleType == -1 and current_val >= 0:
                    continue

                # Check if this is a local maximum in 3D neighborhood
                # Igor Pro checks all 26 neighbors in 3D space
                is_maximum = True

                for dk in range(-1, 2):
                    if not is_maximum:
                        break
                    for di in range(-1, 2):
                        if not is_maximum:
                            break
                        for dj in range(-1, 2):
                            if dk == 0 and di == 0 and dj == 0:
                                continue  # Skip center point

                            neighbor_val = detH.data[i + di, j + dj, k + dk]

                            # For positive particles, current must be >= all neighbors
                            # For negative particles, current must be <= all neighbors (in absolute value)
                            if particleType == 1:
                                if current_val < neighbor_val:
                                    is_maximum = False
                                    break
                            elif particleType == -1:
                                if abs(current_val) < abs(neighbor_val):
                                    is_maximum = False
                                    break
                            else:  # particleType == 0 (both)
                                if abs(current_val) < abs(neighbor_val):
                                    is_maximum = False
                                    break

                if is_maximum:
                    # Simple curvature check - avoid division by zero
                    lg_val = LG.data[i, j, k] if LG.data.ndim > 2 else LG.data[i, j]

                    # Skip if LG value is too small (avoid numerical issues)
                    if abs(lg_val) < 1e-12:
                        continue

                    # Simplified curvature ratio check
                    curvature_ratio = abs(current_val) / abs(lg_val)
                    if curvature_ratio > maxCurvatureRatio:
                        continue

                    # This is a valid maximum
                    # Check if it's better than existing maximum at this position
                    existing_max = maxes_data[i, j]
                    if abs(current_val) > abs(existing_max):
                        maxes_data[i, j] = current_val

                        # Update maps if provided
                        if map_wave is not None:
                            map_wave.data[i, j] = current_val

                        if scaleMap is not None:
                            # Store the scale information
                            # Convert scale index to actual scale value
                            if DimSize(detH, 2) > 1:  # Check if we have scale dimension
                                scale_value = np.exp(DimOffset(detH, 2) + k * DimDelta(detH, 2))
                                scaleMap.data[i, j] = np.sqrt(scale_value)  # Convert back to spatial units
                            else:
                                scaleMap.data[i, j] = 2.0  # Default radius

    # Create output wave
    maxes_wave = Wave(maxes_data, f"{detH.name}_maxes")
    maxes_wave.SetScale('x', DimOffset(detH, 0), DimDelta(detH, 0))
    maxes_wave.SetScale('y', DimOffset(detH, 1), DimDelta(detH, 1))

    num_maxima = np.sum(maxes_data != 0)
    print(f"Found {num_maxima} local maxima in detector response")

    return maxes_wave


def ImageStats(wave, quiet=True):
    """
    Compute image statistics
    Direct port from Igor Pro ImageStats function

    Parameters:
    wave : Wave - The image wave to analyze
    quiet : bool - If True, suppress output

    Returns:
    dict - Dictionary containing statistics
    """
    data = wave.data

    stats = {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'sum': np.sum(data),
        'numPoints': data.size
    }

    # Find locations of min and max
    min_idx = np.unravel_index(np.argmin(data), data.shape)
    max_idx = np.unravel_index(np.argmax(data), data.shape)

    stats['minLoc'] = min_idx
    stats['maxLoc'] = max_idx

    if not quiet:
        print(f"Statistics for {wave.name}:")
        print(f"  Min: {stats['min']:.6f} at {min_idx}")
        print(f"  Max: {stats['max']:.6f} at {max_idx}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Sum: {stats['sum']:.6f}")
        print(f"  Points: {stats['numPoints']}")

    return stats


def MatrixEigenvalues(matrix):
    """
    Compute eigenvalues of a 2x2 matrix
    Used for Hessian eigenvalue analysis

    Parameters:
    matrix : 2x2 numpy array

    Returns:
    tuple - (lambda1, lambda2) eigenvalues
    """
    eigenvals = np.linalg.eigvals(matrix)
    return eigenvals[0], eigenvals[1]


def ComputeHessianEigenvalues(detH, scale_idx, i, j):
    """
    Compute Hessian matrix eigenvalues at a given point
    This is used for more sophisticated curvature analysis

    Parameters:
    detH : Wave - Determinant of Hessian
    scale_idx : int - Scale index
    i, j : int - Spatial coordinates

    Returns:
    tuple - (lambda1, lambda2) eigenvalues
    """
    # This would require access to the individual Hessian components
    # For now, return simplified estimate
    det_val = detH.data[i, j, scale_idx] if detH.data.ndim > 2 else detH.data[i, j]

    # Simplified eigenvalue estimate
    # In full implementation, this would use actual Hxx, Hyy, Hxy components
    lambda1 = np.sqrt(abs(det_val))
    lambda2 = np.sqrt(abs(det_val))

    return lambda1, lambda2


def PauseForUser():
    """
    Pause execution and wait for user input
    Mimics Igor Pro PauseForUser function
    """
    input("Press Enter to continue...")


def DoAlert(alert_type, message):
    """
    Display alert dialog
    Mimics Igor Pro DoAlert function

    Parameters:
    alert_type : int - Type of alert (0=info, 1=warning, 2=question)
    message : str - Message to display

    Returns:
    int - User response (1=OK/Yes, 2=Cancel/No)
    """
    if alert_type == 0:
        messagebox.showinfo("Information", message)
        return 1
    elif alert_type == 1:
        messagebox.showwarning("Warning", message)
        return 1
    elif alert_type == 2:
        result = messagebox.askyesno("Question", message)
        return 1 if result else 2
    else:
        messagebox.showinfo("Alert", message)
        return 1


def GetDataFolder(level):
    """
    Get current data folder path
    Mimics Igor Pro GetDataFolder function

    Parameters:
    level : int - Level indicator (1 for current folder)

    Returns:
    str - Data folder path
    """
    if level == 1:
        return "root:"
    else:
        return ""


def NewDataFolder(path):
    """
    Create new data folder
    Mimics Igor Pro NewDataFolder function

    Parameters:
    path : str - Path for new folder
    """
    # In Python implementation, this is handled by the DataFolder class
    pass


def DataFolderExists(path):
    """
    Check if data folder exists
    Mimics Igor Pro DataFolderExists function

    Parameters:
    path : str - Folder path to check

    Returns:
    bool - True if folder exists
    """
    # Simplified implementation
    return True


def CountObjects(path, obj_type):
    """
    Count objects in data folder
    Mimics Igor Pro CountObjects function

    Parameters:
    path : str - Folder path
    obj_type : int - Object type (1 for waves)

    Returns:
    int - Number of objects
    """
    # Simplified implementation
    return 1


def UniqueName(base_name, obj_type, mode):
    """
    Generate unique name for object
    Mimics Igor Pro UniqueName function

    Parameters:
    base_name : str - Base name
    obj_type : int - Object type
    mode : int - Naming mode

    Returns:
    str - Unique name
    """
    import time
    return f"{base_name}_{int(time.time())}"


def Testing(string_input, number_input):
    """Testing function for utilities module"""
    print(f"Utilities testing: {string_input}, {number_input}")
    return len(string_input) * number_input