"""
Utilities Module
Contains various utility functions used throughout the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
Fixed version with complete FindHessianBlobs implementation
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

    # Initialize info wave (particle information storage)
    max_particles = min(10000, im.data.shape[0] * im.data.shape[1] // 27)  # Reasonable limit
    info.data = np.zeros((max_particles, 15))

    # Info wave columns:
    # 0: x coordinate, 1: y coordinate, 2: scale index, 3: detector response
    # 4: maximum height, 5: area, 6: volume, 7: refined x, 8: refined y
    # 9-14: reserved for additional measurements

    # Get dimensions
    limI, limJ, limK = detH.data.shape
    limI -= 1
    limJ -= 1
    limK -= 1
    cnt = 0  # Particle counter

    print(f"Scanning {limI + 1} x {limJ + 1} x {limK + 1} detector space...")

    # Start with smallest blobs then go to larger blobs (k=0 is smallest scale)
    for k in range(1, limK):  # Skip first and last scale layers
        for i in range(1, limI):  # Skip image boundaries
            for j in range(1, limJ):

                # Does it hit the threshold?
                current_response = detH.data[i, j, k]
                if abs(current_response) < minResponse:
                    continue

                # Check particle type
                if particleType == 1 and current_response <= 0:
                    continue
                elif particleType == -1 and current_response >= 0:
                    continue

                # Is it too edgy? (curvature ratio test)
                if abs(current_response) > 1e-10:  # Avoid division by zero
                    curvature_ratio = LG.data[i, j, k] ** 2 / abs(current_response)
                    max_allowed_ratio = ((maxCurvatureRatio + 1) / maxCurvatureRatio) ** 2
                    if curvature_ratio >= max_allowed_ratio:
                        continue

                # Is there a particle there already with a stronger response?
                if (mapNum.data[i, j] > -1 and
                        abs(current_response) <= abs(info.data[int(mapNum.data[i, j]), 3])):
                    continue

                # Check if it's a local maximum in 3D neighborhood
                is_local_max = True
                for dk in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if dk == 0 and di == 0 and dj == 0:
                                continue  # Skip center point

                            # Check bounds
                            ni, nj, nk = i + di, j + dj, k + dk
                            if ni < 0 or ni > limI or nj < 0 or nj > limJ or nk < 0 or nk > limK:
                                continue

                            neighbor_response = detH.data[ni, nj, nk]

                            if particleType == 1:
                                # For positive particles, current must be maximum
                                if neighbor_response >= current_response:
                                    is_local_max = False
                                    break
                            elif particleType == -1:
                                # For negative particles, current must be minimum (most negative)
                                if neighbor_response <= current_response:
                                    is_local_max = False
                                    break
                            else:  # particleType == 0 (both)
                                # For both types, check absolute values
                                if abs(neighbor_response) >= abs(current_response):
                                    is_local_max = False
                                    break

                        if not is_local_max:
                            break
                    if not is_local_max:
                        break

                if not is_local_max:
                    continue

                # We found a valid blob! Process it
                if cnt >= max_particles:
                    print(f"Warning: Maximum number of particles ({max_particles}) reached")
                    break

                # Calculate blob properties
                blob_properties = CalculateBlobProperties(im, detH, LG, i, j, k, particleType)

                if blob_properties is None:
                    continue  # Skip if properties calculation failed

                x_coord, y_coord, max_height, area, volume, blob_mask = blob_properties

                # Store particle information
                info.data[cnt, 0] = x_coord  # X coordinate in real units
                info.data[cnt, 1] = y_coord  # Y coordinate in real units
                info.data[cnt, 2] = k  # Scale index
                info.data[cnt, 3] = current_response  # Detector response
                info.data[cnt, 4] = max_height  # Maximum height
                info.data[cnt, 5] = area  # Area
                info.data[cnt, 6] = volume  # Volume
                info.data[cnt, 7] = x_coord  # Refined X (initially same as x_coord)
                info.data[cnt, 8] = y_coord  # Refined Y (initially same as y_coord)
                info.data[cnt, 9] = k  # Store scale index for radius calculation

                # Update maps (these use pixel coordinates)
                if blob_mask is not None:
                    mapNum.data[blob_mask] = cnt
                    mapDetH.data[blob_mask] = current_response
                    mapMax.data[blob_mask] = max_height
                else:
                    # Fallback: mark just the center point
                    mapNum.data[i, j] = cnt
                    mapDetH.data[i, j] = current_response
                    mapMax.data[i, j] = max_height

                cnt += 1

                if cnt % 100 == 0:  # Progress indicator
                    print(f"  Found {cnt} particles...")

        if cnt >= max_particles:
            break

    # Trim info wave to actual number of particles found
    if cnt > 0:
        info.data = info.data[:cnt, :]

    print(f"FindHessianBlobs completed: found {cnt} particles")
    return cnt


def CalculateBlobProperties(im, detH, LG, i, j, k, particleType):
    """
    Calculate properties of a detected blob

    Parameters:
    im : Wave - Original image
    detH : Wave - Detector response
    LG : Wave - Laplacian of Gaussian
    i, j, k : int - Blob center coordinates and scale
    particleType : int - Type of particle

    Returns:
    tuple - (x_coord, y_coord, max_height, area, volume, blob_mask) or None if failed
    """
    try:
        # Convert pixel coordinates to real coordinates
        x_coord = DimOffset(im, 0) + j * DimDelta(im, 0)
        y_coord = DimOffset(im, 1) + i * DimDelta(im, 1)

        # Get the scale for this layer
        if k < detH.data.shape[2]:
            scale_value = DimOffset(detH, 2) + k * DimDelta(detH, 2)
            # If using log scale, convert back
            if DimDelta(detH, 2) != 0 and scale_value > 0:
                try:
                    actual_scale = np.exp(scale_value)
                except:
                    actual_scale = scale_value
            else:
                actual_scale = max(scale_value, 1.0)  # Ensure positive scale

            # Calculate characteristic radius
            blob_radius = np.sqrt(2 * actual_scale)
        else:
            blob_radius = 3.0  # Default radius

        # Define blob region (circular region around detected center)
        radius_pixels = max(2, int(blob_radius / DimDelta(im, 0)))

        # Create circular mask
        y_indices, x_indices = np.ogrid[:im.data.shape[0], :im.data.shape[1]]
        mask = ((x_indices - j) ** 2 + (y_indices - i) ** 2) <= radius_pixels ** 2

        # Make sure mask includes at least the center point
        mask[i, j] = True

        if not np.any(mask):
            # Fallback: just use center point
            mask = np.zeros_like(im.data, dtype=bool)
            mask[i, j] = True

        # Calculate properties within the mask
        blob_data = im.data[mask]

        if len(blob_data) == 0:
            return None

        # Calculate measurements
        max_height = np.max(blob_data)
        area = np.sum(mask) * DimDelta(im, 0) * DimDelta(im, 1)
        volume = np.sum(blob_data) * DimDelta(im, 0) * DimDelta(im, 1)

        return x_coord, y_coord, max_height, area, volume, mask

    except Exception as e:
        print(f"Error calculating blob properties: {e}")
        return None


def ScanFill(im, seed_i, seed_j, threshold, connectivity=8):
    """
    Flood fill algorithm to define blob boundaries
    Similar to Igor Pro's ScanFill functionality

    Parameters:
    im : Wave - Image data
    seed_i, seed_j : int - Seed point coordinates
    threshold : float - Threshold value
    connectivity : int - 4 or 8 connectivity

    Returns:
    ndarray - Boolean mask of filled region
    """
    mask = np.zeros(im.data.shape, dtype=bool)

    if (seed_i < 0 or seed_i >= im.data.shape[0] or
            seed_j < 0 or seed_j >= im.data.shape[1]):
        return mask

    # Use scipy's flood fill (more efficient)
    from scipy import ndimage

    # Simple threshold-based region growing
    seed_value = im.data[seed_i, seed_j]

    # Create binary image based on threshold
    if seed_value > threshold:
        binary_im = im.data > threshold
    else:
        binary_im = im.data < threshold

    # Label connected components
    labeled, num_features = ndimage.label(binary_im,
                                          structure=ndimage.generate_binary_structure(2, connectivity // 4))

    if num_features > 0:
        # Find which label contains the seed point
        seed_label = labeled[seed_i, seed_j]
        if seed_label > 0:
            mask = (labeled == seed_label)

    return mask


def RefineParticlePositions(im, info, mapNum, subpixel_factor=1):
    """
    Refine particle positions to sub-pixel accuracy

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information
    mapNum : Wave - Particle map
    subpixel_factor : int - Sub-pixel refinement factor

    Returns:
    bool - Success flag
    """
    if subpixel_factor <= 1:
        return True  # No refinement needed

    print(f"Refining particle positions with factor {subpixel_factor}")

    num_particles = info.data.shape[0]

    for p in range(num_particles):
        try:
            # Get current position
            x_coord = info.data[p, 0]
            y_coord = info.data[p, 1]

            # Convert to pixel coordinates
            i = int((y_coord - DimOffset(im, 1)) / DimDelta(im, 1))
            j = int((x_coord - DimOffset(im, 0)) / DimDelta(im, 0))

            # Define refinement window
            window_size = 5
            i_min = max(0, i - window_size)
            i_max = min(im.data.shape[0], i + window_size + 1)
            j_min = max(0, j - window_size)
            j_max = min(im.data.shape[1], j + window_size + 1)

            # Extract local region
            local_region = im.data[i_min:i_max, j_min:j_max]

            if local_region.size == 0:
                continue

            # Find center of mass for sub-pixel refinement
            yi, xi = np.mgrid[0:local_region.shape[0], 0:local_region.shape[1]]
            total_mass = np.sum(local_region)

            if total_mass > 0:
                # Center of mass
                cm_i = np.sum(yi * local_region) / total_mass + i_min
                cm_j = np.sum(xi * local_region) / total_mass + j_min

                # Convert back to real coordinates
                refined_x = DimOffset(im, 0) + cm_j * DimDelta(im, 0)
                refined_y = DimOffset(im, 1) + cm_i * DimDelta(im, 1)

                # Update refined positions
                info.data[p, 7] = refined_x
                info.data[p, 8] = refined_y

        except Exception as e:
            print(f"Error refining particle {p}: {e}")
            continue

    print("Particle position refinement completed")
    return True


def FilterParticlesByConstraints(info, mapNum, minH, maxH, minV, maxV, minA, maxA):
    """
    Filter particles based on measurement constraints

    Parameters:
    info : Wave - Particle information
    mapNum : Wave - Particle map
    minH, maxH : float - Height constraints
    minV, maxV : float - Volume constraints
    minA, maxA : float - Area constraints

    Returns:
    int - Number of particles remaining after filtering
    """
    if (minH == -np.inf and maxH == np.inf and
            minV == -np.inf and maxV == np.inf and
            minA == -np.inf and maxA == np.inf):
        return info.data.shape[0]  # No constraints

    print("Applying particle constraints...")

    num_particles = info.data.shape[0]
    valid_particles = []

    for p in range(num_particles):
        height = info.data[p, 4]  # Column 4 is max height
        area = info.data[p, 5]  # Column 5 is area
        volume = info.data[p, 6]  # Column 6 is volume

        # Check constraints
        if (minH <= height <= maxH and
                minV <= volume <= maxV and
                minA <= area <= maxA):
            valid_particles.append(p)
        else:
            # Remove this particle from maps
            particle_mask = (mapNum.data == p)
            mapNum.data[particle_mask] = -1

    # Compact info wave and renumber particles
    if len(valid_particles) < num_particles:
        new_info = np.zeros((len(valid_particles), info.data.shape[1]))

        for new_id, old_id in enumerate(valid_particles):
            new_info[new_id, :] = info.data[old_id, :]

            # Update particle numbering in maps
            particle_mask = (mapNum.data == old_id)
            mapNum.data[particle_mask] = new_id

        info.data = new_info

    print(
        f"Constraints applied: {len(valid_particles)} particles remain (removed {num_particles - len(valid_particles)})")
    return len(valid_particles)


def AnalyzeParticleShape(im, particle_mask):
    """
    Analyze the shape properties of a particle

    Parameters:
    im : Wave - Original image
    particle_mask : ndarray - Boolean mask of particle region

    Returns:
    dict - Dictionary of shape properties
    """
    if not np.any(particle_mask):
        return {}

    try:
        # Basic properties
        particle_data = im.data[particle_mask]
        area = np.sum(particle_mask)
        perimeter = calculate_perimeter(particle_mask)

        # Center of mass
        yi, xi = np.where(particle_mask)
        weights = im.data[yi, xi]
        total_weight = np.sum(weights)

        if total_weight > 0:
            cm_x = np.sum(xi * weights) / total_weight
            cm_y = np.sum(yi * weights) / total_weight
        else:
            cm_x = np.mean(xi)
            cm_y = np.mean(yi)

        # Equivalent diameter
        equiv_diameter = np.sqrt(4 * area / np.pi)

        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Aspect ratio (using moments)
        aspect_ratio = calculate_aspect_ratio(particle_mask)

        return {
            'area': area,
            'perimeter': perimeter,
            'center_of_mass_x': cm_x,
            'center_of_mass_y': cm_y,
            'equivalent_diameter': equiv_diameter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'max_intensity': np.max(particle_data),
            'min_intensity': np.min(particle_data),
            'mean_intensity': np.mean(particle_data),
            'total_intensity': np.sum(particle_data)
        }

    except Exception as e:
        print(f"Error analyzing particle shape: {e}")
        return {}


def calculate_perimeter(mask):
    """Calculate perimeter of a binary mask"""
    try:
        from skimage import measure
        contours = measure.find_contours(mask.astype(float), 0.5)
        if contours:
            return len(contours[0])
        else:
            return 0
    except ImportError:
        # Fallback: count edge pixels
        edge_mask = mask & ~ndimage.binary_erosion(mask)
        return np.sum(edge_mask)


def calculate_aspect_ratio(mask):
    """Calculate aspect ratio using second moments"""
    try:
        yi, xi = np.where(mask)

        if len(yi) < 2:
            return 1.0

        # Calculate second moments
        x_mean = np.mean(xi)
        y_mean = np.mean(yi)

        mu20 = np.mean((xi - x_mean) ** 2)
        mu02 = np.mean((yi - y_mean) ** 2)
        mu11 = np.mean((xi - x_mean) * (yi - y_mean))

        # Calculate eigenvalues of covariance matrix
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 ** 2

        if det <= 0:
            return 1.0

        lambda1 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2
        lambda2 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2

        if lambda2 <= 0:
            return 1.0

        return np.sqrt(lambda1 / lambda2)

    except Exception:
        return 1.0


def CreateParticleSummary(info, im_name=""):
    """
    Create a summary of detected particles

    Parameters:
    info : Wave - Particle information
    im_name : str - Name of the image

    Returns:
    dict - Summary statistics
    """
    if info.data.shape[0] == 0:
        return {"num_particles": 0}

    heights = info.data[:, 4]
    areas = info.data[:, 5]
    volumes = info.data[:, 6]

    summary = {
        "image_name": im_name,
        "num_particles": len(heights),
        "height_stats": {
            "mean": np.mean(heights),
            "std": np.std(heights),
            "min": np.min(heights),
            "max": np.max(heights),
            "median": np.median(heights)
        },
        "area_stats": {
            "mean": np.mean(areas),
            "std": np.std(areas),
            "min": np.min(areas),
            "max": np.max(areas),
            "median": np.median(areas)
        },
        "volume_stats": {
            "mean": np.mean(volumes),
            "std": np.std(volumes),
            "min": np.min(volumes),
            "max": np.max(volumes),
            "median": np.median(volumes)
        }
    }

    return summary


def ExportParticleData(info, filename):
    """
    Export particle data to a text file

    Parameters:
    info : Wave - Particle information
    filename : str - Output filename

    Returns:
    bool - Success flag
    """
    try:
        header = ["X_Coord", "Y_Coord", "Scale_Index", "Detector_Response",
                  "Max_Height", "Area", "Volume", "Refined_X", "Refined_Y"]

        np.savetxt(filename, info.data[:, :9], delimiter='\t',
                   header='\t'.join(header), comments='')

        print(f"Particle data exported to {filename}")
        return True

    except Exception as e:
        print(f"Error exporting particle data: {e}")
        return False


def TestUtilities():
    """Test function for utilities module"""
    print("Testing utilities module...")

    # Create test data
    test_image = Wave(np.random.rand(50, 50), "TestImage")
    test_image.SetScale('x', 0, 1)
    test_image.SetScale('y', 0, 1)

    # Test blob properties calculation
    detH = Wave(np.random.rand(50, 50, 10), "TestDetH")
    LG = Wave(np.random.rand(50, 50, 10), "TestLG")

    props = CalculateBlobProperties(test_image, detH, LG, 25, 25, 5, 1)

    if props is not None:
        print("✓ Blob properties calculation working")
    else:
        print("✗ Blob properties calculation failed")

    print("Utilities test completed")


if __name__ == "__main__":
    TestUtilities()