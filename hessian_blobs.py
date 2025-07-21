import numpy as np
from scale_space import scale_space_representation, blob_detectors, otsu_threshold
from measurements import *
from utilities import *
import os
from datetime import datetime


def batch_hessian_blobs(images, params=None):
    """
    Detect Hessian blobs in a series of images

    Parameters:
    images: Dictionary of images {name: image_array}
    params: Optional parameter array with 13 parameters

    Returns:
    Dictionary containing results
    """
    # Create series folder
    series_folder = create_series_folder()

    # Get parameters
    if params is None:
        params = get_parameters_from_user()

    # Store parameters
    series_folder.save_wave("Parameters", params)

    # Initialize collection waves
    all_heights = []
    all_volumes = []
    all_areas = []
    all_avg_heights = []

    # Process each image
    image_results = {}

    for i, (name, im) in enumerate(images.items()):
        print("-------------------------------------------------------")
        print(f"Analyzing image {i + 1} of {len(images)}")
        print("-------------------------------------------------------")

        # Create image folder
        image_folder = series_folder.create_subfolder(f"{name}_Particles")

        # Run Hessian blob algorithm
        result = hessian_blobs(im, params=params, data_folder=image_folder)

        if result:
            # Collect measurements
            all_heights.extend(result['heights'])
            all_volumes.extend(result['volumes'])
            all_areas.extend(result['areas'])
            all_avg_heights.extend(result['avg_heights'])

            image_results[name] = result

    # Save collected measurements
    series_folder.save_wave("AllHeights", np.array(all_heights))
    series_folder.save_wave("AllVolumes", np.array(all_volumes))
    series_folder.save_wave("AllAreas", np.array(all_areas))
    series_folder.save_wave("AllAvgHeights", np.array(all_avg_heights))

    num_particles = len(all_heights)
    print(f"  Series complete. Total particles detected: {num_particles}")

    return {
        'series_folder': series_folder,
        'all_heights': np.array(all_heights),
        'all_volumes': np.array(all_volumes),
        'all_areas': np.array(all_areas),
        'all_avg_heights': np.array(all_avg_heights),
        'image_results': image_results
    }


def hessian_blobs(im, params=None, data_folder=None):
    """
    Execute Hessian blob algorithm on an image

    Parameters:
    im: Image to analyze
    params: Optional parameter array
    data_folder: Optional data folder for saving results

    Returns:
    Dictionary containing detected particles and measurements
    """
    # Get parameters
    if params is None:
        params = get_parameters_from_user()

    # Extract parameters
    scale_start = params[0]
    layers = int(params[1])
    scale_factor = params[2]
    det_h_response_thresh = params[3]
    particle_type = int(params[4])
    sub_pixel_mult = int(params[5])
    allow_overlap = int(params[6])
    min_h = params[7]
    max_h = params[8]
    min_a = params[9]
    max_a = params[10]
    min_v = params[11]
    max_v = params[12]

    # Create coordinate system
    coord_system = CoordinateSystem(im.shape)

    # Convert scale parameters
    scale_start = (scale_start * coord_system.x_delta) ** 2 / 2
    layers = int(np.ceil(np.log((layers * coord_system.x_delta) ** 2 / (2 * scale_start)) / np.log(scale_factor)))
    sub_pixel_mult = max(1, round(sub_pixel_mult))
    scale_factor = max(1.1, scale_factor)

    # Hard coded parameters
    gamma_norm = 1
    max_curvature_ratio = 10
    allow_boundary_particles = 1

    # Create data folder if needed
    if data_folder is None:
        data_folder = DataFolder("./results", "temp_particles")

    # Store original image
    original = im.copy()
    data_folder.save_wave("Original", original)

    # Calculate scale-space representation
    print("Calculating scale-space representation..")
    L, scale_coords = scale_space_representation(im, layers, np.sqrt(scale_start) / coord_system.x_delta,
                                                 scale_factor, coord_system)
    data_folder.save_wave("ScaleSpaceRep", L)

    # Calculate blob detectors
    print("Calculating scale-space derivatives..")
    detH, LapG = blob_detectors(L, gamma_norm, coord_system)
    data_folder.save_wave("detH", detH)
    data_folder.save_wave("LapG", LapG)

    # Determine threshold
    if det_h_response_thresh == -1:
        print("Calculating Otsu's Threshold..")
        det_h_response_thresh = np.sqrt(otsu_threshold(detH, LapG, particle_type, max_curvature_ratio))
        print(f"Otsu's Threshold: {det_h_response_thresh}")
    elif det_h_response_thresh == -2:
        det_h_response_thresh = interactive_threshold(im, detH, LapG, particle_type, max_curvature_ratio)
        if det_h_response_thresh is None:
            return None
        print(f"Chosen Det H Response Threshold: {det_h_response_thresh}")

    # Detect particles
    print("Detecting Hessian blobs..")
    info = find_hessian_blobs(im, detH, LapG, det_h_response_thresh, particle_type,
                              max_curvature_ratio, coord_system)

    num_potential_particles = len(info)

    # Remove overlapping blobs if requested
    if allow_overlap == 0 and num_potential_particles > 0:
        print("Determining scale-maximal particles..")
        info = maximal_blobs(info)

    # Measure particles
    print("Cropping and measuring particles..")
    particles, measurements = measure_particles(im, info, detH, LapG, sub_pixel_mult,
                                                coord_system, data_folder,
                                                min_h, max_h, min_a, max_a, min_v, max_v)

    # Create particle map
    particle_map = create_particle_map(im.shape, particles)
    data_folder.save_wave("ParticleMap", particle_map)

    return {
        'particles': particles,
        'heights': measurements['heights'],
        'volumes': measurements['volumes'],
        'areas': measurements['areas'],
        'avg_heights': measurements['avg_heights'],
        'centers': measurements['centers'],
        'info': info,
        'data_folder': data_folder
    }


def find_hessian_blobs(im, detH, LapG, min_response, particle_type, max_curvature_ratio, coord_system):
    """
    Find Hessian blobs in scale space
    """
    # Square the minimum response
    min_response = min_response ** 2

    info = []

    for k in range(1, detH.shape[2] - 1):
        for i in range(1, detH.shape[0] - 1):
            for j in range(1, detH.shape[1] - 1):
                # Check threshold
                if detH[i, j, k] < min_response:
                    continue

                # Check curvature ratio
                if LapG[i, j, k] ** 2 / detH[i, j, k] >= (max_curvature_ratio + 1) ** 2 / max_curvature_ratio:
                    continue

                # Check particle type
                if particle_type == -1 and LapG[i, j, k] < 0:
                    continue
                elif particle_type == 1 and LapG[i, j, k] > 0:
                    continue

                # Check if local maximum
                is_max = check_local_maximum(detH, i, j, k)

                if is_max:
                    # Find blob extent using scanline fill
                    blob_mask, bbox = scanline_fill_blob(detH[:, :, k], LapG[:, :, k], i, j, 0)

                    particle_info = {
                        'p_seed': i,
                        'q_seed': j,
                        'num_pixels': np.sum(blob_mask),
                        'max_val': detH[i, j, k],
                        'p_start': bbox[0],
                        'p_stop': bbox[1],
                        'q_start': bbox[2],
                        'q_stop': bbox[3],
                        'scale': coord_system.x_start + coord_system.x_delta * (coord_system.y_delta ** k),
                        'layer': k,
                        'is_maximal': 1,
                        'parent': len(info),
                        'num_contained': 0,
                        'edge_quality': 0,
                        'status': 0,
                        'blob_mask': blob_mask
                    }

                    info.append(particle_info)

    return info


def check_local_maximum(detH, i, j, k):
    """Check if position is local maximum in 3x3x3 neighborhood"""
    center = detH[i, j, k]

    # Check 26 neighbors
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue

                ni, nj, nk = i + di, j + dj, k + dk

                # Check bounds
                if ni < 0 or ni >= detH.shape[0]:
                    continue
                if nj < 0 or nj >= detH.shape[1]:
                    continue
                if nk < 0 or nk >= detH.shape[2]:
                    continue

                # For same scale, must be strictly greater
                if dk == 0 and detH[ni, nj, nk] >= center:
                    return False
                # For different scales, can be equal
                elif dk != 0 and detH[ni, nj, nk] > center:
                    return False

    return True


def scanline_fill_blob(detH_layer, LapG_layer, seed_i, seed_j, threshold):
    """
    Scanline fill algorithm to find blob extent
    """
    mask = np.zeros_like(detH_layer)
    stack = [(seed_i, seed_j)]

    min_i = seed_i
    max_i = seed_i
    min_j = seed_j
    max_j = seed_j

    sgn = np.sign(LapG_layer[seed_i, seed_j])

    while stack:
        i, j = stack.pop()

        if i < 0 or i >= detH_layer.shape[0] or j < 0 or j >= detH_layer.shape[1]:
            continue

        if mask[i, j] == 1:
            continue

        if detH_layer[i, j] < threshold:
            continue

        if np.sign(LapG_layer[i, j]) != sgn:
            continue

        # Fill this pixel
        mask[i, j] = 1

        # Update bounding box
        min_i = min(min_i, i)
        max_i = max(max_i, i)
        min_j = min(min_j, j)
        max_j = max(max_j, j)

        # Add neighbors to stack
        stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

    return mask, (min_i, max_i, min_j, max_j)


def maximal_blobs(info):
    """Determine scale-maximal blobs"""
    # Sort by blob strength
    sorted_indices = sorted(range(len(info)), key=lambda i: info[i]['max_val'], reverse=True)

    for idx in sorted_indices:
        blob = info[idx]
        blocked = False

        # Check if this blob overlaps with any already accepted blob
        for other_idx in sorted_indices[:sorted_indices.index(idx)]:
            other = info[other_idx]
            if other['is_maximal'] == 0:
                continue

            # Check for overlap
            if (blob['p_start'] <= other['p_stop'] and blob['p_stop'] >= other['p_start'] and
                    blob['q_start'] <= other['q_stop'] and blob['q_stop'] >= other['q_start']):
                blocked = True
                break

        info[idx]['is_maximal'] = 0 if blocked else 1

    return info


def measure_particles(im, info, detH, LapG, sub_pixel_mult, coord_system, data_folder,
                      min_h, max_h, min_a, max_a, min_v, max_v):
    """Measure detected particles"""
    particles = []
    measurements = {
        'heights': [],
        'volumes': [],
        'areas': [],
        'avg_heights': [],
        'centers': []
    }

    count = 0

    for i, blob_info in enumerate(info):
        # Skip non-maximal blobs if overlap not allowed
        if blob_info['is_maximal'] == 0:
            continue

        # Skip small blobs
        if blob_info['num_pixels'] < 1:
            continue

        # Create particle folder
        particle_folder = data_folder.create_subfolder(f"Particle_{count}")

        # Crop particle region with padding
        padding = int(np.ceil(max(blob_info['p_stop'] - blob_info['p_start'] + 2,
                                  blob_info['q_stop'] - blob_info['q_start'] + 2)))

        p_min = max(blob_info['p_start'] - padding, 0)
        p_max = min(blob_info['p_stop'] + padding, im.shape[0] - 1)
        q_min = max(blob_info['q_start'] - padding, 0)
        q_max = min(blob_info['q_stop'] + padding, im.shape[1] - 1)

        particle = im[p_min:p_max + 1, q_min:q_max + 1].copy()

        # Create mask
        mask = np.zeros_like(particle)
        # Fill mask based on blob detection
        layer = blob_info['layer']

        # Recreate blob mask for this particle
        seed_p = blob_info['p_seed'] - p_min
        seed_q = blob_info['q_seed'] - q_min

        if 0 <= seed_p < particle.shape[0] and 0 <= seed_q < particle.shape[1]:
            particle_detH = detH[p_min:p_max + 1, q_min:q_max + 1, layer]
            particle_LapG = LapG[p_min:p_max + 1, q_min:q_max + 1, layer]
            mask, _ = scanline_fill_blob(particle_detH, particle_LapG, seed_p, seed_q, 0)

        # Create perimeter
        perimeter = create_perimeter(mask)

        # Subpixel refinement if requested
        if sub_pixel_mult > 1:
            # Create subpixel images
            x_pixels = int(mask.shape[0] * sub_pixel_mult)
            y_pixels = int(mask.shape[1] * sub_pixel_mult)

            sub_particle = np.zeros((x_pixels, y_pixels))
            sub_mask = np.zeros((x_pixels, y_pixels))

            # Bilinear interpolation
            for sp in range(x_pixels):
                for sq in range(y_pixels):
                    p = sp / sub_pixel_mult
                    q = sq / sub_pixel_mult
                    sub_particle[sp, sq] = bilinear_interpolate(particle, p, q)
                    sub_mask[sp, sq] = bilinear_interpolate(mask, p, q)

            sub_mask = (sub_mask > 0.5).astype(float)
            sub_perimeter = create_perimeter(sub_mask)
        else:
            sub_particle = particle
            sub_mask = mask
            sub_perimeter = perimeter

        # Measure particle
        bg = m_min_boundary(sub_particle, sub_mask)
        height = m_height(sub_particle, sub_mask, bg)
        volume = m_volume(sub_particle, sub_mask, bg)
        area = m_area(sub_mask)
        center = m_center_of_mass(sub_particle, sub_mask, bg)
        avg_height = volume / area if area > 0 else 0

        # Check constraints
        if not (min_h < height < max_h and min_a < area < max_a and min_v < volume < max_v):
            continue

        # Create wave note
        note = WaveNote()
        note.add_note("Parent", "Original")
        note.add_note("Date", datetime.now().strftime("%Y-%m-%d"))
        note.add_note("Height", f"{height:.6f}")
        note.add_note("Avg Height", f"{avg_height:.6f}")
        note.add_note("Volume", f"{volume:.6f}")
        note.add_note("Area", f"{area:.6f}")
        note.add_note("Scale", f"{blob_info['scale']:.6f}")
        note.add_note("xCOM", f"{center[0]:.6f}")
        note.add_note("yCOM", f"{center[1]:.6f}")
        note.add_note("pSeed", str(blob_info['p_seed']))
        note.add_note("qSeed", str(blob_info['q_seed']))

        # Save particle data
        particle_folder.save_wave(f"Particle_{count}", particle, note)
        particle_folder.save_wave(f"Mask_{count}", mask)
        particle_folder.save_wave(f"Perimeter_{count}", perimeter)

        if sub_pixel_mult > 1:
            particle_folder.save_wave(f"SubPixParticle_{count}", sub_particle)
            particle_folder.save_wave(f"SubPixMask_{count}", sub_mask)
            particle_folder.save_wave(f"SubPixEdges_{count}", sub_perimeter)

        # Store particle info
        particle_data = {
            'image': particle,
            'mask': mask,
            'perimeter': perimeter,
            'height': height,
            'volume': volume,
            'area': area,
            'avg_height': avg_height,
            'center': center,
            'info': blob_info,
            'folder': particle_folder
        }

        particles.append(particle_data)

        # Store measurements
        measurements['heights'].append(height)
        measurements['volumes'].append(volume)
        measurements['areas'].append(area)
        measurements['avg_heights'].append(avg_height)
        measurements['centers'].append(center)

        count += 1

    return particles, measurements


def create_perimeter(mask):
    """Create perimeter from mask"""
    perimeter = np.zeros_like(mask)

    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 1:
                # Check if on edge
                if (mask[i + 1, j] == 0 or mask[i - 1, j] == 0 or
                        mask[i, j + 1] == 0 or mask[i, j - 1] == 0):
                    perimeter[i, j] = 1

    return perimeter


def create_particle_map(shape, particles):
    """Create map showing particle locations"""
    particle_map = np.full(shape, -1, dtype=float)

    for i, particle in enumerate(particles):
        info = particle['info']
        for p in range(info['p_start'], info['p_stop'] + 1):
            for q in range(info['q_start'], info['q_stop'] + 1):
                if 0 <= p < shape[0] and 0 <= q < shape[1]:
                    particle_map[p, q] = i

    return particle_map


def get_parameters_from_user():
    """Get parameters from user input"""
    print("\nHessian Blob Parameters:")

    scale_start = float(input("Minimum Size in Pixels (default=1): ") or "1")
    layers = int(input("Maximum Size in Pixels (default=256): ") or "256")
    scale_factor = float(input("Scaling Factor (default=1.5): ") or "1.5")
    det_h_response_thresh = float(input("Minimum Blob Strength (-2 for interactive, -1 for Otsu): ") or "-2")
    particle_type = int(input("Particle Type (-1 neg, +1 pos, 0 both): ") or "1")
    sub_pixel_mult = int(input("Subpixel Ratio (default=1): ") or "1")
    allow_overlap = int(input("Allow Hessian Blobs to Overlap? (1=yes 0=no): ") or "0")

    use_constraints = input("\nApply constraints? (y/n): ").lower() == 'y'

    if use_constraints:
        min_h = float(input("Minimum height: ") or "-inf")
        max_h = float(input("Maximum height: ") or "inf")
        min_a = float(input("Minimum area: ") or "-inf")
        max_a = float(input("Maximum area: ") or "inf")
        min_v = float(input("Minimum volume: ") or "-inf")
        max_v = float(input("Maximum volume: ") or "inf")
    else:
        min_h = -np.inf
        max_h = np.inf
        min_a = -np.inf
        max_a = np.inf
        min_v = -np.inf
        max_v = np.inf

    return np.array([
        scale_start, layers, scale_factor, det_h_response_thresh,
        particle_type, sub_pixel_mult, allow_overlap,
        min_h, max_h, min_a, max_a, min_v, max_v
    ])