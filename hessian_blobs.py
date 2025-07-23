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


def scanline_fill_8_lg(detH_layer, mask, LG_layer, seed_p, seed_q, threshold,
                       fill_val=1, x0=None, xf=None, y0=None, yf=None):
    """
    Scanline fill algorithm matching Igor Pro's ScanlineFill8_LG exactly

    Returns:
    tuple: (count, is_boundary_particle)
    """
    # Get bounds
    if x0 is None or xf is None:
        x0 = 0
        xf = detH_layer.shape[0] - 1
    else:
        x0 = max(0, min(detH_layer.shape[0] - 1, round(x0)))
        xf = max(x0, min(detH_layer.shape[0] - 1, round(xf)))

    if y0 is None or yf is None:
        y0 = 0
        yf = detH_layer.shape[1] - 1
    else:
        y0 = max(0, min(detH_layer.shape[1] - 1, round(y0)))
        yf = max(y0, min(detH_layer.shape[1] - 1, round(yf)))

    # Check seed validity
    if seed_p < x0 or seed_q < y0 or seed_p > xf or seed_q > yf:
        return 0, -3

    if detH_layer[seed_p, seed_q] <= threshold:
        return 0, -1

    # Get sign of LG at seed
    sgn = np.sign(LG_layer[seed_p, seed_q])

    # Initialize stack
    stack = [(seed_p, seed_p, seed_q, 0)]
    seed_index = 0
    new_seed_index = 1
    count = 0

    # Bounding box
    min_p, max_p = seed_p, seed_p
    min_q, max_q = seed_q, seed_q

    while seed_index < new_seed_index:
        i0, i, j, state = stack[seed_index]
        seed_index += 1
        go_fish = True

        while True:
            if go_fish:
                i0, i, j, state = stack[seed_index - 1]

            go_fish = True

            if state == 0:  # Scan right and left
                # Update bounding box
                max_p = max(max_p, i)
                min_p = min(min_p, i)
                max_q = max(max_q, j)
                min_q = min(min_q, j)

                # Scan right
                i = i0
                while i <= xf and detH_layer[i, j] >= threshold and np.sign(LG_layer[i, j]) == sgn:
                    mask[i, j] = fill_val
                    count += 1
                    max_p = max(max_p, i)
                    i += 1

                # Scan left
                i = i0 - 1
                while i >= x0 and detH_layer[i, j] >= threshold and np.sign(LG_layer[i, j]) == sgn:
                    mask[i, j] = fill_val
                    count += 1
                    min_p = min(min_p, i)
                    i -= 1

                i = i0
                state = 1

            elif state == 1:  # Search up right
                if j != yf:
                    i = i0
                    while i <= xf and (i == i0 or mask[i - 1, j] == fill_val):
                        if mask[i, j + 1] != fill_val and detH_layer[i, j + 1] >= threshold and \
                                np.sign(LG_layer[i, j + 1]) == sgn:
                            stack.append((i0, i, j, 1))
                            new_seed_index += 1
                            i0 = i
                            j += 1
                            state = 0
                            go_fish = False
                            break
                        i += 1

                if go_fish:
                    state = 2
                    i = i0

            elif state == 2:  # Search up left
                if j != yf:
                    i = i0 - 1
                    while i >= x0 and mask[i + 1, j] == fill_val:
                        if mask[i, j + 1] != fill_val and detH_layer[i, j + 1] >= threshold and \
                                np.sign(LG_layer[i, j + 1]) == sgn:
                            stack.append((i0, i, j, 2))
                            new_seed_index += 1
                            i0 = i
                            j += 1
                            state = 0
                            go_fish = False
                            break
                        i -= 1

                if go_fish:
                    state = 3
                    i = i0

            elif state == 3:  # Search down right
                if j != y0:
                    i = i0
                    while i <= xf and (i == i0 or mask[i - 1, j] == fill_val):
                        if mask[i, j - 1] != fill_val and detH_layer[i, j - 1] >= threshold and \
                                np.sign(LG_layer[i, j - 1]) == sgn:
                            stack.append((i0, i, j, 3))
                            new_seed_index += 1
                            i0 = i
                            j -= 1
                            state = 0
                            go_fish = False
                            break
                        i += 1

                if go_fish:
                    state = 4
                    i = i0

            elif state == 4:  # Search down left
                if j != y0:
                    i = i0 - 1
                    while i >= x0 and mask[i + 1, j] == fill_val:
                        if mask[i, j - 1] != fill_val and detH_layer[i, j - 1] >= threshold and \
                                np.sign(LG_layer[i, j - 1]) == sgn:
                            stack.append((i0, i, j, 4))
                            new_seed_index += 1
                            i0 = i
                            j -= 1
                            state = 0
                            go_fish = False
                            break
                        i -= 1

                if go_fish:
                    break

            if not go_fish:
                continue
            else:
                break

    # Check if boundary particle
    is_bp = 0

    # Check edges
    for i in range(x0, xf + 1):
        if mask[i, y0] == fill_val or mask[i, yf] == fill_val:
            is_bp = 1
            break

    if not is_bp:
        for j in range(y0, yf + 1):
            if mask[x0, j] == fill_val or mask[xf, j] == fill_val:
                is_bp = 1
                break

    return count, is_bp


def scanline_fill_blob(detH_layer, LapG_layer, seed_i, seed_j, threshold):
    """
    Simplified scanline fill for blob detection
    """
    mask = np.zeros_like(detH_layer)

    # Use the full scanline fill algorithm
    count, is_bp = scanline_fill_8_lg(detH_layer, mask, LapG_layer, seed_i, seed_j, threshold)

    # Get bounding box
    if count > 0:
        i_coords, j_coords = np.where(mask > 0)
        min_i, max_i = i_coords.min(), i_coords.max()
        min_j, max_j = j_coords.min(), j_coords.max()
    else:
        min_i, max_i = seed_i, seed_i
        min_j, max_j = seed_j, seed_j

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


def calculate_subpixel_offset(detH, p0, q0, r0):
    """
    Calculate subpixel offset using second-order Taylor approximation
    Matching Igor Pro's implementation exactly
    """
    # Create Jacobian (first derivatives)
    jacobian = np.zeros((2, 1))
    jacobian[0, 0] = (detH[p0 + 1, q0, r0] - detH[p0 - 1, q0, r0]) / 2
    jacobian[1, 0] = (detH[p0, q0 + 1, r0] - detH[p0, q0 - 1, r0]) / 2

    # Create Hessian (second derivatives)
    hessian = np.zeros((2, 2))
    hessian[0, 0] = detH[p0 - 1, q0, r0] - 2 * detH[p0, q0, r0] + detH[p0 + 1, q0, r0]
    hessian[1, 1] = detH[p0, q0 - 1, r0] - 2 * detH[p0, q0, r0] + detH[p0, q0 + 1, r0]
    hessian[0, 1] = (detH[p0 + 1, q0 + 1, r0] + detH[p0 - 1, q0 - 1, r0] -
                     detH[p0 + 1, q0 - 1, r0] - detH[p0 - 1, q0 + 1, r0]) / 4
    hessian[1, 0] = hessian[0, 1]

    # Calculate offset
    try:
        offset = -np.linalg.inv(hessian) @ jacobian
        return offset.flatten()
    except:
        return np.array([0.0, 0.0])


def create_subpixel_images(particle, mask, perimeter, particle_detH, particle_LG,
                           sub_pixel_mult, blob_info, p_min, q_min):
    """
    Create subpixel resolution images matching Igor Pro exactly

    This follows the Igor Pro implementation from lines 206-256 in the main function
    """
    # Calculate subpixel dimensions
    x_pixels = int(round(mask.shape[0] * sub_pixel_mult))
    y_pixels = int(round(mask.shape[1] * sub_pixel_mult))

    # Create subpixel arrays with proper scaling
    # SetScale/P x,DimOffset(mask,0)-DimDelta(mask,0)/2+DimDelta(mask,0)/(2*SubPixelMult),DimDelta(mask,0)/SubPixelMult,SubPixDetH
    # SetScale/P y,DimOffset(mask,1)-DimDelta(mask,1)/2+DimDelta(mask,1)/(2*SubPixelMult),DimDelta(mask,1)/SubPixelMult,SubPixDetH

    # For now we assume unit pixel spacing (DimDelta = 1, DimOffset = 0)
    # The key insight is that subpixel coordinates are offset by half a pixel minus half a subpixel

    # Create empty subpixel arrays
    sub_particle = np.zeros((x_pixels, y_pixels))
    sub_detH = np.zeros((x_pixels, y_pixels))
    sub_LG = np.zeros((x_pixels, y_pixels))

    # First, create expanded boundary mask for interpolation decision
    # This matches Igor Pro's ExpandBoundary8 function
    expanded = np.zeros_like(perimeter)
    for i in range(1, perimeter.shape[0] - 1):
        for j in range(1, perimeter.shape[1] - 1):
            if perimeter[i, j] == 1:
                expanded[i, j] = 1
            elif (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                  mask[i, j + 1] == 1 or mask[i, j - 1] == 1 or
                  mask[i + 1, j + 1] == 1 or mask[i - 1, j + 1] == 1 or
                  mask[i + 1, j - 1] == 1 or mask[i - 1, j - 1] == 1):
                expanded[i, j] = 1

    # Bilinear interpolation following Igor Pro logic
    for sp in range(x_pixels):
        for sq in range(y_pixels):
            # Convert subpixel indices to coordinates in original image
            # This matches Igor Pro's coordinate transformation
            x = -0.5 + 1.0 / (2 * sub_pixel_mult) + sp / sub_pixel_mult
            y = -0.5 + 1.0 / (2 * sub_pixel_mult) + sq / sub_pixel_mult

            # Always interpolate the particle image
            sub_particle[sp, sq] = bilinear_interpolate(particle, x, y)

            # For detH and LG, check if we're in expanded region or mask
            # This matches the Igor Pro logic exactly
            if 0 <= int(x) < expanded.shape[0] and 0 <= int(y) < expanded.shape[1]:
                if expanded[int(x), int(y)] == 1:
                    # In expanded boundary - interpolate
                    sub_detH[sp, sq] = bilinear_interpolate(particle_detH, x, y)
                    sub_LG[sp, sq] = bilinear_interpolate(particle_LG, x, y)
                elif mask[int(x), int(y)] == 1:
                    # Inside mask - use direct value at integer coordinates
                    sub_detH[sp, sq] = particle_detH[int(x), int(y)]
                    sub_LG[sp, sq] = particle_LG[int(x), int(y)]
                else:
                    # Outside - mark as invalid
                    sub_detH[sp, sq] = -1
                    sub_LG[sp, sq] = -1
            else:
                # Outside bounds
                sub_detH[sp, sq] = -1
                sub_LG[sp, sq] = -1

    # Create subpixel mask using scanline fill
    sub_mask = np.zeros((x_pixels, y_pixels))

    # Find seed point in subpixel coordinates
    # The seed point comes from the blob info relative to the particle crop
    seed_p = blob_info['p_seed'] - p_min
    seed_q = blob_info['q_seed'] - q_min

    # Convert to subpixel coordinates
    # This matches Igor Pro's ScaleToIndex conversion
    sub_p0 = int(round(seed_p * sub_pixel_mult))
    sub_q0 = int(round(seed_q * sub_pixel_mult))

    # Ensure seed is within bounds
    sub_p0 = max(0, min(x_pixels - 1, sub_p0))
    sub_q0 = max(0, min(y_pixels - 1, sub_q0))

    # Scanline fill in subpixel space
    count_sub, _ = scanline_fill_8_lg(sub_detH, sub_mask, sub_LG,
                                      sub_p0, sub_q0, 0, fill_val=1)

    # Eliminate single pixel width bridges and edges (Igor Pro lines 240-244)
    # First, clear the mask and refill using scanline fill equal
    sub_mask_clean = np.zeros_like(sub_mask)

    # Create a temporary edge detection array
    sub_edges_detH = np.zeros_like(sub_mask)

    # Find edges in the original subpixel mask
    for i in range(x_pixels):
        for j in range(y_pixels):
            if sub_mask[i, j] == 1:
                # Check if on edge (4-connected)
                is_edge = False
                if i == 0 or i == x_pixels - 1 or j == 0 or j == y_pixels - 1:
                    is_edge = True
                elif (sub_mask[min(i + 1, x_pixels - 1), j] == 0 or
                      sub_mask[max(0, i - 1), j] == 0 or
                      sub_mask[i, min(j + 1, y_pixels - 1)] == 0 or
                      sub_mask[i, max(0, j - 1)] == 0):
                    is_edge = True

                if is_edge:
                    sub_edges_detH[i, j] = 1

    # Refill using edges as source
    count_clean, _ = scanline_fill_equal(sub_edges_detH, sub_mask_clean,
                                         sub_p0, sub_q0, fill_val=1)

    # Expand boundary by 4-connectivity
    expand_boundary_4(sub_mask_clean)

    # Recalculate edges on cleaned mask
    sub_edges_final = np.zeros_like(sub_mask_clean)
    for i in range(x_pixels):
        for j in range(y_pixels):
            if sub_mask_clean[i, j] == 1:
                # Check if on edge (4-connected)
                is_edge = False
                if i == 0 or i == x_pixels - 1 or j == 0 or j == y_pixels - 1:
                    is_edge = True
                elif (sub_mask_clean[min(i + 1, x_pixels - 1), j] == 0 or
                      sub_mask_clean[max(0, i - 1), j] == 0 or
                      sub_mask_clean[i, min(j + 1, y_pixels - 1)] == 0 or
                      sub_mask_clean[i, max(0, j - 1)] == 0):
                    is_edge = True

                if is_edge:
                    sub_edges_final[i, j] = 1

    return (sub_particle, sub_mask_clean, sub_edges_final,
            sub_detH, sub_LG, x_pixels, y_pixels)


def bilinear_interpolate(im, x0, y0, r0=0):
    """
    Bilinear interpolation matching Igor Pro implementation exactly

    This matches the BilinearInterpolate function in Igor Pro
    """
    if im.ndim == 2:
        # For 2D images
        # Calculate pixel coordinates
        pMid = x0  # In Igor, this would be (x0-DimOffset(im,0))/DimDelta(im,0)
        p0 = max(0, int(np.floor(pMid)))
        p1 = min(im.shape[0] - 1, int(np.ceil(pMid)))

        qMid = y0  # In Igor, this would be (y0-DimOffset(im,1))/DimDelta(im,1)
        q0 = max(0, int(np.floor(qMid)))
        q1 = min(im.shape[1] - 1, int(np.ceil(qMid)))

        # Bilinear interpolation
        # First interpolate in p direction
        pInterp0 = im[p0, q0] + (im[p1, q0] - im[p0, q0]) * (pMid - p0)
        pInterp1 = im[p0, q1] + (im[p1, q1] - im[p0, q1]) * (pMid - p0)

        # Then interpolate in q direction
        return pInterp0 + (pInterp1 - pInterp0) * (qMid - q0)
    else:
        # For 3D images, use the specified layer
        pMid = x0
        p0 = max(0, int(np.floor(pMid)))
        p1 = min(im.shape[0] - 1, int(np.ceil(pMid)))

        qMid = y0
        q0 = max(0, int(np.floor(qMid)))
        q1 = min(im.shape[1] - 1, int(np.ceil(qMid)))

        # Ensure r0 is valid
        r0 = int(r0)
        if r0 < 0 or r0 >= im.shape[2]:
            r0 = 0

        pInterp0 = im[p0, q0, r0] + (im[p1, q0, r0] - im[p0, q0, r0]) * (pMid - p0)
        pInterp1 = im[p0, q1, r0] + (im[p1, q1, r0] - im[p0, q1, r0]) * (pMid - p0)

        return pInterp0 + (pInterp1 - pInterp0) * (qMid - q0)


def create_subpixel_perimeter(mask):
    """Create perimeter from mask - exact Igor Pro implementation"""
    perimeter = np.zeros_like(mask)
    x_pixels, y_pixels = mask.shape

    for i in range(x_pixels):
        for j in range(y_pixels):
            if mask[i, j] == 1:
                # Check 4-connected neighbors
                if (i == 0 or i == x_pixels - 1 or j == 0 or j == y_pixels - 1 or
                        mask[min(i + 1, x_pixels - 1), j] == 0 or
                        mask[max(0, i - 1), j] == 0 or
                        mask[i, min(j + 1, y_pixels - 1)] == 0 or
                        mask[i, max(0, j - 1)] == 0):
                    perimeter[i, j] = 1

    return perimeter


def measure_particles(im, info, detH, LapG, sub_pixel_mult, coord_system, data_folder,
                      min_h, max_h, min_a, max_a, min_v, max_v, allow_boundary_particles=1):
    """Measure detected particles - exact Igor Pro implementation"""
    particles = []
    measurements = {
        'heights': [],
        'volumes': [],
        'areas': [],
        'avg_heights': [],
        'centers': []
    }

    count = 0

    # Process particles from largest to smallest (reverse order)
    for i in range(len(info) - 1, -1, -1):
        blob_info = info[i]

        # Skip non-maximal blobs if overlap not allowed
        if 'is_maximal' in blob_info and blob_info['is_maximal'] == 0:
            continue

        # Skip small blobs
        if blob_info['num_pixels'] < 1:
            continue

        # Skip invalid blobs
        if (blob_info['p_stop'] - blob_info['p_start']) < 0 or \
                (blob_info['q_stop'] - blob_info['q_start']) < 0:
            continue

        # Consider boundary particles?
        if allow_boundary_particles == 0:
            if (blob_info['p_start'] <= 2 or blob_info['p_stop'] >= im.shape[0] - 3 or
                    blob_info['q_start'] <= 2 or blob_info['q_stop'] >= im.shape[1] - 3):
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

        # Extract particle region
        particle = im[p_min:p_max + 1, q_min:q_max + 1].copy()

        # Create mask using blob detection info
        mask = np.zeros_like(particle)
        layer = blob_info['layer']

        # Fill mask based on stored blob mask or recreate it
        if 'blob_mask' in blob_info:
            # Use stored mask from blob detection
            for pi in range(p_min, p_max + 1):
                for qi in range(q_min, q_max + 1):
                    if 0 <= pi < im.shape[0] and 0 <= qi < im.shape[1]:
                        if blob_info['blob_mask'][pi, qi] > 0:
                            mask[pi - p_min, qi - q_min] = 1
        else:
            # Recreate mask using scanline fill
            particle_detH = detH[p_min:p_max + 1, q_min:q_max + 1, layer].copy()
            particle_LG = LapG[p_min:p_max + 1, q_min:q_max + 1, layer].copy()

            seed_p = blob_info['p_seed'] - p_min
            seed_q = blob_info['q_seed'] - q_min

            if 0 <= seed_p < particle.shape[0] and 0 <= seed_q < particle.shape[1]:
                count_pixels, _ = scanline_fill_8_lg(particle_detH, mask, particle_LG,
                                                     seed_p, seed_q, 0, fill_val=1)

        # Create perimeter
        perimeter = create_perimeter(mask)

        # Extract detH and LG for this particle
        particle_detH = detH[p_min:p_max + 1, q_min:q_max + 1, layer].copy()
        particle_LG = LapG[p_min:p_max + 1, q_min:q_max + 1, layer].copy()

        # Create coordinate system for particle
        particle_coord_system = CoordinateSystem(
            particle.shape,
            x_start=coord_system.x_start + p_min * coord_system.x_delta,
            x_delta=coord_system.x_delta,
            y_start=coord_system.y_start + q_min * coord_system.y_delta,
            y_delta=coord_system.y_delta
        )

        # Subpixel refinement if requested
        if sub_pixel_mult > 1:
            # Calculate subpixel dimensions
            x_pixels = int(round(mask.shape[0] * sub_pixel_mult))
            y_pixels = int(round(mask.shape[1] * sub_pixel_mult))

            # Create subpixel arrays
            sub_particle = np.zeros((x_pixels, y_pixels))
            sub_detH = np.zeros((x_pixels, y_pixels))
            sub_LG = np.zeros((x_pixels, y_pixels))

            # Expand boundary for interpolation
            expanded = expand_boundary_8(perimeter.copy())

            # Bilinear interpolation
            for sp in range(x_pixels):
                for sq in range(y_pixels):
                    # Get coordinates in original particle image
                    p = sp / sub_pixel_mult
                    q = sq / sub_pixel_mult

                    # Check if we should interpolate
                    p_int = int(p)
                    q_int = int(q)

                    if 0 <= p_int < expanded.shape[0] and 0 <= q_int < expanded.shape[1]:
                        if expanded[p_int, q_int] == 1:
                            # On boundary - interpolate
                            sub_detH[sp, sq] = bilinear_interpolate(particle_detH, p, q)
                            sub_LG[sp, sq] = bilinear_interpolate(particle_LG, p, q)
                        elif mask[p_int, q_int] == 1:
                            # Inside particle - use direct value
                            sub_detH[sp, sq] = particle_detH[p_int, q_int]
                            sub_LG[sp, sq] = particle_LG[p_int, q_int]
                        else:
                            # Outside - mark as invalid
                            sub_detH[sp, sq] = -1
                            sub_LG[sp, sq] = -1

                    # Always interpolate particle values
                    sub_particle[sp, sq] = bilinear_interpolate(particle, p, q)

            # Create subpixel mask using scanline fill
            sub_mask = np.zeros((x_pixels, y_pixels))

            # Find seed point in subpixel coordinates
            seed_p = blob_info['p_seed'] - p_min
            seed_q = blob_info['q_seed'] - q_min
            sub_p0 = int(round(seed_p * sub_pixel_mult))
            sub_q0 = int(round(seed_q * sub_pixel_mult))

            # Ensure seed is within bounds
            sub_p0 = max(0, min(x_pixels - 1, sub_p0))
            sub_q0 = max(0, min(y_pixels - 1, sub_q0))

            # Scanline fill in subpixel space
            count_sub, _ = scanline_fill_8_lg(sub_detH, sub_mask, sub_LG,
                                              sub_p0, sub_q0, 0, fill_val=1)

            # Clean up single pixel bridges
            sub_mask_clean = np.zeros_like(sub_mask)
            count_clean, _ = scanline_fill_equal(sub_mask, sub_mask_clean,
                                                 sub_p0, sub_q0, fill_val=1)
            expand_boundary_4(sub_mask_clean)

            # Create final subpixel perimeter
            sub_perimeter = create_subpixel_perimeter(sub_mask_clean)

            # Find subpixel scale-space extrema centers
            p0 = blob_info['p_seed']
            q0 = blob_info['q_seed']
            r0 = blob_info['layer']

            # Check bounds for derivative calculation
            if 1 <= p0 < detH.shape[0] - 1 and 1 <= q0 < detH.shape[1] - 1:
                sub_pixel_offset = calculate_subpixel_offset(detH, p0, q0, r0)

                subpix_x = coord_system.x_start + coord_system.x_delta * (p0 + sub_pixel_offset[0])
                subpix_y = coord_system.y_start + coord_system.y_delta * (q0 + sub_pixel_offset[1])
            else:
                sub_pixel_offset = np.array([0.0, 0.0])
                subpix_x = coord_system.x_start + coord_system.x_delta * p0
                subpix_y = coord_system.y_start + coord_system.y_delta * q0

            # Measure on subpixel images
            bg = m_min_boundary(sub_particle, sub_mask_clean)
            particle -= bg
            sub_particle -= bg

            height = m_height(particle, mask, 0)
            volume = m_volume(sub_particle, sub_mask_clean, 0)
            center = m_center_of_mass(sub_particle, sub_mask_clean, 0)
            area = m_area(sub_mask_clean)
            perimeter_length = m_perimeter(sub_mask_clean)
            avg_height = volume / area if area > 0 else 0

        else:
            # No subpixel refinement
            bg = m_min_boundary(particle, mask)
            particle -= bg

            height = m_height(particle, mask, 0)
            volume = m_volume(particle, mask, 0)
            center = m_center_of_mass(particle, mask, 0)
            area = m_area(mask)
            perimeter_length = m_perimeter(mask)
            avg_height = volume / area if area > 0 else 0

            # No subpixel offset
            sub_pixel_offset = np.array([0.0, 0.0])
            subpix_x = coord_system.x_start + coord_system.x_delta * blob_info['p_seed']
            subpix_y = coord_system.y_start + coord_system.y_delta * blob_info['q_seed']

        # Check constraints
        if not (min_h < height < max_h and min_a < area < max_a and min_v < volume < max_v):
            continue

        # Accept the particle
        blob_info['status'] = count

        # Create wave note
        note = WaveNote()
        note.add_note("Parent", "Original")
        note.add_note("Date", datetime.now().strftime("%a, %b %d, %Y"))
        note.add_note("Height", f"{height:.6e}")
        note.add_note("Avg Height", f"{avg_height:.6e}")
        note.add_note("Volume", f"{volume:.6e}")
        note.add_note("Area", f"{area:.6e}")
        note.add_note("Perimeter", f"{perimeter_length:.6e}")
        note.add_note("Scale", f"{blob_info['scale']:.6e}")
        note.add_note("xCOM", f"{center[0]:.6e}")
        note.add_note("yCOM", f"{center[1]:.6e}")
        note.add_note("pSeed", str(blob_info['p_seed']))
        note.add_note("qSeed", str(blob_info['q_seed']))
        note.add_note("rSeed", str(blob_info['layer']))
        note.add_note("subPixelXOffset", f"{sub_pixel_offset[0]:.6f}")
        note.add_note("subPixelYOffset", f"{sub_pixel_offset[1]:.6f}")
        note.add_note("subPixelXCenter", f"{subpix_x:.6e}")
        note.add_note("subPixelYCenter", f"{subpix_y:.6e}")

        # Save particle data
        particle_folder.save_wave(f"Particle_{count}", particle, note)
        particle_folder.save_wave(f"Mask_{count}", mask)
        particle_folder.save_wave(f"Perimeter_{count}", perimeter)
        particle_folder.save_wave(f"ParticleDetH_{count}", particle_detH)
        particle_folder.save_wave(f"ParticleLG_{count}", particle_LG)

        if sub_pixel_mult > 1:
            particle_folder.save_wave(f"SubPixParticle_{count}", sub_particle)
            particle_folder.save_wave(f"SubPixMask_{count}", sub_mask_clean)
            particle_folder.save_wave(f"SubPixEdges_{count}", sub_perimeter)
            particle_folder.save_wave(f"SubPixDetH_{count}", sub_detH)
            particle_folder.save_wave(f"SubPixLG_{count}", sub_LG)

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


def scanline_fill_equal(image, dest, seed_p, seed_q, fill_val=1, x0=None, xf=None, y0=None, yf=None):
    """
    Scanline fill for equal values - matching Igor Pro's ScanlineFillEqual
    """
    # Get bounds
    if x0 is None or xf is None:
        x0 = 0
        xf = image.shape[0] - 1
    else:
        x0 = max(0, min(image.shape[0] - 1, round(x0)))
        xf = max(x0, min(image.shape[0] - 1, round(xf)))

    if y0 is None or yf is None:
        y0 = 0
        yf = image.shape[1] - 1
    else:
        y0 = max(0, min(image.shape[1] - 1, round(y0)))
        yf = max(y0, min(image.shape[1] - 1, round(yf)))

    # Check seed validity
    if seed_p < x0 or seed_q < y0 or seed_p > xf or seed_q > yf:
        return 0, -3

    # The value to seed fill
    val = image[seed_p, seed_q]

    # Initialize stack
    stack = [(seed_p, seed_p, seed_q, 0)]
    seed_index = 0
    new_seed_index = 1
    count = 0

    while seed_index < new_seed_index:
        i0, i, j, state = stack[seed_index]
        seed_index += 1
        go_fish = True

        while True:
            if go_fish:
                i0, i, j, state = stack[seed_index - 1]

            go_fish = True

            if state == 0:  # Scan right and left
                # Scan right
                i = i0
                while i <= xf and image[i, j] == val:
                    dest[i, j] = fill_val
                    count += 1
                    i += 1

                # Scan left
                i = i0 - 1
                while i >= x0 and image[i, j] == val:
                    dest[i, j] = fill_val
                    count += 1
                    i -= 1

                i = i0
                state = 1

            elif state == 1:  # Search up right
                if j != yf:
                    i = i0
                    while i <= xf and dest[i, j] == fill_val:
                        if dest[i, j + 1] != fill_val and image[i, j + 1] == val:
                            stack.append((i0, i, j, 1))
                            new_seed_index += 1
                            i0 = i
                            j += 1
                            state = 0
                            go_fish = False
                            break
                        i += 1

                if go_fish:
                    state = 2
                    i = i0

            elif state == 2:  # Search up left
                if j != yf:
                    i = i0 - 1
                    while i >= x0 and dest[i, j] == fill_val:
                        if dest[i, j + 1] != fill_val and image[i, j + 1] == val:
                            stack.append((i0, i, j, 2))
                            new_seed_index += 1
                            i0 = i
                            j += 1
                            state = 0
                            go_fish = False
                            break
                        i -= 1

                if go_fish:
                    state = 3
                    i = i0

            elif state == 3:  # Search down right
                if j != y0:
                    i = i0
                    while i <= xf and dest[i, j] == fill_val:
                        if dest[i, j - 1] != fill_val and image[i, j - 1] == val:
                            stack.append((i0, i, j, 3))
                            new_seed_index += 1
                            i0 = i
                            j -= 1
                            state = 0
                            go_fish = False
                            break
                        i += 1

                if go_fish:
                    state = 4
                    i = i0

            elif state == 4:  # Search down left
                if j != y0:
                    i = i0 - 1
                    while i >= x0 and dest[i, j] == fill_val:
                        if dest[i, j - 1] != fill_val and image[i, j - 1] == val:
                            stack.append((i0, i, j, 4))
                            new_seed_index += 1
                            i0 = i
                            j -= 1
                            state = 0
                            go_fish = False
                            break
                        i -= 1

                if go_fish:
                    break

            if not go_fish:
                continue
            else:
                break

    # Check if boundary particle
    is_bp = 0

    # Check edges
    for i in range(x0, xf + 1):
        if dest[i, y0] == fill_val or dest[i, yf] == fill_val:
            is_bp = 1
            break

    if not is_bp:
        for j in range(y0, yf + 1):
            if dest[x0, j] == fill_val or dest[xf, j] == fill_val:
                is_bp = 1
                break

    return count, is_bp