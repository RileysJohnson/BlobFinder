import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from utilities import CoordinateSystem


def scale_space_representation(im, layers, t0, t_factor, coord_system=None):
    """
    Computes the discrete scale-space representation L of an image

    Parameters:
    im: The image to compute L from
    layers: The number of layers of L
    t0: The scale of the first layer of L, provided in pixel units
    t_factor: The scaling factor for the scale between layers of L
    coord_system: Optional coordinate system for the image

    Returns:
    L: Scale-space representation
    scale_coords: Coordinate system for scale dimension
    """
    if coord_system is None:
        coord_system = CoordinateSystem(im.shape)

    # Convert t0 to image units
    t0 = (t0 * coord_system.x_delta) ** 2

    # Go to Fourier space
    im_fft = fft2(im)

    # Get frequency coordinates
    freq_x = fftfreq(im.shape[0], d=coord_system.x_delta)
    freq_y = fftfreq(im.shape[1], d=coord_system.y_delta)
    fx, fy = np.meshgrid(freq_y, freq_x)

    # Make the layers
    L = np.zeros((*im.shape, layers))
    scales = []

    for i in range(layers):
        scale = t0 * (t_factor ** i)
        scales.append(scale)

        # Gaussian in Fourier space
        gaussian_fft = np.exp(-(fx ** 2 + fy ** 2) * np.pi ** 2 * 2 * scale)

        # Convolve and store
        layer_fft = im_fft * gaussian_fft
        L[:, :, i] = np.real(ifft2(layer_fft))

    # Create scale coordinate system
    scale_coords = {
        'start': t0,
        'factor': t_factor,
        'scales': scales
    }

    return L, scale_coords


def blob_detectors(L, gamma_norm, coord_system=None):
    """
    Computes blob detectors: determinant of Hessian and Laplacian of Gaussian

    Parameters:
    L: Scale-space representation
    gamma_norm: Gamma normalization factor
    coord_system: Optional coordinate system

    Returns:
    detH: Determinant of Hessian
    LapG: Laplacian of Gaussian
    """
    if coord_system is None:
        coord_system = CoordinateSystem(L.shape[:2])

    # Define convolution kernels for derivatives
    # Using 5-point stencil for accuracy
    lxx_kernel = np.array([[-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12]]).reshape(5, 1)
    lyy_kernel = lxx_kernel.T

    # For mixed derivative
    lxy_kernel = np.array([
        [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
        [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
        [0, 0, 0, 0, 0],
        [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
        [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
    ])

    # Compute derivatives
    Lxx = np.zeros_like(L)
    Lyy = np.zeros_like(L)
    Lxy = np.zeros_like(L)

    for i in range(L.shape[2]):
        Lxx[:, :, i] = ndimage.convolve(L[:, :, i], lxx_kernel, mode='constant')
        Lyy[:, :, i] = ndimage.convolve(L[:, :, i], lyy_kernel, mode='constant')
        Lxy[:, :, i] = ndimage.convolve(L[:, :, i], lxy_kernel, mode='constant')

    # Compute Laplacian of Gaussian
    LapG = Lxx + Lyy

    # Gamma normalize and account for pixel spacing
    for i in range(L.shape[2]):
        scale = coord_system.x_start + coord_system.x_delta * (coord_system.y_delta ** i)
        LapG[:, :, i] *= (scale ** gamma_norm) / (coord_system.x_delta * coord_system.y_delta)

    # Fix boundaries
    fix_boundaries(LapG)

    # Compute determinant of Hessian
    detH = Lxx * Lyy - Lxy ** 2

    # Gamma normalize
    for i in range(L.shape[2]):
        scale = coord_system.x_start + coord_system.x_delta * (coord_system.y_delta ** i)
        detH[:, :, i] *= (scale ** (2 * gamma_norm)) / ((coord_system.x_delta * coord_system.y_delta) ** 2)

    # Fix boundaries
    fix_boundaries(detH)

    return detH, LapG


def fix_boundaries(arr):
    """Fix boundary issues from convolution"""
    # Handle edges
    limP, limQ = arr.shape[0] - 1, arr.shape[1] - 1

    # Sides
    for i in range(2, limP - 1):
        arr[i, 0, :] = arr[i, 2, :] / 3
        arr[i, 1, :] = arr[i, 2, :] * 2 / 3
        arr[i, limQ, :] = arr[i, limQ - 2, :] / 3
        arr[i, limQ - 1, :] = arr[i, limQ - 2, :] * 2 / 3

    for j in range(2, limQ - 1):
        arr[0, j, :] = arr[2, j, :] / 3
        arr[1, j, :] = arr[2, j, :] * 2 / 3
        arr[limP, j, :] = arr[limP - 2, j, :] / 3
        arr[limP - 1, j, :] = arr[limP - 2, j, :] * 2 / 3

    # Corners - average neighboring edges
    # Top left
    arr[1, 1, :] = (arr[1, 2, :] + arr[2, 1, :]) / 2
    arr[1, 0, :] = (arr[1, 1, :] + arr[2, 0, :]) / 2
    arr[0, 1, :] = (arr[1, 1, :] + arr[0, 2, :]) / 2
    arr[0, 0, :] = (arr[0, 1, :] + arr[1, 0, :]) / 2

    # Bottom right
    arr[limP - 1, limQ - 1, :] = (arr[limP - 1, limQ - 2, :] + arr[limP - 2, limQ - 1, :]) / 2
    arr[limP - 1, limQ, :] = (arr[limP - 1, limQ - 1, :] + arr[limP - 2, limQ, :]) / 2
    arr[limP, limQ - 1, :] = (arr[limP - 1, limQ - 1, :] + arr[limP, limQ - 2, :]) / 2
    arr[limP, limQ, :] = (arr[limP - 1, limQ, :] + arr[limP, limQ - 1, :]) / 2

    # Top right
    arr[limP - 1, 1, :] = (arr[limP - 1, 2, :] + arr[limP - 2, 1, :]) / 2
    arr[limP - 1, 0, :] = (arr[limP - 1, 1, :] + arr[limP - 2, 0, :]) / 2
    arr[limP, 1, :] = (arr[limP - 1, 1, :] + arr[limP, 2, :]) / 2
    arr[limP, 0, :] = (arr[limP - 1, 0, :] + arr[limP, 1, :]) / 2

    # Bottom left
    arr[1, limQ - 1, :] = (arr[1, limQ - 2, :] + arr[2, limQ - 1, :]) / 2
    arr[1, limQ, :] = (arr[1, limQ - 1, :] + arr[2, limQ, :]) / 2
    arr[0, limQ - 1, :] = (arr[1, limQ - 1, :] + arr[0, limQ - 2, :]) / 2
    arr[0, limQ, :] = (arr[1, limQ, :] + arr[0, limQ - 1, :]) / 2


def find_scale_space_maxima(detH, LG, particleType, maxCurvatureRatio):
    """
    Find local maxima in scale space

    Returns:
    maxes: Array of maximum values
    map_data: 2D map of maximum detH values
    scale_map: 2D map of scales at maxima
    """
    maxes = []
    map_data = np.zeros(detH.shape[:2])
    scale_map = np.zeros(detH.shape[:2])

    limI, limJ, limK = detH.shape[0] - 1, detH.shape[1] - 1, detH.shape[2] - 1

    for k in range(1, limK - 1):
        for i in range(1, limI):
            for j in range(1, limJ):
                # Check curvature ratio
                if LG[i, j, k] ** 2 / detH[i, j, k] >= (maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                    continue

                # Check particle type
                if particleType == -1 and LG[i, j, k] < 0:
                    continue
                elif particleType == 1 and LG[i, j, k] > 0:
                    continue

                # Check if local maximum
                center_val = detH[i, j, k]

                # Check 26 neighbors
                is_max = True

                # Check same scale neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if center_val <= detH[i + di, j + dj, k]:
                            is_max = False
                            break
                    if not is_max:
                        break

                # Check scale neighbors
                if is_max and k > 0:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if center_val <= detH[i + di, j + dj, k - 1]:
                                is_max = False
                                break
                        if not is_max:
                            break

                if is_max and k < limK - 1:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if center_val < detH[i + di, j + dj, k + 1]:
                                is_max = False
                                break
                        if not is_max:
                            break

                if is_max:
                    maxes.append(center_val)
                    if center_val > map_data[i, j]:
                        map_data[i, j] = center_val
                        scale_map[i, j] = k  # Store scale index

    return np.array(maxes), map_data, scale_map


def otsu_threshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Use Otsu's method to find optimal threshold
    """
    # Get maxima
    maxes, _, _ = find_scale_space_maxima(detH, LG, particleType, maxCurvatureRatio)

    if len(maxes) == 0:
        return 0

    # Create histogram
    hist, bin_edges = np.histogram(maxes, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find optimal threshold using Otsu's method
    total = len(maxes)
    best_thresh = 0
    min_icv = np.inf

    for i in range(1, len(hist)):
        w0 = np.sum(hist[:i])
        w1 = np.sum(hist[i:])

        if w0 == 0 or w1 == 0:
            continue

        mean0 = np.sum(bin_centers[:i] * hist[:i]) / w0
        mean1 = np.sum(bin_centers[i:] * hist[i:]) / w1

        var0 = np.sum(((bin_centers[:i] - mean0) ** 2) * hist[:i]) / w0
        var1 = np.sum(((bin_centers[i:] - mean1) ** 2) * hist[i:]) / w1

        icv = w0 * var0 + w1 * var1

        if icv < min_icv:
            min_icv = icv
            best_thresh = bin_edges[i]

    return best_thresh