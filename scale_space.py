"""
Scale-Space Functions
Handles scale-space representation and blob detector computations
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2
from igor_compatibility import *
from file_io import data_browser


def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """
    Computes the discrete scale-space representation L of an image.
        im : The image to compute L from.
        layers : The number of layers of L.
        t0 : The scale of the first layer of L, provided in pixel units.
        tFactor : The scaling factor for the scale between layers of L.
    """
    # Convert t0 to image units.
    t0 = (t0 * DimDelta(im, 0)) ** 2

    # Go to Fourier space.
    im_fft = fft2(im.data)

    # Make the layers of the scale-space representation and convolve in Fourier space.
    family = []
    names = []

    for i in range(layers):
        # Create frequency coordinates
        height, width = im.data.shape
        u = np.fft.fftfreq(height, DimDelta(im, 0))
        v = np.fft.fftfreq(width, DimDelta(im, 1))
        U, V = np.meshgrid(v, u)

        # Create Gaussian kernel in frequency domain
        scale = t0 * (tFactor ** i)
        kernel = np.exp(-(U ** 2 + V ** 2) * np.pi ** 2 * 2 * scale)

        # Apply kernel
        layer_fft = im_fft * kernel
        layer_data = np.real(ifft2(layer_fft))

        layer_wave = Wave(layer_data, f"L_{i}")
        layer_wave.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
        layer_wave.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

        family.append(layer_wave)
        names.append(f"L_{i}")

    # Concatenate layers of the scale-space representation into a 3D wave.
    L_data = np.stack([layer.data for layer in family], axis=2)
    L = Wave(L_data, f"{im.name}_L")
    L.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    L.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))
    L.SetScale('z', t0, tFactor)

    return L


def BlobDetectors(L, gammaNorm):
    """
    Computes the two blob detectors, the determinant of the Hessian and the Laplacian of Gaussian.
        L : The scale-space representation of the image.
        gammaNorm : The gamma normalization factor, see Lindeberg 1998. Should be set to 1 in most blob detection cases.
    """
    # Make convolution kernels for calculating central difference derivatives.
    LxxKernel = Wave(np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]), "LxxKernel")

    LyyKernel = Wave(np.array([
        [0, 0, -1 / 12, 0, 0],
        [0, 0, 16 / 12, 0, 0],
        [0, 0, -30 / 12, 0, 0],
        [0, 0, 16 / 12, 0, 0],
        [0, 0, -1 / 12, 0, 0]
    ]), "LyyKernel")

    LxyKernel = Wave(np.array([
        [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
        [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
        [0, 0, 0, 0, 0],
        [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
        [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
    ]), "LxyKernel")

    # Compute Lxx and Lyy. (Second partial derivatives of L).
    Lxx_data = np.zeros_like(L.data)
    Lyy_data = np.zeros_like(L.data)

    for k in range(L.data.shape[2]):
        Lxx_data[:, :, k] = ndimage.convolve(L.data[:, :, k], LxxKernel.data, mode='constant')
        Lyy_data[:, :, k] = ndimage.convolve(L.data[:, :, k], LyyKernel.data, mode='constant')

    Lxx = Wave(Lxx_data, "Lxx")
    Lyy = Wave(Lyy_data, "Lyy")

    # Compute the Laplacian of Gaussian.
    LapG_data = Lxx_data + Lyy_data
    LapG = Wave(LapG_data, "LapG")

    # Set the image scale.
    LapG.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    LapG.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    LapG.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    # Gamma normalize and account for pixel spacing.
    for k in range(LapG.data.shape[2]):
        scale_factor = (DimOffset(L, 2) * (DimDelta(L, 2) ** k)) ** gammaNorm
        pixel_factor = DimDelta(L, 0) * DimDelta(L, 1)
        LapG.data[:, :, k] *= scale_factor / pixel_factor

    # Fix errors on the boundary of the image.
    FixBoundaries(LapG)

    # Compute the determinant of the Hessian.
    Lxy_data = np.zeros_like(L.data)
    for k in range(L.data.shape[2]):
        Lxy_data[:, :, k] = ndimage.convolve(L.data[:, :, k], LxyKernel.data, mode='constant')

    detH_data = Lxx_data * Lyy_data - Lxy_data ** 2
    detH = Wave(detH_data.astype(np.float32), "detH")

    # Set the scaling.
    detH.SetScale('x', DimOffset(L, 0), DimDelta(L, 0))
    detH.SetScale('y', DimOffset(L, 1), DimDelta(L, 1))
    detH.SetScale('z', DimOffset(L, 2), DimDelta(L, 2))

    # Gamma normalize and account for pixel spacing.
    for k in range(detH.data.shape[2]):
        scale_factor = (DimOffset(L, 2) * (DimDelta(L, 2) ** k)) ** (2 * gammaNorm)
        pixel_factor = (DimDelta(L, 0) * DimDelta(L, 1)) ** 2
        detH.data[:, :, k] *= scale_factor / pixel_factor

    # Fix the boundary issues again.
    FixBoundaries(detH)

    # Store results in data browser (simulating Igor's global storage)
    current_folder = data_browser  # Simplified
    current_folder.add_wave(LapG, "LapG")
    current_folder.add_wave(detH, "detH")

    return 0


def FixBoundaries(detH):
    """
    Fixes a boundary issue in the blob detectors. Arises from trying to measure derivatives on the boundary.
        detH : The determinant of Hessian blob detector, but also works for the Laplacian of Gaussian.
    """
    limP = detH.data.shape[0] - 1
    limQ = detH.data.shape[1] - 1

    # Do the sides first. Corners need extra care.
    # Make the edges fade off so that maxima can still be detected.
    for i in range(2, limP - 1):
        detH.data[i, 0, :] = detH.data[i, 2, :] / 3
        detH.data[i, 1, :] = detH.data[i, 2, :] * 2 / 3

    for i in range(2, limP - 1):
        detH.data[i, limQ, :] = detH.data[i, limQ - 2, :] / 3
        detH.data[i, limQ - 1, :] = detH.data[i, limQ - 2, :] * 2 / 3

    for i in range(2, limQ - 1):
        detH.data[0, i, :] = detH.data[2, i, :] / 3
        detH.data[1, i, :] = detH.data[2, i, :] * 2 / 3

    for i in range(2, limQ - 1):
        detH.data[limP, i, :] = detH.data[limP - 2, i, :] / 3
        detH.data[limP - 1, i, :] = detH.data[limP - 2, i, :] * 2 / 3

    # Top Left Corner
    detH.data[1, 1, :] = (detH.data[1, 2, :] + detH.data[2, 1, :]) / 2
    detH.data[1, 0, :] = (detH.data[1, 1, :] + detH.data[2, 0, :]) / 2
    detH.data[0, 1, :] = (detH.data[1, 1, :] + detH.data[0, 2, :]) / 2
    detH.data[0, 0, :] = (detH.data[0, 1, :] + detH.data[1, 0, :]) / 2

    # Bottom Right Corner
    detH.data[limP - 1, limQ - 1, :] = (detH.data[limP - 1, limQ - 2, :] + detH.data[limP - 2, limQ - 1, :]) / 2
    detH.data[limP - 1, limQ, :] = (detH.data[limP - 1, limQ - 1, :] + detH.data[limP - 2, limQ, :]) / 2
    detH.data[limP, limQ - 1, :] = (detH.data[limP - 1, limQ - 1, :] + detH.data[limP, limQ - 2, :]) / 2
    detH.data[limP, limQ, :] = (detH.data[limP - 1, limQ, :] + detH.data[limP, limQ - 1, :]) / 2

    # Top Right Corner
    detH.data[limP - 1, 1, :] = (detH.data[limP - 1, 2, :] + detH.data[limP - 2, 1, :]) / 2
    detH.data[limP - 1, 0, :] = (detH.data[limP - 1, 1, :] + detH.data[limP - 2, 0, :]) / 2
    detH.data[limP, 1, :] = (detH.data[limP - 1, 1, :] + detH.data[limP, 2, :]) / 2
    detH.data[limP, 0, :] = (detH.data[limP - 1, 0, :] + detH.data[limP, 1, :]) / 2

    # Bottom Left Corner
    detH.data[1, limQ - 1, :] = (detH.data[1, limQ - 2, :] + detH.data[2, limQ - 1, :]) / 2
    detH.data[1, limQ, :] = (detH.data[1, limQ - 1, :] + detH.data[2, limQ, :]) / 2
    detH.data[0, limQ - 1, :] = (detH.data[1, limQ - 1, :] + detH.data[0, limQ - 2, :]) / 2
    detH.data[0, limQ, :] = (detH.data[1, limQ, :] + detH.data[0, limQ - 1, :]) / 2

    return 0


def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Uses Otsu's method to automatically define a threshold blob strength.
        detH : The determinant of Hessian blob detector.
        LG : The Laplacian of Gaussian blob detector.
        particleType : If 0, only maximal blob responses are considered. If 1, will consider positive and negative extrema.
        maxCurvatureRatio : Maximum curvature ratio parameter.
    """
    # First identify the maxes
    maxes = Maxes(detH, LG, particleType, maxCurvatureRatio)
    workhorse = Wave(maxes.data.copy(), "SS_OTSU_COPY")

    # Create a histogram using of the maxes
    hist, bin_edges = np.histogram(maxes.data, bins=50)
    hist_wave = Wave(hist, "Hist")

    # Search for the best threshold
    min_ICV = np.inf
    best_thresh = -np.inf

    for i, bin_edge in enumerate(bin_edges[:-1]):
        x_thresh = bin_edge

        # Calculate intra-class variance
        mask_below = maxes.data < x_thresh
        mask_above = maxes.data >= x_thresh

        if np.sum(mask_below) == 0 or np.sum(mask_above) == 0:
            continue

        var_below = np.var(maxes.data[mask_below]) if np.sum(mask_below) > 1 else 0
        var_above = np.var(maxes.data[mask_above]) if np.sum(mask_above) > 1 else 0

        ICV = np.sum(hist[:i + 1]) * var_below + np.sum(hist[i + 1:]) * var_above

        if ICV < min_ICV:
            best_thresh = x_thresh
            min_ICV = ICV

    return best_thresh


def Maxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """
    Returns a wave with the values of the local maxes of the determinant of Hessian.
    """
    name = f"{detH.name}_MaxValues"
    maxes = Wave(np.zeros(detH.data.size // 26), name)

    limI = detH.data.shape[0] - 1
    limJ = detH.data.shape[1] - 1
    limK = detH.data.shape[2] - 1
    cnt = 0

    # Start with smallest blobs then go to larger blobs
    for k in range(1, limK):
        for i in range(1, limI):
            for j in range(1, limJ):

                # Is it too edgy?
                if LG.data[i, j, k] ** 2 / detH.data[i, j, k] >= (maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                    continue

                # Is it the right type of particle?
                if (particleType == -1 and LG.data[i, j, k] < 0) or (particleType == 1 and LG.data[i, j, k] > 0):
                    continue

                # Check if it's a local maximum in 3D neighborhood
                current_val = detH.data[i, j, k]

                # Check 26 neighbors
                is_maximum = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni <= limI and 0 <= nj <= limJ and 0 <= nk <= limK:
                                if detH.data[ni, nj, nk] >= current_val:
                                    is_maximum = False
                                    break
                        if not is_maximum:
                            break
                    if not is_maximum:
                        break

                if not is_maximum:
                    continue

                if cnt < len(maxes.data):
                    maxes.data[cnt] = current_val
                    cnt += 1

                if map_wave is not None:
                    map_wave.data[i, j] = max(map_wave.data[i, j], current_val)

                if scaleMap is not None:
                    scaleMap.data[i, j] = DimOffset(detH, 2) * (DimDelta(detH, 2) ** k)

    # Trim the maxes array
    if cnt < len(maxes.data):
        maxes.data = maxes.data[:cnt]

    return maxes


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Lets the user interactively choose a blob strength for the determinant of Hessian.
    This function would typically show a GUI - here we'll return a default value
    """
    # First identify the maxes
    maxes = Maxes(detH, LG, particleType, maxCurvatureRatio)
    maxes.data = np.sqrt(maxes.data)  # Put it into image units

    # For now, return a default threshold (half of maximum)
    # In a full implementation, this would show an interactive GUI
    if len(maxes.data) > 0:
        return np.max(maxes.data) / 2
    else:
        return 0.0