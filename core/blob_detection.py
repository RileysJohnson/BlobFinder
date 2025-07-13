"""Contains the core algorithms for scale-space and blob detection."""

# #######################################################################
#          CORE: SCALE-SPACE & BLOB DETECTION ALGORITHMS
#
#   CONTENTS:
#       - ScaleSpaceRepresentation: Computes the scale-space of an image.
#       - BlobDetectors: Computes the LoG and DoH blob detectors.
#       - OtsuThreshold: Automatically finds a blob strength threshold.
#       - FindHessianBlobs: Identifies particles as scale-space extrema.
#       - MaximalBlobs: Filters for non-overlapping particles.
#       - ScanlineFill8_LG: A fast seed-fill algorithm.
#       - FixBoundaries: Corrects for derivative errors at image edges.
#
# #######################################################################

import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from utils.error_handler import handle_error, HessianBlobError

def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """Computes the discrete scale-space representation L of an image."""
    try:
        # Convert t0 to image units
        t0 = (t0 * 1.0) ** 2

        # Go to Fourier space
        im_fft = fft2(im)

        # Create frequency grids
        freq_x = fftfreq(im.shape[0])
        freq_y = fftfreq(im.shape[1])
        fx, fy = np.meshgrid(freq_x, freq_y, indexing='ij')

        # Make the layers of the scale-space representation
        L = np.zeros((im.shape[0], im.shape[1], layers))

        for i in range(layers):
            scale = t0 * (tFactor ** i)
            gaussian_kernel = np.exp(-(fx ** 2 + fy ** 2) * np.pi ** 2 * 2 * scale)
            Layer = im_fft * gaussian_kernel
            L[:, :, i] = np.real(ifft2(Layer))

        return L

    except Exception as e:
        handle_error("ScaleSpaceRepresentation", e)
        raise HessianBlobError(f"Failed to compute scale-space representation: {e}")

def BlobDetectors(L, gammaNorm):
    """Computes the two blob detectors, the determinant of the Hessian and the Laplacian of Gaussian."""
    try:
        # Make convolution kernels for calculating central difference derivatives
        LxxKernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        LyyKernel = np.array([
            [0, 0, -1 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -30 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -1 / 12, 0, 0]
        ])

        LxyKernel = np.array([
            [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
            [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
            [0, 0, 0, 0, 0],
            [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
            [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
        ])

        # Compute Lxx, Lyy, and Lxy
        Lxx = np.zeros_like(L)
        Lyy = np.zeros_like(L)
        Lxy = np.zeros_like(L)

        for i in range(L.shape[2]):
            Lxx[:, :, i] = ndimage.convolve(L[:, :, i], LxxKernel, mode='constant')
            Lyy[:, :, i] = ndimage.convolve(L[:, :, i], LyyKernel, mode='constant')
            Lxy[:, :, i] = ndimage.convolve(L[:, :, i], LxyKernel, mode='constant')

        # Compute the Laplacian of Gaussian
        LapG = Lxx + Lyy

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** gammaNorm / (1.0 * 1.0)
            LapG[:, :, r] *= scale_factor

        # Fix errors on the boundary of the image
        FixBoundaries(LapG)

        # Compute the determinant of the Hessian
        detH = Lxx * Lyy - Lxy ** 2

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** (2 * gammaNorm) / (1.0 * 1.0) ** 2
            detH[:, :, r] *= scale_factor

        # Fix the boundary issues again
        FixBoundaries(detH)

        return LapG, detH

    except Exception as e:
        handle_error("BlobDetectors", e)
        raise HessianBlobError(f"Failed to compute blob detectors: {e}")

def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """Uses Otsu's method to automatically define a threshold blob strength."""
    try:
        # First identify the maxes
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio)
        if len(Maxes) == 0:
            return 0.0

        # Create a histogram of the maxes
        Hist, bin_edges = np.histogram(Maxes, bins=50)

        # Search for the best threshold
        minICV = np.inf
        bestThresh = -np.inf

        for i in range(len(Hist)):
            xThresh = bin_edges[i]
            below_thresh = Maxes[Maxes < xThresh]
            above_thresh = Maxes[Maxes >= xThresh]

            if len(below_thresh) == 0 or len(above_thresh) == 0:
                continue

            w1 = len(below_thresh) / len(Maxes)
            w2 = len(above_thresh) / len(Maxes)

            if w1 > 0 and w2 > 0:
                ICV = w1 * np.var(below_thresh) + w2 * np.var(above_thresh)
                if ICV < minICV:
                    bestThresh = xThresh
                    minICV = ICV

        return bestThresh

    except Exception as e:
        handle_error("OtsuThreshold", e)
        return 0.0

def FixBoundaries(detH):
    """Fixes a boundary issue in the blob detectors."""
    try:
        limP, limQ = detH.shape[:2]
        limP -= 1
        limQ -= 1

        # Do the sides first. Corners need extra care.
        for i in range(2, limP - 1):
            detH[i, 0, :] = detH[i, 2, :] / 3
            detH[i, 1, :] = detH[i, 2, :] * 2 / 3

        for i in range(2, limP - 1):
            detH[i, limQ, :] = detH[i, limQ - 2, :] / 3
            detH[i, limQ - 1, :] = detH[i, limQ - 2, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[0, i, :] = detH[2, i, :] / 3
            detH[1, i, :] = detH[2, i, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[limP, i, :] = detH[limP - 2, i, :] / 3
            detH[limP - 1, i, :] = detH[limP - 2, i, :] * 2 / 3

        # Corners
        # Top Left Corner
        detH[1, 1, :] = (detH[1, 2, :] + detH[2, 1, :]) / 2
        detH[1, 0, :] = (detH[1, 1, :] + detH[2, 0, :]) / 2
        detH[0, 1, :] = (detH[1, 1, :] + detH[0, 2, :]) / 2
        detH[0, 0, :] = (detH[0, 1, :] + detH[1, 0, :]) / 2

        # Bottom Right Corner
        detH[limP - 1, limQ - 1, :] = (detH[limP - 1, limQ - 2, :] + detH[limP - 2, limQ - 1, :]) / 2
        detH[limP - 1, limQ, :] = (detH[limP - 1, limQ - 1, :] + detH[limP - 2, limQ, :]) / 2
        detH[limP, limQ - 1, :] = (detH[limP - 1, limQ - 1, :] + detH[limP, limQ - 2, :]) / 2
        detH[limP, limQ, :] = (detH[limP - 1, limQ, :] + detH[limP, limQ - 1, :]) / 2

        # Top Right Corner
        detH[limP - 1, 1, :] = (detH[limP - 1, 2, :] + detH[limP - 2, 1, :]) / 2
        detH[limP - 1, 0, :] = (detH[limP - 1, 1, :] + detH[limP - 2, 0, :]) / 2
        detH[limP, 1, :] = (detH[limP - 1, 1, :] + detH[limP, 2, :]) / 2
        detH[limP, 0, :] = (detH[limP - 1, 0, :] + detH[limP, 1, :]) / 2

        # Bottom Left Corner
        detH[1, limQ - 1, :] = (detH[1, limQ - 2, :] + detH[2, limQ - 1, :]) / 2
        detH[1, limQ, :] = (detH[1, limQ - 1, :] + detH[2, limQ, :]) / 2
        detH[0, limQ - 1, :] = (detH[1, limQ - 1, :] + detH[0, limQ - 2, :]) / 2
        detH[0, limQ, :] = (detH[1, limQ, :] + detH[0, limQ - 1, :]) / 2

        return 0

    except Exception as e:
        handle_error("FixBoundaries", e)
        return -1

def GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """Returns a wave with the values of the local maxes of the determinant of Hessian."""
    try:
        Maxes = []
        limI, limJ, limK = detH.shape

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum
                    is_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < limI and 0 <= nj < limJ and 0 <= nk < limK):
                                    if detH[ni, nj, nk] > detH[i, j, k]:
                                        is_max = False
                                        break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if not is_max:
                        continue

                    Maxes.append(detH[i, j, k])

                    if map_wave is not None:
                        map_wave[i, j] = max(map_wave[i, j], detH[i, j, k])

                    if scaleMap is not None:
                        scaleMap[i, j] = 1.0 * (1.5 ** k)

        return np.array(Maxes)

    except Exception as e:
        handle_error("GetMaxes", e)
        return np.array([])

def FindHessianBlobs(im, detH, LG, minResponse, particleType, maxCurvatureRatio):
    """Find Hessian blobs by detecting scale-space extrema."""
    try:
        # Square the minResponse
        minResponse = minResponse ** 2

        # mapNum: Map identifying particle numbers
        mapNum = np.full(detH.shape, -1, dtype=int)

        # mapLG: Map identifying the value of the LoG at the defined scale
        mapLG = np.zeros_like(detH)

        # mapMax: Map identifying the value of the LoG of the maximum pixel
        mapMax = np.zeros_like(detH)

        # Maintain an info list with particle boundaries and info
        Info = []

        limI, limJ, limK = detH.shape
        cnt = 0

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Does it hit the threshold?
                    if detH[i, j, k] < minResponse:
                        continue

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is there a particle there already?
                    if mapNum[i, j, k] > -1 and detH[i, j, k] <= Info[mapNum[i, j, k]][3]:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum
                    is_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if (0 <= ni < limI and 0 <= nj < limJ and 0 <= nk < limK):
                                    if detH[ni, nj, nk] > detH[i, j, k]:
                                        is_max = False
                                        break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if not is_max:
                        continue

                    # It's a local max, is it overlapped and bigger than another one already?
                    if mapNum[i, j, k] > -1:
                        Info[mapNum[i, j, k]][0] = i
                        Info[mapNum[i, j, k]][1] = j
                        Info[mapNum[i, j, k]][3] = detH[i, j, k]
                        continue

                    # It's a local max, proceed to fill out the feature
                    numPixels, boundingBox = ScanlineFill8_LG(detH, mapMax, LG, i, j, k, 0, mapNum, cnt)

                    if numPixels > 0:
                        particle_info = [0] * 15
                        particle_info[0] = i
                        particle_info[1] = j
                        particle_info[2] = numPixels
                        particle_info[3] = detH[i, j, k]
                        particle_info[4] = boundingBox[0]
                        particle_info[5] = boundingBox[1]
                        particle_info[6] = boundingBox[2]
                        particle_info[7] = boundingBox[3]
                        particle_info[8] = 1.0 * (1.5 ** k)  # scale
                        particle_info[9] = k
                        particle_info[10] = 1
                        Info.append(particle_info)
                        cnt += 1

        # Make the mapLG
        mapLG = np.where(mapNum != -1, detH, 0)

        return mapNum, mapLG, mapMax, Info

    except Exception as e:
        handle_error("FindHessianBlobs", e)
        return np.full(detH.shape, -1, dtype=int), np.zeros_like(detH), np.zeros_like(detH), []

def MaximalBlobs(info, mapNum):
    """Determine scale-maximal particles."""
    try:
        if len(info) == 0:
            return -1

        # Initialize maximality of each particle as undetermined (-1)
        for i in range(len(info)):
            info[i][10] = -1

        # Sort by blob strength
        sorted_indices = sorted(range(len(info)), key=lambda i: info[i][3], reverse=True)

        limK = mapNum.shape[2]

        for idx_pos in range(len(info)):
            blocked = False
            index = sorted_indices[idx_pos]
            k = int(info[index][9])

            for ii in range(int(info[index][4]), int(info[index][5]) + 1):
                for jj in range(int(info[index][6]), int(info[index][7]) + 1):
                    if mapNum[ii, jj, k] == index:
                        for kk in range(limK):
                            if mapNum[ii, jj, kk] != -1 and info[mapNum[ii, jj, kk]][10] == 1:
                                blocked = True
                                break
                        if blocked:
                            break
                if blocked:
                    break

            info[index][10] = 0 if blocked else 1

        return 0

    except Exception as e:
        handle_error("MaximalBlobs", e)
        return -1

def ScanlineFill8_LG(image, dest, LG, seedP, seedQ, layer, Thresh, dest2=None, fillVal2=None):
    """Scanline fill algorithm for blob detection."""
    try:
        fill = image[seedP, seedQ, layer]
        count = 0
        sgn = np.sign(LG[seedP, seedQ, layer])

        x0, xf = 0, image.shape[0] - 1
        y0, yf = 0, image.shape[1] - 1

        if seedP < x0 or seedQ < y0 or seedP > xf or seedQ > yf:
            return 0, [0, 0, 0, 0]
        elif image[seedP, seedQ, layer] <= Thresh:
            return 0, [0, 0, 0, 0]

        stack = [(seedP, seedQ)]
        visited = set()
        pixels = []

        while stack:
            i, j = stack.pop()

            if (i, j) in visited:
                continue
            if i < x0 or i > xf or j < y0 or j > yf:
                continue
            if image[i, j, layer] < Thresh:
                continue
            if np.sign(LG[i, j, layer]) != sgn:
                continue

            visited.add((i, j))
            pixels.append((i, j))
            dest[i, j, layer] = fill
            count += 1

            if dest2 is not None and fillVal2 is not None:
                dest2[i, j, layer] = fillVal2

            # Add 8-connected neighbors
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

        if len(pixels) == 0:
            return 0, [0, 0, 0, 0]

        p_coords = [p for p, q in pixels]
        q_coords = [q for p, q in pixels]

        BoundingBox = [min(p_coords), max(p_coords), min(q_coords), max(q_coords)]
        return count, BoundingBox

    except Exception as e:
        handle_error("ScanlineFill8_LG", e)
        return 0, [0, 0, 0, 0]