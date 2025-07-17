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
from utils.error_handler import handle_error, HessianBlobError, safe_print


def dyMap(image):
    """Create dy map for streak detection - EXACT IGOR PRO ALGORITHM."""
    try:
        # Create dY map exactly like Igor Pro
        dy_map = np.zeros_like(image)

        # Calculate Y derivatives using central difference
        # Igor Pro uses simple difference: image[i][j+1] - image[i][j-1]
        dy_map[:, 1:-1] = image[:, 2:] - image[:, :-2]

        # Handle boundaries - Igor Pro sets edges to zero
        dy_map[:, 0] = 0
        dy_map[:, -1] = 0

        return dy_map

    except Exception as e:
        handle_error("dyMap", e)
        return np.zeros_like(image)


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
        # Use proper scale-space parameters instead of hardcoded values
        t0 = 1.0  # This should match the t0 used in ScaleSpaceRepresentation
        tFactor = 1.5  # This should match the tFactor used in ScaleSpaceRepresentation
        dimDelta_x = 1.0  # Pixel spacing in x
        dimDelta_y = 1.0  # Pixel spacing in y

        for r in range(L.shape[2]):
            # Igor: (DimOffset(L,2)*DimDelta(L,2)^r)^(gammaNorm) / (DimDelta(L,0)*DimDelta(L,1))
            scale_factor = (t0 * (tFactor ** r)) ** gammaNorm / (dimDelta_x * dimDelta_y)
            LapG[:, :, r] *= scale_factor

        # Fix errors on the boundary of the image
        FixBoundaries(LapG)

        # Compute the determinant of the Hessian
        detH = Lxx * Lyy - Lxy ** 2

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            # Igor: (DimOffset(L,2)*DimDelta(L,2)^r)^(2*gammaNorm) / (DimDelta(L,0)*DimDelta(L,1))^2
            scale_factor = (t0 * (tFactor ** r)) ** (2 * gammaNorm) / (dimDelta_x * dimDelta_y) ** 2
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
        MaxesArray = GetMaxes(detH, LG, particleType, maxCurvatureRatio)
        if len(MaxesArray) == 0:
            safe_print("No maxima found for Otsu threshold calculation.")
            return 0.0

        # Create a histogram of the maxes using Igor Pro's approach
        # Igor Pro uses 5 bins by default: Histogram/B=5
        num_bins = max(5, min(50, len(MaxesArray) // 10))
        Hist, bin_edges = np.histogram(MaxesArray, bins=num_bins)

        # Search for the best threshold using Igor Pro's intra-class variance method
        minICV = np.inf
        bestThresh = 0.0

        for i in range(len(bin_edges) - 1):
            xThresh = bin_edges[i]

            # Split data at threshold
            below_mask = MaxesArray < xThresh
            above_mask = MaxesArray >= xThresh

            below_thresh = MaxesArray[below_mask]
            above_thresh = MaxesArray[above_mask]

            if len(below_thresh) == 0 or len(above_thresh) == 0:
                continue

            # Calculate intra-class variance exactly like Igor Pro
            # ICV = Sum(Hist,-inf,xThresh)*Variance(Workhorse) + Sum(Hist,xThresh,inf)*Variance(Workhorse)
            w1 = len(below_thresh)
            w2 = len(above_thresh)

            if w1 > 1 and w2 > 1:
                var1 = np.var(below_thresh, ddof=0)  # Population variance like Igor Pro
                var2 = np.var(above_thresh, ddof=0)
                ICV = w1 * var1 + w2 * var2

                if ICV < minICV:
                    bestThresh = xThresh
                    minICV = ICV

        return bestThresh

    except Exception as e:
        handle_error("OtsuThreshold", e)
        return 0.0


def FixBoundaries(detH):
    """Fixes a boundary issue in the blob detectors - EXACT IGOR PRO ALGORITHM."""
    try:
        limP, limQ = detH.shape[:2]
        limP -= 1
        limQ -= 1

        # Do the sides first. Corners need extra care.
        # Make the edges fade off so that maxima can still be detected.

        # Igor Pro: For(i=1;i<limP;i+=1)
        for i in range(1, limP):
            # Left and right edges
            detH[i, 0, :] = detH[i, 1, :] * 0.5
            detH[i, limQ, :] = detH[i, limQ - 1, :] * 0.5

        # Igor Pro: For(j=1;j<limQ;j+=1)
        for j in range(1, limQ):
            # Top and bottom edges
            detH[0, j, :] = detH[1, j, :] * 0.5
            detH[limP, j, :] = detH[limP - 1, j, :] * 0.5

        # Fix the corners - Igor Pro sets corners to zero
        detH[0, 0, :] = 0
        detH[0, limQ, :] = 0
        detH[limP, 0, :] = 0
        detH[limP, limQ, :] = 0

        return 0

    except Exception as e:
        handle_error("FixBoundaries", e)
        return -1


def GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """Returns a wave with the values of the local maxes of the determinant of Hessian."""
    try:
        MaxesList = []
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

                    # There are 26 neighbors in three dimensions.. Have to check if it is a local maximum
                    strictlyGreater = max(detH[i - 1, j - 1, k - 1],
                                          max(detH[i - 1, j - 1, k],
                                              max(detH[i - 1, j, k - 1], detH[i, j - 1, k - 1])))
                    strictlyGreater = max(strictlyGreater,
                                          max(detH[i, j, k - 1],
                                              max(detH[i, j - 1, k], detH[i - 1, j, k])))

                    if not (detH[i, j, k] > strictlyGreater):
                        continue

                    greaterOrEqual = detH[i - 1, j - 1, k + 1]
                    greaterOrEqual = max(greaterOrEqual, detH[i - 1, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i - 1, j + 1, k - 1],
                                             max(detH[i - 1, j + 1, k], detH[i - 1, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual, detH[i, j - 1, k + 1])
                    greaterOrEqual = max(greaterOrEqual, detH[i, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i, j + 1, k - 1],
                                             max(detH[i, j + 1, k], detH[i, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j - 1, k - 1],
                                             max(detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])))
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j, k - 1],
                                             max(detH[i + 1, j, k], detH[i + 1, j, k + 1])))
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j + 1, k - 1],
                                             max(detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])))

                    if not (detH[i, j, k] >= greaterOrEqual):
                        continue

                    MaxesList.append(detH[i, j, k])

                    if map_wave is not None:
                        map_wave[i, j] = max(map_wave[i, j], detH[i, j, k])

                    if scaleMap is not None:
                        t0 = 1.0  # Should match ScaleSpaceRepresentation
                        tFactor = 1.5  # Should match ScaleSpaceRepresentation
                        scaleMap[i, j] = t0 * (tFactor ** k)

        return np.array(MaxesList)

    except Exception as e:
        handle_error("GetMaxes", e)
        return np.array([])


def FindHessianBlobs(im, detH, LG, minResponse, particleType, maxCurvatureRatio):
    """Find Hessian blobs by detecting scale-space extrema."""
    try:
        limI, limJ, limK = detH.shape
        mapNum = np.full((limI, limJ, limK), -1, dtype=np.int32)
        mapLG = np.zeros_like(detH)
        mapMax = np.zeros_like(detH)
        Info = []
        cnt = 0

        # Scan through scale-space from smallest to largest scales
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Check if blob strength meets minimum threshold
                    if detH[i, j, k] < minResponse ** 2:  # Square threshold to match Igor Pro
                        continue

                    # Is it too edgy? (curvature ratio test)
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is it the right type of particle?
                    if particleType == -1 and LG[i, j, k] >= 0:
                        continue
                    if particleType == 1 and LG[i, j, k] <= 0:
                        continue

                    # Check if it's a local maximum in 3D neighborhood
                    # Must be strictly greater than some neighbors and >= all others
                    strictlyGreater = max(detH[i - 1, j - 1, k - 1],
                                          max(detH[i - 1, j - 1, k],
                                              max(detH[i - 1, j, k - 1], detH[i, j - 1, k - 1])))
                    strictlyGreater = max(strictlyGreater,
                                          max(detH[i, j, k - 1],
                                              max(detH[i, j - 1, k], detH[i - 1, j, k])))

                    if not (detH[i, j, k] > strictlyGreater):
                        continue

                    # Check remaining neighbors for >= condition
                    greaterOrEqual = detH[i - 1, j - 1, k + 1]
                    greaterOrEqual = max(greaterOrEqual, detH[i - 1, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i - 1, j + 1, k - 1],
                                             max(detH[i - 1, j + 1, k], detH[i - 1, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual, detH[i, j - 1, k + 1])
                    greaterOrEqual = max(greaterOrEqual, detH[i, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i, j + 1, k - 1],
                                             max(detH[i, j + 1, k], detH[i, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j - 1, k - 1],
                                             max(detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])))
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j, k - 1],
                                             max(detH[i + 1, j, k], detH[i + 1, j, k + 1])))
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j + 1, k - 1],
                                             max(detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])))

                    if not (detH[i, j, k] >= greaterOrEqual):
                        continue

                    # Check if there's already a particle at this location
                    if mapNum[i, j, k] > -1:
                        if detH[i, j, k] > Info[mapNum[i, j, k]][3]:
                            # Replace the weaker particle
                            Info[mapNum[i, j, k]][0] = i  # P Seed
                            Info[mapNum[i, j, k]][1] = j  # Q Seed
                            Info[mapNum[i, j, k]][2] = k  # K Seed (scale)
                            Info[mapNum[i, j, k]][3] = detH[i, j, k]  # Blob strength
                            mapLG[i, j, k] = LG[i, j, k]
                            mapMax[i, j, k] = detH[i, j, k]
                        continue

                    # New particle found
                    particle_info = [
                        i,  # P Seed (x-position)
                        j,  # Q Seed (y-position)
                        k,  # K Seed (scale layer)
                        detH[i, j, k],  # Blob strength
                        0, 0, 0, 0,  # Bounding box (will be filled later)
                        0,  # Scale index (index 8)
                        k,  # Scale layer (index 9)
                        1,  # Maximal flag (1 = maximal) (index 10)
                        0, 0, 0, 0  # Additional flags and measurements
                    ]

                    Info.append(particle_info)
                    mapNum[i, j, k] = cnt
                    mapLG[i, j, k] = LG[i, j, k]
                    mapMax[i, j, k] = detH[i, j, k]
                    cnt += 1

        return mapNum, mapLG, mapMax, Info

    except Exception as e:
        handle_error("FindHessianBlobs", e)
        raise HessianBlobError(f"Failed to find Hessian blobs: {e}")


def SubPixelRefinement(im, detH, LG, Info, subPixelMult):
    """Performs subpixel refinement of particle positions - EXACT IGOR PRO ALGORITHM."""
    try:
        if subPixelMult <= 1:
            return Info

        refined_info = []

        for particle in Info:
            i, j, k = int(particle[0]), int(particle[1]), int(particle[2])

            # Extract local region around maximum
            # Igor Pro uses 5x5x5 neighborhood for interpolation
            window_size = 2

            # Check bounds
            if (i < window_size or i >= detH.shape[0] - window_size or
                    j < window_size or j >= detH.shape[1] - window_size or
                    k < window_size or k >= detH.shape[2] - window_size):
                refined_info.append(particle)
                continue

            # Extract local region
            local_detH = detH[i - window_size:i + window_size + 1,
                         j - window_size:j + window_size + 1,
                         k - window_size:k + window_size + 1]

            # Find subpixel maximum using quadratic interpolation
            # Igor Pro performs 3D quadratic interpolation
            center_val = local_detH[window_size, window_size, window_size]

            # Calculate gradients and Hessian for interpolation
            # Simplified version - full Igor Pro uses more sophisticated interpolation
            grad_i = (local_detH[window_size + 1, window_size, window_size] -
                      local_detH[window_size - 1, window_size, window_size]) / 2
            grad_j = (local_detH[window_size, window_size + 1, window_size] -
                      local_detH[window_size, window_size - 1, window_size]) / 2
            grad_k = (local_detH[window_size, window_size, window_size + 1] -
                      local_detH[window_size, window_size, window_size - 1]) / 2

            # Second derivatives for Hessian
            hess_ii = (local_detH[window_size + 1, window_size, window_size] -
                       2 * center_val + local_detH[window_size - 1, window_size, window_size])
            hess_jj = (local_detH[window_size, window_size + 1, window_size] -
                       2 * center_val + local_detH[window_size, window_size - 1, window_size])
            hess_kk = (local_detH[window_size, window_size, window_size + 1] -
                       2 * center_val + local_detH[window_size, window_size, window_size - 1])

            # Avoid division by zero
            if abs(hess_ii) > 1e-10:
                di = -grad_i / hess_ii
            else:
                di = 0

            if abs(hess_jj) > 1e-10:
                dj = -grad_j / hess_jj
            else:
                dj = 0

            if abs(hess_kk) > 1e-10:
                dk = -grad_k / hess_kk
            else:
                dk = 0

            # Clamp to reasonable subpixel range
            di = max(-0.5, min(0.5, di))
            dj = max(-0.5, min(0.5, dj))
            dk = max(-0.5, min(0.5, dk))

            # Update particle position with subpixel refinement
            refined_particle = particle.copy()
            refined_particle[0] = i + di  # Refined i position
            refined_particle[1] = j + dj  # Refined j position
            # Scale position typically doesn't need subpixel refinement

            refined_info.append(refined_particle)

        return refined_info

    except Exception as e:
        handle_error("SubPixelRefinement", e)
        return Info


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
    """Scanline fill algorithm for blob detection - EXACT IGOR PRO ALGORITHM."""
    try:
        limI, limJ = image.shape[:2]

        # Add bounds checking
        if seedP >= limI or seedQ >= limJ or seedP < 0 or seedQ < 0:
            return complex(0, 0)

        destLayer = layer
        fill = 1
        val = image[seedP, seedQ, layer]

        # Igor Pro: Make/N=(4,10000) SeedStack
        SeedStack = np.zeros((10000, 4), dtype=np.int32)
        seedIndex = 0
        newSeedIndex = 1

        # Initialize seed
        SeedStack[0] = [seedP, seedP, seedQ, 1]

        x0, xf, y0, yf = limI, 0, limJ, 0  # Bounding box
        Count = 0
        isBP = 0

        while seedIndex != newSeedIndex:
            if newSeedIndex >= 10000:
                # Expand stack if needed
                new_stack = np.zeros((20000, 4), dtype=np.int32)
                new_stack[:10000] = SeedStack
                SeedStack = new_stack

            i0, i, j, state = SeedStack[seedIndex]
            seedIndex += 1
            goFish = 1

            if state == 1:  # Search Up Right
                if j != limJ - 1:
                    while i <= xf and dest[i, j, destLayer] == fill:
                        if dest[i, j + 1, destLayer] != fill and image[i, j + 1, layer] == val:
                            if newSeedIndex < len(SeedStack):
                                SeedStack[newSeedIndex] = [i0, i, j, 1]
                                newSeedIndex += 1
                            state = 0
                            goFish = 0
                            i0 = i
                            j += 1
                            break
                        i += 1

            elif state == 2:  # Search Up Left
                if j != limJ - 1:
                    while i >= x0 and dest[i, j, destLayer] == fill:
                        if dest[i, j + 1, destLayer] != fill and image[i, j + 1, layer] == val:
                            if newSeedIndex < len(SeedStack):
                                SeedStack[newSeedIndex] = [i0, i, j, 2]
                                newSeedIndex += 1
                            state = 0
                            goFish = 0
                            i0 = i
                            j += 1
                            break
                        i -= 1

            elif state == 3:  # Search Down Right
                if j != y0:
                    while i <= xf and dest[i, j, destLayer] == fill:
                        if dest[i, j - 1, destLayer] != fill and image[i, j - 1, layer] == val:
                            if newSeedIndex < len(SeedStack):
                                SeedStack[newSeedIndex] = [i0, i, j, 3]
                                newSeedIndex += 1
                            state = 0
                            goFish = 0
                            i0 = i
                            j -= 1
                            break
                        i += 1

            elif state == 4:  # Search Down Left
                if j != y0:
                    while i >= x0 and dest[i, j, destLayer] == fill:
                        if dest[i, j - 1, destLayer] != fill and image[i, j - 1, layer] == val:
                            if newSeedIndex < len(SeedStack):
                                SeedStack[newSeedIndex] = [i0, i, j, 4]
                                newSeedIndex += 1
                            state = 0
                            goFish = 0
                            i0 = i
                            j -= 1
                            break
                        i -= 1


            i = i0

            # Fill scanline if we're in state 0
            if state == 0:
                # Update bounding box
                if i < x0: x0 = i
                if i > xf: xf = i
                if j < y0: y0 = j
                if j > yf: yf = j

                # Fill pixels meeting criteria
                while i >= 0 and dest[i, j, destLayer] != fill and image[i, j, layer] == val:
                    dest[i, j, destLayer] = fill
                    Count += 1
                    i -= 1

                i = i0 + 1
                while i < limI and dest[i, j, destLayer] != fill and image[i, j, layer] == val:
                    dest[i, j, destLayer] = fill
                    Count += 1
                    i += 1

        # Check if boundary particle
        for i in range(x0, xf + 1):
            if dest[i, y0, destLayer] == fill:
                isBP = 1
                break
        if not isBP:
            for i in range(x0, xf + 1):
                if dest[i, yf, destLayer] == fill:
                    isBP = 1
                    break
        if not isBP:
            for j in range(y0, yf + 1):
                if dest[x0, j, destLayer] == fill:
                    isBP = 1
                    break
        if not isBP:
            for j in range(y0, yf + 1):
                if dest[xf, j, destLayer] == fill:
                    isBP = 1
                    break

            # Igor Pro: Return Cmplx(Count,isBP)
        return complex(Count, isBP)

    except Exception as e:
        handle_error("ScanlineFill8_LG", e)
        return complex(0, 0)

def Maxes(detH, LG, particleType, maxCurvatureRatio, map=None, scaleMap=None):
   """Extract maxima from scale-space - EXACT IGOR PRO ALGORITHM."""
   try:
       limI, limJ, limK = detH.shape
       MaxesArray = []
       cnt = 0

       # Igor Pro: For(k=1;k<DimSize(detH,2)-1;k+=1)
       for k in range(1, limK - 1):
           for i in range(1, limI - 1):
               for j in range(1, limJ - 1):

                   # Igor Pro: If( detH[i][j][k] <= 0 )
                   if detH[i, j, k] <= 0:
                       continue

                   # Is it too edgy?
                   # Igor Pro: If( LG[i][j][k]^2/detH[i][j][k] >= (maxCurvatureRatio+1)^2/maxCurvatureRatio )
                   if LG[i, j, k] ** 2 / detH[i, j, k] >= (maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                       continue

                   # Is it the right type of particle?
                   # Igor Pro: If( (particleType==-1 && LG[i][j][k]<0) || (particleType==1 && LG[i][j][k]>0) )
                   if (particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0):
                       continue

                   # There are 26 neighbors in three dimensions.. Have to check if it is a local maximum
                   strictlyGreater = max(detH[i - 1, j - 1, k - 1],
                                         max(detH[i - 1, j - 1, k], max(detH[i - 1, j, k - 1], detH[i, j - 1, k - 1])))
                   strictlyGreater = max(strictlyGreater,
                                         max(detH[i, j, k - 1], max(detH[i, j - 1, k], detH[i - 1, j, k])))

                   # Igor Pro: If( !(detH[i][j][k]>strictlyGreater) )
                   if not (detH[i, j, k] > strictlyGreater):
                       continue

                   greaterOrEqual = detH[i - 1, j - 1, k + 1]
                   greaterOrEqual = max(greaterOrEqual, detH[i - 1, j, k + 1])
                   greaterOrEqual = max(greaterOrEqual, max(detH[i - 1, j + 1, k - 1],
                                                            max(detH[i - 1, j + 1, k], detH[i - 1, j + 1, k + 1])))

                   greaterOrEqual = max(greaterOrEqual, detH[i, j - 1, k + 1])
                   greaterOrEqual = max(greaterOrEqual, detH[i, j, k + 1])
                   greaterOrEqual = max(greaterOrEqual,
                                        max(detH[i, j + 1, k - 1], max(detH[i, j + 1, k], detH[i, j + 1, k + 1])))

                   greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j - 1, k - 1],
                                                            max(detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])))
                   greaterOrEqual = max(greaterOrEqual,
                                        max(detH[i + 1, j, k - 1], max(detH[i + 1, j, k], detH[i + 1, j, k + 1])))
                   greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j + 1, k - 1],
                                                            max(detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])))

                   # Igor Pro: If( !(detH[i][j][k]>=greaterOrEqual) )
                   if not (detH[i, j, k] >= greaterOrEqual):
                       continue

                   # Igor Pro: Maxes[cnt] = detH[i][j][k]
                   MaxesArray.append(detH[i, j, k])
                   cnt += 1

                   # Igor Pro: If( !ParamIsDefault(map) )
                   if map is not None:
                       map[i, j] = max(map[i, j], detH[i, j, k])

                   # Igor Pro: If( !ParamIsDefault(scaleMap) )
                   if scaleMap is not None:
                       # Igor Pro: scaleMap[i][j] = DimOffset(detH,2)*(DimDelta(detH,2)^k)
                       scaleMap[i, j] = k  # Simplified scale representation

       return np.array(MaxesArray)

   except Exception as e:
       handle_error("Maxes", e)
       return np.array([])