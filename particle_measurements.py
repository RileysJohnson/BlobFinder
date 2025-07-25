"""
Particle Measurements
Contains all measurement functions for analyzing detected particles
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
from igor_compatibility import *

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex


def M_AvgBoundary(im, mask):
    """
    Measure the average pixel value of the particle on the boundary of the particle.
        im : The image containing the particle.
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
    """
    limI = im.data.shape[0]
    limJ = im.data.shape[1]
    bg = 0
    cnt = 0

    for i in range(1, limI - 1):
        for j in range(1, limJ - 1):
            if (mask.data[i, j] == 0 and
                    (mask.data[i + 1, j] == 1 or mask.data[i - 1, j] == 1 or
                     mask.data[i, j + 1] == 1 or mask.data[i, j - 1] == 1)):
                bg += im.data[i, j]
                cnt += 1
            elif (mask.data[i, j] == 1 and
                  (mask.data[i + 1, j] == 0 or mask.data[i - 1, j] == 0 or
                   mask.data[i, j + 1] == 0 or mask.data[i, j - 1] == 0)):
                bg += im.data[i, j]
                cnt += 1

    if cnt > 0:
        return bg / cnt
    else:
        return 0


def M_MinBoundary(im, mask):
    """
    Measure the minimum pixel value of the particle on the boundary of the particle.
        im : The image containing the particle.
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
    """
    limI = im.data.shape[0]
    limJ = im.data.shape[1]
    bg = np.inf

    for i in range(1, limI - 1):
        for j in range(1, limJ - 1):
            if mask.data[i, j] == 1 and im.data[i, j] < bg:
                bg = im.data[i, j]

    return bg


def M_Height(im, mask, bg, negParticle=False):
    """
    Measures the maximum height of the particle above the background level.
        im : The image containing the particle.
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg : The background level of the particle.
        negParticle : An optional parameter that indicates the feature is a hole and has negative height.
    """
    masked_data = np.where(mask.data, im.data, np.nan)

    if not negParticle:
        height = np.nanmax(masked_data) - bg
    else:
        height = bg - np.nanmin(masked_data)

    return height


def M_Volume(im, mask, bg):
    """
    Computes the volume of the particle.
        im : The image containing the particle.
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg : The background level for the particle.
    """
    V = 0
    cnt = 0
    limX = im.data.shape[0]
    limY = im.data.shape[1]

    for i in range(limX):
        for j in range(limY):
            if mask.data[i, j]:
                V += im.data[i, j]
                cnt += 1

    V -= cnt * bg
    V *= DimDelta(im, 0) * DimDelta(im, 1)

    return V


def M_CenterOfMass(im, mask, bg):
    """
    Computes the center of mass of the particle, returning the x center of mass and y
    center of mass in a single complex variable COM. The X center of mass is stored in the real part
    and the Y center of mass in the imaginary part.
        im : The image containing the particle.
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg : The background level for the particle.
    """
    xsum = 0
    ysum = 0
    count = 0
    limI = im.data.shape[0]
    limJ = im.data.shape[1]
    x0 = DimOffset(im, 0)
    dx = DimDelta(im, 0)
    y0 = DimOffset(im, 1)
    dy = DimDelta(im, 1)

    for i in range(limI):
        for j in range(limJ):
            if mask.data[i, j]:
                x_coord = x0 + i * dx
                y_coord = y0 + j * dy
                intensity = im.data[i, j] - bg

                xsum += x_coord * intensity
                ysum += y_coord * intensity
                count += intensity

    if count != 0:
        return complex(xsum / count, ysum / count)
    else:
        return complex(0, 0)


def M_Area(mask):
    """
    Computes the area of the particle using the method employed by Gwyddion:
    http://gwyddion.net/documentation/user-guide-en/grain-analysis.html
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
    """
    a = 0
    limI = mask.data.shape[0]
    limJ = mask.data.shape[1]

    for i in range(limI - 1):
        for j in range(limJ - 1):
            pixels = (mask.data[i, j] + mask.data[i + 1, j] +
                      mask.data[i, j + 1] + mask.data[i + 1, j + 1])

            if pixels == 1:
                a += 0.125  # 1/8
            elif pixels == 2:
                if (mask.data[i, j] == mask.data[i + 1, j] or
                        mask.data[i, j] == mask.data[i, j + 1]):
                    a += 0.5
                else:
                    a += 0.75
            elif pixels == 3:
                a += 0.875  # 7/8
            elif pixels == 4:
                a += 1

    return a * (DimDelta(mask, 0) ** 2)


def M_Perimeter(mask):
    """
    Computes the perimeter of the particle using the method employed by Gwyddion:
    http://gwyddion.net/documentation/user-guide-en/grain-analysis.html
        mask : A mask image of the same size identifying which pixels belong to the particle.
               In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
    """
    l = 0
    limI = mask.data.shape[0]
    limJ = mask.data.shape[1]

    for i in range(limI - 1):
        for j in range(limJ - 1):
            pixels = (mask.data[i, j] + mask.data[i + 1, j] +
                      mask.data[i, j + 1] + mask.data[i + 1, j + 1])

            if pixels == 1 or pixels == 3:
                l += np.sqrt(2) / 2
            elif pixels == 2:
                if (mask.data[i, j] == mask.data[i + 1, j] or
                        mask.data[i, j] == mask.data[i, j + 1]):
                    l += 1
                else:
                    l += np.sqrt(2)

    return l * DimDelta(mask, 0)


def ExpandBoundary8(mask):
    """
    Expands boundary using 8-connectivity
    """
    original = mask.data.copy()

    for i in range(1, mask.data.shape[0] - 1):
        for j in range(1, mask.data.shape[1] - 1):
            if len(mask.data.shape) == 2:
                if (original[i, j] == 0 and
                        (original[i + 1, j] == 1 or original[i - 1, j] == 1 or
                         original[i, j + 1] == 1 or original[i, j - 1] == 1 or
                         original[i + 1, j + 1] == 1 or original[i - 1, j + 1] == 1 or
                         original[i + 1, j - 1] == 1 or original[i - 1, j - 1] == 1)):
                    mask.data[i, j] = 2
            else:
                for k in range(mask.data.shape[2]):
                    if (original[i, j, k] == 0 and
                            (original[i + 1, j, k] == 1 or original[i - 1, j, k] == 1 or
                             original[i, j + 1, k] == 1 or original[i, j - 1, k] == 1 or
                             original[i + 1, j + 1, k] == 1 or original[i - 1, j + 1, k] == 1 or
                             original[i + 1, j - 1, k] == 1 or original[i - 1, j - 1, k] == 1)):
                        mask.data[i, j, k] = 2

    mask.data = (mask.data > 0).astype(int)
    return 0


def ExpandBoundary4(mask):
    """
    Expands boundary using 4-connectivity
    """
    original = mask.data.copy()

    for i in range(1, mask.data.shape[0] - 1):
        for j in range(1, mask.data.shape[1] - 1):
            if len(mask.data.shape) == 2:
                if (original[i, j] == 0 and
                        (original[i + 1, j] == 1 or original[i - 1, j] == 1 or
                         original[i, j + 1] == 1 or original[i, j - 1] == 1)):
                    mask.data[i, j] = 2
            else:
                for k in range(mask.data.shape[2]):
                    if (original[i, j, k] == 0 and
                            (original[i + 1, j, k] == 1 or original[i - 1, j, k] == 1 or
                             original[i, j + 1, k] == 1 or original[i, j - 1, k] == 1)):
                        mask.data[i, j, k] = 2

    mask.data = (mask.data > 0).astype(int)
    return 0


def ScanlineFill8_LG(image, dest, LG, seedP, seedQ, thresh,
                     SeedStack=None, BoundingBox=None, fillVal=1,
                     fillDown=False, perimeter=None, x0=None, xf=None,
                     y0=None, yf=None, layer=0, dest2=None, fillVal2=None, destLayer=0):
    """
    It's a speed demon, about 10x faster than iterative seedfill and more flexible as well.
    The dest wave should be -1 at background locations, and some positive value where a particle is identified.
    If two fills collide with each other, the one with the higher fill value will continue while the other is deleted.
    """
    # Get the parameters straight
    fill = fillVal

    if fillDown:
        image.data *= -1
        thresh *= -1

    if x0 is None or xf is None:
        x0 = 0
        xf = image.data.shape[0] - 1
    else:
        x0 = max(0, min(image.data.shape[0] - 1, int(x0)))
        xf = max(x0, min(image.data.shape[0] - 1, int(xf)))

    if y0 is None or yf is None:
        y0 = 0
        yf = image.data.shape[1] - 1
    else:
        y0 = max(0, min(image.data.shape[1] - 1, int(y0)))
        yf = max(y0, min(image.data.shape[1] - 1, int(yf)))

    # Make sure the seeds are in bounds, and that the seed is valid
    if seedP < x0 or seedQ < y0 or seedP > xf or seedQ > yf:
        return complex(0, -3)
    elif image.data[seedP, seedQ, layer] <= thresh:
        return complex(0, -1)

    # Get the sign of the LG
    sgn = np.sign(LG.data[seedP, seedQ, layer])

    # Simplified flood fill implementation
    # In full implementation, this would use the scanline algorithm from Igor
    count = 0
    is_boundary_particle = 0

    # Use scipy's flood fill for simplicity
    from scipy import ndimage
    from skimage.segmentation import flood_fill

    # Create a binary mask for flood fill
    fill_mask = ((image.data[:, :, layer] >= thresh) &
                 (np.sign(LG.data[:, :, layer]) == sgn))

    # Perform flood fill
    filled = flood_fill(fill_mask.astype(int), (seedP, seedQ), 1, connectivity=2)

    # Update destination
    dest.data[:, :, destLayer] = np.where(filled == 1, fill, dest.data[:, :, destLayer])

    if dest2 is not None:
        dest2.data[:, :, destLayer] = np.where(filled == 1, fillVal2, dest2.data[:, :, destLayer])

    # Count filled pixels
    count = np.sum(filled == 1)

    # Check if boundary particle
    if (np.any(filled[x0, :] == 1) or np.any(filled[xf, :] == 1) or
            np.any(filled[:, y0] == 1) or np.any(filled[:, yf] == 1)):
        is_boundary_particle = 1

    if fillDown:
        image.data *= -1

    if BoundingBox is not None:
        y_indices, x_indices = np.where(filled == 1)
        if len(x_indices) > 0:
            BoundingBox[0] = np.min(x_indices)
            BoundingBox[1] = np.max(x_indices)
            BoundingBox[2] = np.min(y_indices)
            BoundingBox[3] = np.max(y_indices)

            return complex(count, is_boundary_particle)


def ScanlineFillEqual(image, dest, seedP, seedQ, SeedStack=None, BoundingBox=None,
                      fillVal=1, perimeter=None, x0=None, xf=None, y0=None, yf=None,
                      layer=0, dest2=None, fillVal2=None, destLayer=0):
    """
    Scanline fill for equal values
    """
    # Get the parameters straight
    fill = fillVal

    if x0 is None or xf is None:
        x0 = 0
        xf = image.data.shape[0] - 1
    else:
        x0 = max(0, min(image.data.shape[0] - 1, int(x0)))
        xf = max(x0, min(image.data.shape[0] - 1, int(xf)))

    if y0 is None or yf is None:
        y0 = 0
        yf = image.data.shape[1] - 1
    else:
        y0 = max(0, min(image.data.shape[1] - 1, int(y0)))
        yf = max(y0, min(image.data.shape[1] - 1, int(yf)))

    # Make sure the seeds are in bounds
    if seedP < x0 or seedQ < y0 or seedP > xf or seedQ > yf:
        return complex(0, -3)

    # The value to seed fill
    val = image.data[seedP, seedQ, layer] if len(image.data.shape) > 2 else image.data[seedP, seedQ]

    # Simplified implementation using flood fill
    from skimage.segmentation import flood_fill

    if len(image.data.shape) > 2:
        fill_mask = (image.data[:, :, layer] == val)
    else:
        fill_mask = (image.data == val)

    filled = flood_fill(fill_mask.astype(int), (seedP, seedQ), 1, connectivity=2)

    # Update destination
    if len(dest.data.shape) > 2:
        dest.data[:, :, destLayer] = np.where(filled == 1, fill, dest.data[:, :, destLayer])
    else:
        dest.data = np.where(filled == 1, fill, dest.data)

    if dest2 is not None:
        if len(dest2.data.shape) > 2:
            dest2.data[:, :, destLayer] = np.where(filled == 1, fillVal2, dest2.data[:, :, destLayer])
        else:
            dest2.data = np.where(filled == 1, fillVal2, dest2.data)

    # Count filled pixels
    count = np.sum(filled == 1)

    # Check if boundary particle
    is_boundary_particle = 0
    if (np.any(filled[x0, :] == 1) or np.any(filled[xf, :] == 1) or
            np.any(filled[:, y0] == 1) or np.any(filled[:, yf] == 1)):
        is_boundary_particle = 1

    if BoundingBox is not None:
        y_indices, x_indices = np.where(filled == 1)
        if len(x_indices) > 0:
            BoundingBox[0] = np.min(x_indices)
            BoundingBox[1] = np.max(x_indices)
            BoundingBox[2] = np.min(y_indices)
            BoundingBox[3] = np.max(y_indices)

    return complex(count, is_boundary_particle)