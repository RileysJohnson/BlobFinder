"""Contains all functions for measuring the physical properties of particles."""

# #######################################################################
#                   UTILITIES: PARTICLE MEASUREMENTS
#
#   CONTENTS:
#       - M_AvgBoundary: Measures the average value on the particle boundary.
#       - M_MinBoundary: Measures the minimum value on the particle boundary.
#       - M_Height: Measures the maximum height above background.
#       - M_Volume: Computes the integrated volume above background.
#       - M_CenterOfMass: Computes the weighted center of mass.
#       - M_Area: Computes the 2D projected area.
#       - M_Perimeter: Computes the perimeter length.
#
# #######################################################################

import numpy as np
from .error_handler import handle_error

def M_AvgBoundary(im, mask):
    """Measure the average pixel value of the particle on the boundary of the particle."""
    try:
        limI, limJ = im.shape
        bg = 0
        cnt = 0

        for i in range(1, limI - 1):
            for j in range(1, limJ - 1):
                if (mask[i, j] == 0 and
                        (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or mask[i, j + 1] == 1 or mask[i, j - 1] == 1)):
                    bg += im[i, j]
                    cnt += 1
                elif (mask[i, j] == 1 and
                      (mask[i + 1, j] == 0 or mask[i - 1, j] == 0 or mask[i, j + 1] == 0 or mask[i, j - 1] == 0)):
                    bg += im[i, j]
                    cnt += 1

        return bg / cnt if cnt > 0 else 0

    except Exception as e:
        handle_error("M_AvgBoundary", e)
        return 0.0

def M_MinBoundary(im, mask):
    """Measure the minimum pixel value of the particle."""
    try:
        bg = np.inf
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j] == 1 and im[i, j] < bg:
                    bg = im[i, j]
        return bg if bg != np.inf else 0.0

    except Exception as e:
        handle_error("M_MinBoundary", e)
        return 0.0

def M_Height(im, mask, bg, negParticle=False):
    """Measures the maximum height of the particle above the background level."""
    try:
        masked_im = np.where(mask, im, np.nan)
        if negParticle:
            height = bg - np.nanmin(masked_im)
        else:
            height = np.nanmax(masked_im) - bg
        return height if not np.isnan(height) else 0.0

    except Exception as e:
        handle_error("M_Height", e)
        return 0.0

def M_Volume(im, mask, bg):
    """Computes the volume of the particle."""
    try:
        V = 0
        cnt = 0
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j]:
                    V += im[i, j]
                    cnt += 1

        V -= cnt * bg
        V *= 1.0 * 1.0  # DimDelta(im,0) * DimDelta(im,1)
        return V

    except Exception as e:
        handle_error("M_Volume", e)
        return 0.0

def M_CenterOfMass(im, mask, bg):
    """Computes the center of mass of the particle."""
    try:
        xsum = 0
        ysum = 0
        count = 0
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if mask[i, j]:
                    weight = im[i, j] - bg
                    xsum += i * weight
                    ysum += j * weight
                    count += weight

        if count > 0:
            return (xsum / count, ysum / count)
        else:
            return (0.0, 0.0)

    except Exception as e:
        handle_error("M_CenterOfMass", e)
        return (0.0, 0.0)

def M_Area(mask):
    """Computes the area of the particle using the method employed by Gwyddion."""
    try:
        a = 0
        limI, limJ = mask.shape

        for i in range(limI - 1):
            for j in range(limJ - 1):
                pixels = mask[i, j] + mask[i + 1, j] + mask[i, j + 1] + mask[i + 1, j + 1]

                if pixels == 1:
                    a += 0.125  # 1/8
                elif pixels == 2:
                    if mask[i, j] == mask[i + 1, j] or mask[i, j] == mask[i, j + 1]:
                        a += 0.5
                    else:
                        a += 0.75
                elif pixels == 3:
                    a += 0.875  # 7/8
                elif pixels == 4:
                    a += 1

        return a * 1.0 ** 2  # DimDelta(mask,0)^2

    except Exception as e:
        handle_error("M_Area", e)
        return 0.0

def M_Perimeter(mask):
    """Computes the perimeter of the particle using the method employed by Gwyddion."""
    try:
        l = 0
        limI, limJ = mask.shape

        for i in range(limI - 1):
            for j in range(limJ - 1):
                pixels = mask[i, j] + mask[i + 1, j] + mask[i, j + 1] + mask[i + 1, j + 1]

                if pixels == 1 or pixels == 3:
                    l += np.sqrt(2) / 2
                elif pixels == 2:
                    if mask[i, j] == mask[i + 1, j] or mask[i, j] == mask[i, j + 1]:
                        l += 1
                    else:
                        l += np.sqrt(2)

        return l * 1.0  # DimDelta(mask,0)

    except Exception as e:
        handle_error("M_Perimeter", e)
        return 0.0