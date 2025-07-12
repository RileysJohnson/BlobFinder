#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Particle Measurements

Contains particle measurement functions:
- M_AvgBoundary(): Measure average pixel value on particle boundary
- M_MinBoundary(): Measure minimum pixel value on particle boundary
- M_Height(): Measures maximum height of particle above background
- M_Volume(): Computes volume of particle
- M_CenterOfMass(): Computes center of mass of particle
- M_Area(): Computes area using Gwyddion method
- M_Perimeter(): Computes perimeter using Gwyddion method

Corresponds to Section III. Particle Measurements in the original Igor Pro code.
"""

import numpy as np
from core.error_handling import handle_error


# ========================================================================
# PARTICLE MEASUREMENTS
# ========================================================================

def M_AvgBoundary(im, mask):
    """
    Measure the average pixel value of the particle on the boundary of the particle.

    Args:
        im: The image containing the particle
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.

    Returns:
        Average boundary value

    Exact translation of Igor Pro M_AvgBoundary() function.
    """
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

        if cnt > 0:
            return bg / cnt
        else:
            return 0

    except Exception as e:
        handle_error("M_AvgBoundary", e)
        return 0.0


def M_MinBoundary(im, mask):
    """
    Measure the minimum pixel value of the particle on the boundary of the particle.

    Args:
        im: The image containing the particle
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.

    Returns:
        Minimum boundary value

    Exact translation of Igor Pro M_MinBoundary() function.
    """
    try:
        limI, limJ = im.shape
        bg = np.inf

        for i in range(1, limI - 1):
            for j in range(1, limJ - 1):
                if mask[i, j] == 1 and im[i, j] < bg:
                    bg = im[i, j]

        return bg if bg != np.inf else 0.0

    except Exception as e:
        handle_error("M_MinBoundary", e)
        return 0.0


def M_Height(im, mask, bg, negParticle=False):
    """
    Measures the maximum height of the particle above the background level.

    Args:
        im: The image containing the particle
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg: The background level of the particle
        negParticle: An optional parameter that indicates the feature is a hole and has negative height

    Returns:
        Maximum height value

    Exact translation of Igor Pro M_Height() function.
    """
    try:
        # Multithread mask = mask ? im : NaN
        masked_im = np.where(mask, im, np.nan)

        if negParticle:
            height = bg - np.nanmin(masked_im)
        else:
            height = np.nanmax(masked_im) - bg

        # Multithread mask = NumType(mask)==0 ? 1 : 0
        # (This restores the mask - not needed in Python)

        return height if not np.isnan(height) else 0.0

    except Exception as e:
        handle_error("M_Height", e)
        return 0.0


def M_Volume(im, mask, bg):
    """
    Computes the volume of the particle.

    Args:
        im: The image containing the particle
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg: The background level for the particle

    Returns:
        Volume value

    Exact translation of Igor Pro M_Volume() function.
    """
    try:
        # Find the volume and area
        V = 0
        cnt = 0
        LimX, LimY = im.shape

        for i in range(LimX):
            for j in range(LimY):
                if mask[i, j]:
                    V += im[i, j]
                    cnt += 1

        V -= cnt * bg
        V *= 1.0 * 1.0  # DimDelta(im,0) * DimDelta(im,1) = 1.0 * 1.0

        return V

    except Exception as e:
        handle_error("M_Volume", e)
        return 0.0


def M_CenterOfMass(im, mask, bg):
    """
    Computes the center of mass of the particle, returning the x center of mass and y
    center of mass in a single complex variable COM. The X center of mass is stored in the real part
    and the Y center of mass in the imaginary part. Explicitly, the X part is given by Real(COM) and
    the imaginary part by Imag(COM).

    Args:
        im: The image containing the particle
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.
        bg: The background level for the particle

    Returns:
        Tuple of (x_com, y_com) - center of mass coordinates

    Exact translation of Igor Pro M_CenterOfMass() function.
    """
    try:
        xsum = 0
        ysum = 0
        count = 0
        limI, limJ = im.shape
        x0 = 0.0  # DimOffset(im,0) = 0.0
        dx = 1.0  # DimDelta(im,0) = 1.0
        y0 = 0.0  # DimOffset(im,1) = 0.0
        dy = 1.0  # DimDelta(im,1) = 1.0

        for i in range(limI):
            for j in range(limJ):
                if mask[i, j]:
                    weight = im[i, j] - bg
                    xsum += (x0 + i * dx) * weight
                    ysum += (y0 + j * dy) * weight
                    count += weight

        if count > 0:
            return (xsum / count, ysum / count)
        else:
            return (0.0, 0.0)

    except Exception as e:
        handle_error("M_CenterOfMass", e)
        return (0.0, 0.0)


def M_Area(mask):
    """
    Computes the area of the particle using the method employed by Gwyddion:
    http://gwyddion.net/documentation/user-guide-en/grain-analysis.html

    Args:
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.

    Returns:
        Area value

    Exact translation of Igor Pro M_Area() function.
    """
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

        return a * 1.0 ** 2  # DimDelta(mask,0)^2 = 1.0^2

    except Exception as e:
        handle_error("M_Area", e)
        return 0.0


def M_Perimeter(mask):
    """
    Computes the perimeter of the particle using the method employed by Gwyddion:
    http://gwyddion.net/documentation/user-guide-en/grain-analysis.html

    Args:
        mask: A mask image of the same size identifying which pixels belong to the particle.
             In the mask, 0 corresponds to background pixels and 1 corresponds to particle pixels.

    Returns:
        Perimeter value

    Exact translation of Igor Pro M_Perimeter() function.
    """
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

        return l * 1.0  # DimDelta(mask,0) = 1.0

    except Exception as e:
        handle_error("M_Perimeter", e)
        return 0.0