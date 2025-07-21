import numpy as np


def m_avg_boundary(im, mask):
    """
    Measure the average pixel value of the particle on the boundary

    Parameters:
    im: The image containing the particle
    mask: Mask image (0 for background, 1 for particle)

    Returns:
    float: Average boundary value
    """
    limI, limJ = im.shape
    bg = 0
    cnt = 0

    for i in range(1, limI - 1):
        for j in range(1, limJ - 1):
            if mask[i, j] == 0 and (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                                    mask[i, j + 1] == 1 or mask[i, j - 1] == 1):
                bg += im[i, j]
                cnt += 1
            elif mask[i, j] == 1 and (mask[i + 1, j] == 0 or mask[i - 1, j] == 0 or
                                      mask[i, j + 1] == 0 or mask[i, j - 1] == 0):
                bg += im[i, j]
                cnt += 1

    return bg / cnt if cnt > 0 else 0


def m_min_boundary(im, mask):
    """
    Measure the minimum pixel value of the particle on the boundary

    Parameters:
    im: The image containing the particle
    mask: Mask image (0 for background, 1 for particle)

    Returns:
    float: Minimum boundary value
    """
    limI, limJ = im.shape
    bg = np.inf

    for i in range(1, limI - 1):
        for j in range(1, limJ - 1):
            if mask[i, j] == 1 and im[i, j] < bg:
                bg = im[i, j]

    return bg


def m_height(im, mask, bg, neg_particle=False):
    """
    Measures the maximum height of the particle above background

    Parameters:
    im: The image containing the particle
    mask: Mask image (0 for background, 1 for particle)
    bg: Background level
    neg_particle: If True, measure negative height

    Returns:
    float: Height of particle
    """
    # Create masked array
    masked_im = np.where(mask, im, np.nan)

    if not neg_particle:
        height = np.nanmax(masked_im) - bg
    else:
        height = bg - np.nanmin(masked_im)

    return height


def m_volume(im, mask, bg):
    """
    Computes the volume of the particle

    Parameters:
    im: The image containing the particle
    mask: Mask image (0 for background, 1 for particle)
    bg: Background level

    Returns:
    float: Volume of particle
    """
    # Find volume and area
    V = 0
    cnt = 0
    limX, limY = im.shape

    for i in range(limX):
        for j in range(limY):
            if mask[i, j]:
                V += im[i, j]
                cnt += 1

    V -= cnt * bg
    # In Igor Pro, this is multiplied by pixel area, but we assume unit pixels
    # V *= DimDelta(im, 0) * DimDelta(im, 1)

    return V


def m_center_of_mass(im, mask, bg):
    """
    Computes the center of mass of the particle

    Parameters:
    im: The image containing the particle
    mask: Mask image (0 for background, 1 for particle)
    bg: Background level

    Returns:
    tuple: (x_center, y_center)
    """
    xsum = 0
    ysum = 0
    count = 0
    limI, limJ = im.shape

    for i in range(limI):
        for j in range(limJ):
            if mask[i, j]:
                weight = im[i, j] - bg
                xsum += i * weight
                ysum += j * weight
                count += weight

    if count > 0:
        return (xsum / count, ysum / count)
    else:
        return (0, 0)


def m_area(mask):
    """
    Computes the area of the particle using Gwyddion method

    Parameters:
    mask: Mask image (0 for background, 1 for particle)

    Returns:
    float: Area of particle
    """
    a = 0
    limI, limJ = mask.shape

    for i in range(limI - 1):
        for j in range(limJ - 1):
            # Count pixels in 2x2 square
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

    # In Igor Pro, this is multiplied by pixel area, but we assume unit pixels
    # a *= DimDelta(mask, 0) ** 2

    return a


def m_perimeter(mask):
    """
    Computes the perimeter of the particle using Gwyddion method

    Parameters:
    mask: Mask image (0 for background, 1 for particle)

    Returns:
    float: Perimeter of particle
    """
    l = 0
    limI, limJ = mask.shape

    for i in range(limI - 1):
        for j in range(limJ - 1):
            # Count pixels in 2x2 square
            pixels = mask[i, j] + mask[i + 1, j] + mask[i, j + 1] + mask[i + 1, j + 1]

            if pixels == 1 or pixels == 3:
                l += np.sqrt(2) / 2
            elif pixels == 2:
                if mask[i, j] == mask[i + 1, j] or mask[i, j] == mask[i, j + 1]:
                    l += 1
                else:
                    l += np.sqrt(2)

    # In Igor Pro, this is multiplied by pixel size, but we assume unit pixels
    # l *= DimDelta(mask, 0)

    return l