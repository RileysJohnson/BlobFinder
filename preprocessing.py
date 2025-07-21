import numpy as np
from scipy import ndimage
from utilities import flattening_interactive_threshold, CoordinateSystem


def batch_preprocess(images, streak_removal_sdevs=3, flatten_order=2):
    """
    Batch preprocess multiple images

    Parameters:
    images (dict): Dictionary of images
    streak_removal_sdevs: Standard deviations for streak removal
    flatten_order: Polynomial order for flattening

    Returns:
    dict: Preprocessed images
    """
    processed = {}

    for name, im in images.items():
        print(f"Preprocessing {name}...")

        # Apply streak removal if requested
        if streak_removal_sdevs > 0:
            im = remove_streaks(im, sigma=streak_removal_sdevs)

        # Apply flattening if requested
        if flatten_order > 0:
            im = flatten(im, flatten_order)

        processed[name] = im

    return processed


def flatten(im, order, mask=None, no_thresh=False, coord_system=None):
    """
    Flatten every horizontal line scan by subtracting a polynomial fit

    Parameters:
    im: Image to flatten
    order: Polynomial order
    mask: Optional mask for fitting
    no_thresh: If True, don't prompt for threshold
    coord_system: Optional coordinate system

    Returns:
    Flattened image
    """
    im_flattened = im.copy()

    # Get coordinate system if not provided
    if coord_system is None:
        coord_system = CoordinateSystem(im.shape)

    # Get threshold interactively if needed
    if mask is None and not no_thresh:
        threshold = flattening_interactive_threshold(im)
        mask = im <= threshold
        print(f"Flatten Height Threshold: {threshold}")

    # If no mask provided, use all pixels
    if mask is None:
        mask = np.ones_like(im, dtype=bool)

    # Fit and subtract polynomial from each scan line
    for j in range(im.shape[0]):
        scanline = im[j, :]
        mask_line = mask[j, :]

        if np.sum(mask_line) < order + 1:
            continue

        # Get x coordinates for fitting
        x = np.arange(im.shape[1])
        x_scaled = coord_system.x_start + x * coord_system.x_delta

        # Fit polynomial to masked pixels
        if order == 0:
            # Constant offset
            offset = np.mean(scanline[mask_line])
            im_flattened[j, :] -= offset
        elif order == 1:
            # Linear fit
            coefs = np.polyfit(x_scaled[mask_line], scanline[mask_line], 1)
            im_flattened[j, :] -= np.polyval(coefs, x_scaled)
        else:
            # Higher order polynomial
            coefs = np.polyfit(x_scaled[mask_line], scanline[mask_line], order)
            im_flattened[j, :] -= np.polyval(coefs, x_scaled)

    return im_flattened


def remove_streaks(image, sigma=3):
    """
    Remove streak artifacts from image

    Parameters:
    image: Input image
    sigma: Number of standard deviations for streak detection

    Returns:
    Image with streaks removed
    """
    # Create dy map
    dy_map = create_dy_map(image)
    dy_map = np.abs(dy_map)

    # Calculate statistics
    avg_dy = np.mean(dy_map)
    std_dy = np.std(dy_map)
    max_dy = avg_dy + std_dy * sigma

    # Process streaks
    result = image.copy()

    for i in range(image.shape[0]):
        for j in range(1, image.shape[1] - 1):
            if dy_map[i, j] > max_dy:
                # Found a streak starting point
                i0 = i

                # Go left until streak ends
                i = i0
                while i > 0 and dy_map[i, j] > avg_dy:
                    result[i, j] = (result[i, j + 1] + result[i, j - 1]) / 2
                    dy_map[i, j] = 0
                    i -= 1

                # Go right from original point
                i = i0 + 1
                while i < image.shape[0] - 1 and dy_map[i, j] > avg_dy:
                    result[i, j] = (result[i, j + 1] + result[i, j - 1]) / 2
                    dy_map[i, j] = 0
                    i += 1

                i = i0

    return result


def create_dy_map(image):
    """
    Create a map of vertical derivatives

    Parameters:
    image: Input image

    Returns:
    dy_map: Map of vertical derivatives
    """
    dy_map = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            j_plus = min(j + 1, image.shape[1] - 1)
            j_minus = max(j - 1, 0)
            dy_map[i, j] = image[i, j] - (image[i, j_plus] + image[i, j_minus]) / 2

    return dy_map