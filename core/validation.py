"""
Parameter Validation System

Provides comprehensive parameter validation:
- validate_hessian_parameters: Validate Hessian blob parameters
- validate_constraints: Validate particle constraints
- validate_and_convert_parameters: Parameter conversion matching Igor Pro
- Range checking and type validation

This ensures all parameters meet Igor Pro specifications exactly.
"""

import numpy as np
from core.error_handling import HessianBlobError, safe_print


def validate_hessian_parameters(params):
    """
    Validate Hessian blob parameters - EXACT IGOR PRO VALIDATION

    Args:
        params: List/tuple of parameters to validate

    Returns:
        True if valid, raises HessianBlobError if invalid
    """
    required_params = 7
    if len(params) < required_params:
        raise HessianBlobError(f"Parameter array must contain at least {required_params} parameters")

    scale_start, layers, scale_factor, det_h_thresh, particle_type, subpixel_mult, allow_overlap = params[:7]

    # Validate ranges exactly like Igor Pro
    if scale_start <= 0:
        raise HessianBlobError("Minimum size must be positive")
    if layers <= 0:
        raise HessianBlobError("Maximum size must be positive")
    if scale_factor <= 1.0:
        raise HessianBlobError("Scaling factor must be greater than 1.0")
    if particle_type not in [-1, 0, 1]:
        raise HessianBlobError("Particle type must be -1, 0, or 1")
    if subpixel_mult < 1:
        raise HessianBlobError("Subpixel ratio must be >= 1")
    if allow_overlap not in [0, 1]:
        raise HessianBlobError("Allow overlap must be 0 or 1")

    return True


def validate_constraints(constraints):
    """
    Validate particle constraints - EXACT IGOR PRO VALIDATION

    Args:
        constraints: List/tuple of 6 constraint values

    Returns:
        True if valid, raises HessianBlobError if invalid
    """
    if len(constraints) != 6:
        raise HessianBlobError("Constraints must contain 6 values: minH, maxH, minA, maxA, minV, maxV")

    min_h, max_h, min_a, max_a, min_v, max_v = constraints

    if min_h >= max_h and max_h != np.inf:
        raise HessianBlobError("Minimum height must be less than maximum height")
    if min_a >= max_a and max_a != np.inf:
        raise HessianBlobError("Minimum area must be less than maximum area")
    if min_v >= max_v and max_v != np.inf:
        raise HessianBlobError("Minimum volume must be less than maximum volume")

    return True


def validate_and_convert_parameters(params):
    """
    Validate and convert parameters exactly like Igor Pro - EXACT IGOR PRO CONVERSION

    Args:
        params: List/tuple of parameters to validate and convert

    Returns:
        Tuple of converted parameters
    """
    try:
        # Extract parameters
        scaleStart = params[0]
        layers = params[1]
        scaleFactor = params[2]
        detHResponseThresh = params[3]
        particleType = int(params[4])
        subPixelMult = int(params[5])
        allowOverlap = int(params[6])

        # Additional validation matching Igor Pro exactly
        if scaleStart <= 0:
            raise HessianBlobError("Minimum Size must be positive")
        if layers <= 0:
            raise HessianBlobError("Maximum Size must be positive")
        if scaleFactor <= 1.0:
            raise HessianBlobError("Scaling Factor must be greater than 1.0")
        if particleType not in [-1, 0, 1]:
            raise HessianBlobError("Particle Type must be -1, 0, or 1")
        if subPixelMult < 1:
            raise HessianBlobError("Subpixel Ratio must be >= 1")
        if allowOverlap not in [0, 1]:
            raise HessianBlobError("Allow Overlap must be 0 or 1")

        # Parameter conversion exactly like Igor Pro
        dimDelta_im_0 = 1.0  # Pixel spacing

        # Convert scale parameters
        scaleStart_converted = (scaleStart * dimDelta_im_0) ** 2 / 2
        layers_converted = int(
            np.ceil(np.log((layers * dimDelta_im_0) ** 2 / (2 * scaleStart_converted)) / np.log(scaleFactor)))
        subPixelMult_converted = max(1, round(subPixelMult))
        scaleFactor_converted = max(1.1, scaleFactor)

        safe_print(f"Parameter conversion:")
        safe_print(f"  Original scaleStart: {scaleStart} -> {scaleStart_converted}")
        safe_print(f"  Original layers: {layers} -> {layers_converted}")
        safe_print(f"  Original scaleFactor: {scaleFactor} -> {scaleFactor_converted}")
        safe_print(f"  Original subPixelMult: {subPixelMult} -> {subPixelMult_converted}")

        return (scaleStart_converted, layers_converted, scaleFactor_converted,
                detHResponseThresh, particleType, subPixelMult_converted, allowOverlap)

    except Exception as e:
        from core.error_handling import handle_error
        handle_error("validate_and_convert_parameters", e)
        raise


def print_analysis_parameters(params):
    """
    Print analysis parameters like Igor Pro - EXACT IGOR PRO OUTPUT FORMAT

    Args:
        params: List/tuple of parameters to print
    """
    try:
        safe_print("\n" + "=" * 60)
        safe_print("HESSIAN BLOB ANALYSIS PARAMETERS")
        safe_print("=" * 60)

        param_names = [
            "Minimum Size in Pixels",
            "Maximum Size in Pixels",
            "Scaling Factor",
            "Minimum Blob Strength",
            "Particle Type",
            "Subpixel Ratio",
            "Allow Overlap",
            "Min Height Constraint",
            "Max Height Constraint",
            "Min Area Constraint",
            "Max Area Constraint",
            "Min Volume Constraint",
            "Max Volume Constraint"
        ]

        for i, (name, value) in enumerate(zip(param_names, params[:13])):
            if i == 3:  # Blob strength
                if value == -1:
                    safe_print(f"{i + 1:2d}. {name:<25}: Otsu's Method")
                elif value == -2:
                    safe_print(f"{i + 1:2d}. {name:<25}: Interactive Selection")
                else:
                    safe_print(f"{i + 1:2d}. {name:<25}: {value:.3e}")
            elif i == 4:  # Particle type
                type_str = {-1: "Negative only", 0: "Both", 1: "Positive only"}
                safe_print(f"{i + 1:2d}. {name:<25}: {value} ({type_str.get(value, 'Unknown')})")
            elif i == 6:  # Allow overlap
                overlap_str = {0: "No", 1: "Yes"}
                safe_print(f"{i + 1:2d}. {name:<25}: {value} ({overlap_str.get(value, 'Unknown')})")
            elif i >= 7:  # Constraints
                if value == -np.inf:
                    safe_print(f"{i + 1:2d}. {name:<25}: -∞ (no limit)")
                elif value == np.inf:
                    safe_print(f"{i + 1:2d}. {name:<25}: +∞ (no limit)")
                else:
                    safe_print(f"{i + 1:2d}. {name:<25}: {value:.3e}")
            else:
                safe_print(f"{i + 1:2d}. {name:<25}: {value}")

        safe_print("=" * 60)

    except Exception as e:
        from core.error_handling import handle_error
        handle_error("print_analysis_parameters", e)


def validate_image_data(image):
    """
    Validate image data - EXACT IGOR PRO VALIDATION

    Args:
        image: Numpy array containing image data

    Returns:
        True if valid, raises HessianBlobError if invalid
    """
    if image is None:
        raise HessianBlobError("Input image is None")

    if not isinstance(image, np.ndarray):
        raise HessianBlobError("Input must be a numpy array")

    if len(image.shape) < 2:
        raise HessianBlobError("Input must be at least 2D")

    if image.size == 0:
        raise HessianBlobError("Input image is empty")

    if not image.flags.writeable:
        raise HessianBlobError("Input image must be writable")

    return True


def validate_file_path(filepath, must_exist=True):
    """
    Validate file path - EXACT IGOR PRO VALIDATION

    Args:
        filepath: Path to validate
        must_exist: Whether file must already exist

    Returns:
        True if valid, raises HessianBlobError if invalid
    """
    import os

    if not filepath:
        raise HessianBlobError("File path cannot be empty")

    if must_exist and not os.path.exists(filepath):
        raise HessianBlobError(f"File does not exist: {filepath}")

    if must_exist and not os.path.isfile(filepath):
        raise HessianBlobError(f"Path is not a file: {filepath}")

    return True
