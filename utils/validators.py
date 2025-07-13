"""Contains functions for validating user-provided parameters."""

# #######################################################################
#                  UTILITIES: PARAMETER VALIDATION
#
#   CONTENTS:
#       - validate_hessian_parameters: Checks core blob detection params.
#       - validate_constraints: Checks height, area, and volume constraints.
#       - validate_and_convert_parameters: Converts pixel units to scaled units.
#       - print_analysis_parameters: Prints a summary of all parameters.
#
# #######################################################################

import numpy as np
from .error_handler import HessianBlobError, handle_error, safe_print

def validate_hessian_parameters(params):
    """Validate Hessian blob parameters"""
    required_params = 7
    if len(params) < required_params:
        raise HessianBlobError(f"Parameter array must contain at least {required_params} parameters")

    scale_start, layers, scale_factor, det_h_thresh, particle_type, subpixel_mult, allow_overlap = params[:7]

    # Validate ranges
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
    """Validate particle constraints"""
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
    """Validate and convert parameters exactly like Igor Pro"""
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

        # Parameter conversion
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
        handle_error("validate_and_convert_parameters", e)
        raise

def print_analysis_parameters(params):
    """Print analysis parameters like Igor Pro"""
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
        handle_error("print_analysis_parameters", e)