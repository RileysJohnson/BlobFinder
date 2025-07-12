# ========================================================================
# core/__init__.py
# ========================================================================

"""
Core Utilities Package

Contains error handling, validation, and other core utilities
used throughout the Hessian blob detection suite.
"""

from .error_handling import handle_error, safe_print, HessianBlobError
from .validation import (validate_hessian_parameters, validate_constraints,
                        validate_and_convert_parameters, print_analysis_parameters,
                        verify_igor_compatibility)