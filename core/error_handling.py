# ========================================================================
# core/error_handling.py
# ========================================================================

"""
Error Handling and Logging

Centralized error handling and thread-safe printing for the
Hessian blob detection suite.
"""

import queue
import threading

class HessianBlobError(Exception):
    """Custom exception for Hessian Blob operations"""
    pass

def safe_print(message):
    """Thread-safe printing for GUI applications"""
    try:
        print(message)
    except Exception:
        pass

def handle_error(func_name: str, error: Exception, context: str = ""):
    """Centralized error handling with context"""
    error_msg = f"Error in {func_name}"
    if context:
        error_msg += f" ({context})"
    error_msg += f": {str(error)}"
    safe_print(error_msg)
    return error_msg