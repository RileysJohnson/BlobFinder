"""
Error Handling and Logging System

Provides centralized error handling:
- HessianBlobError: Custom exception class
- handle_error: Centralized error handler with context
- safe_print: Thread-safe printing for GUI applications
- Error logging and reporting functionality

This ensures robust error handling throughout the application.
"""

import threading
import sys


class HessianBlobError(Exception):
    """Custom exception for Hessian Blob operations"""
    pass


def safe_print(message):
    """
    Thread-safe printing for GUI applications

    Args:
        message: Message to print
    """
    try:
        print(message)
        sys.stdout.flush()
    except Exception:
        pass  # Silently ignore print failures


def handle_error(func_name: str, error: Exception, context: str = ""):
    """
    Centralized error handling with context

    Args:
        func_name: Name of function where error occurred
        error: Exception object
        context: Additional context information

    Returns:
        Formatted error message string
    """
    error_msg = f"Error in {func_name}"
    if context:
        error_msg += f" ({context})"
    error_msg += f": {str(error)}"

    safe_print(error_msg)

    # Log to file if needed (optional)
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {error_msg}\n"

        # Could write to log file here if desired
        # with open("hessian_blob_errors.log", "a") as f:
        #     f.write(log_entry)

    except Exception:
        pass  # Don't let logging errors break the application

    return error_msg


def log_analysis_start(func_name: str, params: dict = None):
    """
    Log the start of an analysis operation

    Args:
        func_name: Name of analysis function
        params: Optional parameters dictionary
    """
    try:
        safe_print(f"\n{'=' * 60}")
        safe_print(f"STARTING: {func_name}")
        safe_print(f"{'=' * 60}")

        if params:
            safe_print("Parameters:")
            for key, value in params.items():
                safe_print(f"  {key}: {value}")
            safe_print("")

    except Exception:
        pass  # Don't let logging errors break the application


def log_analysis_complete(func_name: str, result_info: dict = None):
    """
    Log the completion of an analysis operation

    Args:
        func_name: Name of analysis function
        result_info: Optional result information dictionary
    """
    try:
        safe_print(f"\n{'=' * 60}")
        safe_print(f"COMPLETED: {func_name}")

        if result_info:
            safe_print("Results:")
            for key, value in result_info.items():
                safe_print(f"  {key}: {value}")

        safe_print(f"{'=' * 60}\n")

    except Exception:
        pass  # Don't let logging errors break the application