# main.py
# Monkey patch for numpy compatibility
import numpy as np
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'object'):
    np.object = object

from gui import launch_gui

if __name__ == "__main__":
    print("Launching Hessian Blob Particle Detection GUI...")
    launch_gui()