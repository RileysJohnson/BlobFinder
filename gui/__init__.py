# ========================================================================
# gui/__init__.py
# ========================================================================

"""
GUI Package

Contains the graphical user interface components that provide
an Igor Pro-like experience for the Hessian blob detection suite.
"""

from .main_window import HessianBlobGUI
from .parameter_dialogs import ParameterDialog
from .particle_viewer import ViewParticles