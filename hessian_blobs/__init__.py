# ========================================================================
# hessian_blobs/__init__.py
# ========================================================================

"""
Hessian Blob Particle Detection Suite - Main Package

Contains the core Hessian blob detection functions organized exactly
like the original Igor Pro code structure.
"""

from .main_functions import BatchHessianBlobs, HessianBlobs
from .scale_space import ScaleSpaceRepresentation, BlobDetectors, OtsuThreshold, InteractiveThreshold
from .particle_measurements import (M_AvgBoundary, M_MinBoundary, M_Height, M_Volume,
                                   M_CenterOfMass, M_Area, M_Perimeter)
from .preprocessing import BatchPreprocess, Flatten, RemoveStreaks
from .utilities import FixBoundaries, GetMaxes, FindHessianBlobs, MaximalBlobs, ViewParticles, Testing

__version__ = "1.0.0"
__author__ = "Brendan Marsh (Original Igor Pro), Python Port"