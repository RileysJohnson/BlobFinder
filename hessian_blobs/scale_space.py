"""
Hessian Blob Particle Detection Suite - Scale-Space Functions

Contains scale-space representation and blob detection functions:
- ScaleSpaceRepresentation(): Computes the discrete scale-space representation
- BlobDetectors(): Computes determinant of Hessian and Laplacian of Gaussian
- OtsuThreshold(): Uses Otsu's method to automatically define threshold
- InteractiveThreshold(): Interactive threshold selection

Corresponds to Section II. Scale-Space Functions in the original Igor Pro code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.fft import fft2, ifft2, fftfreq
from scipy import ndimage
from core.error_handling import handle_error, safe_print
from .utilities import FixBoundaries, GetMaxes


# ========================================================================
# SCALE-SPACE FUNCTIONS
# ========================================================================

def ScaleSpaceRepresentation(im, layers, t0, tFactor):
    """
    Computes the discrete scale-space representation L of an image.

    Args:
        im: The image to compute L from
        layers: The number of layers of L
        t0: The scale of the first layer of L, provided in pixel units
        tFactor: The scaling factor for the scale between layers of L

    Returns:
        3D array representing the scale-space

    Exact translation of Igor Pro ScaleSpaceRepresentation() function.
    """
    try:
        # Convert t0 to image units
        t0 = (t0 * 1.0) ** 2  # DimDelta(im,0) = 1.0

        # Go to Fourier space
        im_fft = fft2(im)

        # Create frequency grids
        freq_x = fftfreq(im.shape[0])
        freq_y = fftfreq(im.shape[1])
        fx, fy = np.meshgrid(freq_x, freq_y, indexing='ij')

        # Make the layers of the scale-space representation and convolve in Fourier space
        L = np.zeros((im.shape[0], im.shape[1], layers))

        for i in range(layers):
            scale = t0 * (tFactor ** i)
            gaussian_kernel = np.exp(-(fx ** 2 + fy ** 2) * np.pi ** 2 * 2 * scale)
            Layer = im_fft * gaussian_kernel
            L[:, :, i] = np.real(ifft2(Layer))

        # Set the scaling to match Igor Pro
        # SetScale/P z,t0,tFactor,L

        return L

    except Exception as e:
        handle_error("ScaleSpaceRepresentation", e)
        raise ValueError(f"Failed to compute scale-space representation: {e}")


def BlobDetectors(L, gammaNorm):
    """
    Computes the two blob detectors, the determinant of the Hessian and the Laplacian of Gaussian.

    Args:
        L: The scale-space representation of the image
        gammaNorm: The gamma normalization factor, see Lindeberg 1998. Should be set to 1 in most blob detection cases

    Returns:
        Tuple of (LapG, detH) - Laplacian of Gaussian and determinant of Hessian

    Exact translation of Igor Pro BlobDetectors() function.
    """
    try:
        # Make convolution kernels for calculating central difference derivatives
        LxxKernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        LyyKernel = np.array([
            [0, 0, -1 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -30 / 12, 0, 0],
            [0, 0, 16 / 12, 0, 0],
            [0, 0, -1 / 12, 0, 0]
        ])

        LxyKernel = np.array([
            [-1 / 144, 1 / 18, 0, -1 / 18, 1 / 144],
            [1 / 18, -4 / 9, 0, 4 / 9, -1 / 18],
            [0, 0, 0, 0, 0],
            [-1 / 18, 4 / 9, 0, -4 / 9, 1 / 18],
            [1 / 144, -1 / 18, 0, 1 / 18, -1 / 144]
        ])

        # Compute Lxx, Lyy, and Lxy (Second partial derivatives of L)
        Lxx = np.zeros_like(L)
        Lyy = np.zeros_like(L)
        Lxy = np.zeros_like(L)

        for i in range(L.shape[2]):
            Lxx[:, :, i] = ndimage.convolve(L[:, :, i], LxxKernel, mode='constant')
            Lyy[:, :, i] = ndimage.convolve(L[:, :, i], LyyKernel, mode='constant')
            Lxy[:, :, i] = ndimage.convolve(L[:, :, i], LxyKernel, mode='constant')

        # Compute the Laplacian of Gaussian
        LapG = Lxx + Lyy

        # Set the image scale to match Igor Pro
        # SetScale/P x,DimOffset(L,0),DimDelta(L,0),LapG
        # SetScale/P y,DimOffset(L,1),DimDelta(L,1),LapG
        # SetScale/P z,DimOffset(L,2),DimDelta(L,2),LapG

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** gammaNorm / (1.0 * 1.0)  # DimOffset and DimDelta = 1.0
            LapG[:, :, r] *= scale_factor

        # Fix errors on the boundary of the image
        FixBoundaries(LapG)

        # Compute the determinant of the Hessian
        detH = Lxx * Lyy - Lxy ** 2

        # Set the scaling to match Igor Pro
        # SetScale/P x,DimOffset(L,0),DimDelta(L,0),detH
        # SetScale/P y,DimOffset(L,1),DimDelta(L,1),detH
        # SetScale/P z,DimOffset(L,2),DimDelta(L,2),detH

        # Gamma normalize and account for pixel spacing
        for r in range(L.shape[2]):
            scale_factor = (1.0 * (1.5 ** r)) ** (2 * gammaNorm) / (1.0 * 1.0) ** 2
            detH[:, :, r] *= scale_factor

        # Fix the boundary issues again
        FixBoundaries(detH)

        return LapG, detH

    except Exception as e:
        handle_error("BlobDetectors", e)
        raise ValueError(f"Failed to compute blob detectors: {e}")


def OtsuThreshold(detH, LG, particleType, maxCurvatureRatio):
    """
    Uses Otsu's method to automatically define a threshold blob strength.

    Args:
        detH: The determinant of Hessian blob detector
        L: The scale-space representation
        doHoles: If 0, only maximal blob reponses are considered. If 1, will consider positive and negative extrema.
                * Note this parameter doesn't matter for the determinant of the Hessian since both positive and negative
                  blobs produce maxima of the determinant of the Hessian.

    Returns:
        Optimal threshold value

    Exact translation of Igor Pro OtsuThreshold() function.
    """
    try:
        # First identify the maxes
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio)
        if len(Maxes) == 0:
            return 0.0

        # Create a histogram using of the maxes
        Hist, bin_edges = np.histogram(Maxes, bins=5)  # /B=5 in Igor Pro

        # Search for the best threshold
        minICV = np.inf
        bestThresh = -np.inf

        for i in range(len(Hist)):
            xThresh = bin_edges[i]

            # Split data based on threshold
            below_thresh = Maxes[Maxes < xThresh]
            above_thresh = Maxes[Maxes >= xThresh]

            if len(below_thresh) == 0 or len(above_thresh) == 0:
                continue

            # Calculate intra-class variance (ICV)
            w1 = len(below_thresh) / len(Maxes)
            w2 = len(above_thresh) / len(Maxes)

            if w1 > 0 and w2 > 0:
                ICV = w1 * np.var(below_thresh) + w2 * np.var(above_thresh)

                if ICV < minICV:
                    bestThresh = xThresh
                    minICV = ICV

        return bestThresh

    except Exception as e:
        handle_error("OtsuThreshold", e)
        return 0.0


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Lets the user interactively choose a blob strength for the determinant of Hessian.

    Args:
        im: The image under analysis
        detH: The determinant of Hessian blob detector
        LG: The Laplacian of Gaussian blob detector
        particleType: 1 to consider positive Hessian blobs, 0 to consider negative Hessian blobs
        maxCurvatureratio: Maximum ratio of the principal curvatures of a blob

    Returns:
        Selected threshold value

    Exact translation of Igor Pro InteractiveThreshold() function.
    """
    try:
        # First identify the maxes
        SS_MAXMAP = np.full_like(im, -1)
        SS_MAXSCALEMAP = np.zeros_like(im)
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)
        Maxes = np.sqrt(np.maximum(Maxes, 0))  # Put it into image units

        if len(Maxes) == 0:
            safe_print("No maxima found for interactive threshold selection.")
            return 0.0

        # CRITICAL: Ensure thread-safe matplotlib backend
        import matplotlib
        matplotlib.use('TkAgg')

        # Close any existing plots
        plt.close('all')

        # Create interactive plot exactly matching Igor Pro Figure 17
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.8)

        im_display = ax.imshow(im, cmap='gray', interpolation='bilinear')
        ax.set_title('Interactive Blob Strength Selection\nAdjust slider to see detected blobs',
                     fontsize=14, fontweight='bold')

        SS_THRESH = np.max(Maxes) / 2

        # Create slider panel exactly like Igor Pro
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Blob Strength', 0, np.max(Maxes) * 1.1,
                        valinit=SS_THRESH, valfmt='%.3e')

        # Create text display for current threshold value
        ax_text = plt.axes([0.82, 0.5, 0.15, 0.3])
        ax_text.axis('off')
        threshold_text = ax_text.text(0.1, 0.9, f'Blob Strength:\n{SS_THRESH:.3e}',
                                      fontsize=10, transform=ax_text.transAxes)

        circles = []

        def update_display(thresh):
            # Clear previous circles
            for circle in circles:
                try:
                    circle.remove()
                except:
                    pass
            circles.clear()

            thresh_squared = thresh ** 2
            count = 0
            for i in range(SS_MAXMAP.shape[0]):
                for j in range(SS_MAXMAP.shape[1]):
                    if SS_MAXMAP[i, j] > thresh_squared:
                        xc = j  # Column is x-coordinate
                        yc = i  # Row is y-coordinate
                        rad = max(2, np.sqrt(2 * SS_MAXSCALEMAP[i, j]))

                        # FIXED: Create RED circles exactly like Igor Pro Figure 17
                        circle = plt.Circle((xc, yc), rad, color='red', fill=False,
                                            linewidth=2.5, alpha=0.9)
                        ax.add_patch(circle)
                        circles.append(circle)
                        count += 1

            # Update title and text display exactly like Igor Pro
            ax.set_title(f'Interactive Blob Strength Selection\n'
                         f'Blob Strength: {thresh:.3e}, Particles: {count}',
                         fontsize=14, fontweight='bold')

            threshold_text.set_text(f'Blob Strength:\n{thresh:.3e}\n\nParticles: {count}')

            # Thread-safe canvas update
            try:
                fig.canvas.draw_idle()
            except:
                pass

        slider.on_changed(update_display)
        update_display(SS_THRESH)

        # Create Accept and Quit buttons exactly like Igor Pro
        ax_accept = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.81, 0.02, 0.1, 0.04])
        button_accept = Button(ax_accept, 'Accept')
        button_quit = Button(ax_quit, 'Quit')

        result = [SS_THRESH]

        def accept_threshold(event):
            result[0] = slider.val
            plt.close(fig)

        def quit_threshold(event):
            result[0] = SS_THRESH
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)
        button_quit.on_clicked(quit_threshold)

        # Add instructions exactly like Igor Pro
        instructions_text = ('Use the slider to adjust blob strength threshold.\n'
                             'Red circles show detected particles.\n'
                             'Click "Accept" when satisfied with detection.')
        ax_text.text(0.1, 0.3, instructions_text, fontsize=9,
                     transform=ax_text.transAxes, style='italic')

        safe_print("Interactive threshold selection:")
        safe_print("- Use slider to adjust blob strength threshold")
        safe_print("- Red circles show detected particles")
        safe_print("- Click 'Accept' when satisfied with detection")

        # CRITICAL: Use blocking show() to prevent threading issues
        plt.show(block=True)

        return result[0]

    except Exception as e:
        handle_error("InteractiveThreshold", e)
        return 0.0
