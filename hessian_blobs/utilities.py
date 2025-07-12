"""
Hessian Blob Particle Detection Suite - Utilities

Contains utility functions:
- FixBoundaries(): Fixes boundary issues in blob detectors
- GetMaxes(): Returns local maxes of determinant of Hessian
- FindHessianBlobs(): Finds Hessian blobs by detecting scale-space extrema
- MaximalBlobs(): Determines scale-maximal particles
- ScanlineFill8_LG(): Fast scanline fill algorithm
- ViewParticles(): Convenient method to view and examine particles
- Various helper functions matching Igor Pro exactly

Corresponds to Section V. Utilities in the original Igor Pro code.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import messagebox, filedialog
import os
import shutil
import json
from PIL import Image, ImageTk
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import threading

from core.error_handling import handle_error, safe_print


# ========================================================================
# UTILITIES - EXACT IGOR PRO FUNCTIONS
# ========================================================================

def FixBoundaries(detH):
    """
    Fixes a boundary issue in the blob detectors. Arises from trying to measure derivatives on the boundary.

    Args:
        detH: The determinant of Hessian blob detector, but also works for the Laplacian of Gaussian

    Returns:
        0 on success, -1 on failure

    Exact translation of Igor Pro FixBoundaries() function.
    """
    try:
        limP, limQ = detH.shape[:2]
        limP -= 1
        limQ -= 1

        # Do the sides first. Corners need extra care.
        # Make the edges fade off so that maxima can still be detected.
        for i in range(2, limP - 1):
            detH[i, 0, :] = detH[i, 2, :] / 3
            detH[i, 1, :] = detH[i, 2, :] * 2 / 3

        for i in range(2, limP - 1):
            detH[i, limQ, :] = detH[i, limQ - 2, :] / 3
            detH[i, limQ - 1, :] = detH[i, limQ - 2, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[0, i, :] = detH[2, i, :] / 3
            detH[1, i, :] = detH[2, i, :] * 2 / 3

        for i in range(2, limQ - 1):
            detH[limP, i, :] = detH[limP - 2, i, :] / 3
            detH[limP - 1, i, :] = detH[limP - 2, i, :] * 2 / 3

        # Corners
        # Top Left Corner
        detH[1, 1, :] = (detH[1, 2, :] + detH[2, 1, :]) / 2
        detH[1, 0, :] = (detH[1, 1, :] + detH[2, 0, :]) / 2
        detH[0, 1, :] = (detH[1, 1, :] + detH[0, 2, :]) / 2
        detH[0, 0, :] = (detH[0, 1, :] + detH[1, 0, :]) / 2

        # Bottom Right Corner
        detH[limP - 1, limQ - 1, :] = (detH[limP - 1, limQ - 2, :] + detH[limP - 2, limQ - 1, :]) / 2
        detH[limP - 1, limQ, :] = (detH[limP - 1, limQ - 1, :] + detH[limP - 2, limQ, :]) / 2
        detH[limP, limQ - 1, :] = (detH[limP - 1, limQ - 1, :] + detH[limP, limQ - 2, :]) / 2
        detH[limP, limQ, :] = (detH[limP - 1, limQ, :] + detH[limP, limQ - 1, :]) / 2

        # Top Right Corner
        detH[limP - 1, 1, :] = (detH[limP - 1, 2, :] + detH[limP - 2, 1, :]) / 2
        detH[limP - 1, 0, :] = (detH[limP - 1, 1, :] + detH[limP - 2, 0, :]) / 2
        detH[limP, 1, :] = (detH[limP - 1, 1, :] + detH[limP, 2, :]) / 2
        detH[limP, 0, :] = (detH[limP - 1, 0, :] + detH[limP, 1, :]) / 2

        # Bottom Left Corner
        detH[1, limQ - 1, :] = (detH[1, limQ - 2, :] + detH[2, limQ - 1, :]) / 2
        detH[1, limQ, :] = (detH[1, limQ - 1, :] + detH[2, limQ, :]) / 2
        detH[0, limQ - 1, :] = (detH[1, limQ - 1, :] + detH[0, limQ - 2, :]) / 2
        detH[0, limQ, :] = (detH[1, limQ, :] + detH[0, limQ - 1, :]) / 2

        return 0

    except Exception as e:
        handle_error("FixBoundaries", e)
        return -1


def GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=None, scaleMap=None):
    """
    Returns a wave with the values of the local maxes of the determinant of Hessian.

    Exact translation of Igor Pro Maxes() function.
    """
    try:
        Maxes = []
        limI, limJ, limK = detH.shape

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum - 26 neighbors in three dimensions
                    strictlyGreater = max(detH[i - 1, j - 1, k - 1],
                                          max(detH[i - 1, j - 1, k], max(detH[i - 1, j, k - 1], detH[i, j - 1, k - 1])))
                    strictlyGreater = max(strictlyGreater,
                                          max(detH[i, j, k - 1], max(detH[i, j - 1, k], detH[i - 1, j, k])))

                    if not (detH[i, j, k] > strictlyGreater):
                        continue

                    greaterOrEqual = detH[i - 1, j - 1, k + 1]
                    greaterOrEqual = max(greaterOrEqual, detH[i - 1, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual, max(detH[i - 1, j + 1, k - 1],
                                                             max(detH[i - 1, j + 1, k], detH[i - 1, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual, detH[i, j - 1, k + 1])
                    greaterOrEqual = max(greaterOrEqual, detH[i, j, k + 1])
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i, j + 1, k - 1], max(detH[i, j + 1, k], detH[i, j + 1, k + 1])))

                    greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j - 1, k - 1],
                                                             max(detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])))
                    greaterOrEqual = max(greaterOrEqual,
                                         max(detH[i + 1, j, k - 1], max(detH[i + 1, j, k], detH[i + 1, j, k + 1])))
                    greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j + 1, k - 1],
                                                             max(detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])))

                    if not (detH[i, j, k] >= greaterOrEqual):
                        continue

                    Maxes.append(detH[i, j, k])

                    if map_wave is not None:
                        map_wave[i, j] = max(map_wave[i, j], detH[i, j, k])

                    if scaleMap is not None:
                        scaleMap[i, j] = 1.0 * (1.5 ** k)

        return np.array(Maxes)

    except Exception as e:
        handle_error("GetMaxes", e)
        return np.array([])


def FindHessianBlobs(im, detH, LG, minResponse, particleType, maxCurvatureRatio):
    """
    Find Hessian blobs by detecting scale-space extrema.
    ParticleType is -1 for negative particles only, 1 for positive only, 0 for both

    Exact translation of Igor Pro FindHessianBlobs() function.
    """
    try:
        # Square the minResponse, since the parameter is provided as the square root
        minResponse = minResponse ** 2

        # mapNum: Map identifying particle numbers
        mapNum = np.full(detH.shape, -1, dtype=int)

        # mapLG: Map identifying the value of the LoG at the defined scale
        mapLG = np.zeros_like(detH)

        # mapMax: Map identifying the value of the LoG of the maximum pixel
        mapMax = np.zeros_like(detH)

        # Maintain an info list with particle boundaries and info
        Info = []

        limI, limJ, limK = detH.shape
        cnt = 0

        # Start with smallest blobs then go to larger blobs
        for k in range(1, limK - 1):
            for i in range(1, limI - 1):
                for j in range(1, limJ - 1):

                    # Does it hit the threshold?
                    if detH[i, j, k] < minResponse:
                        continue

                    # Is it too edgy?
                    if detH[i, j, k] > 0 and LG[i, j, k] ** 2 / detH[i, j, k] >= (
                            maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                        continue

                    # Is there a particle there already?
                    if mapNum[i, j, k] > -1 and detH[i, j, k] <= Info[mapNum[i, j, k]][3]:
                        continue

                    # Is it the right type of particle?
                    if ((particleType == -1 and LG[i, j, k] < 0) or (particleType == 1 and LG[i, j, k] > 0)):
                        continue

                    # Check if it is a local maximum - exact Igor Pro logic
                    if k != 0:
                        strictlyGreater = max(detH[i - 1, j - 1, k - 1], max(detH[i - 1, j - 1, k],
                                                                             max(detH[i - 1, j, k - 1],
                                                                                 detH[i, j - 1, k - 1])))
                        strictlyGreater = max(strictlyGreater,
                                              max(detH[i, j, k - 1], max(detH[i, j - 1, k], detH[i - 1, j, k])))
                    else:
                        strictlyGreater = max(detH[i - 1, j - 1, k], detH[i, j - 1, k], detH[i - 1, j, k])

                    if not (detH[i, j, k] > strictlyGreater):
                        continue

                    if k != 0:
                        greaterOrEqual = detH[i - 1, j - 1, k + 1]
                        greaterOrEqual = max(greaterOrEqual, detH[i - 1, j, k + 1])
                        greaterOrEqual = max(greaterOrEqual, max(detH[i - 1, j + 1, k - 1],
                                                                 max(detH[i - 1, j + 1, k], detH[i - 1, j + 1, k + 1])))

                        greaterOrEqual = max(greaterOrEqual, detH[i, j - 1, k + 1])
                        greaterOrEqual = max(greaterOrEqual, detH[i, j, k + 1])
                        greaterOrEqual = max(greaterOrEqual,
                                             max(detH[i, j + 1, k - 1], max(detH[i, j + 1, k], detH[i, j + 1, k + 1])))

                        greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j - 1, k - 1],
                                                                 max(detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])))
                        greaterOrEqual = max(greaterOrEqual,
                                             max(detH[i + 1, j, k - 1], max(detH[i + 1, j, k], detH[i + 1, j, k + 1])))
                        greaterOrEqual = max(greaterOrEqual, max(detH[i + 1, j + 1, k - 1],
                                                                 max(detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])))
                    else:
                        greaterOrEqual = max(detH[i - 1, j - 1, k + 1], detH[i - 1, j, k + 1], detH[i - 1, j + 1, k],
                                             detH[i - 1, j + 1, k + 1], detH[i, j - 1, k + 1], detH[i, j, k + 1])
                        greaterOrEqual = max(greaterOrEqual, detH[i, j + 1, k], detH[i, j + 1, k + 1],
                                             detH[i + 1, j - 1, k], detH[i + 1, j - 1, k + 1])
                        greaterOrEqual = max(greaterOrEqual, detH[i + 1, j, k], detH[i + 1, j, k + 1],
                                             detH[i + 1, j + 1, k], detH[i + 1, j + 1, k + 1])

                    if not (detH[i, j, k] >= greaterOrEqual):
                        continue

                    # It's a local max, is it overlapped and bigger than another one already?
                    if mapNum[i, j, k] > -1:
                        Info[mapNum[i, j, k]][0] = i
                        Info[mapNum[i, j, k]][1] = j
                        Info[mapNum[i, j, k]][3] = detH[i, j, k]
                        continue

                    # It's a local max, proceed to fill out the feature
                    numPixels, boundingBox = ScanlineFill8_LG(detH, mapMax, LG, i, j, k, 0, mapNum, cnt)

                    if numPixels > 0:
                        particle_info = [0] * 15
                        particle_info[0] = i
                        particle_info[1] = j
                        particle_info[2] = numPixels
                        particle_info[3] = detH[i, j, k]
                        particle_info[4] = boundingBox[0]
                        particle_info[5] = boundingBox[1]
                        particle_info[6] = boundingBox[2]
                        particle_info[7] = boundingBox[3]
                        particle_info[8] = 1.0 * (1.5 ** k)  # scale
                        particle_info[9] = k
                        particle_info[10] = 1
                        Info.append(particle_info)
                        cnt += 1

        # Make the mapLG
        mapLG = np.where(mapNum != -1, detH, 0)

        return mapNum, mapLG, mapMax, Info

    except Exception as e:
        handle_error("FindHessianBlobs", e)
        return np.full(detH.shape, -1, dtype=int), np.zeros_like(detH), np.zeros_like(detH), []


def MaximalBlobs(info, mapNum):
    """
    Determine scale-maximal particles.

    Exact translation of Igor Pro MaximalBlobs() function.
    """
    try:
        if len(info) == 0:
            return -1

        # Initialize maximality of each particle as undetermined (-1)
        for i in range(len(info)):
            info[i][10] = -1

        # Make lists for organizing overlapped particles
        BlobListNumber = list(range(len(info)))
        BlobListStrength = [info[i][3] for i in range(len(info))]

        # Sort by blob strength
        sorted_indices = sorted(range(len(info)), key=lambda i: info[i][3], reverse=True)

        limK = mapNum.shape[2]

        for idx_pos in range(len(info)):
            blocked = False
            index = sorted_indices[idx_pos]
            k = int(info[index][9])

            for ii in range(int(info[index][4]), int(info[index][5]) + 1):
                for jj in range(int(info[index][6]), int(info[index][7]) + 1):
                    if mapNum[ii, jj, k] == index:
                        for kk in range(limK):
                            if mapNum[ii, jj, kk] != -1 and info[mapNum[ii, jj, kk]][10] == 1:
                                blocked = True
                                break
                        if blocked:
                            break
                if blocked:
                    break

            info[index][10] = 0 if blocked else 1

        return 0

    except Exception as e:
        handle_error("MaximalBlobs", e)
        return -1


def ScanlineFill8_LG(image, dest, LG, seedP, seedQ, layer, Thresh, dest2=None, fillVal2=None):
    """
    It's a speed demon, about 10x faster than iterative seedfill and more flexible as well.

    Exact translation of Igor Pro ScanlineFill8_LG() function.
    """
    try:
        fill = image[seedP, seedQ, layer]
        count = 0
        sgn = np.sign(LG[seedP, seedQ, layer])

        x0, xf = 0, image.shape[0] - 1
        y0, yf = 0, image.shape[1] - 1

        if seedP < x0 or seedQ < y0 or seedP > xf or seedQ > yf:
            return 0, [0, 0, 0, 0]
        elif image[seedP, seedQ, layer] <= Thresh:
            return 0, [0, 0, 0, 0]

        stack = [(seedP, seedQ)]
        visited = set()
        pixels = []

        while stack:
            i, j = stack.pop()

            if (i, j) in visited:
                continue
            if i < x0 or i > xf or j < y0 or j > yf:
                continue
            if image[i, j, layer] < Thresh:
                continue
            if np.sign(LG[i, j, layer]) != sgn:
                continue

            visited.add((i, j))
            pixels.append((i, j))
            dest[i, j, layer] = fill
            count += 1

            if dest2 is not None and fillVal2 is not None:
                dest2[i, j, layer] = fillVal2

            # Add 8-connected neighbors
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) not in visited:
                    stack.append((ni, nj))

        if len(pixels) == 0:
            return 0, [0, 0, 0, 0]

        p_coords = [p for p, q in pixels]
        q_coords = [q for p, q in pixels]

        BoundingBox = [min(p_coords), max(p_coords), min(q_coords), max(q_coords)]
        return count, BoundingBox

    except Exception as e:
        handle_error("ScanlineFill8_LG", e)
        return 0, [0, 0, 0, 0]


def ViewParticles():
    """
    View and examine individual detected particles - EXACT IGOR PRO TUTORIAL MATCH.

    Exact translation of Igor Pro ViewParticles() function.
    """
    try:
        # Get the folder containing particles - matching Igor Pro GetBrowserSelection
        particles_folder = filedialog.askdirectory(title="Select particles folder (e.g., YEG_1_Particles)")
        if not particles_folder:
            safe_print("No folder selected.")
            return

        # Look for Particle_X folders - matching Igor Pro structure exactly
        particle_folders = []
        try:
            for item in os.listdir(particles_folder):
                item_path = os.path.join(particles_folder, item)
                if os.path.isdir(item_path) and item.startswith("Particle_"):
                    try:
                        # Verify it's a valid particle number
                        particle_num = int(item.split("_")[-1])
                        particle_folders.append(item_path)
                    except ValueError:
                        continue
        except Exception as e:
            safe_print(f"Error reading folder contents: {e}")
            return

        if len(particle_folders) == 0:
            safe_print("No particle folders found in selected directory.")
            safe_print("Make sure you selected a folder like 'YEG_1_Particles' that contains 'Particle_X' subfolders.")
            return

        # Sort by particle number - matching Igor Pro order
        particle_folders.sort(key=lambda x: int(x.split("_")[-1]))

        safe_print(f"Found {len(particle_folders)} particles to view.")

        # Create Igor Pro-style particle viewer
        class IgorProParticleViewer:
            def __init__(self, folders):
                self.folders = folders
                self.current_index = 0

                # Create main window - matching Igor Pro Particle Viewer size
                self.root = tk.Toplevel()
                self.root.title("Particle Viewer")
                self.root.geometry("1200x900")
                self.root.configure(bg='#e6e6e6')

                # Set up the interface matching Figure 24 from tutorial
                self.setup_igor_interface()

                # Show first particle
                self.show_particle()

                # Bind keyboard events - matching Igor Pro shortcuts
                self.root.bind('<Key>', self.on_key_press)
                self.root.focus_set()

            def setup_igor_interface(self):
                """Set up interface matching Igor Pro tutorial Figure 24"""

                # Main content frame
                main_frame = tk.Frame(self.root, bg='#e6e6e6')
                main_frame.pack(fill='both', expand=True, padx=10, pady=10)

                # Left side - image display (larger, matching Igor Pro)
                left_frame = tk.Frame(main_frame, bg='#e6e6e6', width=800)
                left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
                left_frame.pack_propagate(False)

                # Image frame with border (matching Igor Pro image window)
                image_frame = tk.Frame(left_frame, bg='white', relief='sunken', bd=2)
                image_frame.pack(pady=10, padx=10, fill='both', expand=True)

                # Image canvas - matching Igor Pro image size
                self.canvas = tk.Canvas(image_frame, bg='black', width=700, height=700)
                self.canvas.pack(padx=5, pady=5)

                # Right side - controls panel (matching Igor Pro Controls panel)
                right_frame = tk.Frame(main_frame, bg='#e6e6e6', width=350)
                right_frame.pack(side='right', fill='y')
                right_frame.pack_propagate(False)

                # Controls title - matching Igor Pro "Controls" panel
                controls_title = tk.Label(right_frame, text="Controls",
                                          font=('Arial', 14, 'bold'), bg='#e6e6e6')
                controls_title.pack(pady=(0, 10))

                # Particle title - matching Igor Pro "Particle 21" style
                self.particle_title = tk.Label(right_frame, text="Particle 0",
                                               font=('Arial', 16, 'bold'), bg='#e6e6e6')
                self.particle_title.pack(pady=5)

                # Navigation buttons - matching Igor Pro layout
                nav_frame = tk.Frame(right_frame, bg='#e6e6e6')
                nav_frame.pack(pady=10)

                # Prev and Next buttons side by side
                btn_frame = tk.Frame(nav_frame, bg='#e6e6e6')
                btn_frame.pack()

                self.prev_btn = tk.Button(btn_frame, text="Prev",
                                          command=self.prev_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.prev_btn.pack(side='left', padx=5)

                self.next_btn = tk.Button(btn_frame, text="Next",
                                          command=self.next_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.next_btn.pack(side='left', padx=5)

                # Go To field - matching Igor Pro "Go To: 0"
                goto_frame = tk.Frame(right_frame, bg='#e6e6e6')
                goto_frame.pack(pady=10)

                tk.Label(goto_frame, text="Go To:", font=('Arial', 12), bg='#e6e6e6').pack(side='left')
                self.goto_entry = tk.Entry(goto_frame, width=10, font=('Arial', 12))
                self.goto_entry.pack(side='left', padx=5)
                self.goto_entry.bind('<Return>', self.goto_particle)

                # Counter display - matching Igor Pro "1/10" style
                self.counter_label = tk.Label(right_frame, text="1/1",
                                              font=('Arial', 12), bg='#e6e6e6')
                self.counter_label.pack(pady=5)

                # Measurements section - matching Igor Pro Figure 24
                measurements_frame = tk.LabelFrame(right_frame, text="Measurements",
                                                   font=('Arial', 12, 'bold'), bg='#e6e6e6')
                measurements_frame.pack(fill='x', pady=10, padx=10)

                # Height display - matching Igor Pro "Height (nm)"
                height_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                height_frame.pack(fill='x', pady=5)
                tk.Label(height_frame, text="Height (nm)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.height_label = tk.Label(height_frame, text="0.0000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.height_label.pack(fill='x', padx=5)

                # Volume display - matching Igor Pro "Volume (m^3 e-25)"
                volume_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                volume_frame.pack(fill='x', pady=5)
                tk.Label(volume_frame, text="Volume (m^3 e-25)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.volume_label = tk.Label(volume_frame, text="0.000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.volume_label.pack(fill='x', padx=5)

                # Area display - additional measurement
                area_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                area_frame.pack(fill='x', pady=5)
                tk.Label(area_frame, text="Area", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.area_label = tk.Label(area_frame, text="0.0",
                                           font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.area_label.pack(fill='x', padx=5)

                # DELETE button - matching Igor Pro red DELETE button
                self.delete_btn = tk.Button(right_frame, text="DELETE",
                                            command=self.delete_particle,
                                            bg='#ff6b6b', fg='white',
                                            font=('Arial', 12, 'bold'),
                                            width=20, height=2, relief='raised', bd=2)
                self.delete_btn.pack(pady=20)

            def show_particle(self):
                """Display current particle - matching Igor Pro tutorial visualization"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Load particle files
                    particle_file = os.path.join(current_folder, f"Particle_{particle_num}.npy")
                    mask_file = os.path.join(current_folder, f"Mask_{particle_num}.npy")
                    info_file = os.path.join(current_folder, f"Particle_{particle_num}_info.txt")

                    if not os.path.exists(particle_file):
                        safe_print(f"Particle file not found: {particle_file}")
                        return

                    particle = np.load(particle_file)

                    # Update labels - matching Igor Pro style
                    self.particle_title.config(text=f"Particle {particle_num}")
                    self.counter_label.config(text=f"{self.current_index + 1}/{len(self.folders)}")
                    self.goto_entry.delete(0, tk.END)
                    self.goto_entry.insert(0, str(self.current_index))

                    # Display particle image with hot colormap and green contour
                    self.display_igor_image(particle, mask_file)

                    # Update measurements - matching Igor Pro format
                    self.update_measurements(info_file)

                    # Update window title
                    self.root.title(f"Particle Viewer - Particle {particle_num}")

                except Exception as e:
                    handle_error("show_particle", e)

            def display_igor_image(self, particle, mask_file):
                """Display particle image matching Igor Pro tutorial Figure 24 exactly"""
                try:
                    # Clear canvas
                    self.canvas.delete("all")

                    # Apply hot colormap exactly like Igor Pro
                    vmin, vmax = np.min(particle), np.max(particle)
                    norm = Normalize(vmin=vmin, vmax=vmax)

                    hot_cmap = cm.get_cmap('hot')
                    colored = hot_cmap(norm(particle))
                    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

                    # Convert to PIL Image
                    img = Image.fromarray(colored_rgb)

                    # Resize to fit canvas (matching Igor Pro size)
                    canvas_size = 650
                    img_resized = img.resize((canvas_size, canvas_size), Image.NEAREST)

                    # Convert to PhotoImage
                    self.photo = ImageTk.PhotoImage(img_resized)

                    # Display on canvas
                    self.canvas.create_image(350, 350, image=self.photo)

                    # Add GREEN contour exactly like Igor Pro Figure 24
                    if os.path.exists(mask_file):
                        try:
                            mask = np.load(mask_file)

                            # Resize mask to match image
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_resized = np.array(mask_img.resize((canvas_size, canvas_size), Image.NEAREST))

                            # Create GREEN contour - matching Igor Pro exactly
                            for i in range(1, canvas_size - 1):
                                for j in range(1, canvas_size - 1):
                                    if mask_resized[i, j] > 127:  # Inside mask
                                        # Check if it's an edge pixel
                                        neighbors = [
                                            mask_resized[i - 1, j], mask_resized[i + 1, j],
                                            mask_resized[i, j - 1], mask_resized[i, j + 1]
                                        ]
                                        if any(n < 127 for n in neighbors):
                                            x, y = j + 25, i + 25  # Offset for centering
                                            # Draw GREEN pixels - matching Igor Pro green contour
                                            self.canvas.create_rectangle(x, y, x + 2, y + 2,
                                                                         fill='#00ff00', outline='#00ff00')

                        except Exception as e:
                            safe_print(f"Could not display mask contour: {e}")

                    # Add coordinate axes labels if needed (matching Igor Pro)
                    self.canvas.create_text(350, 680, text="Pixels", font=('Arial', 10))

                except Exception as e:
                    handle_error("display_igor_image", e)

            def update_measurements(self, info_file):
                """Update measurements display - matching Igor Pro format"""
                try:
                    height_val = "0.0000"
                    volume_val = "0.000"
                    area_val = "0.0"

                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith('Height:'):
                                    height_val = line.split(':')[1].strip()
                                elif line.startswith('Volume:'):
                                    volume_val = line.split(':')[1].strip()
                                elif line.startswith('Area:'):
                                    area_val = line.split(':')[1].strip()

                    # Update labels with proper formatting
                    self.height_label.config(text=height_val)
                    self.volume_label.config(text=volume_val)
                    self.area_label.config(text=area_val)

                except Exception as e:
                    handle_error("update_measurements", e)

            def prev_particle(self):
                """Navigate to previous particle - matching Igor Pro Prev button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index - 1) % len(self.folders)
                    self.show_particle()

            def next_particle(self):
                """Navigate to next particle - matching Igor Pro Next button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index + 1) % len(self.folders)
                    self.show_particle()

            def goto_particle(self, event=None):
                """Go to specific particle - matching Igor Pro Go To functionality"""
                try:
                    particle_index = int(self.goto_entry.get())
                    if 0 <= particle_index < len(self.folders):
                        self.current_index = particle_index
                        self.show_particle()
                except ValueError:
                    pass  # Invalid input, ignore

            def delete_particle(self):
                """Delete current particle - matching Igor Pro DELETE button"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Matching Igor Pro delete dialog exactly
                    result = messagebox.askyesno(
                        f"Deleting Particle {particle_num}..",
                        f"Are you sure you want to delete Particle {particle_num}?",
                        parent=self.root
                    )

                    if result:
                        shutil.rmtree(current_folder)
                        self.folders.pop(self.current_index)

                        if len(self.folders) == 0:
                            safe_print("No more particles to view.")
                            self.root.destroy()
                            return

                        if self.current_index >= len(self.folders):
                            self.current_index = len(self.folders) - 1

                        self.show_particle()

                except Exception as e:
                    handle_error("delete_particle", e)

            def on_key_press(self, event):
                """Handle keyboard shortcuts - matching Igor Pro tutorial"""
                if event.keysym == 'Left':
                    self.prev_particle()
                elif event.keysym == 'Right':
                    self.next_particle()
                elif event.keysym == 'space' or event.keysym == 'Down':
                    self.delete_particle()
                elif event.keysym == 'Return':
                    self.goto_particle()

        # Create and run viewer
        viewer = IgorProParticleViewer(particle_folders)

        # Print Igor Pro-style instructions
        safe_print("=" * 60)
        safe_print("PARTICLE VIEWER CONTROLS:")
        safe_print("- Left/Right arrows: Navigate between particles")
        safe_print("- Space bar or Down arrow: Delete current particle")
        safe_print("- Enter in Go To field: Jump to particle")
        safe_print("- Close window when finished")
        safe_print("=" * 60)

    except Exception as e:
        error_msg = handle_error("ViewParticles", e)
        try:
            messagebox.showerror("Viewer Error", error_msg)
        except:
            safe_print(error_msg)


def BilinearInterpolate(im, x0, y0, r0=0):
    """
    Threadsafe Function BilinearInterpolate(im,x0,y0,[r0])

    Exact translation of Igor Pro BilinearInterpolate() function.
    """
    try:
        pMid = x0
        p0 = max(0, int(np.floor(pMid)))
        p1 = min(im.shape[0] - 1, int(np.ceil(pMid)))
        qMid = y0
        q0 = max(0, int(np.floor(qMid)))
        q1 = min(im.shape[1] - 1, int(np.ceil(qMid)))

        if len(im.shape) == 3:
            pInterp0 = im[p0, q0, r0] + (im[p1, q0, r0] - im[p0, q0, r0]) * (pMid - p0)
            pInterp1 = im[p0, q1, r0] + (im[p1, q1, r0] - im[p0, q1, r0]) * (pMid - p0)
        else:
            pInterp0 = im[p0, q0] + (im[p1, q0] - im[p0, q0]) * (pMid - p0)
            pInterp1 = im[p0, q1] + (im[p1, q1] - im[p0, q1]) * (pMid - p0)

        return pInterp0 + (pInterp1 - pInterp0) * (qMid - q0)

    except Exception as e:
        handle_error("BilinearInterpolate", e)
        return 0.0


def ExpandBoundary8(mask):
    """
    Function ExpandBoundary8(mask)

    Exact translation of Igor Pro ExpandBoundary8() function.
    """
    try:
        # Create expanded mask using 8-connectivity
        expanded = mask.copy()
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if mask.shape[2] > 1:  # 3D case
                    for k in range(mask.shape[2]):
                        if (mask[i, j, k] == 0 and
                                (mask[i + 1, j, k] == 1 or mask[i - 1, j, k] == 1 or
                                 mask[i, j + 1, k] == 1 or mask[i, j - 1, k] == 1 or
                                 mask[i + 1, j + 1, k] == 1 or mask[i - 1, j + 1, k] == 1 or
                                 mask[i + 1, j - 1, k] == 1 or mask[i - 1, j - 1, k] == 1)):
                            expanded[i, j, k] = 2
                else:  # 2D case
                    if (mask[i, j] == 0 and
                            (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                             mask[i, j + 1] == 1 or mask[i, j - 1] == 1 or
                             mask[i + 1, j + 1] == 1 or mask[i - 1, j + 1] == 1 or
                             mask[i + 1, j - 1] == 1 or mask[i - 1, j - 1] == 1)):
                        expanded[i, j] = 2

        # Convert expanded pixels to 1
        mask[:] = (expanded > 0).astype(mask.dtype)
        return 0

    except Exception as e:
        handle_error("ExpandBoundary8", e)
        return -1


def ExpandBoundary4(mask):
    """
    Function ExpandBoundary4(mask)

    Exact translation of Igor Pro ExpandBoundary4() function.
    """
    try:
        # Create expanded mask using 4-connectivity
        expanded = mask.copy()
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                if mask.shape[2] > 1:  # 3D case
                    for k in range(mask.shape[2]):
                        if (mask[i, j, k] == 0 and
                                (mask[i + 1, j, k] == 1 or mask[i - 1, j, k] == 1 or
                                 mask[i, j + 1, k] == 1 or mask[i, j - 1, k] == 1)):
                            expanded[i, j, k] = 2
                else:  # 2D case
                    if (mask[i, j] == 0 and
                            (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                             mask[i, j + 1] == 1 or mask[i, j - 1] == 1)):
                        expanded[i, j] = 2

        # Convert expanded pixels to 1
        mask[:] = (expanded > 0).astype(mask.dtype)
        return 0

    except Exception as e:
        handle_error("ExpandBoundary4", e)
        return -1


def Testing(str_input, num):
    """
    Testing function to demonstrate how user-defined functions work.

    Exact translation of Igor Pro Testing() function.
    """
    try:
        safe_print(f"You typed: {str_input}")
        safe_print(f"Your number plus two is {num + 2}")

    except Exception as e:
        handle_error("Testing", e)
