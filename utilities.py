"""
Utilities Module
Contains various utility functions used throughout the blob detection algorithm
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
from igor_compatibility import *
from file_io import *
from particle_measurements import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import messagebox


def FindHessianBlobs(im, detH, LG, detHResponseThresh, mapNum, mapDetH, mapMax, info,
                     particleType, maxCurvatureRatio):
    """
    The map and info must be fed into the function since Igor doesn't return multiple objects..
    ParticleType is -1 for negative particles only, 1 for positive only, 0 for both
    """
    # Square the minResponse, since the parameter is provided as the square root
    # of the actual minimum detH response so that it is in normal image units
    minResponse = detHResponseThresh ** 2

    # mapNum: Map identifying particle numbers
    mapNum.data = np.full(detH.data.shape, -1)
    mapNum.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapNum.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    # mapLG: Map identifying the value of the LoG at the defined scale
    mapDetH.data = np.zeros(detH.data.shape)
    mapDetH.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapDetH.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    # mapMax: Map identifying the value of the maximum pixel in the particle
    mapMax.data = np.zeros(detH.data.shape)
    mapMax.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    mapMax.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))

    # Maintain an info wave with particle boundaries and info
    max_particles = im.data.shape[0] * im.data.shape[1] * detH.data.shape[2] // 27
    info.data = np.zeros((max_particles, 15))

    # Make a bounding box wave for scanfill
    box = np.zeros(4)

    limI = detH.data.shape[0] - 1
    limJ = detH.data.shape[1] - 1
    limK = detH.data.shape[2] - 1
    cnt = 0

    # Start with smallest blobs then go to larger blobs
    for k in range(1, limK):
        for i in range(1, limI):
            for j in range(1, limJ):

                # Does it hit the threshold?
                if detH.data[i, j, k] < minResponse:
                    continue

                # Is it too edgy?
                if LG.data[i, j, k] ** 2 / detH.data[i, j, k] >= (maxCurvatureRatio + 1) ** 2 / maxCurvatureRatio:
                    continue

                # Is there a particle there already?
                if mapNum.data[i, j, k] > -1 and detH.data[i, j, k] <= info.data[int(mapNum.data[i, j, k]), 3]:
                    continue

                # Is it the right type of particle?
                if (particleType == -1 and LG.data[i, j, k] < 0) or (particleType == 1 and LG.data[i, j, k] > 0):
                    continue

                # Check if it's a local maximum in 3D
                is_maximum = True
                current_val = detH.data[i, j, k]

                # Check strictly greater neighbors (scale below)
                if k > 0:
                    neighbors = [
                        detH.data[i - 1, j - 1, k - 1], detH.data[i - 1, j - 1, k], detH.data[i - 1, j, k - 1],
                        detH.data[i, j - 1, k - 1], detH.data[i, j, k - 1], detH.data[i, j - 1, k],
                        detH.data[i - 1, j, k]
                    ]
                    strictly_greater = max(neighbors)
                else:
                    strictly_greater = max(detH.data[i - 1, j - 1, k], detH.data[i, j - 1, k], detH.data[i - 1, j, k])

                if not (current_val > strictly_greater):
                    continue

                # Check greater or equal neighbors (scale above and same scale)
                greater_or_equal = -np.inf

                # Same scale and above scale neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [0, 1] if k < limK else [0]:
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            ni, nj, nk = i + di, j + dj, k + dk
                            if 0 <= ni <= limI and 0 <= nj <= limJ and 0 <= nk <= limK:
                                greater_or_equal = max(greater_or_equal, detH.data[ni, nj, nk])

                if not (current_val >= greater_or_equal):
                    continue

                # It's a local max, is it overlapped and bigger than another one already?
                if mapNum.data[i, j, k] > -1:
                    info.data[int(mapNum.data[i, j, k]), 0] = i
                    info.data[int(mapNum.data[i, j, k]), 1] = j
                    info.data[int(mapNum.data[i, j, k]), 3] = current_val
                    continue

                # It's a local max, proceed to fill out the feature.
                numPixels = ScanlineFill8_LG(detH, mapMax, LG, i, j, 0,
                                             BoundingBox=box, fillVal=current_val, layer=k,
                                             dest2=mapNum, fillVal2=cnt, destLayer=k)

                if numPixels == -2:
                    mapMax.data[i, j, k] = current_val
                    mapNum.data[i, j, k] = cnt
                    numPixels = 1
                    box[0] = box[1] = i
                    box[2] = box[3] = j

                # Store particle info
                if cnt < info.data.shape[0]:
                    info.data[cnt, 0] = i
                    info.data[cnt, 1] = j
                    info.data[cnt, 2] = real(numPixels)
                    info.data[cnt, 3] = current_val
                    info.data[cnt, 4] = box[0]
                    info.data[cnt, 5] = box[1]
                    info.data[cnt, 6] = box[2]
                    info.data[cnt, 7] = box[3]
                    info.data[cnt, 8] = DimOffset(detH, 2) * (DimDelta(detH, 2) ** k)
                    info.data[cnt, 9] = k
                    info.data[cnt, 10] = 1

                cnt += 1

    # Remove unused rows in the info wave
    if cnt < info.data.shape[0]:
        info.data = info.data[:cnt, :]

    # Make the mapLG
    mapDetH.data = np.where(mapNum.data != -1, detH.data, 0)

    return 0


def MaximalBlobs(info, mapNum):
    """
    Determines which blobs are scale-maximal (non-overlapping)
    """
    if info.data.shape[0] == 0:
        return -1

    # Initialize maximality of each particle as undetermined (-1)
    info.data[:, 10] = -1

    # Make lists for organizing overlapped particles
    blob_numbers = np.arange(info.data.shape[0])
    blob_strengths = info.data[:, 3].copy()

    # Sort by blob strength (descending)
    sort_indices = np.argsort(-blob_strengths)
    blob_numbers = blob_numbers[sort_indices]
    blob_strengths = blob_strengths[sort_indices]

    limK = mapNum.data.shape[2]

    for i in range(len(blob_numbers)):
        # See if there's room for the i'th strongest particle
        blocked = False
        index = blob_numbers[i]
        k = int(info.data[index, 9])

        # Check if this particle overlaps with any already accepted particle
        for ii in range(int(info.data[index, 4]), int(info.data[index, 5]) + 1):
            for jj in range(int(info.data[index, 6]), int(info.data[index, 7]) + 1):
                if mapNum.data[ii, jj, k] == index:
                    # Check all scales at this position
                    for kk in range(limK):
                        if (mapNum.data[ii, jj, kk] != -1 and
                                mapNum.data[ii, jj, kk] != index and
                                info.data[int(mapNum.data[ii, jj, kk]), 10] == 1):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if blocked:
                break

        info.data[index, 10] = 0 if blocked else 1

    return 0


def InteractiveThresholdGUI(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Interactive threshold selection GUI
    """
    # First identify the maxes
    im_copy = Wave(im.data.copy(), "SS_MAXMAP")
    im_copy.data.fill(-1)
    scale_map = Wave(im_copy.data.copy(), "SS_MAXSCALEMAP")

    maxes = Maxes(detH, LG, particleType, maxCurvatureRatio, im_copy, scale_map)
    maxes.data = np.sqrt(maxes.data)  # Put it into image units

    if len(maxes.data) == 0:
        return 0.0

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    # Display image
    im_display = ax.imshow(im.data, cmap='gray', origin='lower',
                           extent=[DimOffset(im, 0),
                                   DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
                                   DimOffset(im, 1),
                                   DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)])
    ax.set_title("Interactive Blob Strength Selection")

    # Initial threshold
    initial_thresh = np.max(maxes.data) / 2

    # Create slider
    ax_thresh = plt.axes([0.2, 0.1, 0.5, 0.03])
    thresh_slider = Slider(ax_thresh, 'Blob Strength',
                           0, np.max(maxes.data) * 1.1,
                           valinit=initial_thresh)

    # Create buttons
    ax_accept = plt.axes([0.75, 0.1, 0.1, 0.04])
    ax_quit = plt.axes([0.75, 0.05, 0.1, 0.04])
    accept_button = Button(ax_accept, 'Accept')
    quit_button = Button(ax_quit, 'Quit')

    # Store circles for updating
    circles = []

    result = {'threshold': initial_thresh, 'quit': False}

    def update_threshold(val):
        # Clear previous circles
        for circle in circles:
            circle.remove()
        circles.clear()

        threshold = thresh_slider.val

        # Draw circles for blobs above threshold
        for i in range(im_copy.data.shape[0]):
            for j in range(im_copy.data.shape[1]):
                if im_copy.data[i, j] > threshold ** 2:
                    xc = DimOffset(im_copy, 0) + i * DimDelta(im_copy, 0)
                    yc = DimOffset(im_copy, 1) + j * DimDelta(im_copy, 1)
                    rad = np.sqrt(2 * scale_map.data[i, j])

                    circle = plt.Circle((xc, yc), rad, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)
                    circles.append(circle)

        fig.canvas.draw()

    def accept(event):
        result['threshold'] = thresh_slider.val
        plt.close()

    def quit_app(event):
        result['quit'] = True
        plt.close()

    thresh_slider.on_changed(update_threshold)
    accept_button.on_clicked(accept)
    quit_button.on_clicked(quit_app)

    # Initial update
    update_threshold(initial_thresh)

    plt.show()

    if result['quit']:
        raise SystemExit("User quit the analysis")

    return result['threshold']


def Testing(string, num):
    """
    Testing function from original Igor code
    """
    print(f"You typed: {string}")
    print(f"Your number plus two is {num + 2}")


def KernelDensity(data, points=250, start=None, stop=None, bandwidth=None):
    """
    Kernel density estimation with an Epanechnikov kernel.
    """
    name = f"{data.name}_Epan"

    if start is None or stop is None or bandwidth is None:
        # Make some appropriate starting guesses
        data_clean = data.data[~np.isnan(data.data)]
        if len(data_clean) == 0:
            return Wave(np.array([]), name)

        min_val, max_val = np.min(data_clean), np.max(data_clean)

        if bandwidth is None:
            bandwidth = 3.5 * np.sqrt(np.var(data_clean)) / (len(data_clean) ** (1 / 3))
        if start is None:
            start = min_val - bandwidth
        if stop is None:
            stop = max_val + bandwidth

    # Create output wave
    epdf = Wave(np.zeros(points), name)
    epdf.SetScale('x', start, (stop - start) / (points - 1))

    # Create x coordinates
    x_coords = np.linspace(start, stop, points)

    # Compute kernel density
    count = 0
    for i, data_point in enumerate(data.data):
        if not np.isnan(data_point):
            # Epanechnikov kernel
            u = (x_coords - data_point) / bandwidth
            kernel_vals = np.maximum(1 - u ** 2, 0)
            epdf.data += kernel_vals
            count += 1

    # Normalize
    if count > 0:
        epdf.data /= count * 4 * bandwidth / 3

    # Add note with parameters
    epdf.note = f"Start:{start}\nStop:{stop}\nBandwidth:{bandwidth}\nPoints:{points}\nDataPoints:{count}"

    return epdf


def ViewParticles():
    """
    Particle viewer function (simplified version)
    This would be implemented as a full GUI in a complete version
    """
    particles_df = GetBrowserSelection(0)
    folder = data_browser.get_folder(particles_df.rstrip(':'))

    if folder is None or len(folder.subfolders) == 0:
        messagebox.showerror("Error", "Please select the folder containing the crop folders.")
        return -1

    print(f"Viewing particles in {particles_df}")
    print(f"Found {len(folder.subfolders)} particle folders")

    # This would implement the full particle viewer GUI
    # For now, just print information
    for subfolder_name in folder.subfolders:
        subfolder = folder.subfolders[subfolder_name]
        if 'Particle_' in subfolder_name and len(subfolder.waves) > 0:
            particle_wave = list(subfolder.waves.values())[0]
            note_info = particle_wave.note

            height = NumberByKey("Height", note_info, ":", "\n")
            volume = NumberByKey("Volume", note_info, ":", "\n")
            area = NumberByKey("Area", note_info, ":", "\n")

            print(f"{subfolder_name}: Height={height:.4f}, Volume={volume:.4e}, Area={area:.4f}")

    return 0


def ParticleNumber(name):
    """
    Extract particle number from particle name
    """
    parts = name.split('_')
    if len(parts) > 1:
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    return 0


def SubfoldersList(folder):
    """
    Get list of subfolders in a data folder
    """
    if folder is None:
        return []
    return list(folder.subfolders.keys())