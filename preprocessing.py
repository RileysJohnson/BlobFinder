"""
Preprocessing Functions
Handles image flattening and streak removal
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
from scipy.optimize import curve_fit
from igor_compatibility import *
from file_io import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import messagebox


def BatchPreprocess():
    """
    Allows you to preprocess multiple images in a data folder successively. Make sure to highlight the
    folder containing the images in the data browser before executing.
    """
    ImagesDF = GetBrowserSelection(0)
    CurrentDF = GetDataFolder(1)

    if not DataFolderExists(ImagesDF) or CountObjects(ImagesDF, 1) < 1:
        messagebox.showerror("Error", "Select the folder with your images in it in the data browser, then try again.")
        return -1

    # Declare algorithm parameters.
    flattenOrder = 2
    streakRemovalSDevs = 3

    # Get parameters from user (simplified GUI)
    root = tk.Tk()
    root.title("Preprocessing Parameters")
    root.geometry("400x200")

    tk.Label(root, text="Polynomial order for flattening:").pack()
    flatten_var = tk.IntVar(value=flattenOrder)
    tk.Entry(root, textvariable=flatten_var).pack()

    tk.Label(root, text="Std. Deviations for streak removal:").pack()
    streak_var = tk.DoubleVar(value=streakRemovalSDevs)
    tk.Entry(root, textvariable=streak_var).pack()

    result = {'confirmed': False}

    def confirm():
        result['flattenOrder'] = flatten_var.get()
        result['streakRemovalSDevs'] = streak_var.get()
        result['confirmed'] = True
        root.destroy()

    def cancel():
        root.destroy()

    tk.Button(root, text="OK", command=confirm).pack(side=tk.LEFT, padx=20, pady=20)
    tk.Button(root, text="Cancel", command=cancel).pack(side=tk.RIGHT, padx=20, pady=20)

    root.mainloop()

    if not result['confirmed']:
        return -1

    flattenOrder = result['flattenOrder']
    streakRemovalSDevs = result['streakRemovalSDevs']

    # Preprocess the images
    folder = data_browser.get_folder(ImagesDF.rstrip(':'))
    NumImages = len(folder.waves)

    for i in range(NumImages):
        wave_name = list(folder.waves.keys())[i]
        im = folder.waves[wave_name]

        if streakRemovalSDevs > 0:
            RemoveStreaks(im, sigma=streakRemovalSDevs)
        if flattenOrder > 0:
            Flatten(im, flattenOrder)

    return 0


def Flatten(im, order, mask=None, noThresh=False):
    """
    Flattens every horizontal line scan of the image by subtracting off a least-squares
    fitted polynomial of a given order.
        im : The image to be flattened.
        order : The order of the polynomial to be subtracted off.
        mask : An optional mask identifying pixels to fit the polynomial to.
               In the mask, 1 corresponds to pixels which will be used for fitting, 0 for pixels to be ignored.
        noThresh : An optional parameter, if given any value will not prompt the user to set the threshold level.
    """
    # Want to interactively determine a threshold?
    if not noThresh:
        threshold = InteractiveFlattenThreshold(im)
        if threshold is None:
            return  # User cancelled

        # Make the mask wave
        mask = Wave((im.data <= threshold).astype(int), "FLATTEN_MASK")
        print(f"Flatten Height Threshold: {threshold}")

    # Make a 1D wave for fitting and masking
    scanline = Wave(np.zeros(im.data.shape[0]), "FLATTEN_SCANLINE")
    scanline.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))

    mask1D = Wave(np.ones(im.data.shape[0]), "FLATTEN_MASK")

    # Make the coefficient wave
    coefs = Wave(np.zeros(max(2, order + 1)), "FLATTEN_COEFS")

    # Fit to each scan line
    lines = im.data.shape[1]
    x_coords = np.arange(im.data.shape[0]) * DimDelta(im, 0) + DimOffset(im, 0)

    for i in range(lines):
        scanline.data = im.data[:, i]

        if mask is not None or not noThresh:
            if mask is not None:
                mask1D.data = mask.data[:, i]
            else:
                mask1D.data = (scanline.data <= threshold).astype(int)

        # Do a fit to the scan line
        valid_indices = mask1D.data.astype(bool)
        if np.sum(valid_indices) < order + 1:
            continue  # Not enough points for fit

        x_fit = x_coords[valid_indices]
        y_fit = scanline.data[valid_indices]

        try:
            if order == 1:
                # Linear fit
                def line_func(x, a, b):
                    return a + b * x

                popt, _ = curve_fit(line_func, x_fit, y_fit)
                im.data[:, i] -= (popt[0] + x_coords * popt[1])

            elif order == 0:
                # Constant fit (mean)
                mean_val = np.mean(y_fit)
                im.data[:, i] -= mean_val

            elif order > 1:
                # Polynomial fit
                popt = np.polyfit(x_fit, y_fit, order)
                poly_vals = np.polyval(popt, x_coords)
                im.data[:, i] -= poly_vals

        except Exception as e:
            print(f"Fit failed for line {i}: {e}")
            continue

    return 0


def InteractiveFlattenThreshold(im):
    """
    Interactive threshold selection for flattening
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # Display image
    im_display = ax.imshow(im.data, cmap='gray', origin='lower')
    ax.set_title("Set Flatten Threshold")

    # Initial threshold
    initial_thresh = np.mean(im.data)

    # Create slider
    ax_thresh = plt.axes([0.2, 0.1, 0.5, 0.03])
    thresh_slider = Slider(ax_thresh, 'Threshold',
                           np.min(im.data), np.max(im.data),
                           valinit=initial_thresh)

    # Create buttons
    ax_accept = plt.axes([0.8, 0.1, 0.1, 0.04])
    ax_cancel = plt.axes([0.8, 0.05, 0.1, 0.04])
    accept_button = Button(ax_accept, 'Accept')
    cancel_button = Button(ax_cancel, 'Cancel')

    result = {'threshold': None, 'cancelled': False}

    def update_threshold(val):
        threshold = thresh_slider.val
        # Update image to show masked regions in blue
        masked_image = im.data.copy()
        masked_image = np.stack([masked_image, masked_image, masked_image], axis=2)
        masked_image = (masked_image - np.min(masked_image)) / (np.max(masked_image) - np.min(masked_image))

        # Set masked pixels to blue
        mask = im.data <= threshold
        masked_image[mask, 0] = 0.1  # Low red
        masked_image[mask, 1] = 0.4  # Medium green
        masked_image[mask, 2] = 1.0  # High blue

        im_display.set_array(masked_image)
        fig.canvas.draw()

    def accept(event):
        result['threshold'] = thresh_slider.val
        plt.close()

    def cancel(event):
        result['cancelled'] = True
        plt.close()

    thresh_slider.on_changed(update_threshold)
    accept_button.on_clicked(accept)
    cancel_button.on_clicked(cancel)

    # Initial update
    update_threshold(initial_thresh)

    plt.show()

    if result['cancelled']:
        return None
    return result['threshold']


def RemoveStreaks(image, sigma=3):
    """
    Removes streak artifacts from the image.
        image : The image from which streaks will be removed.
        sigma : The number of standard deviations away from mean streak level to smooth a streak.
    """
    # Produce the dY map
    dyMap_wave = dyMap(image)
    dyMap_data = np.abs(dyMap_wave.data)

    # Calculate statistics
    avg_dy = np.mean(dyMap_data)
    sdev_dy = np.std(dyMap_data)
    max_dy = avg_dy + sdev_dy * sigma

    limI = image.data.shape[0]
    limJ = image.data.shape[1] - 1

    for i in range(limI):
        for j in range(1, limJ):
            if dyMap_data[i, j] > max_dy:
                i0 = i

                # Go left until the left side of the streak is gone
                while i >= 0 and dyMap_data[i, j] > avg_dy:
                    image.data[i, j] = (image.data[i, j + 1] + image.data[i, j - 1]) / 2
                    dyMap_data[i, j] = 0
                    i -= 1

                i = i0

                # Go right from the original point doing the same thing
                while i < limI and dyMap_data[i, j] > avg_dy:
                    image.data[i, j] = (image.data[i, j + 1] + image.data[i, j - 1]) / 2
                    dyMap_data[i, j] = 0
                    i += 1

                i = i0

    return 0


def dyMap(image):
    """
    Computes the y-derivative map for streak detection
    """
    name = f"{image.name}_dyMap"
    dy_map = Wave(image.data.copy(), name)

    limQ = image.data.shape[1] - 1

    for i in range(image.data.shape[0]):
        for j in range(image.data.shape[1]):
            j_next = min(j + 1, limQ)
            j_prev = max(j - 1, 0)
            dy_map.data[i, j] = image.data[i, j] - (image.data[i, j_next] + image.data[i, j_prev]) / 2

    return dy_map


class FlattenGUI:
    """
    GUI for interactive flattening (if needed for more complex implementations)
    """

    def __init__(self, image):
        self.image = image
        self.threshold = np.mean(image.data)
        self.accepted = False

    def show(self):
        # This would implement the full flattening GUI
        # For now, return a simple threshold
        return self.threshold if not self.cancelled else None


def CleanWaveStats():
    """
    Igor CleanWaveStats equivalent - cleans up statistical variables
    """
    # In Igor, this cleans global variables. In Python, we'll just pass
    pass