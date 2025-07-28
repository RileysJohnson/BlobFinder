"""
Preprocessing Module
Contains image preprocessing functions for blob detection
Direct port from Igor Pro code maintaining same variable names and structure
COMPLETE IMPLEMENTATION: Streak removal, flattening, batch processing
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from scipy import ndimage
from pathlib import Path

from igor_compatibility import *
from file_io import *
from utilities import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def RemoveStreaks(im, sigma=3):
    """
    Remove horizontal streaks from image (matching Igor Pro implementation)

    Parameters:
    im : Wave - Input image to process
    sigma : float - Number of standard deviations for streak identification
    """
    print(f"Removing streaks with sigma = {sigma}")

    if sigma <= 0:
        return  # Skip if disabled

    data = im.data
    height, width = data.shape

    # Calculate streakiness for each row
    streakiness = np.zeros(height)

    for i in range(height):
        row = data[i, :]

        # Calculate horizontal variation (streakiness measure)
        diff = np.diff(row)
        streakiness[i] = np.std(diff)

    # Find mean and std of streakiness
    mean_streak = np.mean(streakiness)
    std_streak = np.std(streakiness)

    # Identify streaky rows
    threshold = mean_streak + sigma * std_streak
    streaky_rows = streakiness > threshold

    streak_count = np.sum(streaky_rows)
    print(f"Found {streak_count} streaky rows to correct")

    # Correct streaky rows by replacing with interpolated values
    for i in range(height):
        if streaky_rows[i]:
            # Find nearest non-streaky rows above and below
            above_row = None
            below_row = None

            # Search upward
            for j in range(i - 1, -1, -1):
                if not streaky_rows[j]:
                    above_row = j
                    break

            # Search downward
            for j in range(i + 1, height):
                if not streaky_rows[j]:
                    below_row = j
                    break

            # Interpolate replacement row
            if above_row is not None and below_row is not None:
                # Linear interpolation between above and below
                weight = (i - above_row) / (below_row - above_row)
                im.data[i, :] = (1 - weight) * data[above_row, :] + weight * data[below_row, :]
            elif above_row is not None:
                # Use row above
                im.data[i, :] = data[above_row, :]
            elif below_row is not None:
                # Use row below
                im.data[i, :] = data[below_row, :]


def Flatten(im, order=2):
    """
    Flatten image by removing polynomial background from each row
    Direct port from Igor Pro Flatten function

    Parameters:
    im : Wave - Input image to process
    order : int - Order of polynomial for fitting (0 = disable)
    """
    print(f"Flattening image with polynomial order = {order}")

    if order <= 0:
        return  # Skip if disabled

    data = im.data
    height, width = data.shape

    # Process each row
    for i in range(height):
        row = data[i, :]

        # Create x coordinates for fitting
        x = np.arange(width)

        try:
            # Fit polynomial to the row
            coeffs = np.polyfit(x, row, order)

            # Calculate polynomial background
            background = np.polyval(coeffs, x)

            # Subtract background from row
            im.data[i, :] = row - background

        except np.linalg.LinAlgError:
            # If fitting fails, skip this row
            continue

    print("Flattening complete")


def BatchPreprocess():
    """
    Batch preprocessing interface for multiple image files
    Matching Igor Pro BatchPreprocess functionality exactly
    """

    class PreprocessingGUI:
        def __init__(self):
            self.root = tk.Toplevel()
            self.root.title("Batch Image Preprocessing")
            self.root.geometry("800x600")
            self.root.transient()
            self.root.grab_set()

            self.input_files = []
            self.output_folder = ""

            self.setup_ui()

        def setup_ui(self):
            """Setup the preprocessing GUI - matching Igor Pro interface"""
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            ttk.Label(main_frame, text="Batch Image Preprocessing",
                      font=('TkDefaultFont', 14, 'bold')).pack(pady=(0, 20))

            # Input files section
            input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
            input_frame.pack(fill=tk.X, pady=(0, 10))

            buttons_frame = ttk.Frame(input_frame)
            buttons_frame.pack(fill=tk.X)

            ttk.Button(buttons_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
            ttk.Button(buttons_frame, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=5)
            ttk.Button(buttons_frame, text="Clear", command=self.clear_files).pack(side=tk.LEFT, padx=5)

            # File list
            list_frame = ttk.Frame(input_frame)
            list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

            self.file_listbox = tk.Listbox(list_frame, height=6)
            scrollbar_files = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
            self.file_listbox.configure(yscrollcommand=scrollbar_files.set)

            self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar_files.pack(side=tk.RIGHT, fill=tk.Y)

            # Preprocessing parameters - matching Igor Pro exactly
            params_frame = ttk.LabelFrame(main_frame, text="Preprocessing Parameters", padding="10")
            params_frame.pack(fill=tk.X, pady=(0, 10))

            # Streak removal parameters
            streak_frame = ttk.Frame(params_frame)
            streak_frame.pack(fill=tk.X, pady=5)

            ttk.Label(streak_frame, text="Std. Deviations for streak removal (0 = disable):").pack(side=tk.LEFT)
            self.streak_sdevs_var = tk.DoubleVar(value=3)  # Igor Pro default
            ttk.Entry(streak_frame, textvariable=self.streak_sdevs_var, width=10).pack(side=tk.LEFT, padx=5)

            # Flattening parameters
            flatten_frame = ttk.Frame(params_frame)
            flatten_frame.pack(fill=tk.X, pady=5)

            ttk.Label(flatten_frame, text="Polynomial order for flattening (0 = disable):").pack(side=tk.LEFT)
            self.flatten_order_var = tk.IntVar(value=2)  # Igor Pro default
            ttk.Entry(flatten_frame, textvariable=self.flatten_order_var, width=10).pack(side=tk.LEFT, padx=5)

            # Output folder section
            output_frame = ttk.LabelFrame(main_frame, text="Output Folder", padding="10")
            output_frame.pack(fill=tk.X, pady=(0, 10))

            output_buttons_frame = ttk.Frame(output_frame)
            output_buttons_frame.pack(fill=tk.X)

            ttk.Button(output_buttons_frame, text="Select Output Folder",
                       command=self.select_output_folder).pack(side=tk.LEFT, padx=5)

            self.output_label = ttk.Label(output_frame, text="No output folder selected")
            self.output_label.pack(anchor=tk.W, pady=(5, 0))

            # Progress and control
            progress_frame = ttk.Frame(main_frame)
            progress_frame.pack(fill=tk.X, pady=(0, 10))

            self.progress = ttk.Progressbar(progress_frame, mode='determinate')
            self.progress.pack(fill=tk.X, pady=(0, 5))

            self.status_label = ttk.Label(progress_frame, text="Ready")
            self.status_label.pack(anchor=tk.W)

            # Control buttons
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X)

            ttk.Button(control_frame, text="Start Preprocessing",
                       command=self.start_preprocessing).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Close",
                       command=self.root.destroy).pack(side=tk.RIGHT, padx=5)

        def add_files(self):
            """Add individual files"""
            files = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=[
                    ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.ibw"),
                    ("All files", "*.*")
                ]
            )

            for file in files:
                if file not in self.input_files:
                    self.input_files.append(file)
                    self.file_listbox.insert(tk.END, Path(file).name)

        def add_folder(self):
            """Add all images from a folder"""
            folder = filedialog.askdirectory(title="Select Folder Containing Images")

            if folder:
                extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.ibw']

                for ext in extensions:
                    for file_path in Path(folder).glob(f"*{ext}"):
                        file_str = str(file_path)
                        if file_str not in self.input_files:
                            self.input_files.append(file_str)
                            self.file_listbox.insert(tk.END, file_path.name)

                    # Also check uppercase extensions
                    for file_path in Path(folder).glob(f"*{ext.upper()}"):
                        file_str = str(file_path)
                        if file_str not in self.input_files:
                            self.input_files.append(file_str)
                            self.file_listbox.insert(tk.END, file_path.name)

        def clear_files(self):
            """Clear the file list"""
            self.input_files.clear()
            self.file_listbox.delete(0, tk.END)

        def select_output_folder(self):
            """Select output folder"""
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_folder = folder
                self.output_label.config(text=f"Output: {folder}")

        def start_preprocessing(self):
            """Start the preprocessing operation"""
            if not self.input_files:
                messagebox.showwarning("No Files", "Please select input files.")
                return

            if not self.output_folder:
                messagebox.showwarning("No Output", "Please select an output folder.")
                return

            try:
                total_files = len(self.input_files)
                self.progress['maximum'] = total_files

                streak_sdevs = self.streak_sdevs_var.get()
                flatten_order = self.flatten_order_var.get()

                print(f"Starting batch preprocessing of {total_files} files")
                print(f"Parameters: streak_sdevs={streak_sdevs}, flatten_order={flatten_order}")

                for i, file_path in enumerate(self.input_files):
                    self.status_label.config(text=f"Processing {Path(file_path).name}...")
                    self.root.update_idletasks()

                    try:
                        # Load image
                        im = LoadImage(file_path)
                        if im is None:
                            print(f"Failed to load {file_path}")
                            continue

                        # Apply preprocessing - matches Igor Pro BatchPreprocess exactly
                        if streak_sdevs > 0:
                            RemoveStreaks(im, sigma=streak_sdevs)

                        if flatten_order > 0:
                            Flatten(im, flatten_order)

                        # Save processed image
                        output_path = Path(self.output_folder) / f"preprocessed_{Path(file_path).name}"

                        # For now, save as numpy array (could extend to save in original format)
                        np.save(str(output_path).replace('.tif', '.npy').replace('.png', '.npy'), im.data)
                        print(f"Saved preprocessed image: {output_path}")

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

                    self.progress['value'] = i + 1
                    self.root.update_idletasks()

                self.status_label.config(text=f"Complete! Processed {total_files} files.")
                messagebox.showinfo("Complete", f"Preprocessing complete! Processed {total_files} files.")

            except Exception as e:
                messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
                self.status_label.config(text="Error occurred")

    # Create and show the GUI
    try:
        gui = PreprocessingGUI()
        return gui
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open preprocessing interface: {str(e)}")
        return None


def GroupPreprocess(images_dict):
    """
    Group preprocessing for loaded images
    Apply preprocessing to all loaded images in memory
    """
    if not images_dict:
        messagebox.showwarning("No Images", "No images loaded for preprocessing.")
        return

    # Get preprocessing parameters
    result = get_preprocessing_params()
    if result is None:
        return

    streak_sdevs, flatten_order = result

    try:
        total_images = len(images_dict)
        processed = 0

        print(f"Starting group preprocessing of {total_images} images")
        print(f"Parameters: streak_sdevs={streak_sdevs}, flatten_order={flatten_order}")

        for image_name, wave in images_dict.items():
            print(f"Preprocessing {image_name}...")

            # Apply preprocessing - matches Igor Pro exactly
            if streak_sdevs > 0:
                RemoveStreaks(wave, sigma=streak_sdevs)

            if flatten_order > 0:
                Flatten(wave, flatten_order)

            processed += 1

        messagebox.showinfo("Complete", f"Group preprocessing complete! Processed {processed} images.")
        print(f"Group preprocessing complete: {processed} images processed")

    except Exception as e:
        messagebox.showerror("Error", f"Group preprocessing failed: {str(e)}")
        print(f"Group preprocessing error: {str(e)}")


def get_preprocessing_params():
    """Get preprocessing parameters from user"""
    # Create parameter dialog
    root = tk.Tk()
    root.withdraw()

    dialog = tk.Toplevel()
    dialog.title("Preprocessing Parameters")
    dialog.geometry("600x300")
    dialog.transient()
    dialog.grab_set()
    dialog.focus_set()

    result = [None]

    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main_frame, text="Preprocessing Parameters",
              font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 20))

    # Parameters frame
    params_frame = ttk.Frame(main_frame)
    params_frame.pack(fill=tk.X, pady=10)

    # Streak removal
    ttk.Label(params_frame, text="Std. Deviations for streak removal (0 = disable):").grid(row=0, column=0, sticky=tk.W,
                                                                                           pady=5)
    streak_var = tk.DoubleVar(value=3)
    ttk.Entry(params_frame, textvariable=streak_var, width=10).grid(row=0, column=1, padx=10)

    # Flattening
    ttk.Label(params_frame, text="Polynomial order for flattening (0 = disable):").grid(row=1, column=0, sticky=tk.W,
                                                                                        pady=5)
    flatten_var = tk.IntVar(value=2)
    ttk.Entry(params_frame, textvariable=flatten_var, width=10).grid(row=1, column=1, padx=10)

    def ok_clicked():
        result[0] = (streak_var.get(), flatten_var.get())
        dialog.destroy()

    def cancel_clicked():
        result[0] = None
        dialog.destroy()

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side=tk.BOTTOM, pady=20)

    ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

    dialog.wait_window()
    root.destroy()

    return result[0]


def NoiseReduction(image, method='gaussian', sigma=1.0):
    """
    Apply noise reduction to image

    Parameters:
    image : Wave - Input image
    method : str - Noise reduction method ('gaussian', 'median')
    sigma : float - Parameter for method
    """
    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter
        image.data = gaussian_filter(image.data, sigma=sigma)
    elif method == 'median':
        from scipy.ndimage import median_filter
        kernel_size = int(2 * sigma + 1)
        image.data = median_filter(image.data, size=kernel_size)

    print(f"Applied {method} noise reduction with sigma={sigma}")


def ContrastEnhancement(image, method='clahe', clip_limit=0.01):
    """
    Enhance image contrast

    Parameters:
    image : Wave - Input image
    method : str - Enhancement method
    clip_limit : float - Clipping limit for CLAHE
    """
    try:
        from skimage import exposure

        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            image.data = exposure.equalize_adapthist(image.data, clip_limit=clip_limit)
        elif method == 'histogram':
            # Standard histogram equalization
            image.data = exposure.equalize_hist(image.data)
        elif method == 'stretch':
            # Linear contrast stretching
            p2, p98 = np.percentile(image.data, (2, 98))
            image.data = exposure.rescale_intensity(image.data, in_range=(p2, p98))

        print(f"Applied {method} contrast enhancement")

    except ImportError:
        # Fallback simple contrast stretching
        data_min = np.min(image.data)
        data_max = np.max(image.data)
        image.data = (image.data - data_min) / (data_max - data_min)
        print("Applied simple contrast stretching (skimage not available)")


def CreateMask(image, threshold_method='otsu', threshold_value=None):
    """
    Create binary mask from image

    Parameters:
    image : Wave - Input image
    threshold_method : str - Thresholding method
    threshold_value : float - Manual threshold value

    Returns:
    Wave - Binary mask
    """
    if threshold_value is not None:
        threshold = threshold_value
    elif threshold_method == 'otsu':
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(image.data)
        except ImportError:
            # Fallback
            threshold = np.mean(image.data) + np.std(image.data)
    elif threshold_method == 'mean':
        threshold = np.mean(image.data)
    else:
        threshold = 0.5 * (np.min(image.data) + np.max(image.data))

    mask_data = image.data > threshold
    mask = Wave(mask_data.astype(np.uint8), f"{image.name}_mask")

    print(f"Created mask using {threshold_method} threshold: {threshold:.6f}")
    return mask


def TestingPreprocessing(string_input, number_input):
    """Testing function for preprocessing module"""
    print(f"Preprocessing testing: {string_input}, {number_input}")
    return f"Preprocessed: {string_input}_{number_input}"


# Alias for Igor Pro compatibility
Testing = TestingPreprocessing