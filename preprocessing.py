"""
Preprocessing Module
Contains image preprocessing functions for blob detection
Direct port from Igor Pro code maintaining same variable names and structure
Complete implementation with all preprocessing functionality
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


def BatchPreprocess():
    """
    Batch preprocessing interface
    Allows user to apply various preprocessing operations to multiple images
    """

    class PreprocessingGUI:
        def __init__(self):
            self.root = tk.Toplevel()
            self.root.title("Batch Image Preprocessing")
            self.root.geometry("800x600")

            self.input_files = []
            self.output_folder = ""
            self.operations = []

            self.setup_ui()

        def setup_ui(self):
            """Setup the preprocessing GUI"""
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Input files section
            input_frame = ttk.LabelFrame(main_frame, text="Input Files", padding="10")
            input_frame.pack(fill=tk.X, pady=(0, 10))

            buttons_frame = ttk.Frame(input_frame)
            buttons_frame.pack(fill=tk.X)

            ttk.Button(buttons_frame, text="Add Files",
                       command=self.add_files).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(buttons_frame, text="Add Folder",
                       command=self.add_folder).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(buttons_frame, text="Clear List",
                       command=self.clear_files).pack(side=tk.LEFT, padx=(0, 5))

            # File list
            self.file_listbox = tk.Listbox(input_frame, height=6)
            self.file_listbox.pack(fill=tk.X, pady=(10, 0))

            # Output folder section
            output_frame = ttk.LabelFrame(main_frame, text="Output Folder", padding="10")
            output_frame.pack(fill=tk.X, pady=(0, 10))

            output_buttons_frame = ttk.Frame(output_frame)
            output_buttons_frame.pack(fill=tk.X)

            ttk.Button(output_buttons_frame, text="Select Output Folder",
                       command=self.select_output_folder).pack(side=tk.LEFT)

            self.output_label = ttk.Label(output_frame, text="No output folder selected")
            self.output_label.pack(anchor=tk.W, pady=(5, 0))

            # Preprocessing operations section
            ops_frame = ttk.LabelFrame(main_frame, text="Preprocessing Operations", padding="10")
            ops_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

            # Operations list
            ops_list_frame = ttk.Frame(ops_frame)
            ops_list_frame.pack(fill=tk.BOTH, expand=True)

            self.ops_listbox = tk.Listbox(ops_list_frame, height=8)
            self.ops_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

            ops_buttons_frame = ttk.Frame(ops_list_frame)
            ops_buttons_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

            ttk.Button(ops_buttons_frame, text="Add Gaussian Blur",
                       command=self.add_gaussian_blur).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Add Median Filter",
                       command=self.add_median_filter).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Add Background Subtract",
                       command=self.add_background_subtract).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Add Normalize",
                       command=self.add_normalize).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Add Crop",
                       command=self.add_crop).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Remove Selected",
                       command=self.remove_operation).pack(fill=tk.X, pady=2)
            ttk.Button(ops_buttons_frame, text="Clear All",
                       command=self.clear_operations).pack(fill=tk.X, pady=2)

            # Process button
            process_frame = ttk.Frame(main_frame)
            process_frame.pack(fill=tk.X)

            ttk.Button(process_frame, text="Start Processing",
                       command=self.start_processing).pack(side=tk.RIGHT)
            ttk.Button(process_frame, text="Cancel",
                       command=self.root.destroy).pack(side=tk.RIGHT, padx=(0, 10))

        def add_files(self):
            """Add individual files"""
            filetypes = [
                ("All Supported", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.ibw"),
                ("All files", "*.*")
            ]

            files = filedialog.askopenfilenames(
                title="Select Image Files",
                filetypes=filetypes
            )

            for file in files:
                if file not in self.input_files:
                    self.input_files.append(file)
                    self.file_listbox.insert(tk.END, Path(file).name)

        def add_folder(self):
            """Add all images from a folder"""
            folder = filedialog.askdirectory(title="Select Folder")
            if folder:
                extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.ibw']
                folder_path = Path(folder)

                for ext in extensions:
                    for file in folder_path.glob(f"*{ext}"):
                        file_str = str(file)
                        if file_str not in self.input_files:
                            self.input_files.append(file_str)
                            self.file_listbox.insert(tk.END, file.name)

        def clear_files(self):
            """Clear the file list"""
            self.input_files = []
            self.file_listbox.delete(0, tk.END)

        def select_output_folder(self):
            """Select output folder"""
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_folder = folder
                self.output_label.config(text=f"Output: {folder}")

        def add_gaussian_blur(self):
            """Add Gaussian blur operation"""
            sigma = tk.simpledialog.askfloat("Gaussian Blur", "Enter sigma value:", initialvalue=1.0)
            if sigma is not None:
                op = {"type": "gaussian_blur", "sigma": sigma}
                self.operations.append(op)
                self.ops_listbox.insert(tk.END, f"Gaussian Blur (σ={sigma})")

        def add_median_filter(self):
            """Add median filter operation"""
            size = tk.simpledialog.askinteger("Median Filter", "Enter filter size:", initialvalue=3)
            if size is not None:
                op = {"type": "median_filter", "size": size}
                self.operations.append(op)
                self.ops_listbox.insert(tk.END, f"Median Filter (size={size})")

        def add_background_subtract(self):
            """Add background subtraction operation"""
            method = tk.messagebox.askyesno("Background Subtract",
                                            "Use rolling ball? (No = polynomial)")
            if method is not None:
                if method:
                    radius = tk.simpledialog.askfloat("Rolling Ball", "Enter ball radius:", initialvalue=50.0)
                    if radius is not None:
                        op = {"type": "background_subtract", "method": "rolling_ball", "radius": radius}
                        self.operations.append(op)
                        self.ops_listbox.insert(tk.END, f"Background Subtract (rolling ball, r={radius})")
                else:
                    order = tk.simpledialog.askinteger("Polynomial", "Enter polynomial order:", initialvalue=2)
                    if order is not None:
                        op = {"type": "background_subtract", "method": "polynomial", "order": order}
                        self.operations.append(op)
                        self.ops_listbox.insert(tk.END, f"Background Subtract (polynomial, order={order})")

        def add_normalize(self):
            """Add normalization operation"""
            method = tk.messagebox.askyesnocancel("Normalize",
                                                  "Yes = Min-Max, No = Z-score, Cancel = abort")
            if method is not None:
                if method:
                    op = {"type": "normalize", "method": "minmax"}
                    self.operations.append(op)
                    self.ops_listbox.insert(tk.END, "Normalize (Min-Max)")
                else:
                    op = {"type": "normalize", "method": "zscore"}
                    self.operations.append(op)
                    self.ops_listbox.insert(tk.END, "Normalize (Z-score)")

        def add_crop(self):
            """Add crop operation"""
            # Simple crop dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Crop Parameters")
            dialog.geometry("300x200")

            result = {}

            ttk.Label(dialog, text="Crop Parameters (pixels):").pack(pady=10)

            ttk.Label(dialog, text="Left:").pack()
            left_var = tk.IntVar(value=0)
            ttk.Entry(dialog, textvariable=left_var, width=10).pack()

            ttk.Label(dialog, text="Top:").pack()
            top_var = tk.IntVar(value=0)
            ttk.Entry(dialog, textvariable=top_var, width=10).pack()

            ttk.Label(dialog, text="Right:").pack()
            right_var = tk.IntVar(value=0)
            ttk.Entry(dialog, textvariable=right_var, width=10).pack()

            ttk.Label(dialog, text="Bottom:").pack()
            bottom_var = tk.IntVar(value=0)
            ttk.Entry(dialog, textvariable=bottom_var, width=10).pack()

            def ok_clicked():
                result['ok'] = True
                result['left'] = left_var.get()
                result['top'] = top_var.get()
                result['right'] = right_var.get()
                result['bottom'] = bottom_var.get()
                dialog.destroy()

            def cancel_clicked():
                result['ok'] = False
                dialog.destroy()

            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=10)
            ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

            dialog.wait_window()

            if result.get('ok'):
                op = {"type": "crop",
                      "left": result['left'], "top": result['top'],
                      "right": result['right'], "bottom": result['bottom']}
                self.operations.append(op)
                self.ops_listbox.insert(tk.END,
                                        f"Crop (L:{result['left']}, T:{result['top']}, R:{result['right']}, B:{result['bottom']})")

        def remove_operation(self):
            """Remove selected operation"""
            selection = self.ops_listbox.curselection()
            if selection:
                index = selection[0]
                self.operations.pop(index)
                self.ops_listbox.delete(index)

        def clear_operations(self):
            """Clear all operations"""
            self.operations = []
            self.ops_listbox.delete(0, tk.END)

        def start_processing(self):
            """Start the preprocessing"""
            if not self.input_files:
                messagebox.showerror("Error", "No input files selected")
                return

            if not self.output_folder:
                messagebox.showerror("Error", "No output folder selected")
                return

            if not self.operations:
                messagebox.showerror("Error", "No preprocessing operations specified")
                return

            # Create progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing...")
            progress_window.geometry("400x150")

            ttk.Label(progress_window, text="Processing images...").pack(pady=10)

            progress = ttk.Progressbar(progress_window, length=300, mode='determinate')
            progress.pack(pady=10)
            progress['maximum'] = len(self.input_files)

            status_label = ttk.Label(progress_window, text="")
            status_label.pack(pady=5)

            # Process files
            for i, file_path in enumerate(self.input_files):
                try:
                    status_label.config(text=f"Processing {Path(file_path).name}")
                    progress_window.update()

                    # Load image
                    wave = LoadWave(file_path)
                    if wave is None:
                        print(f"Failed to load {file_path}")
                        continue

                    # Apply operations
                    processed_wave = self.apply_operations(wave)

                    # Save result
                    output_path = Path(self.output_folder) / f"processed_{Path(file_path).stem}.tif"
                    SaveWave(processed_wave, output_path, format="tiff")

                    progress['value'] = i + 1
                    progress_window.update()

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            progress_window.destroy()
            messagebox.showinfo("Complete", f"Processed {len(self.input_files)} images")
            self.root.destroy()

        def apply_operations(self, wave):
            """Apply all operations to a wave"""
            result = Duplicate(wave, f"{wave.name}_processed")

            for op in self.operations:
                if op["type"] == "gaussian_blur":
                    result = GaussianBlur(result, op["sigma"])
                elif op["type"] == "median_filter":
                    result = MedianFilter(result, op["size"])
                elif op["type"] == "background_subtract":
                    if op["method"] == "rolling_ball":
                        result = RollingBallBackground(result, op["radius"])
                    else:
                        result = PolynomialBackground(result, op["order"])
                elif op["type"] == "normalize":
                    if op["method"] == "minmax":
                        result = NormalizeMinMax(result)
                    else:
                        result = NormalizeZScore(result)
                elif op["type"] == "crop":
                    result = CropImage(result, op["left"], op["top"], op["right"], op["bottom"])

            return result

    # Launch the preprocessing GUI
    gui = PreprocessingGUI()
    gui.root.mainloop()


def GaussianBlur(wave, sigma):
    """
    Apply Gaussian blur to image
    """
    blurred = ndimage.gaussian_filter(wave.data, sigma)
    result = Wave(blurred, f"{wave.name}_blur")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def MedianFilter(wave, size):
    """
    Apply median filter to image
    """
    filtered = ndimage.median_filter(wave.data, size)
    result = Wave(filtered, f"{wave.name}_median")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def RollingBallBackground(wave, radius):
    """
    Subtract rolling ball background
    """
    from scipy import ndimage

    # Create structuring element (disk)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = x * x + y * y <= radius * radius

    # Rolling ball is erosion followed by dilation
    background = ndimage.grey_erosion(wave.data, structure=disk)
    background = ndimage.grey_dilation(background, structure=disk)

    # Subtract background
    corrected = wave.data - background
    result = Wave(corrected, f"{wave.name}_bgcorr")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def PolynomialBackground(wave, order):
    """
    Subtract polynomial background
    """
    height, width = wave.data.shape

    # Create coordinate meshgrid
    y, x = np.mgrid[0:height, 0:width]
    x = x.flatten()
    y = y.flatten()
    data = wave.data.flatten()

    # Build polynomial terms
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append((x ** i) * (y ** j))

    # Fit polynomial
    A = np.column_stack(terms)
    coeffs, residuals, rank, s = np.linalg.lstsq(A, data, rcond=None)

    # Compute background
    background = A @ coeffs
    background = background.reshape(height, width)

    # Subtract background
    corrected = wave.data - background
    result = Wave(corrected, f"{wave.name}_polybg")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def NormalizeMinMax(wave):
    """
    Normalize using min-max scaling
    """
    data = wave.data
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    if max_val > min_val:
        normalized = (data - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(data)

    result = Wave(normalized, f"{wave.name}_norm")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def NormalizeZScore(wave):
    """
    Normalize using z-score
    """
    data = wave.data
    mean_val = np.nanmean(data)
    std_val = np.nanstd(data)

    if std_val > 0:
        normalized = (data - mean_val) / std_val
    else:
        normalized = np.zeros_like(data)

    result = Wave(normalized, f"{wave.name}_zscore")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def CropImage(wave, left, top, right, bottom):
    """
    Crop image by specified margins
    """
    height, width = wave.data.shape

    # Calculate crop bounds
    left = max(0, left)
    top = max(0, top)
    right = max(0, right)
    bottom = max(0, bottom)

    # Ensure valid crop region
    if left + right >= width or top + bottom >= height:
        raise ValueError("Crop margins too large")

    # Crop data
    cropped = wave.data[top:height - bottom, left:width - right]
    result = Wave(cropped, f"{wave.name}_crop")

    # Update scaling
    x_scale = wave.GetScale('x')
    y_scale = wave.GetScale('y')

    new_x_offset = x_scale['offset'] + left * x_scale['delta']
    new_y_offset = y_scale['offset'] + top * y_scale['delta']

    result.SetScale('x', new_x_offset, x_scale['delta'], x_scale['units'])
    result.SetScale('y', new_y_offset, y_scale['delta'], y_scale['units'])

    return result


def FlatFieldCorrection(wave, flat_field_wave):
    """
    Apply flat field correction
    """
    if wave.data.shape != flat_field_wave.data.shape:
        raise ValueError("Image and flat field must have same dimensions")

    # Avoid division by zero
    flat_field = flat_field_wave.data.copy()
    flat_field[flat_field == 0] = 1

    corrected = wave.data / flat_field
    result = Wave(corrected, f"{wave.name}_flatcorr")

    # Copy scaling
    for axis in ['x', 'y']:
        scale_info = wave.GetScale(axis)
        result.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return result


def Testing(string_input, number_input):
    """Testing function for preprocessing module"""
    print(f"Preprocessing testing: {string_input}, {number_input}")
    return len(string_input) + number_input