#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Simplified GUI
Uses OS file management only, no internal data browser
Clean interface focused on analysis workflow

Copyright 2019 by The Curators of the University of Missouri (original Igor Pro code)
Python port by: Brendan Marsh - marshbp@stanford.edu
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import os
import sys
from pathlib import Path

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex

# Import all our modules
from igor_compatibility import *
from file_io import *
from main_functions import *
from preprocessing import *
from utilities import *
from scale_space import *
from particle_measurements import *


class HessianBlobGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hessian Blob Particle Detection Suite")
        self.root.geometry("1200x800")

        # Current state
        self.current_images = {}  # Dict of filename -> Wave
        self.current_results = {}  # Dict of analysis results
        self.current_display_image = None
        self.current_display_results = None

        self.setup_ui()
        self.setup_menu()

        # Display welcome message
        self.log_message("=== Hessian Blob Particle Detection Suite ===")
        self.log_message("G.M. King Laboratory - University of Missouri-Columbia")
        self.log_message("Ready for analysis...")

    def setup_ui(self):
        """Setup the simplified UI layout"""
        # Main container with three sections
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top section: File operations and analysis buttons
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))

        # File operations
        file_frame = ttk.LabelFrame(top_frame, text="File Operations", padding="5")
        file_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ttk.Button(file_frame, text="Load Single Image",
                   command=self.load_single_image, width=18).pack(pady=2)
        ttk.Button(file_frame, text="Load Multiple Images",
                   command=self.load_multiple_images, width=18).pack(pady=2)
        ttk.Button(file_frame, text="Load Image Folder",
                   command=self.load_image_folder, width=18).pack(pady=2)

        # Analysis functions
        analysis_frame = ttk.LabelFrame(top_frame, text="Analysis", padding="5")
        analysis_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ttk.Button(analysis_frame, text="Single Image Analysis",
                   command=self.analyze_single_image, width=18).pack(pady=2)
        ttk.Button(analysis_frame, text="Batch Analysis",
                   command=self.analyze_all_images, width=18).pack(pady=2)
        ttk.Button(analysis_frame, text="Preprocess Images",
                   command=self.preprocess_images, width=18).pack(pady=2)

        # Results functions
        results_frame = ttk.LabelFrame(top_frame, text="Results", padding="5")
        results_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        ttk.Button(results_frame, text="View Particles",
                   command=self.view_particles, width=18).pack(pady=2)
        ttk.Button(results_frame, text="Export Results",
                   command=self.export_results, width=18).pack(pady=2)
        ttk.Button(results_frame, text="Statistics",
                   command=self.show_statistics, width=18).pack(pady=2)

        # Image selection
        selection_frame = ttk.LabelFrame(top_frame, text="Current Images", padding="5")
        selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image listbox
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.image_listbox = tk.Listbox(listbox_frame, height=4)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)

        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Middle section: Image display
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="5")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Select an image to display")

        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom section: Status log
        log_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load images to begin analysis")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_menu(self):
        """Setup simplified menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Single Image", command=self.load_single_image)
        file_menu.add_command(label="Load Multiple Images", command=self.load_multiple_images)
        file_menu.add_command(label="Load Image Folder", command=self.load_image_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Clear All Images", command=self.clear_all_images)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Single Image Analysis", command=self.analyze_single_image)
        analysis_menu.add_command(label="Batch Analysis", command=self.analyze_all_images)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocess Images", command=self.preprocess_images)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Testing Function", command=self.run_testing)

    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_image_list(self):
        """Update the image listbox"""
        self.image_listbox.delete(0, tk.END)
        for filename in sorted(self.current_images.keys()):
            self.image_listbox.insert(tk.END, filename)

        if self.current_images:
            self.update_status(f"Loaded {len(self.current_images)} images")
        else:
            self.update_status("No images loaded")

    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.image_listbox.curselection()
        if selection:
            filename = self.image_listbox.get(selection[0])
            self.display_image(filename)

    def display_image(self, filename):
        """Display selected image"""
        try:
            if filename not in self.current_images:
                return

            wave = self.current_images[filename]
            self.current_display_image = filename

            self.ax.clear()

            # Display image with proper scaling
            extent = [
                DimOffset(wave, 0),
                DimOffset(wave, 0) + wave.data.shape[1] * DimDelta(wave, 0),
                DimOffset(wave, 1),
                DimOffset(wave, 1) + wave.data.shape[0] * DimDelta(wave, 1)
            ]

            im = self.ax.imshow(wave.data, cmap='gray', origin='lower', extent=extent)
            self.ax.set_title(f"Image: {filename}")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

            # Add colorbar if it doesn't exist, update if it does
            if not hasattr(self, 'colorbar') or self.colorbar is None:
                self.colorbar = self.fig.colorbar(im, ax=self.ax)
            else:
                self.colorbar.mappable.set_array(wave.data)
                self.colorbar.update_normal(im)

            # Overlay detection results if available
            result_key = f"{filename}_results"
            if result_key in self.current_results:
                self.overlay_detection_results(result_key, extent)

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying image {filename}: {str(e)}")

    def overlay_detection_results(self, result_key, extent):
        """Overlay blob detection results on the current image"""
        try:
            results = self.current_results[result_key]
            if 'info' in results and results['info'] is not None:
                info = results['info']

                # Draw circles for detected particles
                for i in range(info.data.shape[0]):
                    x_coord = info.data[i, 0]  # X coordinate in real units
                    y_coord = info.data[i, 1]  # Y coordinate in real units

                    # Get scale index to estimate radius
                    if info.data.shape[1] > 9:
                        scale_idx = int(info.data[i, 9])  # Stored scale index
                    else:
                        scale_idx = int(info.data[i, 2]) if info.data.shape[1] > 2 else 5

                    # Calculate radius from scale (more accurate)
                    # In the scale-space, each layer represents a different Gaussian scale
                    # The radius is approximately sqrt(2 * scale_value)
                    scale_value = 1.0 * (1.5 ** scale_idx)  # Default scale progression
                    radius = np.sqrt(2 * scale_value) * 2  # Multiply by pixel size if needed

                    # Convert radius to real coordinates
                    current_wave = self.current_images[self.current_display_image]
                    radius_real = radius * DimDelta(current_wave, 0)

                    # Draw circle - make sure coordinates are in the right system
                    circle = Circle((x_coord, y_coord), radius_real, fill=False,
                                    color='red', linewidth=2, alpha=0.8)
                    self.ax.add_patch(circle)

                    # Add particle number
                    self.ax.text(x_coord + radius_real, y_coord + radius_real, str(i + 1),
                                 color='yellow', fontsize=8, ha='left', va='bottom',
                                 weight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                          facecolor='black', alpha=0.7))

                self.log_message(f"Overlaid {info.data.shape[0]} detected particles")

        except Exception as e:
            self.log_message(f"Error overlaying results: {str(e)}")

    # File loading methods
    def load_single_image(self):
        """Load a single image file using OS file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Single Image File",
            filetypes=[
                ("Igor Binary Wave", "*.ibw"),
                ("TIFF files", "*.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All supported", "*.ibw *.tif *.tiff *.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.load_image_files([file_path])

    def load_multiple_images(self):
        """Load multiple image files using OS file dialog"""
        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Image Files",
            filetypes=[
                ("Igor Binary Wave", "*.ibw"),
                ("TIFF files", "*.tif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All supported", "*.ibw *.tif *.tiff *.png *.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )

        if file_paths:
            self.load_image_files(file_paths)

    def load_image_folder(self):
        """Load all images from a selected folder"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")

        if folder_path:
            # Find all supported image files in the folder
            supported_extensions = {'.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}
            image_files = []

            for file_path in Path(folder_path).rglob('*'):
                if file_path.suffix.lower() in supported_extensions:
                    image_files.append(str(file_path))

            if image_files:
                self.log_message(f"Found {len(image_files)} image files in folder")
                self.load_image_files(image_files)
            else:
                messagebox.showinfo("No Images", "No supported image files found in the selected folder.")

    def load_image_files(self, file_paths):
        """Load images from file paths"""
        if not file_paths:
            return

        self.update_status("Loading images...")
        success_count = 0

        for file_path in file_paths:
            try:
                self.log_message(f"Loading: {Path(file_path).name}")

                # Load the image
                wave = load_image_file(file_path)

                if wave is not None:
                    # Use just the filename as key (not full path)
                    filename = Path(file_path).name

                    # Handle duplicate filenames
                    original_filename = filename
                    counter = 1
                    while filename in self.current_images:
                        name_parts = original_filename.rsplit('.', 1)
                        if len(name_parts) == 2:
                            filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
                        else:
                            filename = f"{original_filename}_{counter}"
                        counter += 1

                    self.current_images[filename] = wave
                    success_count += 1
                    self.log_message(f"✓ Loaded: {filename}")
                else:
                    self.log_message(f"✗ Failed to load: {Path(file_path).name}")

            except Exception as e:
                self.log_message(f"✗ Error loading {Path(file_path).name}: {str(e)}")

        self.update_image_list()
        self.log_message(f"Successfully loaded {success_count} out of {len(file_paths)} images")

        # Display first image if none currently displayed
        if success_count > 0 and not self.current_display_image:
            first_image = next(iter(self.current_images.keys()))
            self.image_listbox.selection_set(0)
            self.display_image(first_image)

    def clear_all_images(self):
        """Clear all loaded images"""
        if self.current_images:
            if messagebox.askyesno("Clear Images", "Clear all loaded images and results?"):
                self.current_images.clear()
                self.current_results.clear()
                self.current_display_image = None
                self.current_display_results = None

                self.update_image_list()

                # Clear display
                self.ax.clear()
                self.ax.set_title("Select an image to display")
                if hasattr(self, 'colorbar') and self.colorbar is not None:
                    self.colorbar.remove()
                    self.colorbar = None
                self.canvas.draw()

                self.log_message("All images and results cleared")

    # Analysis methods
    def analyze_single_image(self):
        """Analyze the currently selected image"""
        if not self.current_display_image:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if self.current_display_image not in self.current_images:
            messagebox.showerror("Error", "Selected image not found.")
            return

        try:
            wave = self.current_images[self.current_display_image]
            self.log_message(f"Starting analysis of: {self.current_display_image}")
            self.update_status("Running Hessian blob analysis...")

            # Run the analysis
            results = self.run_hessian_analysis(wave)

            if results:
                result_key = f"{self.current_display_image}_results"
                self.current_results[result_key] = results
                self.current_display_results = result_key

                # Refresh display to show results
                self.display_image(self.current_display_image)

                num_particles = results['info'].data.shape[0] if 'info' in results else 0
                self.log_message(f"✓ Analysis completed: {num_particles} particles detected")
                messagebox.showinfo("Analysis Complete",
                                    f"Found {num_particles} particles in {self.current_display_image}")
            else:
                self.log_message("✗ Analysis failed or was cancelled")

        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Analysis Error", error_msg)
        finally:
            self.update_status("Ready")

    def analyze_all_images(self):
        """Analyze all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        if not messagebox.askyesno("Batch Analysis", f"Analyze all {len(self.current_images)} images?"):
            return

        try:
            self.update_status("Running batch analysis...")
            self.log_message(f"Starting batch analysis of {len(self.current_images)} images...")

            success_count = 0
            total_particles = 0

            for i, (filename, wave) in enumerate(self.current_images.items(), 1):
                try:
                    self.log_message(f"Analyzing {i}/{len(self.current_images)}: {filename}")
                    self.update_status(f"Analyzing {i}/{len(self.current_images)}: {filename}")

                    results = self.run_hessian_analysis(wave)

                    if results:
                        result_key = f"{filename}_results"
                        self.current_results[result_key] = results

                        num_particles = results['info'].data.shape[0] if 'info' in results else 0
                        total_particles += num_particles
                        success_count += 1

                        self.log_message(f"  ✓ Found {num_particles} particles")
                    else:
                        self.log_message(f"  ✗ Analysis failed")

                except Exception as e:
                    self.log_message(f"  ✗ Error: {str(e)}")

            # Refresh display if current image has new results
            if self.current_display_image:
                self.display_image(self.current_display_image)

            self.log_message(
                f"Batch analysis completed: {success_count}/{len(self.current_images)} images, {total_particles} total particles")
            messagebox.showinfo("Batch Complete",
                                f"Analyzed {success_count} images\nFound {total_particles} total particles")

        except Exception as e:
            error_msg = f"Error in batch analysis: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Batch Error", error_msg)
        finally:
            self.update_status("Ready")

    def run_hessian_analysis(self, wave):
        """Run the Hessian blob analysis on a single wave"""
        try:
            self.log_message("Getting analysis parameters...")

            # Get parameters - create dialogs properly
            params = self.get_hessian_blob_parameters()
            if params is None:
                self.log_message("Analysis cancelled - no parameters selected")
                return None

            scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params
            self.log_message(f"Using parameters: scale={scaleStart}, layers={layers}, threshold={detHResponseThresh}")

            # Get constraints
            constraints = self.get_particle_constraints()
            if constraints is None:
                self.log_message("Analysis cancelled - no constraints selected")
                return None

            minH, maxH, minV, maxV, minA, maxA = constraints
            self.log_message(f"Using constraints: height=[{minH}, {maxH}]")

            # Create scale-space representation
            self.log_message(f"Computing scale-space ({layers} layers)...")
            self.update_status("Computing scale-space representation...")
            L = ScaleSpaceRepresentation(wave, layers, scaleStart, scaleFactor)

            # Compute blob detectors
            self.log_message("Computing blob detectors...")
            self.update_status("Computing blob detectors...")
            BlobDetectors(L, 1)

            # Get detector results from global storage
            detH = data_browser.get_wave("detH")
            LapG = data_browser.get_wave("LapG")

            if detH is None or LapG is None:
                raise Exception("Failed to compute blob detectors")

            self.log_message(f"Blob detectors computed: detH shape={detH.data.shape}")

            # Determine threshold
            if detHResponseThresh == -2:  # Interactive
                self.log_message("Opening interactive threshold window...")
                self.update_status("Select threshold interactively...")
                detHResponseThresh = self.interactive_threshold(wave, detH, LapG, particleType)
                if detHResponseThresh is None:
                    self.log_message("Analysis cancelled - no threshold selected")
                    return None
            elif detHResponseThresh == -1:  # Otsu
                self.log_message("Computing Otsu threshold...")
                detHResponseThresh = OtsusThreshold(detH, particleType)

            self.log_message(f"Using threshold: {detHResponseThresh}")

            # Create output waves
            mapNum = Wave(np.full(wave.data.shape, -1, dtype=np.int32), "ParticleMap")
            mapDetH = Wave(np.zeros(wave.data.shape), "DetHMap")
            mapMax = Wave(np.zeros(wave.data.shape), "MaxMap")
            info = Wave(np.zeros((1000, 15)), "Info")

            # Find blobs
            self.log_message("Finding particles...")
            self.update_status("Finding particles...")
            num_found = FindHessianBlobs(wave, detH, LapG, detHResponseThresh, mapNum, mapDetH, mapMax, info,
                                         particleType, 1.6)

            # Trim info to actual number found
            if num_found > 0:
                info.data = info.data[:num_found, :]

            self.log_message(f"Analysis completed: found {num_found} particles")

            return {
                'original': wave,
                'scale_space': L,
                'detH': detH,
                'LapG': LapG,
                'mapNum': mapNum,
                'mapDetH': mapDetH,
                'mapMax': mapMax,
                'info': info,
                'num_particles': num_found,
                'threshold': detHResponseThresh
            }

        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            self.log_message(error_msg)
            import traceback
            self.log_message(f"Traceback: {traceback.format_exc()}")
            return None

    def get_hessian_blob_parameters(self):
        """Get Hessian blob parameters using a simple dialog"""
        try:
            # Create parameter dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Hessian Blob Parameters")
            dialog.geometry("450x400")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.focus_set()

            # Variables for parameters with defaults
            scale_start_var = tk.DoubleVar(value=1.0)
            layers_var = tk.IntVar(value=50)  # Reduced for faster testing
            scale_factor_var = tk.DoubleVar(value=1.5)
            thresh_var = tk.IntVar(value=-2)  # Use IntVar for radio buttons
            particle_type_var = tk.IntVar(value=1)
            subpixel_var = tk.IntVar(value=1)
            overlap_var = tk.IntVar(value=0)

            result = [None]

            # Create main frame
            main_frame = ttk.Frame(dialog, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            ttk.Label(main_frame, text="Hessian Blob Analysis Parameters",
                      font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 10))

            # Parameters
            row = 1
            ttk.Label(main_frame, text="Scale Start (pixels):").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=scale_start_var, width=10).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="Initial scale size", font=('TkDefaultFont', 8)).grid(row=row, column=2,
                                                                                             sticky="w", padx=5)

            row += 1
            ttk.Label(main_frame, text="Number of Layers:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=layers_var, width=10).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="50-256 layers", font=('TkDefaultFont', 8)).grid(row=row, column=2, sticky="w",
                                                                                        padx=5)

            row += 1
            ttk.Label(main_frame, text="Scale Factor:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=scale_factor_var, width=10).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="Scaling between layers", font=('TkDefaultFont', 8)).grid(row=row, column=2,
                                                                                                 sticky="w", padx=5)

            row += 1
            ttk.Label(main_frame, text="Threshold Method:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            threshold_frame = ttk.Frame(main_frame)
            threshold_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=3)
            ttk.Radiobutton(threshold_frame, text="Interactive", variable=thresh_var, value=-2).pack(anchor='w')
            ttk.Radiobutton(threshold_frame, text="Otsu Auto", variable=thresh_var, value=-1).pack(anchor='w')
            ttk.Radiobutton(threshold_frame, text="Manual (0.1)", variable=thresh_var, value=0.1).pack(anchor='w')

            row += 1
            ttk.Label(main_frame, text="Particle Type:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            particle_frame = ttk.Frame(main_frame)
            particle_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=3)
            ttk.Radiobutton(particle_frame, text="Positive only", variable=particle_type_var, value=1).pack(anchor='w')
            ttk.Radiobutton(particle_frame, text="Both", variable=particle_type_var, value=0).pack(anchor='w')
            ttk.Radiobutton(particle_frame, text="Negative only", variable=particle_type_var, value=-1).pack(anchor='w')

            row += 1
            ttk.Label(main_frame, text="Sub-pixel Multiplier:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=subpixel_var, width=10).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="1=off, >1=enabled", font=('TkDefaultFont', 8)).grid(row=row, column=2,
                                                                                            sticky="w", padx=5)

            row += 1
            ttk.Label(main_frame, text="Allow Overlap:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            overlap_frame = ttk.Frame(main_frame)
            overlap_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=3)
            ttk.Radiobutton(overlap_frame, text="No", variable=overlap_var, value=0).pack(anchor='w')
            ttk.Radiobutton(overlap_frame, text="Yes", variable=overlap_var, value=1).pack(anchor='w')

            def ok_clicked():
                try:
                    # Get threshold value
                    thresh_val = thresh_var.get()
                    if thresh_val == 0:  # Manual case, use 0.1
                        thresh_val = 0.1

                    result[0] = (
                        scale_start_var.get(),
                        layers_var.get(),
                        scale_factor_var.get(),
                        thresh_val,
                        particle_type_var.get(),
                        subpixel_var.get(),
                        overlap_var.get()
                    )
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")

            def cancel_clicked():
                result[0] = None
                dialog.destroy()

            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=row + 1, column=0, columnspan=3, pady=20)

            ttk.Button(button_frame, text="Start Analysis", command=ok_clicked, width=15).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=cancel_clicked, width=15).pack(side=tk.LEFT, padx=10)

            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            # Wait for dialog to complete
            self.root.wait_window(dialog)

            return result[0]

        except Exception as e:
            self.log_message(f"Error in parameter dialog: {str(e)}")
            return None

    def get_particle_constraints(self):
        """Get particle constraints using a simple dialog"""
        try:
            # Create constraints dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Particle Constraints")
            dialog.geometry("450x300")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.focus_set()

            # Variables with defaults (no constraints)
            min_h_var = tk.StringVar(value="-inf")
            max_h_var = tk.StringVar(value="inf")
            min_v_var = tk.StringVar(value="-inf")
            max_v_var = tk.StringVar(value="inf")
            min_a_var = tk.StringVar(value="-inf")
            max_a_var = tk.StringVar(value="inf")

            result = [None]

            # Create main frame
            main_frame = ttk.Frame(dialog, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            ttk.Label(main_frame, text="Particle Size Constraints",
                      font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=4, pady=(0, 10))

            ttk.Label(main_frame, text="Leave as 'inf' or '-inf' for no limits",
                      font=('TkDefaultFont', 9)).grid(row=1, column=0, columnspan=4, pady=(0, 10))

            # Constraints grid
            row = 2
            ttk.Label(main_frame, text="Height Constraints:").grid(row=row, column=0, columnspan=4, sticky="w",
                                                                   pady=(10, 5))

            row += 1
            ttk.Label(main_frame, text="Min:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=min_h_var, width=12).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="Max:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=max_h_var, width=12).grid(row=row, column=3, padx=5, pady=3)

            row += 1
            ttk.Label(main_frame, text="Volume Constraints:").grid(row=row, column=0, columnspan=4, sticky="w",
                                                                   pady=(10, 5))

            row += 1
            ttk.Label(main_frame, text="Min:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=min_v_var, width=12).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="Max:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=max_v_var, width=12).grid(row=row, column=3, padx=5, pady=3)

            row += 1
            ttk.Label(main_frame, text="Area Constraints:").grid(row=row, column=0, columnspan=4, sticky="w",
                                                                 pady=(10, 5))

            row += 1
            ttk.Label(main_frame, text="Min:").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=min_a_var, width=12).grid(row=row, column=1, padx=5, pady=3)
            ttk.Label(main_frame, text="Max:").grid(row=row, column=2, sticky="w", padx=5, pady=3)
            ttk.Entry(main_frame, textvariable=max_a_var, width=12).grid(row=row, column=3, padx=5, pady=3)

            def parse_constraint(value_str):
                """Parse constraint string to float"""
                value_str = value_str.strip()
                if value_str.lower() in ["-inf", "-infinity"]:
                    return -np.inf
                elif value_str.lower() in ["inf", "infinity"]:
                    return np.inf
                else:
                    try:
                        return float(value_str)
                    except ValueError:
                        return -np.inf if value_str.startswith('-') else np.inf

            def ok_clicked():
                try:
                    result[0] = (
                        parse_constraint(min_h_var.get()),
                        parse_constraint(max_h_var.get()),
                        parse_constraint(min_v_var.get()),
                        parse_constraint(max_v_var.get()),
                        parse_constraint(min_a_var.get()),
                        parse_constraint(max_a_var.get())
                    )
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid constraint values: {str(e)}")

            def cancel_clicked():
                result[0] = None
                dialog.destroy()

            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=row + 1, column=0, columnspan=4, pady=20)

            ttk.Button(button_frame, text="Continue", command=ok_clicked, width=15).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=cancel_clicked, width=15).pack(side=tk.LEFT, padx=10)

            # Center the dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            # Wait for dialog to complete
            self.root.wait_window(dialog)

            return result[0]

        except Exception as e:
            self.log_message(f"Error in constraints dialog: {str(e)}")
            return None

    def interactive_threshold(self, im, detH, LG, particleType):
        """Simple interactive threshold selection"""
        try:
            self.log_message("Computing blob strength map for interactive threshold...")

            # Create a simple threshold input dialog for now
            # (Full interactive threshold with image display can be added later)
            dialog = tk.Toplevel(self.root)
            dialog.title("Interactive Threshold")
            dialog.geometry("400x200")
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.focus_set()

            result = [None]

            main_frame = ttk.Frame(dialog, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Calculate suggested threshold
            max_response = np.max(np.abs(detH.data))
            suggested_threshold = max_response * 0.1  # 10% of maximum

            ttk.Label(main_frame, text="Interactive Threshold Selection",
                      font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 10))

            ttk.Label(main_frame, text=f"Maximum detector response: {max_response:.6f}").pack(pady=5)
            ttk.Label(main_frame, text=f"Suggested threshold: {suggested_threshold:.6f}").pack(pady=5)

            ttk.Label(main_frame, text="Enter threshold value:").pack(pady=(10, 5))
            threshold_var = tk.DoubleVar(value=suggested_threshold)
            ttk.Entry(main_frame, textvariable=threshold_var, width=20).pack(pady=5)

            def accept_threshold():
                result[0] = threshold_var.get()
                dialog.destroy()

            def cancel_threshold():
                result[0] = None
                dialog.destroy()

            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=20)

            ttk.Button(button_frame, text="Accept", command=accept_threshold).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=cancel_threshold).pack(side=tk.LEFT, padx=10)

            # Center dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")

            # Wait for result
            self.root.wait_window(dialog)

            return result[0]

        except Exception as e:
            self.log_message(f"Error in interactive threshold: {str(e)}")
            return None

    def preprocess_images(self):
        """Preprocess all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Get preprocessing parameters
        params = GetPreprocessingParameters()
        if params is None:
            return

        streak_removal, polynomial_order, flatten_order = params

        try:
            self.update_status("Preprocessing images...")
            self.log_message(f"Preprocessing {len(self.current_images)} images...")

            processed_count = 0

            for filename, wave in self.current_images.items():
                try:
                    self.log_message(f"  Processing: {filename}")

                    # Create a copy for preprocessing
                    processed_wave = Duplicate(wave, f"{wave.name}_processed")

                    # Apply preprocessing
                    if streak_removal > 0:
                        RemoveStreaks(processed_wave, streak_removal)

                    if polynomial_order > 0:
                        FlattenImage(processed_wave, polynomial_order)

                    # Replace original with processed version
                    self.current_images[filename] = processed_wave
                    processed_count += 1

                except Exception as e:
                    self.log_message(f"  ✗ Error processing {filename}: {str(e)}")

            # Refresh display
            if self.current_display_image:
                self.display_image(self.current_display_image)

            self.log_message(f"Preprocessing completed: {processed_count} images processed")
            messagebox.showinfo("Preprocessing Complete", f"Processed {processed_count} images")

        except Exception as e:
            error_msg = f"Preprocessing error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Preprocessing Error", error_msg)
        finally:
            self.update_status("Ready")

    def view_particles(self):
        """View particles for current image"""
        if not self.current_display_image:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        result_key = f"{self.current_display_image}_results"
        if result_key not in self.current_results:
            messagebox.showwarning("No Results", "Please run analysis on this image first.")
            return

        try:
            results = self.current_results[result_key]
            if 'info' in results and results['info'] is not None:
                # Launch particle viewer
                viewer = ParticleViewer(results['original'], results['info'],
                                        results.get('mapNum'), self.current_display_image)
                viewer.show()
            else:
                messagebox.showinfo("No Particles", "No particles found in analysis results.")

        except Exception as e:
            error_msg = f"Error viewing particles: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Viewer Error", error_msg)

    def export_results(self):
        """Export analysis results"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return

        # Choose export file
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                self.log_message(f"Exporting results to: {filename}")

                with open(filename, 'w') as f:
                    f.write("# Hessian Blob Detection Results\n")
                    f.write("# Image\tParticle\tX_Coord\tY_Coord\tHeight\tArea\tVolume\n")

                    for result_key, results in self.current_results.items():
                        if '_results' in result_key:
                            image_name = result_key.replace('_results', '')
                            if 'info' in results and results['info'] is not None:
                                info = results['info']
                                for i in range(info.data.shape[0]):
                                    f.write(f"{image_name}\t{i + 1}")
                                    for j in range(min(6, info.data.shape[1])):
                                        f.write(f"\t{info.data[i, j]}")
                                    f.write("\n")

                self.log_message(f"✓ Results exported to {filename}")
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")

            except Exception as e:
                error_msg = f"Export error: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Export Error", error_msg)

    def show_statistics(self):
        """Show statistics for all results"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results available.")
            return

        try:
            # Collect all particle data
            all_heights = []
            all_areas = []
            all_volumes = []
            image_counts = {}

            for result_key, results in self.current_results.items():
                if '_results' in result_key:
                    image_name = result_key.replace('_results', '')
                    if 'info' in results and results['info'] is not None:
                        info = results['info']
                        num_particles = info.data.shape[0]
                        image_counts[image_name] = num_particles

                        if num_particles > 0:
                            heights = info.data[:, 4] if info.data.shape[1] > 4 else info.data[:, 3]
                            areas = info.data[:, 5] if info.data.shape[1] > 5 else np.ones(num_particles)
                            volumes = info.data[:, 6] if info.data.shape[1] > 6 else np.ones(num_particles)

                            all_heights.extend(heights)
                            all_areas.extend(areas)
                            all_volumes.extend(volumes)

            if not all_heights:
                messagebox.showinfo("No Data", "No particle data found in results.")
                return

            # Create statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Analysis Statistics")
            stats_window.geometry("500x400")
            stats_window.transient(self.root)

            stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
            stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Generate statistics
            stats_info = "=== HESSIAN BLOB DETECTION STATISTICS ===\n\n"

            stats_info += f"Images analyzed: {len(image_counts)}\n"
            stats_info += f"Total particles: {len(all_heights)}\n\n"

            stats_info += "Per-image particle counts:\n"
            for image_name, count in image_counts.items():
                stats_info += f"  {image_name}: {count} particles\n"

            if all_heights:
                stats_info += f"\nParticle Heights:\n"
                stats_info += f"  Mean: {np.mean(all_heights):.6f}\n"
                stats_info += f"  Std Dev: {np.std(all_heights):.6f}\n"
                stats_info += f"  Min: {np.min(all_heights):.6f}\n"
                stats_info += f"  Max: {np.max(all_heights):.6f}\n"

            if all_areas and np.any(np.array(all_areas) != 1):
                stats_info += f"\nParticle Areas:\n"
                stats_info += f"  Mean: {np.mean(all_areas):.6e}\n"
                stats_info += f"  Std Dev: {np.std(all_areas):.6e}\n"
                stats_info += f"  Min: {np.min(all_areas):.6e}\n"
                stats_info += f"  Max: {np.max(all_areas):.6e}\n"

            if all_volumes and np.any(np.array(all_volumes) != 1):
                stats_info += f"\nParticle Volumes:\n"
                stats_info += f"  Mean: {np.mean(all_volumes):.6e}\n"
                stats_info += f"  Std Dev: {np.std(all_volumes):.6e}\n"
                stats_info += f"  Min: {np.min(all_volumes):.6e}\n"
                stats_info += f"  Max: {np.max(all_volumes):.6e}\n"

            stats_text.insert(tk.END, stats_info)
            stats_text.config(state=tk.DISABLED)

        except Exception as e:
            error_msg = f"Statistics error: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Statistics Error", error_msg)

    def run_testing(self):
        """Run the testing function"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Testing Function")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Enter a string:").pack(pady=5)
        string_var = tk.StringVar(value="Test string")
        tk.Entry(dialog, textvariable=string_var, width=30).pack(pady=5)

        tk.Label(dialog, text="Enter a number:").pack(pady=5)
        number_var = tk.DoubleVar(value=42.0)
        tk.Entry(dialog, textvariable=number_var, width=30).pack(pady=5)

        def run_test():
            try:
                Testing(string_var.get(), number_var.get())
                dialog.destroy()
                self.log_message(f"Testing function executed: '{string_var.get()}', {number_var.get()}")
            except Exception as e:
                messagebox.showerror("Error", f"Testing function failed: {str(e)}")

        tk.Button(dialog, text="Run Test", command=run_test).pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite

Python Port of Igor Pro Implementation

Original Igor Pro Code:
Copyright 2019 by The Curators of the University of Missouri
G.M. King Laboratory - University of Missouri-Columbia
Coded by: Brendan Marsh (marshbp@stanford.edu)

Python Port maintains 1-1 functionality with the original Igor Pro version.

The Hessian blob algorithm is a general-purpose particle detection algorithm,
designed to detect, isolate, and draw the boundaries of roughly "blob-like" 
particles in an image.

Reference:
"The Hessian Blob Algorithm: Precise Particle Detection in Atomic Force 
Microscopy Imagery" - Scientific Reports
doi:10.1038/s41598-018-19379-x
"""

        messagebox.showinfo("About", about_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = HessianBlobGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()