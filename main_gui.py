#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main GUI
Complete 1-1 port from Igor Pro implementation
Fixed version with proper interactive threshold functionality

Copyright 2019 by The Curators of the University of Missouri (original Igor Pro code)
Python port maintains 1-1 functionality with Igor Pro version
G.M. King Laboratory - University of Missouri-Columbia
Original coded by: Brendan Marsh - marshbp@stanford.edu
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
        self.log_message("Python Port of Igor Pro Implementation")
        self.log_message("Original Igor Pro code by: Brendan Marsh - marshbp@stanford.edu")
        self.log_message("=" * 60)
        self.log_message("")
        self.log_message("Instructions:")
        self.log_message("1. Load image(s) using File menu")
        self.log_message("2. Run Hessian Blob Detection")
        self.log_message("3. Use interactive threshold to select blob strength")
        self.log_message("4. View results and statistics")
        self.log_message("")

    def setup_ui(self):
        """Setup the main user interface"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls and file list
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=0)

        # Right panel for image display
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # === LEFT PANEL ===

        # File management section
        file_frame = ttk.LabelFrame(left_frame, text="File Management", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Single Image",
                   command=self.load_single_image, width=20).pack(pady=2)
        ttk.Button(file_frame, text="Load Multiple Images",
                   command=self.load_multiple_images, width=20).pack(pady=2)
        ttk.Button(file_frame, text="Load Folder",
                   command=self.load_folder, width=20).pack(pady=2)

        # Loaded images list
        list_frame = ttk.LabelFrame(left_frame, text="Loaded Images", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create listbox with scrollbar
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.images_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.images_listbox.yview)
        self.images_listbox.configure(yscrollcommand=scrollbar.set)

        self.images_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Analysis controls section
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis Controls", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Hessian Blob Detection",
                   command=self.run_hessian_blob_detection, width=25).pack(pady=2)
        ttk.Button(analysis_frame, text="Batch Process All Images",
                   command=self.batch_process_images, width=25).pack(pady=2)

        # View controls section
        view_frame = ttk.LabelFrame(left_frame, text="View Controls", padding="10")
        view_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(view_frame, text="Show Original Image",
                   command=self.show_original_image, width=20).pack(pady=2)
        ttk.Button(view_frame, text="Show Results Overlay",
                   command=self.show_results_overlay, width=20).pack(pady=2)
        ttk.Button(view_frame, text="Show Statistics",
                   command=self.show_statistics, width=20).pack(pady=2)

        # === RIGHT PANEL ===

        # Image display area
        display_frame = ttk.LabelFrame(right_frame, text="Image Display", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No image loaded")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Log area
        log_frame = ttk.LabelFrame(right_frame, text="Log Output", padding="10")
        log_frame.pack(fill=tk.X)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Single Image", command=self.load_single_image)
        file_menu.add_command(label="Load Multiple Images", command=self.load_multiple_images)
        file_menu.add_command(label="Load Folder", command=self.load_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Hessian Blob Detection", command=self.run_hessian_blob_detection)
        analysis_menu.add_command(label="Batch Process All", command=self.batch_process_images)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocess Images", command=self.preprocess_images)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Original Image", command=self.show_original_image)
        view_menu.add_command(label="Results Overlay", command=self.show_results_overlay)
        view_menu.add_command(label="Statistics", command=self.show_statistics)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Testing Function", command=self.run_testing)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def log_message(self, message):
        """Add a message to the log output"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.images_listbox.curselection()
        if selection:
            filename = self.images_listbox.get(selection[0])
            self.current_display_image = filename
            self.show_original_image()

    # === FILE LOADING METHODS ===

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

    def load_folder(self):
        """Load all supported images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            # Find all supported image files in the folder
            supported_extensions = ['.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
            image_files = []

            for ext in supported_extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))
                image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

            if image_files:
                self.load_image_files([str(f) for f in image_files])
            else:
                messagebox.showwarning("No Images", "No supported image files found in the selected folder.")

    def load_image_files(self, file_paths):
        """Load image files into the application"""
        loaded_count = 0

        for file_path in file_paths:
            try:
                # Load the image
                wave = load_image(file_path)
                if wave is not None:
                    filename = os.path.basename(file_path)

                    # Store the image
                    self.current_images[filename] = wave

                    # Add to listbox
                    self.images_listbox.insert(tk.END, filename)

                    loaded_count += 1
                    self.log_message(f"Loaded: {filename}")

                else:
                    self.log_message(f"Failed to load: {os.path.basename(file_path)}")

            except Exception as e:
                self.log_message(f"Error loading {os.path.basename(file_path)}: {str(e)}")

        if loaded_count > 0:
            self.log_message(f"Successfully loaded {loaded_count} image(s)")

            # Select and display the first loaded image
            if self.current_display_image is None and self.current_images:
                first_image = list(self.current_images.keys())[0]
                self.current_display_image = first_image
                self.images_listbox.selection_set(0)
                self.show_original_image()
        else:
            messagebox.showerror("Load Error", "No images could be loaded.")

    # === ANALYSIS METHODS ===

    def run_hessian_blob_detection(self):
        """Run Hessian blob detection on selected image"""
        if not self.current_display_image:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            self.log_message("Starting Hessian blob detection...")

            # Get the current image
            im = self.current_images[self.current_display_image]

            # Get parameters from user
            params = GetBlobDetectionParams()
            if params is None:
                self.log_message("Blob detection cancelled by user.")
                return

            self.log_message(
                f"Parameters: Scale={params['scaleStart']}, Layers={params['layers']}, Factor={params['scaleFactor']}")

            # Compute scale-space representation
            self.log_message("Computing scale-space representation...")
            L = ScaleSpaceRepresentation(im, params['layers'], params['scaleStart'], params['scaleFactor'])

            # Compute blob detectors
            self.log_message("Computing blob detectors...")
            detH, LG = BlobDetectors(L, True)  # gammaNorm=True

            # Interactive threshold selection - this is the key fix!
            self.log_message("Opening interactive threshold selection...")
            threshold = InteractiveThreshold(im, detH, LG, params['particleType'], params['maxCurvatureRatio'])

            if threshold is None:
                self.log_message("Threshold selection cancelled.")
                return

            self.log_message(f"Selected threshold: {threshold:.6f}")

            # Initialize output waves
            mapNum = Wave(np.zeros(im.data.shape, dtype=np.int32), "ParticleMap")
            mapDetH = Wave(np.zeros(im.data.shape), "DetHMap")
            mapMax = Wave(np.zeros(im.data.shape), "MaxMap")
            info = Wave(np.zeros((0, 10)), "ParticleInfo")

            # Find blobs
            self.log_message("Finding Hessian blobs...")
            num_particles = FindHessianBlobs(im, detH, LG, threshold, mapNum, mapDetH, mapMax, info,
                                             params['particleType'], params['maxCurvatureRatio'])

            # Store results
            results = {
                'num_particles': num_particles,
                'threshold': threshold,
                'parameters': params,
                'mapNum': mapNum,
                'mapDetH': mapDetH,
                'mapMax': mapMax,
                'info': info,
                'detH': detH,
                'LG': LG,
                'L': L
            }

            self.current_results[self.current_display_image] = results
            self.current_display_results = self.current_display_image

            self.log_message(f"Detection completed! Found {num_particles} particles.")

            # Automatically show results overlay
            self.show_results_overlay()

        except Exception as e:
            error_msg = f"Error during blob detection: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Detection Error", error_msg)

    def batch_process_images(self):
        """Run blob detection on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Get parameters once for all images
        params = GetBlobDetectionParams()
        if params is None:
            return

        total_images = len(self.current_images)
        processed = 0

        for filename, im in self.current_images.items():
            try:
                self.log_message(f"Processing {filename} ({processed + 1}/{total_images})...")

                # Compute scale-space representation
                L = ScaleSpaceRepresentation(im, params['layers'], params['scaleStart'], params['scaleFactor'])

                # Compute blob detectors
                detH, LG = BlobDetectors(L, True)

                # Use interactive threshold for first image, then apply to all
                if processed == 0:
                    threshold = InteractiveThreshold(im, detH, LG, params['particleType'], params['maxCurvatureRatio'])
                    if threshold is None:
                        self.log_message("Batch processing cancelled.")
                        return
                    batch_threshold = threshold
                else:
                    threshold = batch_threshold

                # Initialize output waves
                mapNum = Wave(np.zeros(im.data.shape, dtype=np.int32), "ParticleMap")
                mapDetH = Wave(np.zeros(im.data.shape), "DetHMap")
                mapMax = Wave(np.zeros(im.data.shape), "MaxMap")
                info = Wave(np.zeros((0, 10)), "ParticleInfo")

                # Find blobs
                num_particles = FindHessianBlobs(im, detH, LG, threshold, mapNum, mapDetH, mapMax, info,
                                                 params['particleType'], params['maxCurvatureRatio'])

                # Store results
                results = {
                    'num_particles': num_particles,
                    'threshold': threshold,
                    'parameters': params,
                    'mapNum': mapNum,
                    'mapDetH': mapDetH,
                    'mapMax': mapMax,
                    'info': info,
                    'detH': detH,
                    'LG': LG,
                    'L': L
                }

                self.current_results[filename] = results
                processed += 1

                self.log_message(f"  Found {num_particles} particles in {filename}")

            except Exception as e:
                self.log_message(f"Error processing {filename}: {str(e)}")
                continue

        self.log_message(f"Batch processing completed! Processed {processed}/{total_images} images.")

        # Show results for current image if available
        if self.current_display_image in self.current_results:
            self.current_display_results = self.current_display_image
            self.show_results_overlay()

    # === DISPLAY METHODS ===

    def show_original_image(self):
        """Display the original image"""
        if not self.current_display_image or self.current_display_image not in self.current_images:
            return

        try:
            im = self.current_images[self.current_display_image]

            self.ax.clear()
            self.ax.imshow(im.data, cmap='gray', origin='lower',
                           extent=[DimOffset(im, 0),
                                   DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
                                   DimOffset(im, 1),
                                   DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)])

            self.ax.set_title(f"Original Image: {self.current_display_image}")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")

    def show_results_overlay(self):
        """Display image with detected particles overlaid"""
        if (not self.current_display_image or
                self.current_display_image not in self.current_images or
                self.current_display_image not in self.current_results):
            messagebox.showwarning("No Results", "No detection results available for current image.")
            return

        try:
            im = self.current_images[self.current_display_image]
            results = self.current_results[self.current_display_image]
            info = results['info']

            self.ax.clear()
            self.ax.imshow(im.data, cmap='gray', origin='lower',
                           extent=[DimOffset(im, 0),
                                   DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
                                   DimOffset(im, 1),
                                   DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)])

            # Draw circles for detected particles
            if info.data.shape[0] > 0:
                for i in range(info.data.shape[0]):
                    x_coord = info.data[i, 0]
                    y_coord = info.data[i, 1]

                    # Calculate radius from scale
                    if info.data.shape[1] > 3:
                        scale_value = info.data[i, 3]
                    else:
                        scale_idx = int(info.data[i, 2]) if info.data.shape[1] > 2 else 5
                        scale_value = 1.0 * (1.5 ** scale_idx)

                    radius = np.sqrt(2 * scale_value)

                    # Draw circle
                    circle = Circle((x_coord, y_coord), radius, fill=False,
                                    color='red', linewidth=2, alpha=0.8)
                    self.ax.add_patch(circle)

                    # Add particle number
                    self.ax.text(x_coord + radius, y_coord + radius, str(i + 1),
                                 color='yellow', fontsize=8, ha='left', va='bottom',
                                 weight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                          facecolor='black', alpha=0.7))

                self.log_message(f"Displayed {info.data.shape[0]} detected particles")

            self.ax.set_title(f"Detected Particles: {self.current_display_image} ({results['num_particles']} found)")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying results: {str(e)}")

    def show_statistics(self):
        """Show detection statistics"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No detection results available.")
            return

        try:
            # Create statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Detection Statistics")
            stats_window.geometry("600x500")
            stats_window.transient(self.root)

            # Create text widget with scrollbar
            frame = ttk.Frame(stats_window)
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            stats_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD)
            stats_text.pack(fill=tk.BOTH, expand=True)

            # Generate statistics
            stats_info = "=== HESSIAN BLOB DETECTION STATISTICS ===\n\n"

            total_particles = 0
            all_strengths = []
            all_areas = []
            all_volumes = []

            for filename, results in self.current_results.items():
                info = results['info']
                num_particles = results['num_particles']
                threshold = results['threshold']

                stats_info += f"Image: {filename}\n"
                stats_info += f"  Particles Found: {num_particles}\n"
                stats_info += f"  Threshold Used: {threshold:.6f}\n"

                if info.data.shape[0] > 0:
                    strengths = info.data[:, 4] if info.data.shape[1] > 4 else []
                    areas = info.data[:, 7] if info.data.shape[1] > 7 else []
                    volumes = info.data[:, 8] if info.data.shape[1] > 8 else []

                    if len(strengths) > 0:
                        stats_info += f"  Blob Strength - Mean: {np.mean(strengths):.6f}, Std: {np.std(strengths):.6f}\n"
                        all_strengths.extend(strengths)

                    if len(areas) > 0 and np.any(np.array(areas) != 1):
                        stats_info += f"  Area - Mean: {np.mean(areas):.6e}, Std: {np.std(areas):.6e}\n"
                        all_areas.extend(areas)

                    if len(volumes) > 0 and np.any(np.array(volumes) != 1):
                        stats_info += f"  Volume - Mean: {np.mean(volumes):.6e}, Std: {np.std(volumes):.6e}\n"
                        all_volumes.extend(volumes)

                stats_info += "\n"
                total_particles += num_particles

            # Overall statistics
            stats_info += f"=== OVERALL STATISTICS ===\n"
            stats_info += f"Total Images Processed: {len(self.current_results)}\n"
            stats_info += f"Total Particles Found: {total_particles}\n"
            stats_info += f"Average Particles per Image: {total_particles / len(self.current_results):.2f}\n\n"

            if all_strengths:
                stats_info += f"All Blob Strengths:\n"
                stats_info += f"  Mean: {np.mean(all_strengths):.6f}\n"
                stats_info += f"  Std Dev: {np.std(all_strengths):.6f}\n"
                stats_info += f"  Min: {np.min(all_strengths):.6f}\n"
                stats_info += f"  Max: {np.max(all_strengths):.6f}\n"

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

    # === UTILITY METHODS ===

    def preprocess_images(self):
        """Preprocess all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        try:
            # Run batch preprocessing
            BatchPreprocess()
            self.log_message("Preprocessing completed.")
        except Exception as e:
            self.log_message(f"Preprocessing error: {str(e)}")

    def save_results(self):
        """Save detection results to file"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Hessian Blob Detection Results\n")
                    f.write("=" * 40 + "\n\n")

                    for filename, results in self.current_results.items():
                        info = results['info']
                        f.write(f"Image: {filename}\n")
                        f.write(f"Particles: {results['num_particles']}\n")
                        f.write(f"Threshold: {results['threshold']:.6f}\n")

                        if info.data.shape[0] > 0:
                            f.write("Particle Data:\n")
                            f.write("ID\tX\tY\tScale\tStrength\tMaxValue\n")
                            for i in range(info.data.shape[0]):
                                x, y = info.data[i, 0], info.data[i, 1]
                                scale = info.data[i, 2] if info.data.shape[1] > 2 else 0
                                strength = info.data[i, 4] if info.data.shape[1] > 4 else 0
                                max_val = info.data[i, 5] if info.data.shape[1] > 5 else 0
                                f.write(f"{i + 1}\t{x:.4f}\t{y:.4f}\t{scale:.2f}\t{strength:.6f}\t{max_val:.4f}\n")

                        f.write("\n")

                self.log_message(f"Results saved to: {file_path}")

            except Exception as e:
                self.log_message(f"Error saving results: {str(e)}")

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