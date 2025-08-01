#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main GUI
Complete 1-1 port from Igor Pro implementation

// Copyright 2019 by The Curators of the University of Missouri, a public corporation //
//																					   //
// Hessian Blob Particle Detection Suite - Python Port  //
//                                                       //
// G.M. King Laboratory                                  //
// University of Missouri-Columbia	                     //
// Original Igor Pro coded by: Brendan Marsh             //
// Email: marshbp@stanford.edu		                     //
// Python port maintains 1-1 functionality              //
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
        self.current_results = {}  # Dict of image_name -> results (most recent analysis)
        self.current_display_image = None
        self.current_display_results = None
        self.figure = None
        self.canvas = None
        self.ax = None
        self.show_blobs = False

        # FIXED: Initialize color table variable
        self.color_table_var = None

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
        self.log_message("4. Toggle 'Show Blobs' to view detected particles")
        self.log_message("")
        self.log_message("Ready for analysis...")

    def setup_ui(self):
        """Setup the main user interface - based on Igor Pro panel layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Igor Pro: File management section with proper button names
        file_frame = ttk.LabelFrame(left_panel, text="File Management", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Image",
                   command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Folder",
                   command=self.load_folder).pack(fill=tk.X, pady=2)

        # Image list
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.image_listbox = tk.Listbox(list_frame, height=6)
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_listbox.config(yscrollcommand=scrollbar.set)

        # Igor Pro: Preprocessing section (single and batch, not batch and group)
        preprocess_frame = ttk.LabelFrame(left_panel, text="Preprocessing", padding="5")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(preprocess_frame, text="Single Preprocess",
                   command=self.single_preprocess).pack(fill=tk.X, pady=2)
        ttk.Button(preprocess_frame, text="Batch Preprocess",
                   command=self.batch_preprocess).pack(fill=tk.X, pady=2)

        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis", padding="5")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Single Analysis",
                   command=self.run_single_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Run Batch Analysis",
                   command=self.run_batch_analysis).pack(fill=tk.X, pady=2)

        # Display controls
        display_frame = ttk.LabelFrame(left_panel, text="Display", padding="5")
        display_frame.pack(fill=tk.X, pady=(0, 10))

        # Igor Pro: Add color table selection like Igor Pro
        ttk.Label(display_frame, text="Color Table:").pack(anchor=tk.W)
        self.color_table_var = tk.StringVar(value="gray")
        color_combo = ttk.Combobox(display_frame, textvariable=self.color_table_var,
                                   values=["gray", "rainbow", "hot", "cool", "viridis", "plasma"],
                                   width=18, state="readonly")
        color_combo.pack(anchor=tk.W, pady=(2, 5))
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.display_image())

        self.blob_toggle_var = tk.BooleanVar()
        self.blob_toggle = ttk.Checkbutton(display_frame, text="Show Blob Regions",
                                           variable=self.blob_toggle_var,
                                           command=self.toggle_blob_display,
                                           state=tk.DISABLED)
        self.blob_toggle.pack(anchor=tk.W, pady=2)

        # Igor Pro: View particles button (matching Igor Pro)
        self.view_particles_button = ttk.Button(display_frame, text="View Particles",
                                                command=self.view_particles,
                                                state=tk.DISABLED)
        self.view_particles_button.pack(fill=tk.X, pady=2)
        
        # Igor Pro: Plot histogram button (matching Igor Pro)
        self.plot_histogram_button = ttk.Button(display_frame, text="Plot Histogram",
                                               command=self.plot_histogram,
                                               state=tk.DISABLED)
        self.plot_histogram_button.pack(fill=tk.X, pady=2)

        # Results info
        info_frame = ttk.LabelFrame(left_panel, text="Results Info", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=30, font=("Courier", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        toolbar.update()

        # Bottom panel for logging
        log_frame = ttk.LabelFrame(self.root, text="Log", padding="5")
        log_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, font=("Courier", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_menu(self):
        """Setup the application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image", command=self.load_image)
        file_menu.add_command(label="Load Folder", command=self.load_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Hessian Blob Detection", command=self.run_single_analysis)
        analysis_menu.add_command(label="Batch Analysis", command=self.run_batch_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Single Preprocess", command=self.single_preprocess)
        analysis_menu.add_command(label="Batch Preprocess", command=self.batch_preprocess)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Blob Display", command=self.toggle_blob_display)
        view_menu.add_command(label="View Particles", command=self.view_particles)
        view_menu.add_command(label="Zoom Fit", command=self.zoom_fit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def log_message(self, message):
        """Add a message to the log"""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def load_image(self):
        """FIXED: Load single or multiple image files (renamed from load_single_image)"""
        filetypes = [
            ("All supported", "*.ibw *.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy"),
            ("Igor Binary Wave", "*.ibw"),
            ("Preprocessed NumPy", "*.npy"),
            ("TIFF files", "*.tif *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]

        file_paths = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=filetypes
        )

        if file_paths:
            for file_path in file_paths:
                try:
                    # Load the image
                    wave = LoadWave(file_path)
                    if wave is not None:
                        # Store in current images
                        filename = os.path.basename(file_path)
                        self.current_images[filename] = wave
                        self.log_message(f"Loaded: {filename}")

                except Exception as e:
                    self.log_message(f"Error loading {file_path}: {str(e)}")
                    messagebox.showerror("Load Error", f"Failed to load {file_path}:\n{str(e)}")

            # Update the image list and display the first loaded image
            self.update_image_list()
            if self.current_images and not self.current_display_image:
                first_image = next(iter(self.current_images.values()))
                self.current_display_image = first_image
                self.display_image()

    def load_folder(self):
        """Load all supported images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")

        if folder_path:
            supported_extensions = ['.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.npy']
            loaded_count = 0

            try:
                for file_path in Path(folder_path).iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        try:
                            wave = LoadWave(str(file_path))
                            if wave is not None:
                                filename = file_path.name
                                self.current_images[filename] = wave
                                loaded_count += 1
                        except Exception as e:
                            self.log_message(f"Error loading {file_path}: {str(e)}")

                self.log_message(f"Loaded {loaded_count} images from folder")
                self.update_image_list()

                # Display the first image if none is currently displayed
                if self.current_images and not self.current_display_image:
                    first_image = next(iter(self.current_images.values()))
                    self.current_display_image = first_image
                    self.display_image()

            except Exception as e:
                messagebox.showerror("Folder Load Error", f"Failed to load folder:\n{str(e)}")

    # FIXED: Preprocessing methods (matching Igor Pro)
    def single_preprocess(self):
        """Run single preprocessing on current image"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            # Get preprocessing parameters and output folder
            result = self.get_single_preprocess_params()
            if result is None:
                return

            streak_sdevs, flatten_order, output_folder = result

            if not output_folder:
                self.log_message("No output folder selected")
                return

            # Create a copy for preprocessing
            original_name = self.current_display_image.name
            base_name = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
            preprocessed_name = f"{base_name}_preprocessed"

            preprocessed_image = Duplicate(self.current_display_image, preprocessed_name)

            # Apply preprocessing to the copy
            if streak_sdevs > 0:
                RemoveStreaks(preprocessed_image, sigma=streak_sdevs)
                self.log_message(f"Applied streak removal (σ={streak_sdevs}) to preprocessed image")

            if flatten_order > 0:
                Flatten(preprocessed_image, flatten_order)
                self.log_message(f"Applied flattening (order={flatten_order}) to preprocessed image")

            # Igor Pro: Save preprocessed image to selected folder
            # Igor Pro: Save preprocessed image to selected folder
            try:
                from pathlib import Path
                import numpy as np
                import os

                # Ensure output folder exists
                output_folder_path = Path(output_folder)
                output_folder_path.mkdir(parents=True, exist_ok=True)

                # Create output file path
                output_file = output_folder_path / f"{preprocessed_name}.npy"

                self.log_message(f"Saving to: {output_file}")

                # Save the numpy array
                np.save(str(output_file), preprocessed_image.data)

                # Verify the file was created
                import time
                time.sleep(0.1)  # Brief pause to ensure file system sync

                if output_file.exists():
                    file_size = output_file.stat().st_size
                    self.log_message(f"SUCCESS: Saved {preprocessed_name}.npy ({file_size} bytes)")
                else:
                    raise IOError(f"Failed to create output file: {output_file}")

            except Exception as save_error:
                error_msg = str(save_error)
                self.log_message(f"ERROR saving file: {error_msg}")
                messagebox.showerror("Save Error", f"Failed to save preprocessed image:\n{error_msg}")
                return

            # Add preprocessed image to current images
            self.current_images[preprocessed_name] = preprocessed_image
            self.update_image_list()

            # Display the preprocessed image
            self.current_display_image = preprocessed_image
            self.display_image()
            self.log_message(f"Single preprocessing completed. Created: {preprocessed_name}")

        except Exception as e:
            self.log_message(f"Error in single preprocessing: {str(e)}")
            messagebox.showerror("Error", f"Error in single preprocessing: {str(e)}")

    def get_single_preprocess_params(self):
        """Get parameters for single image preprocessing"""
        # Create parameter dialog
        root = tk.Tk()
        root.withdraw()

        dialog = tk.Toplevel()
        dialog.title("Single Image Preprocessing")
        dialog.geometry("600x300")
        dialog.transient()
        dialog.grab_set()
        dialog.focus_set()

        result = [None]
        output_folder = [None]

        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Preprocessing Parameters",
                  font=('TkDefaultFont', 12, 'bold')).pack(pady=(0, 15))

        # Output folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=10)

        ttk.Label(folder_frame, text="Output folder:").pack(anchor=tk.W)
        folder_display = ttk.Label(folder_frame, text="No folder selected",
                                   foreground="red", font=('TkDefaultFont', 9))
        folder_display.pack(anchor=tk.W, pady=2)

        def select_folder():
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                output_folder[0] = folder
                folder_display.config(text=folder, foreground="green")

        ttk.Button(folder_frame, text="Select Output Folder",
                   command=select_folder).pack(anchor=tk.W, pady=5)

        # Parameters frame
        params_frame = ttk.Frame(main_frame)
        params_frame.pack(fill=tk.X, pady=10)

        # Streak removal
        streak_frame = ttk.Frame(params_frame)
        streak_frame.pack(fill=tk.X, pady=5)

        ttk.Label(streak_frame, text="Std. Deviations for streak removal (0 = disable):").pack(side=tk.LEFT)
        streak_sdevs_var = tk.DoubleVar(value=3)  # Igor Pro default
        ttk.Entry(streak_frame, textvariable=streak_sdevs_var, width=10).pack(side=tk.LEFT, padx=5)

        # Flattening parameters
        flatten_frame = ttk.Frame(params_frame)
        flatten_frame.pack(fill=tk.X, pady=5)

        ttk.Label(flatten_frame, text="Polynomial order for flattening (0 = disable):").pack(side=tk.LEFT)
        flatten_order_var = tk.IntVar(value=2)  # Igor Pro default
        ttk.Entry(flatten_frame, textvariable=flatten_order_var, width=10).pack(side=tk.LEFT, padx=5)

        def ok_clicked():
            if not output_folder[0]:
                messagebox.showwarning("No Folder", "Please select an output folder.")
                return
            result[0] = (streak_sdevs_var.get(), flatten_order_var.get(), output_folder[0])
            dialog.destroy()

        def cancel_clicked():
            result[0] = None
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT, padx=5)

        dialog.wait_window()
        return result[0]

    def batch_preprocess(self):
        """Run batch preprocessing on multiple images"""
        try:
            BatchPreprocess()
            self.log_message("Batch preprocessing interface opened.")
        except Exception as e:
            self.log_message(f"Error in batch preprocessing: {str(e)}")
            messagebox.showerror("Error", f"Error in batch preprocessing: {str(e)}")

    def update_image_list(self):
        """FIXED: Update the image list display with analysis status indicators"""
        self.image_listbox.delete(0, tk.END)
        for name in self.current_images.keys():
            # Check if this image has analysis results
            if name in self.current_results:
                results = self.current_results[name]
                if results and 'info' in results and results['info'].data.shape[0] > 0:
                    blob_count = results['info'].data.shape[0]
                    display_name = f"{name} [{blob_count} blobs]"
                else:
                    display_name = f"{name} [analyzed, 0 blobs]"
            else:
                display_name = name
            self.image_listbox.insert(tk.END, display_name)

    def update_button_states(self):
        """Update ViewParticles, histogram, and blob toggle button states"""
        # Enable histogram button if image is loaded (same logic as Igor Pro)
        if self.current_display_image is not None:
            self.plot_histogram_button.configure(state=tk.NORMAL)
        else:
            self.plot_histogram_button.configure(state=tk.DISABLED)
        
        # FIXED: Only enable ViewParticles if we have valid analysis results
        if (self.current_display_results and
                'info' in self.current_display_results and
                self.current_display_results['info'] is not None):

            info = self.current_display_results['info']
            blob_count = info.data.shape[0]

            # Enable ViewParticles only if we have analysis results (even with 0 blobs)
            self.view_particles_button.configure(state=tk.NORMAL)

            # Enable blob toggle only if blobs exist
            if blob_count > 0:
                self.blob_toggle.configure(state=tk.NORMAL)
                # Auto-enable blob display for images with blobs (user can toggle off if desired)
                if not self.show_blobs:  # Only auto-enable if not already enabled
                    self.show_blobs = True
                    self.blob_toggle_var.set(True)
                    print(f"Auto-enabled blob display for restored image with {blob_count} blobs")
            else:
                self.blob_toggle.configure(state=tk.DISABLED)
                self.show_blobs = False
                self.blob_toggle_var.set(False)

            print(
                f"Button states updated: ViewParticles={self.view_particles_button['state']}, BlobToggle={self.blob_toggle['state']}, ShowBlobs={self.show_blobs}")
        else:
            # NO valid results - disable both buttons
            self.view_particles_button.configure(state=tk.DISABLED)
            self.blob_toggle.configure(state=tk.DISABLED)
            self.show_blobs = False
            self.blob_toggle_var.set(False)
            print(f"No valid results - disabled ViewParticles and blob toggle")

    def on_image_select(self, event):
        """FIXED: Handle image selection from list with analysis status"""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            display_name = self.image_listbox.get(index)

            # Extract actual image name from display format "name [X blobs]" or "name"
            if '[' in display_name and ']' in display_name:
                image_name = display_name.split(' [')[0]
            else:
                image_name = display_name

            if image_name in self.current_images:
                self.current_display_image = self.current_images[image_name]

                # Restore saved analysis results for this image
                if image_name in self.current_results:
                    self.current_display_results = self.current_results[image_name]
                    print(f"RESTORED analysis results for {image_name}")

                    # Log what we restored
                    if self.current_display_results and 'info' in self.current_display_results:
                        info = self.current_display_results['info']
                        blob_count = info.data.shape[0] if info else 0
                        threshold_mode = self.current_display_results.get('detHResponseThresh', 'unknown')
                        print(f"  Restored {blob_count} blobs from threshold mode {threshold_mode}")
                else:
                    self.current_display_results = None
                    print(f"No saved analysis results for {image_name}")

                # Update display and UI
                self.display_image()
                self.update_info_display()
                self.update_button_states()

                # Log status for user
                if self.current_display_results and 'info' in self.current_display_results:
                    info = self.current_display_results['info']
                    if info is not None:
                        blob_count = info.data.shape[0]
                        threshold_mode = self.current_display_results.get('detHResponseThresh', 'unknown')
                        if blob_count > 0:
                            self.log_message(f"Restored analysis: {blob_count} blobs (threshold {threshold_mode})")
                            self.log_message(f"ViewParticles and Show Blob Regions are ENABLED")
                        else:
                            self.log_message(f"Restored analysis: 0 blobs (threshold {threshold_mode})")
                            self.log_message(f"ViewParticles is ENABLED (shows empty list)")
                else:
                    self.log_message(f"No analysis results for this image")

                # CRITICAL FIX: Force GUI refresh to ensure button states are updated
                self.root.update_idletasks()

    def display_image(self):
        """Display the currently selected image"""
        if self.current_display_image is None:
            return

        try:
            self.ax.clear()

            # Get color map
            cmap = self.color_table_var.get() if self.color_table_var else 'gray'

            # Display the image
            self.ax.imshow(self.current_display_image.data, cmap=cmap, aspect='equal')
            self.ax.set_title(f"Image: {self.current_display_image.name}")

            # Igor Pro: Add blob overlay if enabled and results exist
            print(
                f"DEBUG display_image: show_blobs={self.show_blobs}, has_results={self.current_display_results is not None}")
            if self.current_display_results:
                print(f"DEBUG: Results keys: {self.current_display_results.keys()}")
                if 'info' in self.current_display_results:
                    print(f"DEBUG: Info shape: {self.current_display_results['info'].data.shape}")
                    print(
                        f"DEBUG: Manual threshold used: {self.current_display_results.get('manual_threshold_used', False)}")

            if self.show_blobs and self.current_display_results:
                print("DEBUG: Calling add_blob_overlay")
                self.add_blob_overlay()
            else:
                print(
                    f"DEBUG: NOT calling add_blob_overlay - show_blobs={self.show_blobs}, has_results={self.current_display_results is not None}")

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")

    def add_blob_overlay(self):
        """FIXED: Add blob region overlay to current display - works for ALL threshold modes"""
        try:
            if not self.current_display_results or 'info' not in self.current_display_results:
                print("DEBUG: No display results or info in add_blob_overlay")
                return

            info = self.current_display_results['info']
            if info is None or info.data.shape[0] == 0:
                print("DEBUG: No blob info or empty blob data")
                return

            print(f"DEBUG: add_blob_overlay called with {info.data.shape[0]} blobs")

            # Remove any existing blob overlays first
            for patch in self.ax.patches[:]:
                patch.remove()

            # Clear any existing overlays (keep main image, remove overlays)
            for image in self.ax.images[1:]:
                image.remove()

            blob_count = 0
            # Igor Pro ShowBlobRegions implementation: Create mask for all blob regions
            blob_mask = np.zeros(self.current_display_image.data.shape, dtype=bool)

            for i in range(info.data.shape[0]):
                x_coord = info.data[i, 0]
                y_coord = info.data[i, 1]
                radius = info.data[i, 2]

                # Igor Pro: Create circular mask for this blob using radius
                y_coords, x_coords = np.ogrid[:self.current_display_image.data.shape[0],
                                     :self.current_display_image.data.shape[1]]
                distance = np.sqrt((x_coords - x_coord) ** 2 + (y_coords - y_coord) ** 2)
                blob_region = distance <= radius

                blob_mask |= blob_region

                # Igor Pro: Draw perimeter circle (green like Igor Pro)
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)
                blob_count += 1

            # Igor Pro: Create red tinted overlay for blob regions
            red_overlay = np.zeros((*self.current_display_image.data.shape, 4))
            red_overlay[blob_mask] = [1, 0, 0, 0.3]  # Red with transparency

            # Igor Pro: Apply the overlay
            self.ax.imshow(red_overlay, aspect='equal', alpha=0.5)

            self.log_message(f"Displaying {blob_count} detected blobs with region overlay")
            print(f"DEBUG: Successfully added overlay for {blob_count} blobs")

        except Exception as e:
            self.log_message(f"Error adding blob overlay: {str(e)}")
            print(f"DEBUG: Exception in add_blob_overlay: {str(e)}")

    def toggle_blob_display(self):
        """FIXED: Toggle blob overlay display"""
        print(f"=== TOGGLE BLOB DISPLAY DEBUG ===")
        print(f"toggle_blob_display called")
        print(f"blob_toggle_var.get(): {self.blob_toggle_var.get()}")

        self.show_blobs = self.blob_toggle_var.get()
        print(f"show_blobs set to: {self.show_blobs}")

        self.log_message(f"Blob display: {'ON' if self.show_blobs else 'OFF'}")
        print(f"current_display_results exists: {self.current_display_results is not None}")
        print(f"current_display_image exists: {self.current_display_image is not None}")

        if self.current_display_results:
            info = self.current_display_results.get('info')
            if info:
                print(f"Info available with {info.data.shape[0]} blobs")
            else:
                print("No info in current_display_results")
        else:
            print("No current_display_results")

        print(f"About to call display_image...")
        print(f"================================")

        if self.current_display_image is not None:
            self.display_image()

    def update_info_display(self):
        """Update the results info display"""
        self.info_text.delete(1.0, tk.END)

        if self.current_display_results:
            info = self.current_display_results.get('info')
            if info is not None:
                blob_count = info.data.shape[0]
                threshold = self.current_display_results.get('threshold', 'N/A')

                self.info_text.insert(tk.END, f"Analysis Results:\n")
                self.info_text.insert(tk.END, f"Blobs detected: {blob_count}\n")
                self.info_text.insert(tk.END, f"Threshold: {threshold:.6f}\n\n")

                if blob_count > 0:
                    self.info_text.insert(tk.END, "Blob Statistics:\n")
                    self.info_text.insert(tk.END, f"Avg radius: {np.mean(info.data[:, 2]):.2f}\n")
                    self.info_text.insert(tk.END, f"Avg response: {np.mean(info.data[:, 3]):.6f}\n")
        else:
            self.info_text.insert(tk.END, "No analysis results")

    def run_single_analysis(self):
        """Run Hessian blob detection on current image"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            # Get parameters
            params = GetBlobDetectionParams()
            if params is None:
                self.log_message("Analysis cancelled by user")
                return

            self.log_message("Starting Hessian blob analysis...")

            # Run the analysis
            results = HessianBlobs(
                self.current_display_image,
                scaleStart=params['scaleStart'],
                layers=params['layers'],
                scaleFactor=params['scaleFactor'],
                detHResponseThresh=params['detHResponseThresh'],
                particleType=params['particleType'],
                maxCurvatureRatio=params['maxCurvatureRatio'],
                subPixelMult=params['subPixelMult'],
                allowOverlap=params['allowOverlap'],
                minH=params.get('minH', float('-inf')),
                maxH=params.get('maxH', float('inf')),
                minA=params.get('minA', float('-inf')),
                maxA=params.get('maxA', float('inf')),
                minV=params.get('minV', float('-inf')),
                maxV=params.get('maxV', float('inf'))
            )

            print(f"=== SINGLE ANALYSIS DEBUG ===")
            print(f"HessianBlobs returned: {type(results)}")
            print(f"Results is truthy: {bool(results)}")
            if results:
                print(f"Results keys: {list(results.keys())}")
                info = results.get('info')
                print(f"Info exists: {info is not None}")
                if info:
                    print(f"Info data shape: {info.data.shape}")
                    print(f"Blob count: {info.data.shape[0]}")
                    print(f"Info has measurements (>=11 cols): {info.data.shape[1] >= 11}")
                print(f"Has SS_MAXMAP: {'SS_MAXMAP' in results}")
                print(f"Has SS_MAXSCALEMAP: {'SS_MAXSCALEMAP' in results}")
            else:
                print("Results is None or empty!")
            print(f"=============================")

            if results:
                # Store results (most recent analysis overwrites previous)
                image_name = self.current_display_image.name
                self.current_results[image_name] = results
                self.current_display_results = results

                threshold_mode = results.get('detHResponseThresh', 'unknown')
                print(f"Stored results for {image_name}, threshold {threshold_mode}")

                print(f"=== GUI STATE DEBUG ===")
                print(f"Stored results for image: {image_name}")
                print(f"current_display_results is not None: {self.current_display_results is not None}")
                print(f"current_results keys: {list(self.current_results.keys())}")

                blob_count = results['info'].data.shape[0] if results['info'] else 0
                print(f"Blob count: {blob_count}")
                threshold_mode = results.get('detHResponseThresh', 'unknown')
                manual_used = results.get('manual_threshold_used', False)
                interactive_used = (threshold_mode == -2)
                manual_value_used = results.get('manual_value_used', False)

                print(f"=== THRESHOLD MODE DEBUG ===")
                print(f"detHResponseThresh: {threshold_mode}")
                print(f"Is interactive (-2): {interactive_used}")
                print(f"manual_threshold_used flag: {manual_used}")
                print(f"manual_value_used flag: {manual_value_used}")
                print(f"==========================")

                self.log_message(f"Analysis complete: {blob_count} blobs detected")
                self.log_message(f"Threshold mode: {threshold_mode} (manual={manual_used})")

                # FIXED: Enable ViewParticles for ALL threshold modes (regardless of blob count)
                print(f"Enabling ViewParticles for ANY analysis result...")
                self.view_particles_button.configure(state=tk.NORMAL)
                print(f"ViewParticles button state after enable: {self.view_particles_button['state']}")

                # Enable blob toggle only if blobs found
                if blob_count > 0:
                    print(f"Enabling blob toggle for {blob_count} blobs...")
                    self.blob_toggle.configure(state=tk.NORMAL)
                    print(f"Blob toggle state after enable: {self.blob_toggle['state']}")
                else:
                    print(f"Keeping blob toggle disabled - no blobs found")
                    self.blob_toggle.configure(state=tk.DISABLED)

                # FIXED: Auto-enable blob display for ALL modes with detected blobs
                if blob_count > 0:
                    print(f"Auto-enabling blob display for {blob_count} blobs...")
                    self.blob_toggle_var.set(True)
                    self.show_blobs = True
                    print(f"show_blobs set to: {self.show_blobs}")
                    print(f"blob_toggle_var set to: {self.blob_toggle_var.get()}")
                    self.log_message("Show Blob Regions enabled automatically")

                # Update displays and image list
                self.update_info_display()
                self.update_image_list()  # FIXED: Update image list to show analysis status
                self.update_button_states()  # Update button states

                # Force refresh display to show blobs if enabled
                self.display_image()

                # CRITICAL FIX: Force complete GUI state refresh
                self.root.update_idletasks()

                # EXTRA DEBUG: Check button states after interactive threshold
                if threshold_mode == -2:
                    print(f"=== POST-INTERACTIVE BUTTON CHECK ===")
                    print(f"ViewParticles button state: {self.view_particles_button['state']}")
                    print(f"Blob toggle state: {self.blob_toggle['state']}")
                    print(f"Button should be enabled for {blob_count} blobs")
                    print(f"=======================================")

                # Igor Pro: Automatic save prompt for single image analysis
                if blob_count > 0:
                    save_response = messagebox.askyesno("Save Results", 
                        f"Analysis complete. {blob_count} blobs detected.\n\nSave results to file?")
                    if save_response:
                        self.prompt_single_image_save(results, image_name)
                
                if blob_count > 0:
                    self.log_message("=" * 50)
                    self.log_message("ANALYSIS COMPLETE!")
                    self.log_message(
                        f"Found {blob_count} blobs - ViewParticles and Show Blob Regions are now AVAILABLE!")
                    self.log_message("You can now:")
                    self.log_message("- Click 'View Particles' to browse detected particles")
                    self.log_message("- Toggle 'Show Blob Regions' to see blob overlays")
                    self.log_message("=" * 50)

                    print(f"=== FINAL STATE CHECK ===")
                    print(f"current_display_results ready: {self.current_display_results is not None}")
                    print(
                        f"ViewParticles ready: {self.current_display_results is not None and 'info' in self.current_display_results}")
                    print(f"Show Blob Regions ready: {self.show_blobs}")
                    print(f"Blob toggle enabled: {self.blob_toggle['state'] == 'normal'}")
                    print(f"=========================")
                else:
                    self.log_message("Analysis complete - no blobs detected above threshold")

            else:
                print("ERROR: HessianBlobs returned None or empty results")
                self.log_message("Analysis failed or was cancelled - please try again")

        except Exception as e:
            self.log_message(f"Error in analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{str(e)}")

    def run_batch_analysis(self):
        """Run batch analysis on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        try:
            # Get parameters once for all images
            params = GetBlobDetectionParams()
            if params is None:
                return

            # Igor Pro: Handle interactive threshold mode for batch analysis
            interactive_mode = (params['detHResponseThresh'] == -2)
            if interactive_mode:
                self.log_message("Interactive threshold mode - will prompt for threshold on each image")

            total_images = len(self.current_images)
            processed = 0

            self.log_message(f"Starting batch analysis of {total_images} images...")

            print(f"=== BATCH ANALYSIS DEBUG ===")
            print(f"Total images to process: {total_images}")
            print(f"Final threshold mode: {params['detHResponseThresh']}")
            print(f"============================")

            for image_name, wave in self.current_images.items():
                print(f"=== Processing image {processed + 1}/{total_images}: {image_name} ===")
                self.log_message(f"Processing {image_name}...")

                try:
                    # Igor Pro: Handle interactive threshold for each image individually
                    current_threshold = params['detHResponseThresh']
                    if interactive_mode:
                        # Show interactive threshold dialog for this specific image
                        self.log_message(f"Showing interactive threshold dialog for {image_name}...")
                        from main_functions import InteractiveThreshold
                        from scale_space import ScaleSpaceRepresentation, BlobDetectors  
                        from igor_compatibility import DimDelta
                        import numpy as np

                        # Create scale-space for this image
                        scaleStart_converted = (params['scaleStart'] * DimDelta(wave, 0)) ** 2 / 2
                        layers_calculated = np.log(
                            (params['layers'] * DimDelta(wave, 0)) ** 2 / (2 * scaleStart_converted)) / np.log(
                            params['scaleFactor'])
                        layers = max(1, int(np.ceil(layers_calculated)))
                        igor_scale_start = np.sqrt(params['scaleStart']) / DimDelta(wave, 0)

                        L = ScaleSpaceRepresentation(wave, layers, igor_scale_start, params['scaleFactor'])
                        if L is None:
                            self.log_message(f"Failed to create scale-space for {image_name}, skipping...")
                            continue

                        # Get detectors for this image
                        detH, LG = BlobDetectors(L, 1)
                        if detH is None or LG is None:
                            self.log_message(f"Failed to compute detectors for {image_name}, skipping...")
                            continue

                        # Get threshold interactively for this image
                        try:
                            threshold_result = InteractiveThreshold(wave, detH, LG, params['particleType'],
                                                                  params['maxCurvatureRatio'])
                            if threshold_result[0] is None:
                                self.log_message(f"Threshold selection cancelled for {image_name}, skipping...")
                                continue
                            current_threshold = threshold_result[0]
                            self.log_message(f"Selected threshold {current_threshold:.6f} for {image_name}")
                        except Exception as e:
                            self.log_message(f"Error in threshold selection for {image_name}: {e}")
                            continue

                    print(f"Calling HessianBlobs with threshold: {current_threshold}")
                    results = HessianBlobs(
                        wave,
                        scaleStart=params['scaleStart'],
                        layers=params['layers'],
                        scaleFactor=params['scaleFactor'],
                        detHResponseThresh=current_threshold,
                        particleType=params['particleType'],
                        maxCurvatureRatio=params['maxCurvatureRatio'],
                        subPixelMult=params['subPixelMult'],
                        allowOverlap=params['allowOverlap'],
                        minH=params.get('minH', float('-inf')),
                        maxH=params.get('maxH', float('inf')),
                        minA=params.get('minA', float('-inf')),
                        maxA=params.get('maxA', float('inf')),
                        minV=params.get('minV', float('-inf')),
                        maxV=params.get('maxV', float('inf'))
                    )
                    print(f"HessianBlobs returned: {type(results)}")

                    if results:
                        # Store results (overwrites any previous analysis for this image)
                        self.current_results[image_name] = results

                        blob_count = results['info'].data.shape[0] if results['info'] else 0
                        threshold_mode = results.get('detHResponseThresh', 'unknown')
                        self.log_message(f"  -> {blob_count} blobs detected (threshold {threshold_mode})")
                    else:
                        self.log_message(f"  -> Analysis failed")

                    processed += 1

                except Exception as e:
                    self.log_message(f"  -> Error: {str(e)}")

            self.log_message(f"Batch analysis complete: {processed}/{total_images} images processed")

            # FIXED: After batch analysis, ensure ALL images show results available
            total_blobs_found = 0
            successful_analyses = 0

            # Count total results
            for image_name, results in self.current_results.items():
                if results and 'info' in results:
                    blob_count = results['info'].data.shape[0] if results['info'] else 0
                    total_blobs_found += blob_count
                    successful_analyses += 1

            self.log_message(f"Total blobs found across all images: {total_blobs_found}")
            self.log_message(f"Images with successful analysis: {successful_analyses}")

            # CRITICAL FIX: Update display for current image AND ensure image list reflects analysis status
            if self.current_display_image:
                image_name = self.current_display_image.name
                if image_name in self.current_results:
                    self.current_display_results = self.current_results[image_name]

                    # CRITICAL FIX: Always enable ViewParticles after analysis, enable blob toggle only if blobs exist
                    info = self.current_display_results['info']
                    blob_count = info.data.shape[0] if info else 0
                else:
                    self.current_display_results = None
                    blob_count = 0

                    # ALWAYS enable ViewParticles after analysis
                    self.view_particles_button.configure(state=tk.NORMAL)

                    if blob_count > 0:
                        self.blob_toggle.configure(state=tk.NORMAL)
                        self.log_message(
                            f"Current image '{image_name}': {blob_count} blobs - ViewParticles and Show Blob Regions AVAILABLE")
                    else:
                        self.blob_toggle.configure(state=tk.DISABLED)
                        self.log_message(
                            f"Current image '{image_name}': No blobs detected - ViewParticles AVAILABLE (empty list)")

                    self.update_info_display()
                    self.display_image()

                    # CRITICAL FIX: Force GUI refresh after batch analysis
                    self.root.update_idletasks()

            # CRITICAL FIX: Update the image list to show analysis results
            self.update_image_list()

            # CRITICAL FIX: Highlight that users can now browse through all analyzed images
            self.log_message("=" * 60)
            self.log_message("BATCH ANALYSIS COMPLETE!")
            self.log_message("You can now:")
            self.log_message("1. Select any image from the list to view its results")
            self.log_message("2. Use 'View Particles' to browse detected particles")
            self.log_message("3. Toggle 'Show Blob Regions' to see overlays")
            self.log_message("4. All analyzed images are available in the image list")
            self.log_message("5. Image list now shows '[X blobs]' for analyzed images")
            self.log_message("=" * 60)

            # Igor Pro-style save dialog after batch processing
            self.prompt_save_batch_results()

        except Exception as e:
            self.log_message(f"Error in batch analysis: {str(e)}")
            messagebox.showerror("Batch Analysis Error", f"Batch analysis failed:\n{str(e)}")

    def view_particles(self):
        """FIXED: Launch particle viewer for current results - works for ALL threshold modes"""
        print(f"=== VIEW PARTICLES DEBUG ===")
        print(f"view_particles called")
        print(f"current_display_results is None: {self.current_display_results is None}")

        # Check if we have valid analysis results
        if (self.current_display_results is None or
                'info' not in self.current_display_results or
                self.current_display_results['info'] is None):
            print("ERROR: No valid analysis results - showing warning")
            messagebox.showwarning("No Analysis Results",
                                   "Please run analysis on this image first.\n\n" +
                                   "Click 'Single Analysis' to analyze the current image.")
            return

        info = self.current_display_results['info']
        blob_count = info.data.shape[0]
        print(f"info data shape: {info.data.shape}")
        print(f"blob count: {blob_count}")

        # Check threshold mode for debugging
        threshold_mode = self.current_display_results.get('detHResponseThresh', 'unknown')
        print(f"Current threshold mode: {threshold_mode}")

        if blob_count == 0:
            print("INFO: No particles found - showing empty viewer")
            messagebox.showinfo("No Particles",
                                f"No particles were detected in this analysis.\n\n" +
                                f"Threshold mode: {threshold_mode}\n" +
                                f"Try adjusting the analysis parameters or threshold value.")
            return

        print(f"ViewParticles should work - launching...")
        print(f"==============================")

        # Import and launch particle viewer using the working version from particle_measurements.py
        try:
            from particle_measurements import ViewParticles
            ViewParticles(self.current_display_image, info)
        except Exception as e:
            messagebox.showerror("Viewer Error", f"Failed to open particle viewer:\n{str(e)}")

    def export_results(self):
        """Export analysis results to file"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if file_path:
                ExportResults(self.current_results, file_path)
                self.log_message(f"Results exported to: {file_path}")

        except Exception as e:
            self.log_message(f"Error exporting results: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

    def zoom_fit(self):
        """Fit image to display area"""
        if self.current_display_image is not None:
            self.ax.set_xlim(0, self.current_display_image.data.shape[1])
            self.ax.set_ylim(self.current_display_image.data.shape[0], 0)
            self.canvas.draw()
    
    def plot_histogram(self):
        """Igor Pro: Plot histogram of detected blob measurements"""
        if self.current_display_results is None:
            messagebox.showwarning("No Analysis", "Please run blob detection first.")
            return
        
        try:
            # Igor Pro: Try to use measurement waves first (Heights, Areas, Volumes)
            measurement_data = None
            measurement_name = ""
            
            if ('Heights' in self.current_display_results and 
                self.current_display_results['Heights'] is not None and 
                len(self.current_display_results['Heights'].data) > 0):
                measurement_data = self.current_display_results['Heights'].data
                measurement_name = "Heights"
            elif ('Areas' in self.current_display_results and 
                  self.current_display_results['Areas'] is not None and 
                  len(self.current_display_results['Areas'].data) > 0):
                measurement_data = self.current_display_results['Areas'].data
                measurement_name = "Areas"
            elif ('Volumes' in self.current_display_results and 
                  self.current_display_results['Volumes'] is not None and 
                  len(self.current_display_results['Volumes'].data) > 0):
                measurement_data = self.current_display_results['Volumes'].data
                measurement_name = "Volumes"
            elif ('info' in self.current_display_results and 
                  self.current_display_results['info'] is not None and 
                  self.current_display_results['info'].data.shape[0] > 0):
                # Fallback to blob sizes from info wave
                measurement_data = self.current_display_results['info'].data[:, 2]  # Column 2 = radius
                measurement_name = "Blob Sizes"
            else:
                messagebox.showwarning("No Data", "No measurement data available for histogram.")
                return
                
            if len(measurement_data) == 0:
                messagebox.showwarning("No Blobs", "No blobs detected to plot.")
                return
            
            import matplotlib.pyplot as plt
            
            # Create new figure for histogram (Igor Pro style)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram with Igor Pro style (15 bins, gray color)
            n_bins = min(15, max(5, len(measurement_data) // 3))  # 5-15 bins based on data
            ax.hist(measurement_data, bins=n_bins, color='gray', alpha=0.7, edgecolor='black')
            
            # Igor Pro style formatting
            ax.set_title(f"Histogram - {measurement_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel(measurement_name, fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text (Igor Pro style)
            mean_val = np.mean(measurement_data)
            std_val = np.std(measurement_data)
            min_val = np.min(measurement_data)
            max_val = np.max(measurement_data)
            total_count = len(measurement_data)
            
            stats_text = f'Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nTotal: {total_count}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            self.log_message(f"{measurement_name} histogram plotted ({total_count} measurements)")
            
        except Exception as e:
            self.log_message(f"Error plotting histogram: {str(e)}")
            messagebox.showerror("Histogram Error", f"Failed to plot histogram:\n{str(e)}")

    def prompt_save_batch_results(self):
        """Igor Pro-style save dialog for batch analysis results"""
        if not self.current_results:
            return

        # Count total results
        total_images = len(self.current_results)
        total_blobs = sum(
            results['info'].data.shape[0] if results and 'info' in results and results['info'] is not None else 0
            for results in self.current_results.values()
        )

        # Create dialog matching Igor Pro style
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Batch Analysis Results")
        dialog.geometry("500x400")
        dialog.resizable(False, False)
        dialog.grab_set()

        # Center the dialog
        dialog.transient(self.root)
        dialog.focus_set()

        # Main frame
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header info
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(header_frame, text="Hessian Blob Analysis Results",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Processed {total_images} images, found {total_blobs} total blobs").pack(
            anchor=tk.W)

        # Save options (Igor Pro style)
        options_frame = ttk.LabelFrame(main_frame, text="Save Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 15))

        # Variables for save options
        save_vars = {
            'particle_info': tk.BooleanVar(value=True),
            'scale_space': tk.BooleanVar(value=False),
            'blob_maps': tk.BooleanVar(value=False),
            'summary_report': tk.BooleanVar(value=True),
            'individual_files': tk.BooleanVar(value=False)
        }

        # Checkboxes for save options
        ttk.Checkbutton(options_frame, text="Particle Information (coordinates, sizes, measurements)",
                        variable=save_vars['particle_info']).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Scale-space representations (detH, LapG)",
                        variable=save_vars['scale_space']).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Blob detection maps (maxima locations)",
                        variable=save_vars['blob_maps']).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Analysis summary report",
                        variable=save_vars['summary_report']).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Save individual files for each image",
                        variable=save_vars['individual_files']).pack(anchor=tk.W, pady=2)

        # File format selection
        format_frame = ttk.LabelFrame(main_frame, text="Output Format", padding="10")
        format_frame.pack(fill=tk.X, pady=(0, 15))

        format_var = tk.StringVar(value="csv")
        ttk.Radiobutton(format_frame, text="CSV files (Excel compatible)",
                        variable=format_var, value="csv").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text="Tab-delimited text files",
                        variable=format_var, value="txt").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(format_frame, text="NumPy binary files (.npy)",
                        variable=format_var, value="npy").pack(anchor=tk.W, pady=2)

        # Output directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Output Location", padding="10")
        dir_frame.pack(fill=tk.X, pady=(0, 15))

        output_dir = tk.StringVar(value=os.getcwd())
        dir_entry_frame = ttk.Frame(dir_frame)
        dir_entry_frame.pack(fill=tk.X)

        ttk.Entry(dir_entry_frame, textvariable=output_dir, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_entry_frame, text="Browse...",
                   command=lambda: self.browse_output_directory(output_dir)).pack(side=tk.RIGHT, padx=(5, 0))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))

        def save_results():
            """Save the batch results with selected options"""
            if not any(save_vars[key].get() for key in save_vars):
                messagebox.showwarning("No Options Selected", "Please select at least one save option.")
                return

            try:
                # FIXED: Use Igor Pro BatchHessianBlobs format for batch saving
                batch_results = self.prepare_batch_results_for_save()
                
                if batch_results and batch_results['numParticles'] > 0:
                    from main_functions import SaveBatchResults
                    SaveBatchResults(batch_results, output_dir.get(), format_var.get())
                    self.log_message(f"Batch results saved successfully to {output_dir.get()}!")
                else:
                    messagebox.showwarning("No Data", "No analysis results to save.")
                    return
                    
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
                import traceback
                print(f"Save error details: {traceback.format_exc()}")

        ttk.Button(button_frame, text="Save Results", command=save_results).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

        # Wait for dialog to close
        dialog.wait_window()

    def browse_output_directory(self, dir_var):
        """Browse for output directory"""
        directory = filedialog.askdirectory(initialdir=dir_var.get())
        if directory:
            dir_var.set(directory)
    
    def prepare_batch_results_for_save(self):
        """FIXED: Prepare current analysis results in Igor Pro BatchHessianBlobs format"""
        if not self.current_results:
            return None
            
        from utilities import Wave
        import numpy as np
        
        # Collect all measurement waves (Igor Pro style concatenation)
        all_heights_data = []
        all_volumes_data = []
        all_areas_data = []
        all_avg_heights_data = []
        
        valid_results = {}
        
        for image_name, results in self.current_results.items():
            if results and 'Heights' in results and results['Heights'] is not None:
                heights = results['Heights'].data
                volumes = results['Volumes'].data
                areas = results['Areas'].data
                avg_heights = results['AvgHeights'].data
                
                if len(heights) > 0:
                    all_heights_data.extend(heights)
                    all_volumes_data.extend(volumes)
                    all_areas_data.extend(areas)
                    all_avg_heights_data.extend(avg_heights)
                    valid_results[image_name] = results
        
        if not all_heights_data:
            return None
            
        # Create Igor Pro style batch results structure
        batch_results = {
            'series_folder': f'BatchAnalysis_{len(valid_results)}Images',
            'Parameters': Wave(np.array([1, 256, 1.5, -2, 1, 1, 0, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]), "Parameters"),
            'AllHeights': Wave(np.array(all_heights_data), "AllHeights"),
            'AllVolumes': Wave(np.array(all_volumes_data), "AllVolumes"),
            'AllAreas': Wave(np.array(all_areas_data), "AllAreas"),
            'AllAvgHeights': Wave(np.array(all_avg_heights_data), "AllAvgHeights"),
            'numParticles': len(all_heights_data),
            'numImages': len(valid_results),
            'image_results': valid_results
        }
        
        return batch_results

    def save_batch_results_to_files(self, output_dir, save_vars, file_format):
        """Save batch analysis results to files (Igor Pro compatible format)"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save particle information (main results)
        if save_vars['particle_info'].get():
            self.save_particle_info(output_dir, file_format, timestamp, save_vars['individual_files'].get())

        # Save scale-space data
        if save_vars['scale_space'].get():
            self.save_scale_space_data(output_dir, file_format, timestamp, save_vars['individual_files'].get())

        # Save blob detection maps
        if save_vars['blob_maps'].get():
            self.save_blob_maps(output_dir, file_format, timestamp, save_vars['individual_files'].get())

        # Save summary report
        if save_vars['summary_report'].get():
            self.save_analysis_summary(output_dir, timestamp)

    def save_particle_info(self, output_dir, file_format, timestamp, individual_files):
        """Save particle information data (coordinates, sizes, measurements)"""
        if individual_files:
            # Save separate file for each image
            for image_name, results in self.current_results.items():
                if results and 'info' in results and results['info'] is not None:
                    info = results['info']
                    if info.data.shape[0] > 0:  # Has particles
                        safe_name = "".join(c for c in image_name if c.isalnum() or c in '._-')
                        filename = f"particles_{safe_name}_{timestamp}.{file_format}"
                        filepath = os.path.join(output_dir, filename)
                        self.save_info_data(info, filepath, file_format, image_name)
        else:
            # Save combined file for all images
            filename = f"batch_particles_{timestamp}.{file_format}"
            filepath = os.path.join(output_dir, filename)
            self.save_combined_particle_info(filepath, file_format)

    def save_info_data(self, info, filepath, file_format, image_name):
        """Save individual particle info data"""
        data = info.data

        # Igor Pro HessianBlobs column headers (matching original implementation)
        headers = [
            'X_Center', 'Y_Center', 'Scale', 'DetH_Response', 'LapG_Response',
            'Eccentricity', 'Orientation', 'Area', 'Mean_Intensity', 'Max_Intensity',
            'Min_Intensity', 'Std_Intensity', 'Integrated_Intensity'
        ]

        if file_format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Hessian Blob Analysis Results'])
                writer.writerow(['# Image:', image_name])
                writer.writerow(['# Particles found:', data.shape[0]])
                writer.writerow(['# Columns:', ', '.join(headers)])
                writer.writerow([])
                writer.writerow(headers)
                for row in data:
                    writer.writerow(row)

        elif file_format == 'txt':
            with open(filepath, 'w') as f:
                f.write('# Hessian Blob Analysis Results\n')
                f.write(f'# Image: {image_name}\n')
                f.write(f'# Particles found: {data.shape[0]}\n')
                f.write(f'# Columns: {", ".join(headers)}\n')
                f.write('\n')
                f.write('\t'.join(headers) + '\n')
                for row in data:
                    f.write('\t'.join(map(str, row)) + '\n')

        elif file_format == 'npy':
            np.save(filepath, data)
            # Also save metadata
            metadata_file = filepath.replace('.npy', '_metadata.txt')
            with open(metadata_file, 'w') as f:
                f.write(f'Image: {image_name}\n')
                f.write(f'Particles: {data.shape[0]}\n')
                f.write(f'Columns: {", ".join(headers)}\n')

    def save_combined_particle_info(self, filepath, file_format):
        """FIXED: Save combined particle info from all images with proper Igor Pro format"""
        all_data = []
        image_labels = []

        for image_name, results in self.current_results.items():
            if results and 'info' in results and results['info'] is not None:
                info = results['info']
                heights = results.get('Heights')
                areas = results.get('Areas') 
                volumes = results.get('Volumes')
                avg_heights = results.get('AvgHeights')
                com = results.get('COM')
                
                if info.data.shape[0] > 0:
                    # Create combined data with image name and measurements
                    for i in range(info.data.shape[0]):
                        row_data = [
                            image_name,
                            info.data[i, 0],  # X_Center
                            info.data[i, 1],  # Y_Center
                            heights.data[i] if heights and i < len(heights.data) else 0,
                            areas.data[i] if areas and i < len(areas.data) else 0,
                            volumes.data[i] if volumes and i < len(volumes.data) else 0,
                            avg_heights.data[i] if avg_heights and i < len(avg_heights.data) else 0,
                            com.data[i, 0] if com and i < len(com.data) else info.data[i, 0],
                            com.data[i, 1] if com and i < len(com.data) else info.data[i, 1],
                        ]
                        all_data.append(row_data)

        if not all_data:
            return

        headers = [
            'Image_Name', 'X_Center', 'Y_Center', 'Height', 'Area', 'Volume', 
            'AvgHeight', 'COM_X', 'COM_Y'
        ]

        if file_format == 'csv':
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['# Hessian Blob Batch Analysis Results'])
                writer.writerow(['# Total images:', len(self.current_results)])
                writer.writerow(['# Total particles:', len(all_data)])
                writer.writerow(['# Columns:', ', '.join(headers)])
                writer.writerow([])
                writer.writerow(headers)
                for row in all_data:
                    writer.writerow(row)

        elif file_format == 'txt':
            with open(filepath, 'w') as f:
                f.write('# Hessian Blob Batch Analysis Results\n')
                f.write(f'# Total images: {len(self.current_results)}\n')
                f.write(f'# Total particles: {len(all_data)}\n')
                f.write(f'# Columns: {", ".join(headers)}\n')
                f.write('\n')
                f.write('\t'.join(headers) + '\n')
                for row in all_data:
                    f.write('\t'.join(map(str, row)) + '\n')

        elif file_format == 'npy':
            # Save data and headers separately for NumPy format
            np.save(filepath, np.array(all_data, dtype=object))
            metadata_file = filepath.replace('.npy', '_headers.txt')
            with open(metadata_file, 'w') as f:
                f.write('\n'.join(headers))

    def save_scale_space_data(self, output_dir, file_format, timestamp, individual_files):
        """Save scale-space representation data (detH, LapG)"""
        for image_name, results in self.current_results.items():
            if results and 'detH' in results and 'LG' in results:
                safe_name = "".join(c for c in image_name if c.isalnum() or c in '._-')

                # Save detH
                detH_file = f"detH_{safe_name}_{timestamp}.npy"
                np.save(os.path.join(output_dir, detH_file), results['detH'].data)

                # Save LapG
                lapG_file = f"LapG_{safe_name}_{timestamp}.npy"
                np.save(os.path.join(output_dir, lapG_file), results['LG'].data)

    def save_blob_maps(self, output_dir, file_format, timestamp, individual_files):
        """Save blob detection maps (maxima locations)"""
        for image_name, results in self.current_results.items():
            if results and 'SS_MAXMAP' in results and 'SS_MAXSCALEMAP' in results:
                safe_name = "".join(c for c in image_name if c.isalnum() or c in '._-')

                # Save maxima map
                maxmap_file = f"maxmap_{safe_name}_{timestamp}.npy"
                np.save(os.path.join(output_dir, maxmap_file), results['SS_MAXMAP'].data)

                # Save scale map
                scalemap_file = f"scalemap_{safe_name}_{timestamp}.npy"
                np.save(os.path.join(output_dir, scalemap_file), results['SS_MAXSCALEMAP'].data)

    def save_analysis_summary(self, output_dir, timestamp):
        """Save analysis summary report"""
        filename = f"analysis_summary_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write('Hessian Blob Batch Analysis Summary Report\n')
            f.write('=' * 50 + '\n\n')
            import datetime
            f.write(f'Analysis Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Total Images Processed: {len(self.current_results)}\n\n')

            total_blobs = 0
            for image_name, results in self.current_results.items():
                if results and 'info' in results and results['info'] is not None:
                    blob_count = results['info'].data.shape[0]
                    total_blobs += blob_count

                    threshold_mode = results.get('detHResponseThresh', 'unknown')
                    f.write(f'{image_name}: {blob_count} blobs (threshold: {threshold_mode})\n')
                else:
                    f.write(f'{image_name}: No analysis results\n')

            f.write(f'\nTotal Blobs Found: {total_blobs}\n')
            f.write(f'Average Blobs per Image: {total_blobs / len(self.current_results):.2f}\n')

    def prompt_single_image_save(self, results, image_name):
        """Igor Pro: Save dialog for single image analysis results"""
        try:
            # Create save dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Save Single Image Results")
            dialog.geometry("500x200")
            dialog.resizable(False, False)
            dialog.grab_set()
            
            # Center the dialog
            dialog.transient(self.root)
            self.root.update_idletasks()
            x = (self.root.winfo_width() // 2) - (500 // 2) + self.root.winfo_x()
            y = (self.root.winfo_height() // 2) - (200 // 2) + self.root.winfo_y()
            dialog.geometry(f"500x200+{x}+{y}")
            
            main_frame = ttk.Frame(dialog, padding="15")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Title
            title_label = ttk.Label(main_frame, text=f"Save Results for: {image_name}", 
                                  font=('TkDefaultFont', 11, 'bold'))
            title_label.pack(pady=(0, 15))
            
            # Output directory selection
            dir_frame = ttk.Frame(main_frame)
            dir_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(dir_frame, text="Output Directory:").pack(anchor=tk.W)
            
            dir_select_frame = ttk.Frame(dir_frame)
            dir_select_frame.pack(fill=tk.X, pady=(5, 0))
            
            output_dir = tk.StringVar(value=os.getcwd())
            dir_entry = ttk.Entry(dir_select_frame, textvariable=output_dir, width=50)
            dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            browse_btn = ttk.Button(dir_select_frame, text="Browse...", 
                                  command=lambda: self.browse_output_directory(output_dir))
            browse_btn.pack(side=tk.RIGHT)
            
            # Format selection
            format_frame = ttk.Frame(main_frame)
            format_frame.pack(fill=tk.X, pady=(10, 15))
            
            ttk.Label(format_frame, text="Save Format:").pack(anchor=tk.W)
            format_var = tk.StringVar(value="igor")
            
            format_select_frame = ttk.Frame(format_frame)
            format_select_frame.pack(anchor=tk.W, pady=(5, 0))
            
            ttk.Radiobutton(format_select_frame, text="Igor Pro (.csv, .txt, .h5)", 
                          variable=format_var, value="igor").pack(anchor=tk.W)
            ttk.Radiobutton(format_select_frame, text="CSV only", 
                          variable=format_var, value="csv").pack(anchor=tk.W)
            
            # Buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(15, 0))
            
            def save_results():
                """Save the single image results"""
                try:
                    from main_functions import SaveSingleImageResults
                    SaveSingleImageResults(results, image_name, output_dir.get(), format_var.get())
                    self.log_message(f"Single image results saved to {output_dir.get()}!")
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
            
            ttk.Button(button_frame, text="Save", command=save_results).pack(side=tk.RIGHT, padx=(5, 0))
            ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
            
            # Wait for dialog to close
            dialog.wait_window()
            
        except Exception as e:
            self.log_message(f"Error in save dialog: {str(e)}")
            messagebox.showerror("Dialog Error", f"Failed to show save dialog:\n{str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite
Python Port of Igor Pro Implementation

G.M. King Laboratory - University of Missouri-Columbia
Original Igor Pro code by: Brendan Marsh - marshbp@stanford.edu

This Python port maintains 1-1 functionality with the original Igor Pro version.

For more information, visit:
https://www.physicsandastronomy.missouri.edu/kinggm"""

        messagebox.showinfo("About", about_text)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = HessianBlobGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()