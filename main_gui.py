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
                        # Store in current images - prevent duplicates
                        filename = os.path.basename(file_path)
                        if filename in self.current_images:
                            self.log_message(f"Already loaded: {filename} (skipping)")
                        else:
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
                                if filename in self.current_images:
                                    self.log_message(f"Already loaded: {filename} (skipping)")
                                else:
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

    def show_data_source_dialog(self, title="Select Data Source"):
        """Show data source selection dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.transient(self.root)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"400x200+{x}+{y}")
        
        result = {"choice": None}
        
        # Main frame
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Choose data source:", 
                  font=('TkDefaultFont', 11, 'bold')).pack(pady=(0, 15))
        
        # Radio button variable
        choice_var = tk.StringVar(value="current")
        
        # Radio buttons
        ttk.Radiobutton(main_frame, text="Use current image analysis results", 
                        variable=choice_var, value="current").pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(main_frame, text="Load saved results from file", 
                        variable=choice_var, value="load").pack(anchor=tk.W, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))
        
        def ok_clicked():
            result["choice"] = choice_var.get()
            dialog.destroy()
            
        def cancel_clicked():
            result["choice"] = None
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=ok_clicked).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=cancel_clicked).pack(side=tk.LEFT)
        
        # Wait for dialog to close
        dialog.wait_window()
        return result["choice"]

    def view_particles(self):
        """Launch particle viewer with data source selection"""
        # Show data source dialog
        choice = self.show_data_source_dialog("View Particles - Select Data Source")
        
        if choice is None:  # User cancelled
            return
        elif choice == "current":
            # Use current analysis results
            if (self.current_display_results is None or 
                'info' not in self.current_display_results or 
                self.current_display_results['info'] is None):
                messagebox.showwarning("No Results", "No analysis results available for current image.")
                return
                
            info = self.current_display_results['info']
            if info.data.shape[0] == 0:
                messagebox.showinfo("No Particles", "No particles found in current analysis.")
                return
                
            # Launch viewer with current results
            try:
                from particle_measurements import ViewParticles
                ViewParticles(self.current_display_image, info)
            except Exception as e:
                messagebox.showerror("Viewer Error", f"Failed to open particle viewer:\n{str(e)}")
                
        elif choice == "load":
            # Load saved results
            try:
                results_folder = filedialog.askdirectory(
                    title="Select Saved Analysis Results Folder",
                    initialdir=os.getcwd()
                )
                
                if not results_folder:
                    return
                
                # Load and launch viewer with saved data
                from particle_measurements import ViewParticles
                ViewParticles(None, None, saved_data_path=results_folder)
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load saved results:\n{str(e)}")

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
        """Plot histogram with data source selection"""
        # Show data source dialog
        choice = self.show_data_source_dialog("Plot Histogram - Select Data Source")
        
        if choice is None:  # User cancelled
            return
        elif choice == "current":
            # Use current analysis results
            if self.current_display_results is None:
                messagebox.showwarning("No Analysis", "No analysis results available for current image.")
                return
            self.plot_histogram_from_results(self.current_display_results)
            
        elif choice == "load":
            # Load saved results
            try:
                results_folder = filedialog.askdirectory(
                    title="Select Saved Analysis Results Folder",
                    initialdir=os.getcwd()
                )
                
                if not results_folder:
                    return
                
                # Load results and plot histogram
                loaded_results = self.load_saved_results_for_histogram(results_folder)
                if loaded_results:
                    self.plot_histogram_from_results(loaded_results)
                else:
                    messagebox.showerror("Load Error", "Could not load histogram data from selected folder.")
                    
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load saved results:\n{str(e)}")

    def plot_histogram_from_results(self, results):
        """Plot histogram from analysis results"""
        try:
            # Igor Pro: Try to use measurement waves first (Heights, Areas, Volumes)
            measurement_data = None
            measurement_name = ""
            
            if ('Heights' in results and 
                results['Heights'] is not None and 
                len(results['Heights'].data) > 0):
                measurement_data = results['Heights'].data
                measurement_name = "Heights"
            elif ('Areas' in results and 
                  results['Areas'] is not None and 
                  len(results['Areas'].data) > 0):
                measurement_data = results['Areas'].data
                measurement_name = "Areas"
            elif ('Volumes' in results and 
                  results['Volumes'] is not None and 
                  len(results['Volumes'].data) > 0):
                measurement_data = results['Volumes'].data
                measurement_name = "Volumes"
            elif ('info' in results and 
                  results['info'] is not None and 
                  results['info'].data.shape[0] > 0):
                # Fallback to blob sizes from info wave
                measurement_data = results['info'].data[:, 2]  # Column 2 = radius
                measurement_name = "Blob Sizes"
            else:
                messagebox.showwarning("No Data", "No measurement data available for histogram.")
                return
                
            if len(measurement_data) == 0:
                messagebox.showwarning("No Blobs", "No blobs detected to plot.")
                return
            
            import matplotlib.pyplot as plt
            
            # Igor Pro: Create isolated histogram window (prevent ALL extra displays)
            import matplotlib
            matplotlib.use('TkAgg')  # Ensure consistent backend
            
            # Close any existing matplotlib figures to prevent interference
            plt.close('all')
            
            # Create histogram in isolated environment
            plt.ioff()  # Turn off interactive mode completely
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
            
            # Show only the histogram (no other windows)
            plt.tight_layout()
            plt.show()  # Simple blocking show for clean display
            
            # Clean up and restore normal matplotlib state
            plt.ion()  # Restore interactive mode for other parts of application
            
            self.log_message(f"{measurement_name} histogram plotted ({total_count} measurements)")
            
        except Exception as e:
            self.log_message(f"Error plotting histogram: {str(e)}")
            messagebox.showerror("Histogram Error", f"Failed to plot histogram:\n{str(e)}")

    def load_saved_results_for_histogram(self, results_folder):
        """Load saved analysis results for histogram plotting"""
        try:
            from utilities import Wave
            import numpy as np
            
            # Try to load Igor Pro format measurement data directly
            measurement_files = {
                'Heights': os.path.join(results_folder, 'Heights.txt'),
                'Areas': os.path.join(results_folder, 'Areas.txt'), 
                'Volumes': os.path.join(results_folder, 'Volumes.txt'),
                'Info': os.path.join(results_folder, 'Info.txt')
            }
            
            # Check if required files exist
            if not os.path.exists(measurement_files['Info']):
                return None
                
            # Load the measurement data for histogram
            results = {}
            
            # Load Heights, Areas, Volumes
            for wave_name in ['Heights', 'Areas', 'Volumes']:
                file_path = measurement_files[wave_name]
                if os.path.exists(file_path):
                    data = self.load_wave_data(file_path)
                    if data is not None:
                        results[wave_name] = Wave(data, wave_name)
            
            # Load info data for fallback
            info_data = []
            with open(measurement_files['Info'], 'r') as f:
                lines = f.readlines()
                reading_data = False
                for line in lines:
                    line = line.strip()
                    if "P_Seed" in line and "Q_Seed" in line:  # Header line
                        reading_data = True
                        continue
                    if reading_data and line:
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 4:  # At least X, Y, scale, response
                                info_data.append([float(p) for p in parts])
                        except ValueError:
                            continue
            
            if info_data:
                results['info'] = Wave(np.array(info_data), "info")
            
            return results if results else None
            
        except Exception as e:
            print(f"Error loading saved results for histogram: {e}")
            return None

    def prompt_save_batch_results(self):
        """Automatically save all batch analysis results"""
        if not self.current_results:
            return

        # Count total results
        total_images = len(self.current_results)
        total_blobs = sum(
            results['info'].data.shape[0] if results and 'info' in results and results['info'] is not None else 0
            for results in self.current_results.values()
        )

        # Get output directory from user
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Batch Results",
            initialdir=os.getcwd()
        )
        
        if not output_dir:
            return
        
        try:
            # Auto-save all results without dialog
            batch_results = self.prepare_batch_results_for_save()
            
            if batch_results is not None:
                from main_functions import SaveBatchResults
                SaveBatchResults(batch_results, output_dir, "igor")  # Default to Igor format
                particles_msg = f"with {batch_results['numParticles']} particles" if batch_results['numParticles'] > 0 else "with no particles detected"
                self.log_message(f"All batch results automatically saved to {output_dir} {particles_msg}!")
                messagebox.showinfo("Save Complete", 
                                  f"All batch analysis results saved successfully!\n\n"
                                  f"Location: {output_dir}\n"
                                  f"Images processed: {total_images}\n"
                                  f"Total blobs detected: {total_blobs}")
            else:
                messagebox.showwarning("No Data", "No analysis results to save - no images have been analyzed.")
                return
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
            import traceback
            print(f"Save error details: {traceback.format_exc()}")


    def browse_output_directory(self, dir_var):
        """Browse for output directory"""
        directory = filedialog.askdirectory(initialdir=dir_var.get())
        if directory:
            dir_var.set(directory)
    
    def load_saved_results_for_viewing(self):
        """Igor Pro: Load saved analysis results from Igor Pro format files"""
        try:
            # Ask user to select saved results folder
            results_folder = filedialog.askdirectory(
                title="Select Saved Analysis Results Folder",
                initialdir=os.getcwd()
            )
            
            if not results_folder:
                return
                
            # Try to load Igor Pro format results
            if self.load_igor_pro_results(results_folder):
                self.log_message(f"Loaded saved analysis results from {results_folder}")
                # Update displays
                self.update_info_display()
                self.update_button_states()
                self.display_image()
                # Now launch ViewParticles with loaded data
                self.view_particles()
            else:
                messagebox.showerror("Load Error", 
                    "Could not load analysis results from selected folder.\n\n" +
                    "Please select a folder containing Igor Pro format analysis results.")
                    
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading saved results:\n{str(e)}")
            
    def load_igor_pro_results(self, results_folder):
        """Load Igor Pro format analysis results from folder"""
        try:
            from utilities import Wave
            import numpy as np
            
            # Check for Igor Pro folder structure (ImageName_Particles or Series_X)
            folder_name = os.path.basename(results_folder)
            
            # Try to load measurement waves
            measurement_files = {
                'Heights': os.path.join(results_folder, 'Heights.txt'),
                'Areas': os.path.join(results_folder, 'Areas.txt'), 
                'Volumes': os.path.join(results_folder, 'Volumes.txt'),
                'AvgHeights': os.path.join(results_folder, 'AvgHeights.txt'),
                'COM': os.path.join(results_folder, 'COM.txt'),
                'Info': os.path.join(results_folder, 'Info.txt')
            }
            
            # Check if required files exist
            if not all(os.path.exists(f) for f in [measurement_files['Heights'], measurement_files['Info']]):
                return False
                
            # Load the measurement data
            results = {}
            
            # Load Heights, Areas, Volumes, AvgHeights
            for wave_name in ['Heights', 'Areas', 'Volumes', 'AvgHeights']:
                file_path = measurement_files[wave_name]
                if os.path.exists(file_path):
                    data = self.load_wave_data(file_path)
                    if data is not None:
                        results[wave_name] = Wave(data, wave_name)
            
            # Load COM (2D data)
            com_file = measurement_files['COM']
            if os.path.exists(com_file):
                com_data = self.load_com_data(com_file)
                if com_data is not None:
                    results['COM'] = Wave(com_data, 'COM')
            
            # Create basic info wave from particle count
            if 'Heights' in results:
                num_particles = len(results['Heights'].data)
                # Create minimal info wave (simplified)
                info_data = np.zeros((num_particles, 15))  # 15 columns for Igor Pro info wave
                for i in range(num_particles):
                    if 'COM' in results and i < len(results['COM'].data):
                        info_data[i, 0] = results['COM'].data[i, 0]  # X center
                        info_data[i, 1] = results['COM'].data[i, 1]  # Y center
                    info_data[i, 2] = 1.0  # Default radius
                    if 'Heights' in results and i < len(results['Heights'].data):
                        info_data[i, 3] = results['Heights'].data[i]  # Max blob strength
                
                results['info'] = Wave(info_data, 'info')
                results['numParticles'] = num_particles
                
                # Store as current display results
                if self.current_display_image:
                    image_name = self.current_display_image.name
                    self.current_results[image_name] = results
                    self.current_display_results = results
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error loading Igor Pro results: {e}")
            return False
            
    def load_wave_data(self, file_path):
        """Load 1D wave data from Igor Pro format text file"""
        try:
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                reading_data = False
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Look for data section markers
                    if line in ["Data:", "BEGIN", "DATA"] or "Data:" in line:
                        reading_data = True
                        continue
                    
                    # Stop reading if we hit end markers
                    if line in ["END", "end"] or line.startswith("END"):
                        break
                    
                    # Try to parse numerical data
                    if reading_data:
                        try:
                            # Handle tab-separated or space-separated values
                            if '\t' in line:
                                values = line.split('\t')
                            else:
                                values = line.split()
                            
                            # Take the first numerical value
                            for val in values:
                                try:
                                    data.append(float(val))
                                    break  # Only take first value per line for 1D data
                                except ValueError:
                                    continue
                        except ValueError:
                            continue
                    else:
                        # If no data marker found, try to parse all numerical lines
                        try:
                            if '\t' in line:
                                values = line.split('\t')
                            else:
                                values = line.split()
                            
                            for val in values:
                                try:
                                    data.append(float(val))
                                    break  # Only take first value per line
                                except ValueError:
                                    continue
                        except ValueError:
                            continue
                            
            return np.array(data) if data else None
        except Exception as e:
            return None
            
    def load_com_data(self, file_path):
        """Load 2D COM data from Igor Pro format text file"""
        try:
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                reading_data = False
                for line in lines:
                    line = line.strip()
                    if "X_Center" in line and "Y_Center" in line:
                        reading_data = True
                        continue
                    if reading_data and line:
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                x = float(parts[0])
                                y = float(parts[1])
                                data.append([x, y])
                        except ValueError:
                            continue
            return np.array(data) if data else None
        except Exception:
            return None
    
    def prepare_batch_results_for_save(self):
        """FIXED: Prepare current analysis results in Igor Pro BatchHessianBlobs format"""
        if not self.current_results:
            print("DEBUG: No current_results to save")
            return None
            
        from utilities import Wave
        import numpy as np
        
        print(f"DEBUG: Preparing batch results from {len(self.current_results)} images")
        
        # Collect all measurement data from info waves
        all_info_data = []
        valid_results = {}
        total_particles = 0
        
        for image_name, results in self.current_results.items():
            print(f"DEBUG: Processing {image_name}")
            if results and 'info' in results and results['info'] is not None:
                info = results['info']
                if info.data.shape[0] > 0:
                    print(f"  Found {info.data.shape[0]} particles")
                    all_info_data.extend(info.data)
                    valid_results[image_name] = results
                    total_particles += info.data.shape[0]
                else:
                    print("  No particles in this image")
            else:
                print("  No info data found")
        
        print(f"DEBUG: Total particles across all images: {total_particles}")
        
        if total_particles == 0:
            print("DEBUG: No particles found in any image")
            # Still create a batch results structure even with 0 particles (Igor Pro compatibility)
            batch_results = {
                'series_folder': f'BatchAnalysis_{len(self.current_results)}Images',
                'numParticles': 0,
                'numImages': len(self.current_results),
                'image_results': self.current_results,
                'AllHeights': Wave(np.array([]), "AllHeights"),
                'AllVolumes': Wave(np.array([]), "AllVolumes"),
                'AllAreas': Wave(np.array([]), "AllAreas"),
                'AllAvgHeights': Wave(np.array([]), "AllAvgHeights")
            }
            return batch_results
        
        # Extract measurement data from info waves
        all_info_array = np.array(all_info_data)
        
        # Create measurement waves from info data
        all_heights_data = []
        all_volumes_data = []
        all_areas_data = []
        all_avg_heights_data = []
        
        for image_name, results in valid_results.items():
            if 'Heights' in results and results['Heights'] is not None:
                # Use existing measurement waves if available
                all_heights_data.extend(results['Heights'].data)
                all_volumes_data.extend(results['Volumes'].data)
                all_areas_data.extend(results['Areas'].data)
                all_avg_heights_data.extend(results['AvgHeights'].data)
            else:
                # Extract from info wave if measurement waves not available
                info = results['info']
                if info.data.shape[1] > 10:  # Has measurement columns
                    all_heights_data.extend(info.data[:, 10])  # Height column
                    all_areas_data.extend(info.data[:, 8])     # Area column
                    all_volumes_data.extend(info.data[:, 9])   # Volume column
                    all_avg_heights_data.extend(info.data[:, 10])  # Use height as avg height
                else:
                    # Default values if no measurements available
                    n_particles = info.data.shape[0]
                    all_heights_data.extend([0.0] * n_particles)
                    all_areas_data.extend([0.0] * n_particles)
                    all_volumes_data.extend([0.0] * n_particles)
                    all_avg_heights_data.extend([0.0] * n_particles)
        
        # Create Igor Pro style batch results structure
        batch_results = {
            'series_folder': f'BatchAnalysis_{len(valid_results)}Images',
            'Parameters': Wave(np.array([1, 256, 1.5, -2, 1, 1, 0, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]), "Parameters"),
            'AllHeights': Wave(np.array(all_heights_data), "AllHeights"),
            'AllVolumes': Wave(np.array(all_volumes_data), "AllVolumes"),
            'AllAreas': Wave(np.array(all_areas_data), "AllAreas"),
            'AllAvgHeights': Wave(np.array(all_avg_heights_data), "AllAvgHeights"),
            'numParticles': total_particles,
            'numImages': len(valid_results),
            'image_results': valid_results
        }
        
        print(f"DEBUG: Created batch results with {batch_results['numParticles']} particles from {batch_results['numImages']} images")
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
        """Automatically save single image analysis results"""
        try:
            # Count particles for display
            particle_count = results['info'].data.shape[0] if results and 'info' in results else 0
            
            # Get output directory from user
            output_dir = filedialog.askdirectory(
                title="Select Output Directory for Single Image Results",
                initialdir=os.getcwd()
            )
            
            if not output_dir:
                return
            
            # Auto-save all results without dialog
            from main_functions import SaveSingleImageResults
            SaveSingleImageResults(results, image_name, output_dir, "igor")  # Default to Igor format
            self.log_message(f"Single image results automatically saved to {output_dir}!")
            messagebox.showinfo("Save Complete", 
                              f"Single image analysis results saved successfully!\n\n"
                              f"Location: {output_dir}\n"
                              f"Image: {image_name}\n"
                              f"Particles detected: {particle_count}")
                              
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")
            import traceback
            print(f"Save error details: {traceback.format_exc()}")


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