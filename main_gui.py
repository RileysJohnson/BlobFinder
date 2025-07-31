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
        """Update ViewParticles and blob toggle button states"""
        if self.current_display_results and 'info' in self.current_display_results:
            info = self.current_display_results['info']
            if info is not None:
                blob_count = info.data.shape[0]
                
                # ALWAYS enable ViewParticles after analysis (even with 0 blobs)
                self.view_particles_button.configure(state=tk.NORMAL)
                
                # Enable blob toggle only if blobs exist
                if blob_count > 0:
                    self.blob_toggle.configure(state=tk.NORMAL)
                else:
                    self.blob_toggle.configure(state=tk.DISABLED)
                    self.show_blobs = False
                    self.blob_toggle_var.set(False)
            else:
                self.view_particles_button.configure(state=tk.DISABLED)
                self.blob_toggle.configure(state=tk.DISABLED)
        else:
            self.view_particles_button.configure(state=tk.DISABLED)
            self.blob_toggle.configure(state=tk.DISABLED)

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
                self.current_display_results = self.current_results.get(image_name, None)
                
                self.display_image()
                self.update_info_display()

                # Update button states using centralized method
                self.update_button_states()
                
                # Log status
                if self.current_display_results and 'info' in self.current_display_results:
                    info = self.current_display_results['info']
                    if info is not None:
                        blob_count = info.data.shape[0]
                        if blob_count > 0:
                            self.log_message(f"ViewParticles and Show Blob Regions are now ENABLED for {blob_count} blobs")
                        else:
                            self.log_message(f"ViewParticles is ENABLED (shows empty list)")
                
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
            print(f"DEBUG display_image: show_blobs={self.show_blobs}, has_results={self.current_display_results is not None}")
            if self.current_display_results:
                print(f"DEBUG: Results keys: {self.current_display_results.keys()}")
                if 'info' in self.current_display_results:
                    print(f"DEBUG: Info shape: {self.current_display_results['info'].data.shape}")
                    print(f"DEBUG: Manual threshold used: {self.current_display_results.get('manual_threshold_used', False)}")
            
            if self.show_blobs and self.current_display_results:
                print("DEBUG: Calling add_blob_overlay")
                self.add_blob_overlay()
            else:
                print(f"DEBUG: NOT calling add_blob_overlay - show_blobs={self.show_blobs}, has_results={self.current_display_results is not None}")

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
                allowOverlap=params['allowOverlap']
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
                
                if blob_count > 0:
                    self.log_message("="*50)
                    self.log_message("ANALYSIS COMPLETE!")
                    self.log_message(f"Found {blob_count} blobs - ViewParticles and Show Blob Regions are now AVAILABLE!")
                    self.log_message("You can now:")
                    self.log_message("- Click 'View Particles' to browse detected particles")
                    self.log_message("- Toggle 'Show Blob Regions' to see blob overlays")
                    self.log_message("="*50)
                    
                    print(f"=== FINAL STATE CHECK ===")
                    print(f"current_display_results ready: {self.current_display_results is not None}")
                    print(f"ViewParticles ready: {self.current_display_results is not None and 'info' in self.current_display_results}")
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
                
            # Handle manual threshold for batch analysis
            if params['detHResponseThresh'] == -2:
                # Get threshold once for all images in batch
                from main_functions import InteractiveThreshold
                
                # Use first image to determine threshold
                first_image = next(iter(self.current_images.values()))
                
                # Need to set up detectors for threshold selection  
                from scale_space import ScaleSpaceRepresentation, BlobDetectors
                from igor_compatibility import DimDelta
                import numpy as np
                
                # Create scale-space for first image
                scaleStart_converted = (params['scaleStart'] * DimDelta(first_image, 0)) ** 2 / 2
                layers_calculated = np.log((params['layers'] * DimDelta(first_image, 0)) ** 2 / (2 * scaleStart_converted)) / np.log(params['scaleFactor'])
                layers = max(1, int(np.ceil(layers_calculated)))
                igor_scale_start = np.sqrt(params['scaleStart']) / DimDelta(first_image, 0)
                
                L = ScaleSpaceRepresentation(first_image, layers, igor_scale_start, params['scaleFactor'])
                if L is None:
                    messagebox.showerror("Error", "Failed to create scale-space for threshold selection")
                    return
                    
                # FIXED: Get detectors directly from BlobDetectors return instead of using non-existent GetWave
                detH, LG = BlobDetectors(L, 1)
                
                if detH is None or LG is None:
                    messagebox.showerror("Error", "Failed to compute detectors for threshold selection")
                    return
                
                # Get threshold interactively  
                try:
                    threshold_result = InteractiveThreshold(first_image, detH, LG, params['particleType'], params['maxCurvatureRatio'])
                    if threshold_result[0] is None:
                        self.log_message("Batch analysis cancelled - no threshold selected")
                        return
                except Exception as e:
                    self.log_message(f"Error in interactive threshold selection: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                    
                # Use the selected threshold for all images
                batch_threshold = threshold_result[0]
                params['detHResponseThresh'] = batch_threshold
                self.log_message(f"Using threshold {batch_threshold:.6f} for all images in batch")

            total_images = len(self.current_images)
            processed = 0

            self.log_message(f"Starting batch analysis of {total_images} images...")

            print(f"=== BATCH ANALYSIS DEBUG ===")
            print(f"Total images to process: {total_images}")
            print(f"Final threshold mode: {params['detHResponseThresh']}")
            print(f"============================")
            
            for image_name, wave in self.current_images.items():
                print(f"=== Processing image {processed+1}/{total_images}: {image_name} ===")
                self.log_message(f"Processing {image_name}...")

                try:
                    print(f"Calling HessianBlobs with threshold: {params['detHResponseThresh']}")
                    results = HessianBlobs(
                        wave,
                        scaleStart=params['scaleStart'],
                        layers=params['layers'],
                        scaleFactor=params['scaleFactor'],
                        detHResponseThresh=params['detHResponseThresh'],
                        particleType=params['particleType'],
                        maxCurvatureRatio=params['maxCurvatureRatio'],
                        subPixelMult=params['subPixelMult'],
                        allowOverlap=params['allowOverlap']
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
                        self.log_message(f"Current image '{image_name}': {blob_count} blobs - ViewParticles and Show Blob Regions AVAILABLE")
                    else:
                        self.blob_toggle.configure(state=tk.DISABLED)
                        self.log_message(f"Current image '{image_name}': No blobs detected - ViewParticles AVAILABLE (empty list)")
                    
                    self.update_info_display()
                    self.display_image()
                    
                    # CRITICAL FIX: Force GUI refresh after batch analysis
                    self.root.update_idletasks()
            
            # CRITICAL FIX: Update the image list to show analysis results
            self.update_image_list()
            
            # CRITICAL FIX: Highlight that users can now browse through all analyzed images
            self.log_message("="*60)
            self.log_message("BATCH ANALYSIS COMPLETE!")  
            self.log_message("You can now:")
            self.log_message("1. Select any image from the list to view its results")
            self.log_message("2. Use 'View Particles' to browse detected particles")
            self.log_message("3. Toggle 'Show Blob Regions' to see overlays")
            self.log_message("4. All analyzed images are available in the image list")
            self.log_message("5. Image list now shows '[X blobs]' for analyzed images")
            self.log_message("="*60)

        except Exception as e:
            self.log_message(f"Error in batch analysis: {str(e)}")
            messagebox.showerror("Batch Analysis Error", f"Batch analysis failed:\n{str(e)}")

    def view_particles(self):
        """FIXED: Launch particle viewer for current results - works for ALL threshold modes"""
        print(f"=== VIEW PARTICLES DEBUG ===")
        print(f"view_particles called")
        print(f"current_display_results is None: {self.current_display_results is None}")
        
        if self.current_display_results is None:
            print("ERROR: current_display_results is None - showing warning")
            messagebox.showwarning("No Results", "Please run analysis first.")
            return

        print(f"current_display_results keys: {list(self.current_display_results.keys())}")
        info = self.current_display_results.get('info')
        print(f"info is None: {info is None}")
        if info:
            print(f"info data shape: {info.data.shape}")
            
        # Check threshold mode for debugging
        threshold_mode = self.current_display_results.get('detHResponseThresh', 'unknown')
        print(f"Current threshold mode: {threshold_mode}")
        if threshold_mode == -2:
            print("This is from INTERACTIVE THRESHOLD - should work!")
            print(f"blob count: {info.data.shape[0]}")
            
        if info is None or info.data.shape[0] == 0:
            print("ERROR: No particles - showing warning")
            messagebox.showwarning("No Particles", "No particles detected in current analysis.")
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