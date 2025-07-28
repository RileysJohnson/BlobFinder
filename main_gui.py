#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main GUI
Complete 1-1 port from Igor Pro implementation
COMPLETE FIX: Proper buttons, blob regions, preprocessing, view particles

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
        """Setup the main user interface - FIXED: Enhanced layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        # FIXED: File management section with proper button names
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

        # FIXED: Preprocessing section (matching Igor Pro)
        preprocess_frame = ttk.LabelFrame(left_panel, text="Preprocessing", padding="5")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(preprocess_frame, text="Batch Preprocess",
                   command=self.batch_preprocess).pack(fill=tk.X, pady=2)
        ttk.Button(preprocess_frame, text="Group Preprocess",
                   command=self.group_preprocess).pack(fill=tk.X, pady=2)

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

        # FIXED: Add color table selection like Igor Pro
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

        # FIXED: View particles button (matching Igor Pro)
        ttk.Button(display_frame, text="View Particles",
                   command=self.view_particles).pack(fill=tk.X, pady=2)

        # Results info
        info_frame = ttk.LabelFrame(left_panel, text="Results Info", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_text = tk.Text(info_frame, height=8, width=30, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=info_scrollbar.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Image display area
        self.setup_matplotlib_canvas(right_panel)

        # Bottom panel for log messages
        bottom_panel = ttk.Frame(self.root)
        bottom_panel.pack(fill=tk.X, padx=5, pady=(0, 5))

        log_frame = ttk.LabelFrame(bottom_panel, text="Log Messages", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_matplotlib_canvas(self, parent):
        """Setup matplotlib canvas for image display"""
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot
        self.ax.set_title("Load an image to begin analysis")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def setup_menu(self):
        """Setup the menu bar"""
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
        analysis_menu.add_command(label="Batch Preprocess", command=self.batch_preprocess)
        analysis_menu.add_command(label="Group Preprocess", command=self.group_preprocess)

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
            ("All supported", "*.ibw *.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
            ("Igor Binary Wave", "*.ibw"),
            ("TIFF files", "*.tif *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]

        filenames = filedialog.askopenfilenames(  # Changed to allow multiple selection
            title="Select Image File(s)",
            filetypes=filetypes
        )

        if filenames:
            loaded_count = 0
            for filename in filenames:
                try:
                    wave = LoadWave(filename)
                    if wave is not None:
                        base_name = os.path.basename(filename)
                        self.current_images[base_name] = wave
                        loaded_count += 1
                        self.log_message(f"Loaded image: {base_name}")
                    else:
                        messagebox.showerror("Error", f"Failed to load image: {filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading image: {str(e)}")
                    self.log_message(f"Error loading {filename}: {str(e)}")

            if loaded_count > 0:
                self.update_image_list()
                # Select the first loaded image
                self.image_listbox.selection_set(0)
                self.on_image_select(None)
                self.log_message(f"Loaded {loaded_count} image(s)")

    def load_folder(self):
        """FIXED: Load all images from a selected folder"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")

        if folder_path:
            # Supported image extensions
            extensions = ['.ibw', '.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']

            image_files = []
            # FIXED: Only search once per extension to avoid duplicates
            for ext in extensions:
                image_files.extend(Path(folder_path).glob(f"*{ext}"))

            if not image_files:
                messagebox.showinfo("No Images", "No supported image files found in the selected folder.")
                return

            loaded_count = 0
            for file_path in image_files:
                try:
                    wave = LoadWave(str(file_path))
                    if wave is not None:
                        base_name = file_path.name
                        self.current_images[base_name] = wave
                        loaded_count += 1
                        self.log_message(f"Loaded image: {base_name}")
                    else:
                        self.log_message(f"Failed to load: {file_path.name}")
                except Exception as e:
                    self.log_message(f"Error loading {file_path.name}: {str(e)}")

            if loaded_count > 0:
                self.update_image_list()
                # Select the first loaded image
                self.image_listbox.selection_set(0)
                self.on_image_select(None)
                self.log_message(f"Loaded {loaded_count} images from folder")
            else:
                messagebox.showerror("Error", "No images could be loaded from the folder.")

    # FIXED: Preprocessing methods (matching Igor Pro)
    def batch_preprocess(self):
        """Run batch preprocessing on multiple images"""
        try:
            BatchPreprocess()
            self.log_message("Batch preprocessing interface opened.")
        except Exception as e:
            self.log_message(f"Error in batch preprocessing: {str(e)}")
            messagebox.showerror("Error", f"Error in batch preprocessing: {str(e)}")

    def group_preprocess(self):
        """Run group preprocessing on loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        try:
            GroupPreprocess(self.current_images)
            self.log_message("Group preprocessing completed.")
            # Refresh display if current image was processed
            if self.current_display_image:
                self.display_image()
        except Exception as e:
            self.log_message(f"Error in group preprocessing: {str(e)}")
            messagebox.showerror("Error", f"Error in group preprocessing: {str(e)}")

    def update_image_list(self):
        """Update the image list display"""
        self.image_listbox.delete(0, tk.END)
        for name in self.current_images.keys():
            self.image_listbox.insert(tk.END, name)

    def on_image_select(self, event):
        """Handle image selection from list"""
        selection = self.image_listbox.curselection()
        if selection:
            index = selection[0]
            image_name = self.image_listbox.get(index)

            if image_name in self.current_images:
                self.current_display_image = self.current_images[image_name]
                self.current_display_results = self.current_results.get(image_name, None)
                self.display_image()
                self.update_info_display()

                # Enable blob toggle if results exist
                if self.current_display_results:
                    self.blob_toggle.configure(state=tk.NORMAL)
                else:
                    self.blob_toggle.configure(state=tk.DISABLED)
                    self.show_blobs = False
                    self.blob_toggle_var.set(False)

    def display_image(self):
        """Display the currently selected image"""
        if self.current_display_image is None:
            return

        try:
            self.ax.clear()

            # FIXED: Use selected color table
            colormap = self.color_table_var.get()

            # Display the image
            self.ax.imshow(self.current_display_image.data, cmap=colormap, aspect='equal')
            self.ax.set_title(f"{self.current_display_image.name}")

            # FIXED: Add blob overlay if requested and available
            if self.show_blobs and self.current_display_results:
                self.add_blob_overlay()

            # Remove ticks for cleaner display
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")

    def add_blob_overlay(self):
        """FIXED: Add blob regions with red tinting like Igor Pro"""
        try:
            if (self.current_display_results is None or
                    'info' not in self.current_display_results or
                    self.current_display_results['info'] is None):
                return

            info_wave = self.current_display_results['info']
            if info_wave.data.shape[0] == 0:
                return

            # Create mask for all blob regions
            blob_mask = np.zeros(self.current_display_image.data.shape, dtype=bool)

            # Draw each blob region
            blob_count = 0
            for i in range(info_wave.data.shape[0]):
                x_coord = info_wave.data[i, 0]
                y_coord = info_wave.data[i, 1]
                radius = info_wave.data[i, 2]

                # Create circular mask for this blob
                y_coords, x_coords = np.ogrid[:self.current_display_image.data.shape[0],
                                     :self.current_display_image.data.shape[1]]
                distance = np.sqrt((x_coords - x_coord) ** 2 + (y_coords - y_coord) ** 2)
                blob_region = distance <= radius

                blob_mask |= blob_region

                # Draw perimeter circle (green like Igor Pro)
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)
                blob_count += 1

            # Create red tinted overlay for blob regions
            red_overlay = np.zeros((*self.current_display_image.data.shape, 4))
            red_overlay[blob_mask] = [1, 0, 0, 0.3]  # Red with transparency

            # Apply the overlay
            self.ax.imshow(red_overlay, aspect='equal', alpha=0.5)

            self.log_message(f"Displaying {blob_count} detected blobs with region overlay")

        except Exception as e:
            self.log_message(f"Error adding blob overlay: {str(e)}")

    def toggle_blob_display(self):
        """FIXED: Toggle blob overlay display"""
        self.show_blobs = self.blob_toggle_var.get()
        self.log_message(f"Blob display: {'ON' if self.show_blobs else 'OFF'}")
        if self.current_display_image is not None:
            self.display_image()

    def update_info_display(self):
        """Update the results info display"""
        self.info_text.delete(1.0, tk.END)

        def format_scientific(value):
            """Format like Igor Pro - scientific notation for very small/large numbers"""
            if abs(value) < 1e-3 or abs(value) > 1e6:
                return f"{value:.3e}"
            else:
                return f"{value:.6f}"

        if self.current_display_image:
            info = f"Image: {self.current_display_image.name}\n"
            info += f"Size: {self.current_display_image.data.shape}\n"
            info += f"Data type: {self.current_display_image.data.dtype}\n"
            info += f"Range: [{format_scientific(np.min(self.current_display_image.data))}, {format_scientific(np.max(self.current_display_image.data))}]\n\n"

            if self.current_display_results:
                info += f"Analysis Results:\n"
                info += f"Particles found: {self.current_display_results['num_particles']}\n"
                info += f"Threshold used: {format_scientific(self.current_display_results['threshold'])}\n"

                if self.current_display_results['num_particles'] > 0:
                    sizes = self.current_display_results['info'].data[:, 2]  # Radii
                    info += f"Size range: [{format_scientific(np.min(sizes))}, {format_scientific(np.max(sizes))}]\n"
                    info += f"Mean size: {format_scientific(np.mean(sizes))}\n"

            self.info_text.insert(1.0, info)

    def run_single_analysis(self):
        """Run blob detection on selected image - FIXED: Proper parameter passing"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            self.log_message(f"Starting blob detection on {self.current_display_image.name}...")

            # Run analysis - this will prompt for parameters
            results = FindHessianBlobs(self.current_display_image)

            if results:
                # Store results
                self.current_results[self.current_display_image.name] = results
                self.current_display_results = results

                # Enable blob toggle
                self.blob_toggle.configure(state=tk.NORMAL)

                self.log_message(f"Detection complete. Found {results['num_particles']} particles.")

                # Update display
                self.update_info_display()
                self.display_image()

            else:
                self.log_message("Detection cancelled or failed.")

        except Exception as e:
            self.log_message(f"Error in analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {str(e)}")

    def run_batch_analysis(self):
        """Run batch analysis on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load some images first.")
            return

        try:
            self.log_message("Starting batch analysis...")

            # Run batch analysis
            batch_results = BatchHessianBlobs(self.current_images)

            if batch_results:
                # Store all results
                self.current_results.update(batch_results)

                # Update display if current image has results
                if (self.current_display_image and
                        self.current_display_image.name in batch_results):
                    self.current_display_results = batch_results[self.current_display_image.name]
                    self.blob_toggle.configure(state=tk.NORMAL)
                    self.update_info_display()
                    self.display_image()

                self.log_message(f"Batch analysis complete. Processed {len(batch_results)} images.")
            else:
                self.log_message("Batch analysis failed or was cancelled.")

        except Exception as e:
            self.log_message(f"Error in batch analysis: {str(e)}")
            messagebox.showerror("Batch Analysis Error", f"An error occurred: {str(e)}")

    # FIXED: View particles function (matching Igor Pro)
    def view_particles(self):
        """Launch particle viewer (matching Igor Pro ViewParticles)"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if self.current_display_results is None:
            messagebox.showwarning("No Results", "Please run blob detection first.")
            return

        try:
            # Launch particle viewer
            ViewParticles(self.current_display_image,
                          self.current_display_results['info'])
            self.log_message("Particle viewer opened.")
        except Exception as e:
            self.log_message(f"Error opening particle viewer: {str(e)}")
            messagebox.showerror("Viewer Error", f"Error opening particle viewer: {str(e)}")

    def zoom_fit(self):
        """Fit image to display area"""
        if self.current_display_image is not None:
            self.ax.set_xlim(0, self.current_display_image.data.shape[1])
            self.ax.set_ylim(self.current_display_image.data.shape[0], 0)
            self.canvas.draw()

    def export_results(self):
        """Export analysis results to file"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return

        filename = filedialog.asksavename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                ExportResults(self.current_results, filename)
                self.log_message(f"Results exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite

Python Port of Igor Pro Implementation

G.M. King Laboratory - University of Missouri-Columbia
Original Igor Pro code by: Brendan Marsh - marshbp@stanford.edu

This software implements the Hessian blob algorithm for precise 
particle detection in atomic force microscopy images and other 
scientific imaging applications.

Key Features:
• Scale-space blob detection
• Interactive threshold selection  
• Batch processing capabilities
• Multiple file format support
• 1-to-1 port maintaining Igor Pro functionality

For more information, see the included documentation."""

        messagebox.showinfo("About", about_text)


def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = HessianBlobGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Application Error", f"An error occurred: {e}")


if __name__ == "__main__":
    main()