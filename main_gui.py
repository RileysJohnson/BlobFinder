#!/usr/bin/env python3
"""
Hessian Blob Particle Detection Suite - Main GUI
Complete 1-1 port from Igor Pro implementation
FIXED: Simplified layout with working blob toggle and image selection

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

    def setup_ui(self):
        """Setup the main user interface - SIMPLIFIED LAYOUT"""
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

        ttk.Button(file_frame, text="Load Image",
                   command=self.load_image, width=20).pack(pady=2)
        ttk.Button(file_frame, text="Load Folder",
                   command=self.load_folder, width=20).pack(pady=2)

        # Loaded images list
        list_frame = ttk.LabelFrame(left_frame, text="Loaded Images", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create listbox with scrollbar - FIXED
        listbox_frame = ttk.Frame(list_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.images_listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.images_listbox.yview)
        self.images_listbox.configure(yscrollcommand=scrollbar.set)

        self.images_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # FIXED: Proper event binding
        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Analysis controls section
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis Controls", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Blob Detection",
                   command=self.run_single_analysis, width=20).pack(pady=2)
        ttk.Button(analysis_frame, text="Batch Analyze All",
                   command=self.run_batch_analysis, width=20).pack(pady=2)
        ttk.Button(analysis_frame, text="View Particles",
                   command=self.view_particles, width=20).pack(pady=2)

        # Results section
        results_frame = ttk.LabelFrame(left_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(results_frame, text="Show Statistics",
                   command=self.show_statistics, width=20).pack(pady=2)
        ttk.Button(results_frame, text="Export Results",
                   command=self.export_results, width=20).pack(pady=2)

        # === RIGHT PANEL ===

        # Image display area with blob toggle
        image_container = ttk.Frame(right_frame)
        image_container.pack(fill=tk.BOTH, expand=True)

        # FIXED: Blob toggle in top right corner of image area
        toggle_frame = ttk.Frame(image_container)
        toggle_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(toggle_frame, text="").pack(side=tk.LEFT, expand=True)  # Spacer

        self.blob_toggle_var = tk.BooleanVar()
        self.blob_toggle = ttk.Checkbutton(toggle_frame,
                                           text="Show Detected Blobs",
                                           variable=self.blob_toggle_var,
                                           command=self.toggle_blob_display,
                                           state=tk.DISABLED)
        self.blob_toggle.pack(side=tk.RIGHT)

        # Create matplotlib figure - FIXED: Single persistent figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, image_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log section at bottom
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.X, pady=(5, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

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
        analysis_menu.add_command(label="Run Blob Detection", command=self.run_single_analysis)
        analysis_menu.add_command(label="Batch Process All", command=self.run_batch_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocessing", command=self.run_preprocessing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Particles", command=self.view_particles)
        view_menu.add_command(label="Show Statistics", command=self.show_statistics)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def log_message(self, message):
        """Add a message to the log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def load_image(self):
        """Load a single image file"""
        file_types = [
            ("All Supported", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.ibw"),
            ("TIFF files", "*.tif *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("Igor Binary Wave", "*.ibw"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=file_types
        )

        if filename:
            try:
                wave = LoadWave(filename)
                if wave is not None:
                    self.current_images[wave.name] = wave
                    self.update_image_list()
                    self.log_message(f"Loaded image: {wave.name}")
                    # Auto-select the newly loaded image
                    self.select_image_by_name(wave.name)
                else:
                    self.log_message(f"Failed to load: {filename}")
            except Exception as e:
                self.log_message(f"Error loading {filename}: {str(e)}")
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def load_folder(self):
        """Load all images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            supported_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.ibw']
            loaded_count = 0

            try:
                for file_path in Path(folder_path).iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                        try:
                            wave = LoadWave(str(file_path))
                            if wave is not None:
                                self.current_images[wave.name] = wave
                                loaded_count += 1
                                self.log_message(f"Loaded: {wave.name}")
                        except Exception as e:
                            self.log_message(f"Error loading {file_path.name}: {str(e)}")

                self.update_image_list()
                self.log_message(f"Loaded {loaded_count} images from folder")

                if loaded_count > 0:
                    # Auto-select first image
                    first_name = list(self.current_images.keys())[0]
                    self.select_image_by_name(first_name)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load folder: {str(e)}")

    def update_image_list(self):
        """Update the images listbox"""
        self.images_listbox.delete(0, tk.END)
        for name in sorted(self.current_images.keys()):
            self.images_listbox.insert(tk.END, name)

    def select_image_by_name(self, image_name):
        """Select an image by name in the listbox"""
        try:
            items = list(self.images_listbox.get(0, tk.END))
            if image_name in items:
                index = items.index(image_name)
                self.images_listbox.selection_clear(0, tk.END)
                self.images_listbox.selection_set(index)
                self.images_listbox.see(index)
                self.on_image_select(None)  # Trigger the selection event
        except Exception as e:
            self.log_message(f"Error selecting image {image_name}: {str(e)}")

    def on_image_select(self, event):
        """Handle image selection from listbox - FIXED"""
        try:
            selection = self.images_listbox.curselection()
            if selection:
                index = selection[0]
                image_name = self.images_listbox.get(index)
                if image_name in self.current_images:
                    self.current_display_image = self.current_images[image_name]
                    self.current_display_results = self.current_results.get(image_name, None)
                    self.display_image()
                    self.log_message(f"Selected image: {image_name}")

                    # Enable/disable blob toggle based on whether results exist
                    if self.current_display_results is not None:
                        self.blob_toggle.configure(state=tk.NORMAL)
                    else:
                        self.blob_toggle.configure(state=tk.DISABLED)
                        self.blob_toggle_var.set(False)
                        self.show_blobs = False
        except Exception as e:
            self.log_message(f"Error in image selection: {str(e)}")

    def display_image(self):
        """Display the current image - FIXED: Proper matplotlib handling"""
        if self.current_display_image is None:
            return

        try:
            # Clear the axes
            self.ax.clear()

            # Get image dimensions and scaling
            height, width = self.current_display_image.data.shape
            x_min = DimOffset(self.current_display_image, 1)
            x_max = x_min + width * DimDelta(self.current_display_image, 1)
            y_min = DimOffset(self.current_display_image, 0)
            y_max = y_min + height * DimDelta(self.current_display_image, 0)

            # Display image with proper scaling
            im = self.ax.imshow(self.current_display_image.data,
                                extent=[x_min, x_max, y_max, y_min],
                                cmap='gray', aspect='equal', origin='upper')

            # FIXED: Show blobs if toggle is enabled and results exist
            if self.show_blobs and self.current_display_results is not None:
                self.add_blob_overlay()

            # Set labels and title
            self.ax.set_xlabel(f"X ({DimUnits(self.current_display_image, 1)})")
            self.ax.set_ylabel(f"Y ({DimUnits(self.current_display_image, 0)})")
            self.ax.set_title(self.current_display_image.name)

            # Update the canvas
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying image: {str(e)}")

    def add_blob_overlay(self):
        """Add red circle overlay for detected blobs - FIXED"""
        try:
            if (self.current_display_results is None or
                    'info' not in self.current_display_results or
                    self.current_display_results['info'] is None):
                return

            info_wave = self.current_display_results['info']
            if info_wave.data.shape[0] == 0:
                return

            # Draw red circles for each detected blob
            for i in range(info_wave.data.shape[0]):
                x_coord = info_wave.data[i, 0]
                y_coord = info_wave.data[i, 1]
                radius = info_wave.data[i, 2]

                # Create red circle overlay (matches Igor Pro)
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='red', linewidth=2)
                self.ax.add_patch(circle)

            self.log_message(f"Showing {info_wave.data.shape[0]} detected blobs")

        except Exception as e:
            self.log_message(f"Error adding blob overlay: {str(e)}")

    def toggle_blob_display(self):
        """Toggle blob overlay display - FIXED"""
        self.show_blobs = self.blob_toggle_var.get()
        if self.current_display_image is not None:
            self.display_image()

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
                self.display_image()

            else:
                self.log_message("Detection cancelled or failed.")

        except Exception as e:
            self.log_message(f"Error in blob detection: {str(e)}")
            messagebox.showerror("Error", f"Error in blob detection: {str(e)}")

    def run_batch_analysis(self):
        """Run blob detection on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        if not messagebox.askyesno("Batch Analysis",
                                   f"Run blob detection on all {len(self.current_images)} images?"):
            return

        try:
            # Get parameters once for all images
            params = GetBlobDetectionParams()
            if params is None:
                return  # User cancelled

            processed = 0
            for image_name, wave in self.current_images.items():
                self.log_message(f"Processing {image_name}...")
                self.root.update_idletasks()

                results = FindHessianBlobs(wave, params)
                if results:
                    self.current_results[image_name] = results
                    processed += 1
                    self.log_message(f"  Found {results['num_particles']} particles")

            self.log_message(f"Batch analysis complete. Processed {processed} images.")

            # Update display if current image was processed
            if (self.current_display_image and
                    self.current_display_image.name in self.current_results):
                self.current_display_results = self.current_results[self.current_display_image.name]
                self.blob_toggle.configure(state=tk.NORMAL)
                self.display_image()

        except Exception as e:
            self.log_message(f"Error in batch analysis: {str(e)}")
            messagebox.showerror("Error", f"Error in batch analysis: {str(e)}")

    def view_particles(self):
        """Open particle viewer window"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        if (self.current_display_results is None or
                'info' not in self.current_display_results):
            messagebox.showwarning("No Results", "Please run blob detection first.")
            return

        try:
            ViewParticles(self.current_display_image,
                          self.current_display_results['info'])
        except Exception as e:
            self.log_message(f"Error opening particle viewer: {str(e)}")
            messagebox.showerror("Error", f"Error opening particle viewer: {str(e)}")

    def run_preprocessing(self):
        """Run preprocessing on selected image"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            BatchPreprocess()
        except Exception as e:
            self.log_message(f"Error in preprocessing: {str(e)}")

    def show_statistics(self):
        """Show image statistics"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        stats = ImageStats(self.current_display_image, quiet=False)

        stats_window = tk.Toplevel(self.root)
        stats_window.title("Image Statistics")
        stats_window.geometry("500x400")

        text_widget = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget.insert(tk.END, f"Statistics for {self.current_display_image.name}\n")
        text_widget.insert(tk.END, "=" * 40 + "\n\n")
        text_widget.insert(tk.END, f"Minimum: {stats['min']:.6f}\n")
        text_widget.insert(tk.END, f"Maximum: {stats['max']:.6f}\n")
        text_widget.insert(tk.END, f"Mean: {stats['mean']:.6f}\n")
        text_widget.insert(tk.END, f"Standard Deviation: {stats['std']:.6f}\n")
        text_widget.insert(tk.END, f"Sum: {stats['sum']:.6f}\n")
        text_widget.insert(tk.END, f"Number of Points: {stats['numPoints']}\n")
        text_widget.insert(tk.END, f"Min Location: {stats['minLoc']}\n")
        text_widget.insert(tk.END, f"Max Location: {stats['maxLoc']}\n")

    def export_results(self):
        """Export analysis results"""
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