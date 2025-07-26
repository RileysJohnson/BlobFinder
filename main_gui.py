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
        self.root.geometry("1400x900")  # FIXED: Increased size for better visibility

        # Current state
        self.current_images = {}  # Dict of filename -> Wave
        self.current_results = {}  # Dict of analysis results
        self.current_display_image = None
        self.current_display_results = None
        self.current_figure = None
        self.current_canvas = None
        self.current_colorbar = None
        self.show_blobs_var = tk.BooleanVar(value=True)

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
        left_frame = ttk.Frame(main_paned, width=400)  # FIXED: Increased width
        main_paned.add(left_frame, weight=0)

        # Right panel for image display
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # === LEFT PANEL ===

        # File management section - FIXED: Only "Load Image" and "Load Folder"
        file_frame = ttk.LabelFrame(left_frame, text="File Management", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Image",
                   command=self.load_image, width=25).pack(pady=2)  # FIXED: Increased width
        ttk.Button(file_frame, text="Load Folder",
                   command=self.load_folder, width=25).pack(pady=2)  # FIXED: Increased width

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

        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Hessian Blob Detection",
                   command=self.run_blob_detection, width=30).pack(pady=2)  # FIXED: Increased width
        ttk.Button(analysis_frame, text="Run Preprocessing",
                   command=self.run_preprocessing, width=30).pack(pady=2)  # FIXED: Increased width

        # === RIGHT PANEL ===

        # Image display area
        image_display_frame = ttk.Frame(right_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True)

        # Add a checkbox for toggling blob visibility
        toggle_blobs_check = ttk.Checkbutton(
            image_display_frame,
            text="Show Detected Blobs",
            variable=self.show_blobs_var,
            command=self.on_toggle_blobs
        )
        toggle_blobs_check.pack(anchor=tk.NE, padx=5, pady=2)

        self.image_frame = ttk.Frame(image_display_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Status and log
        status_frame = ttk.LabelFrame(right_frame, text="Status Log", padding="5")
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.log_text = scrolledtext.ScrolledText(status_frame, height=8, wrap=tk.WORD)
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
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Hessian Blob Detection", command=self.run_blob_detection)
        analysis_menu.add_command(label="View Particles", command=self.view_particles)
        analysis_menu.add_command(label="Preprocessing", command=self.run_preprocessing)
        analysis_menu.add_command(label="Image Statistics", command=self.show_statistics)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Check Dependencies", command=self.check_dependencies)

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
        """Load a single image file - handles both single and multiple selection"""
        filetypes = [
            ("All supported", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp *.ibw *.pxp"),
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
            ("Igor files", "*.ibw *.pxp"),
            ("All files", "*.*")
        ]

        # Allow multiple selection since user might want to select multiple images at once
        filenames = filedialog.askopenfilenames(
            title="Select Image(s)",
            filetypes=filetypes
        )

        if filenames:
            for filename in filenames:
                self.load_single_file(filename)

    def load_folder(self):
        """Load all supported images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return

        supported_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.ibw', '.pxp'}
        loaded_count = 0

        try:
            for file_path in Path(folder_path).iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    if self.load_single_file(str(file_path)):
                        loaded_count += 1

            if loaded_count > 0:
                self.log_message(f"Loaded {loaded_count} images from folder: {folder_path}")
            else:
                messagebox.showwarning("No Images", "No supported image files found in the selected folder.")

        except Exception as e:
            self.log_message(f"Error loading folder: {str(e)}")
            messagebox.showerror("Error", f"Error loading folder: {str(e)}")

    def load_single_file(self, filename):
        """Load a single file and add to the list"""
        try:
            self.log_message(f"Loading: {filename}")

            # Use the file_io module to load the image
            image_wave = LoadWave(filename)

            if image_wave is not None:
                # Store the image
                key = Path(filename).name
                self.current_images[key] = image_wave

                # Add to listbox
                self.images_listbox.insert(tk.END, key)

                self.log_message(f"Successfully loaded: {key}")
                return True
            else:
                self.log_message(f"Failed to load: {filename}")
                return False

        except Exception as e:
            self.log_message(f"Error loading {filename}: {str(e)}")
            messagebox.showerror("Error", f"Error loading {filename}: {str(e)}")
            return False

    def on_toggle_blobs(self):
        """Handle toggling of blob visibility."""
        if self.current_canvas:
            self.display_results_overlay()

    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.images_listbox.curselection()
        if not selection:
            return

        selected_name = self.images_listbox.get(selection[0])
        self.current_display_image = self.current_images.get(selected_name)

        if self.current_display_image:
            self.log_message(f"Selected image: {selected_name}")
            self.current_display_results = self.current_results.get(selected_name)
            self.display_image(self.current_display_image)

    def display_image(self, image_wave):
        """
        Display an image, and if results are available, overlay them.
        """
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            plt.close(self.current_figure)

        self.current_figure = Figure(figsize=(8, 6), dpi=100)
        ax = self.current_figure.add_subplot(111)

        extent = [
            DimOffset(image_wave, 0),
            DimOffset(image_wave, 0) + DimSize(image_wave, 0) * DimDelta(image_wave, 0),
            DimOffset(image_wave, 1),
            DimOffset(image_wave, 1) + DimSize(image_wave, 1) * DimDelta(image_wave, 1)
        ]

        im = ax.imshow(image_wave.data, cmap='gray', origin='lower', extent=extent, aspect='equal')
        ax.set_title(image_wave.name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if self.current_colorbar:
            self.current_colorbar.remove()
        self.current_colorbar = self.current_figure.colorbar(im, ax=ax)

        self.current_canvas = FigureCanvasTkAgg(self.current_figure, self.image_frame)
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.display_results_overlay()
        self.current_canvas.draw()

    def display_results_overlay(self):
        """Display or hide blob detection results based on the toggle."""
        if not self.current_figure:
            return

        ax = self.current_figure.axes[0]
        # Remove only circles previously added by this function
        for patch in self.get_blob_patches(ax):
            patch.remove()

        show_blobs = self.show_blobs_var.get()
        if show_blobs and self.current_display_results:
            results = self.current_display_results
            if 'info' in results and len(results['info'].data) > 0:
                info_data = results['info'].data
                num_particles = len(info_data)

                for i in range(num_particles):
                    x_pos = info_data[i, 0]
                    y_pos = info_data[i, 1]
                    radius = info_data[i, 2] if info_data.shape[1] > 2 else 3.0
                    circle = Circle((x_pos, y_pos), radius, fill=False, color='red', linewidth=1.5,
                                    picker=True, gid=f'blob_{i}')
                    ax.add_patch(circle)

                ax.set_title(f"{self.current_display_image.name} - {num_particles} particles detected")
            else:
                ax.set_title(self.current_display_image.name)
        else:
            ax.set_title(self.current_display_image.name if self.current_display_image else "")

        self.current_canvas.draw_idle()

    def get_blob_patches(self, ax):
        """Helper to get only blob-related patches."""
        return [p for p in ax.patches if isinstance(p, Circle) and p.get_gid() and p.get_gid().startswith('blob_')]

    def run_blob_detection(self):
        """Run blob detection on selected image"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            # Get parameters from user
            params = GetBlobDetectionParams()
            if params is None:
                return  # User cancelled

            self.log_message("Starting Hessian blob detection...")
            self.log_message(f"Parameters: {params}")

            # Run detection
            results = HessianBlobDetection(
                self.current_display_image,
                scaleStart=params['scaleStart'],
                scaleLayers=params['scaleLayers'],
                scaleFactor=params['scaleFactor'],
                minResponse=params['minResponse'],
                particleType=params['particleType'],
                maxCurvatureRatio=params['maxCurvatureRatio'],
                subPixelMult=params['subPixelMult'],
                allowOverlap=params['allowOverlap']
            )

            if results is not None:
                # Store results
                selected_name = self.images_listbox.get(self.images_listbox.curselection()[0])
                self.current_results[selected_name] = results
                self.current_display_results = results

                self.log_message(f"Detection complete! Found {results['num_particles']} particles.")

                # Update display with results
                self.display_image(self.current_display_image)

            else:
                self.log_message("Detection cancelled or failed.")

        except Exception as e:
            self.log_message(f"Error in blob detection: {str(e)}")
            messagebox.showerror("Error", f"Error in blob detection: {str(e)}")

    def view_particles(self):
        """Launch the particle viewer for the current image and results."""
        if self.current_display_image is None or self.current_display_results is None:
            messagebox.showwarning("No Results", "Please run blob detection on an image first.")
            return

        try:
            # The ViewParticles function is in the particle_measurements module
            ViewParticles(self.current_display_image, self.current_display_results['info'])
        except Exception as e:
            self.log_message(f"Error opening particle viewer: {str(e)}")
            messagebox.showerror("Error", f"Could not open particle viewer: {e}")

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
        stats_window.geometry("500x400")  # FIXED: Increased size

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

    def check_dependencies(self):
        """Check and display dependency status"""
        check_window = tk.Toplevel(self.root)
        check_window.title("Dependency Check")
        check_window.geometry("600x500")  # FIXED: Increased size

        text_widget = scrolledtext.ScrolledText(check_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Redirect stdout to capture check output
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            check_file_io_dependencies()

        output = f.getvalue()
        text_widget.insert(tk.END, output)

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