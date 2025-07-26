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

        # Bind selection event
        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Analysis section
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Hessian Blob Detection",
                   command=self.run_hessian_blobs, width=25).pack(pady=2)
        ttk.Button(analysis_frame, text="View Results",
                   command=self.view_results, width=25).pack(pady=2)
        ttk.Button(analysis_frame, text="View Particles",
                   command=self.view_particles, width=25).pack(pady=2)

        # === RIGHT PANEL ===

        # Image display area
        display_frame = ttk.LabelFrame(right_frame, text="Image Display", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Display controls
        controls_frame = ttk.Frame(display_frame)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="Zoom Reset",
                   command=self.zoom_reset).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Show Original",
                   command=self.show_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Show with Blobs",
                   command=self.show_with_blobs).pack(side=tk.LEFT, padx=5)

        # === BOTTOM PANEL ===

        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Log", padding="5")
        log_frame.pack(fill=tk.X, padx=5, pady=5, side=tk.BOTTOM)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Clear initial display
        self.clear_display()

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
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Hessian Blob Detection", command=self.run_hessian_blobs)
        analysis_menu.add_command(label="Batch Processing", command=self.batch_process)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocessing", command=self.run_preprocessing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Results", command=self.view_results)
        view_menu.add_command(label="View Particles", command=self.view_particles)
        view_menu.add_command(label="Show Statistics", command=self.show_statistics)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Check Dependencies", command=self.check_dependencies)

    def log_message(self, message):
        """Add message to log area"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_display(self):
        """Clear the image display"""
        self.ax.clear()
        self.ax.set_title("No Image Loaded")
        self.ax.text(0.5, 0.5, "Load an image to begin analysis",
                     ha='center', va='center', transform=self.ax.transAxes)
        self.canvas.draw()

    def load_single_image(self):
        """Load a single image file"""
        filetypes = [
            ("All Supported", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.ibw"),
            ("TIFF files", "*.tif;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("Igor Binary Wave", "*.ibw"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )

        if filename:
            self.load_image_file(filename)

    def load_multiple_images(self):
        """Load multiple image files"""
        filetypes = [
            ("All Supported", "*.tif;*.tiff;*.png;*.jpg;*.jpeg;*.ibw"),
            ("TIFF files", "*.tif;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("Igor Binary Wave", "*.ibw"),
            ("All files", "*.*")
        ]

        filenames = filedialog.askopenfilenames(
            title="Select Image Files",
            filetypes=filetypes
        )

        for filename in filenames:
            self.load_image_file(filename)

    def load_folder(self):
        """Load all images from a folder"""
        folder = filedialog.askdirectory(title="Select Folder Containing Images")

        if folder:
            # Find all image files in folder
            extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.ibw']
            folder_path = Path(folder)

            image_files = []
            for ext in extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))

            self.log_message(f"Found {len(image_files)} image files in {folder}")

            for filepath in image_files:
                self.load_image_file(str(filepath))

    def load_image_file(self, filepath):
        """Load an individual image file"""
        try:
            # Check if already loaded
            filename = os.path.basename(filepath)
            if filename in self.current_images:
                self.log_message(f"Image {filename} already loaded, skipping...")
                return

            self.log_message(f"Loading {filename}...")

            # Load the image
            wave = LoadWave(filepath)

            if wave is None:
                self.log_message(f"Failed to load {filepath}")
                return

            # Store in our dictionary
            self.current_images[filename] = wave

            # Add to listbox
            self.images_listbox.insert(tk.END, filename)

            self.log_message(f"Successfully loaded {filename}: {wave.data.shape}")

            # Auto-select the first image
            if len(self.current_images) == 1:
                self.images_listbox.select_set(0)
                self.on_image_select(None)

        except Exception as e:
            self.log_message(f"Error loading {filepath}: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load {filepath}:\n{str(e)}")

    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.images_listbox.curselection()
        if selection:
            filename = self.images_listbox.get(selection[0])
            self.current_display_image = self.current_images[filename]
            self.display_image(self.current_display_image)

    def display_image(self, wave, title=None):
        """Display a wave as an image"""
        self.ax.clear()

        if title is None:
            title = f"Image: {wave.name}"

        # Get coordinate arrays
        height, width = wave.data.shape
        x_coords = np.arange(width) * DimDelta(wave, 0) + DimOffset(wave, 0)
        y_coords = np.arange(height) * DimDelta(wave, 1) + DimOffset(wave, 1)

        extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]

        # Display image
        im = self.ax.imshow(wave.data, extent=extent, cmap='gray', aspect='auto')
        self.ax.set_title(title)
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        # Add colorbar
        self.fig.colorbar(im, ax=self.ax)

        self.canvas.draw()

    def run_hessian_blobs(self):
        """Run Hessian blob detection on selected image"""
        if self.current_display_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            # Get parameters from user
            params = GetBlobDetectionParams()
            if params is None:
                return

            self.log_message("Starting Hessian blob detection...")

            # Run analysis
            results = RunHessianBlobs(
                self.current_display_image,
                scaleStart=params['scaleStart'],
                scaleLayers=params['scaleLayers'],
                scaleFactor=params['scaleFactor'],
                particleType=params['particleType'],
                maxCurvatureRatio=params['maxCurvatureRatio'],
                minResponse=params['minResponse'],
                interactive=True
            )

            if results is not None:
                # Store results
                image_name = self.current_display_image.name
                self.current_results[image_name] = results
                self.current_display_results = results

                self.log_message(f"Analysis complete: Found {results['num_particles']} particles")
                self.show_with_blobs()
            else:
                self.log_message("Analysis was cancelled.")

        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}")

    def view_results(self):
        """View analysis results"""
        if not self.current_results:
            messagebox.showinfo("No Results", "No analysis results available.")
            return

        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results")
        results_window.geometry("600x400")

        # Create text widget for results
        text_widget = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Display results for all analyzed images
        for image_name, results in self.current_results.items():
            text_widget.insert(tk.END, f"=== Results for {image_name} ===\n")
            text_widget.insert(tk.END, f"Number of particles: {results['num_particles']}\n")
            text_widget.insert(tk.END, f"Threshold used: {results['threshold']:.6f}\n")

            params = results['parameters']
            text_widget.insert(tk.END, f"Scale start: {params['scaleStart']}\n")
            text_widget.insert(tk.END, f"Scale layers: {params['scaleLayers']}\n")
            text_widget.insert(tk.END, f"Scale factor: {params['scaleFactor']}\n")
            text_widget.insert(tk.END, f"Particle type: {params['particleType']}\n")
            text_widget.insert(tk.END, f"Max curvature ratio: {params['maxCurvatureRatio']}\n")
            text_widget.insert(tk.END, "\n")

            if results['num_particles'] > 0:
                info = results['info']
                text_widget.insert(tk.END, "Particle Information:\n")
                text_widget.insert(tk.END, "ID\tX\tY\tScale\tDetH\tLG\n")
                for i in range(min(10, results['num_particles'])):  # Show first 10
                    text_widget.insert(tk.END,
                                       f"{i}\t{info.data[i, 0]:.2f}\t{info.data[i, 1]:.2f}\t"
                                       f"{info.data[i, 2]:.2f}\t{info.data[i, 3]:.4f}\t{info.data[i, 4]:.4f}\n")

                if results['num_particles'] > 10:
                    text_widget.insert(tk.END, f"... and {results['num_particles'] - 10} more particles\n")

            text_widget.insert(tk.END, "\n" + "=" * 50 + "\n\n")

    def view_particles(self):
        """View individual particles"""
        if self.current_display_results is None:
            messagebox.showinfo("No Results", "No analysis results available for current image.")
            return

        try:
            MeasureParticles(
                self.current_display_results['image'],
                self.current_display_results['mapNum'],
                self.current_display_results['info']
            )
        except Exception as e:
            self.log_message(f"Error viewing particles: {str(e)}")

    def show_original(self):
        """Show original image"""
        if self.current_display_image is not None:
            self.display_image(self.current_display_image)

    def show_with_blobs(self):
        """Show image with detected blobs overlaid"""
        if self.current_display_image is None or self.current_display_results is None:
            return

        self.ax.clear()

        # Display original image
        wave = self.current_display_image
        height, width = wave.data.shape
        x_coords = np.arange(width) * DimDelta(wave, 0) + DimOffset(wave, 0)
        y_coords = np.arange(height) * DimDelta(wave, 1) + DimOffset(wave, 1)
        extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]

        self.ax.imshow(wave.data, extent=extent, cmap='gray', aspect='auto')

        # Overlay detected blobs
        results = self.current_display_results
        if results['num_particles'] > 0:
            info = results['info']
            for i in range(results['num_particles']):
                x = info.data[i, 0]  # X coordinate
                y = info.data[i, 1]  # Y coordinate
                scale = info.data[i, 2]  # Scale (radius)

                circle = Circle((x, y), scale, fill=False, color='red', linewidth=1.5)
                self.ax.add_patch(circle)

        self.ax.set_title(f"Detected Blobs: {results['num_particles']} particles")
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        self.canvas.draw()

    def zoom_reset(self):
        """Reset zoom to fit image"""
        if self.current_display_image is not None:
            self.ax.set_xlim(None)
            self.ax.set_ylim(None)
            self.canvas.draw()

    def batch_process(self):
        """Run batch processing on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load some images first.")
            return

        # Get parameters
        params = GetBlobDetectionParams()
        if params is None:
            return

        self.log_message("Starting batch processing...")

        for filename, wave in self.current_images.items():
            try:
                self.log_message(f"Processing {filename}...")

                results = RunHessianBlobs(
                    wave,
                    scaleStart=params['scaleStart'],
                    scaleLayers=params['scaleLayers'],
                    scaleFactor=params['scaleFactor'],
                    particleType=params['particleType'],
                    maxCurvatureRatio=params['maxCurvatureRatio'],
                    minResponse=params['minResponse'],
                    interactive=False  # No interactive threshold for batch
                )

                if results is not None:
                    self.current_results[filename] = results
                    self.log_message(f"  Found {results['num_particles']} particles")
                else:
                    self.log_message(f"  Failed to process {filename}")

            except Exception as e:
                self.log_message(f"  Error processing {filename}: {str(e)}")

        self.log_message("Batch processing complete.")

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
        stats_window.geometry("400x300")

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
        check_window.geometry("500x400")

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
    # Create the main window
    root = tk.Tk()

    # Create the application
    app = HessianBlobGUI(root)

    # Handle window closing
    def on_closing():
        try:
            # Clean up matplotlib figures
            plt.close('all')
        except:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Run the application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user.")
    finally:
        # Cleanup
        try:
            plt.close('all')
        except:
            pass


if __name__ == "__main__":
    main()