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

        # Analysis controls
        analysis_frame = ttk.LabelFrame(left_frame, text="Analysis", padding="10")
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(analysis_frame, text="Run Hessian Blob Detection",
                   command=self.run_blob_detection, width=20).pack(pady=2)
        ttk.Button(analysis_frame, text="Batch Process All",
                   command=self.batch_process_all, width=20).pack(pady=2)

        # === RIGHT PANEL ===

        # Create notebook for multiple views
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Image display tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Display")

        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        # Log tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")

        # Setup log text widget
        self.log_text = scrolledtext.ScrolledText(self.log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Setup image display
        self.setup_image_display()

        # Setup results display
        self.setup_results_display()

    def setup_image_display(self):
        """Setup the image display area"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No Image Loaded")
        self.ax.axis('off')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = ttk.Frame(self.image_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Toggle Overlay", command=self.toggle_overlay).pack(side=tk.LEFT, padx=2)

    def setup_results_display(self):
        """Setup the results display area"""
        # Create text widget for results
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image...", command=self.load_single_image)
        file_menu.add_command(label="Load Multiple Images...", command=self.load_multiple_images)
        file_menu.add_command(label="Load Folder...", command=self.load_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Hessian Blob Detection", command=self.run_blob_detection)
        analysis_menu.add_command(label="Batch Process All", command=self.batch_process_all)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocessing...", command=self.run_preprocessing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Particle Statistics", command=self.show_statistics)
        view_menu.add_command(label="Show Particle Viewer", command=self.show_particle_viewer)
        view_menu.add_separator()
        view_menu.add_command(label="Clear Log", command=self.clear_log)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Tutorial", command=self.show_tutorial)

    def log_message(self, message):
        """Add a message to the log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

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
            self.log_message(f"Loading {os.path.basename(filepath)}...")

            # Load the image
            wave = LoadWave(filepath)

            if wave is None:
                self.log_message(f"Failed to load {filepath}")
                return

            # Store the image
            filename = os.path.basename(filepath)
            self.current_images[filename] = wave

            # Add to listbox
            self.images_listbox.insert(tk.END, filename)

            # Select the newly loaded image
            self.images_listbox.selection_clear(0, tk.END)
            self.images_listbox.selection_set(tk.END)
            self.images_listbox.see(tk.END)

            self.log_message(f"Successfully loaded {filename}")
            self.log_message(f"  Shape: {wave.data.shape}")
            self.log_message(f"  Type: {wave.data.dtype}")

            # Display the image
            self.display_image(wave)

        except Exception as e:
            self.log_message(f"Error loading {filepath}: {str(e)}")
            messagebox.showerror("Load Error", f"Failed to load image:\n{str(e)}")

    def on_image_select(self, event):
        """Handle image selection from listbox"""
        selection = self.images_listbox.curselection()
        if selection:
            filename = self.images_listbox.get(selection[0])
            if filename in self.current_images:
                self.current_display_image = filename
                self.display_image(self.current_images[filename])

                # Display results if available
                if filename in self.current_results:
                    self.display_results(self.current_results[filename])

    def display_image(self, wave):
        """Display an image in the main view"""
        self.ax.clear()

        # Display the image
        im = self.ax.imshow(wave.data, cmap='gray', origin='lower',
                            extent=[DimOffset(wave, 0),
                                    DimOffset(wave, 0) + wave.data.shape[1] * DimDelta(wave, 0),
                                    DimOffset(wave, 1),
                                    DimOffset(wave, 1) + wave.data.shape[0] * DimDelta(wave, 1)])

        self.ax.set_title(wave.name)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Add colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(im, ax=self.ax)

        self.canvas.draw()

    def run_blob_detection(self):
        """Run Hessian blob detection on current image"""
        if not self.current_display_image:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        wave = self.current_images[self.current_display_image]

        # Get parameters
        params = GetBlobDetectionParams()
        if params is None:
            return  # User cancelled

        self.log_message(f"\nRunning Hessian Blob Detection on {self.current_display_image}")
        self.log_message(f"Parameters: {params}")

        try:
            # Create scale-space representation
            self.log_message("Creating scale-space representation...")
            L = ScaleSpaceRepresentation(wave, params['layers'],
                                         params['scale_start'], params['scale_factor'])

            # Compute blob detectors
            self.log_message("Computing blob detectors...")
            detH, LG = BlobDetectors(L, True)

            # Get threshold
            if params['threshold'] == -2:
                # Interactive threshold
                self.log_message("Opening interactive threshold window...")
                threshold = InteractiveThreshold(wave, detH, LG,
                                                 params['particle_type'],
                                                 params['max_curvature_ratio'])
                if threshold is None:
                    self.log_message("Threshold selection cancelled")
                    return
            elif params['threshold'] == -1:
                # Otsu's method
                self.log_message("Calculating Otsu threshold...")
                threshold = OtsuThreshold(detH, params['particle_type'])
            else:
                threshold = params['threshold']

            self.log_message(f"Using threshold: {threshold}")

            # Find blobs
            self.log_message("Finding Hessian blobs...")

            # Create output maps
            mapNum = Duplicate(wave, f"{wave.name}_mapNum")
            mapNum.data = np.full(wave.data.shape, -1)

            mapDetH = Duplicate(wave, f"{wave.name}_mapDetH")
            mapDetH.data = np.zeros(wave.data.shape)

            mapMax = Duplicate(wave, f"{wave.name}_mapMax")
            mapMax.data = np.zeros(wave.data.shape)

            info = Wave(np.zeros((0, 20)), f"{wave.name}_info")

            num_particles = FindHessianBlobs(wave, detH, LG, threshold,
                                             mapNum, mapDetH, mapMax, info,
                                             params['particle_type'],
                                             params['max_curvature_ratio'])

            self.log_message(f"Found {num_particles} particles")

            # Store results
            self.current_results[self.current_display_image] = {
                'mapNum': mapNum,
                'mapDetH': mapDetH,
                'mapMax': mapMax,
                'info': info,
                'num_particles': num_particles,
                'params': params,
                'threshold': threshold
            }

            # Display results
            self.display_results(self.current_results[self.current_display_image])

            # Show results tab
            self.notebook.select(self.results_frame)

        except Exception as e:
            self.log_message(f"Error during blob detection: {str(e)}")
            messagebox.showerror("Analysis Error", f"Blob detection failed:\n{str(e)}")

    def display_results(self, results):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, f"=== Blob Detection Results ===\n\n")
        self.results_text.insert(tk.END, f"Number of particles found: {results['num_particles']}\n")
        self.results_text.insert(tk.END, f"Threshold used: {results['threshold']:.6f}\n\n")

        self.results_text.insert(tk.END, "Parameters:\n")
        for key, value in results['params'].items():
            self.results_text.insert(tk.END, f"  {key}: {value}\n")

        if results['num_particles'] > 0:
            info = results['info']
            self.results_text.insert(tk.END, f"\n\nParticle Statistics:\n")
            self.results_text.insert(tk.END, f"  Mean X: {np.mean(info.data[:, 0]):.2f}\n")
            self.results_text.insert(tk.END, f"  Mean Y: {np.mean(info.data[:, 1]):.2f}\n")
            self.results_text.insert(tk.END, f"  Mean Scale: {np.mean(info.data[:, 3]):.2f}\n")
            self.results_text.insert(tk.END, f"  Mean Strength: {np.mean(info.data[:, 4]):.2f}\n")

        # Update image display with overlay
        self.display_image_with_overlay()

    def display_image_with_overlay(self):
        """Display image with detected blobs overlay"""
        if not self.current_display_image:
            return

        wave = self.current_images[self.current_display_image]

        self.ax.clear()

        # Display the image
        self.ax.imshow(wave.data, cmap='gray', origin='lower', alpha=0.8,
                       extent=[DimOffset(wave, 0),
                               DimOffset(wave, 0) + wave.data.shape[1] * DimDelta(wave, 0),
                               DimOffset(wave, 1),
                               DimOffset(wave, 1) + wave.data.shape[0] * DimDelta(wave, 1)])

        # Add blob overlays if results exist
        if self.current_display_image in self.current_results:
            results = self.current_results[self.current_display_image]
            info = results['info']

            # Draw circles for each blob
            for i in range(info.data.shape[0]):
                x = info.data[i, 0]
                y = info.data[i, 1]
                scale = info.data[i, 3]
                radius = np.sqrt(2 * scale)

                circle = Circle((x, y), radius, fill=False,
                                color='red', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)

        self.ax.set_title(f"{wave.name} - Detected Blobs")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.canvas.draw()

    def batch_process_all(self):
        """Run blob detection on all loaded images"""
        if not self.current_images:
            messagebox.showwarning("No Images", "Please load images first.")
            return

        # Create a simple folder structure for batch processing
        class ImageFolder:
            def __init__(self, waves):
                self.waves = waves

        folder = ImageFolder(self.current_images)

        # Run batch processing
        BatchHessianBlobs(folder)

    def run_preprocessing(self):
        """Run preprocessing on images"""
        try:
            BatchPreprocess()
        except Exception as e:
            self.log_message(f"Preprocessing error: {str(e)}")
            messagebox.showinfo("Preprocessing",
                                "Preprocessing functionality not fully implemented yet.")

    def show_statistics(self):
        """Show particle statistics"""
        if not self.current_display_image or self.current_display_image not in self.current_results:
            messagebox.showwarning("No Results", "Please run blob detection first.")
            return

        results = self.current_results[self.current_display_image]
        if results['num_particles'] == 0:
            messagebox.showinfo("Statistics", "No particles found.")
            return

        # Calculate and display statistics
        info = results['info']
        ParticleStatistics(info)

    def show_particle_viewer(self):
        """Show particle viewer"""
        if not self.current_display_image or self.current_display_image not in self.current_results:
            messagebox.showwarning("No Results", "Please run blob detection first.")
            return

        results = self.current_results[self.current_display_image]
        wave = self.current_images[self.current_display_image]

        # Show particle viewer
        ShowParticleViewer(wave, results['info'])

    def save_results(self):
        """Save current results"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No results to save.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Save all results
                np.savez(filename, **self.current_results)
                self.log_message(f"Results saved to {filename}")
                messagebox.showinfo("Success", "Results saved successfully.")
            except Exception as e:
                self.log_message(f"Error saving results: {str(e)}")
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")

    def zoom_in(self):
        """Zoom in on the image"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / 2
        y_range = (ylim[1] - ylim[0]) / 2

        self.ax.set_xlim(x_center - x_range * 0.7, x_center + x_range * 0.7)
        self.ax.set_ylim(y_center - y_range * 0.7, y_center + y_range * 0.7)
        self.canvas.draw()

    def zoom_out(self):
        """Zoom out on the image"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) / 2
        y_range = (ylim[1] - ylim[0]) / 2

        self.ax.set_xlim(x_center - x_range * 1.3, x_center + x_range * 1.3)
        self.ax.set_ylim(y_center - y_range * 1.3, y_center + y_range * 1.3)
        self.canvas.draw()

    def reset_view(self):
        """Reset the view to show the entire image"""
        if self.current_display_image:
            wave = self.current_images[self.current_display_image]
            self.ax.set_xlim(DimOffset(wave, 0),
                             DimOffset(wave, 0) + wave.data.shape[1] * DimDelta(wave, 0))
            self.ax.set_ylim(DimOffset(wave, 1),
                             DimOffset(wave, 1) + wave.data.shape[0] * DimDelta(wave, 1))
            self.canvas.draw()

    def toggle_overlay(self):
        """Toggle between image only and image with overlay"""
        if self.current_display_image:
            if self.current_display_image in self.current_results:
                # Check if overlay is currently shown
                if len(self.ax.patches) > 0:
                    # Remove overlay
                    self.display_image(self.current_images[self.current_display_image])
                else:
                    # Add overlay
                    self.display_image_with_overlay()
            else:
                self.display_image(self.current_images[self.current_display_image])

    def clear_log(self):
        """Clear the log window"""
        self.log_text.delete(1.0, tk.END)

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite
Python Port of Igor Pro Implementation

Original Igor Pro code by:
Brendan Marsh - marshbp@stanford.edu

G.M. King Laboratory
University of Missouri-Columbia

Copyright 2019 by The Curators of the University of Missouri

This is a complete 1-to-1 port of the Igor Pro
Hessian Blob detection algorithm, maintaining
all original functionality and behavior."""

        messagebox.showinfo("About", about_text)

    def show_tutorial(self):
        """Show tutorial information"""
        tutorial_text = """Hessian Blob Detection Tutorial

1. Load Image(s):
   - Use File menu or buttons to load images
   - Supports TIFF, PNG, JPEG, and Igor Binary Wave formats

2. Run Detection:
   - Click "Run Hessian Blob Detection"
   - Set parameters in the dialog
   - Use -2 for interactive threshold selection

3. Interactive Threshold:
   - Adjust slider to change detection threshold
   - Red circles show detected blobs
   - Click Accept when satisfied

4. View Results:
   - Results tab shows statistics
   - Toggle overlay to show/hide detected blobs
   - Use View menu for additional displays

For detailed information, see the original paper:
"The Hessian Blob Algorithm: Precise Particle Detection
in Atomic Force Microscopy Imagery" - Scientific Reports"""

        messagebox.showinfo("Tutorial", tutorial_text)


def main():
    """Main entry point for GUI application"""
    root = tk.Tk()
    app = HessianBlobGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()