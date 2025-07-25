"""
Main GUI Application
Provides the main interface for the Hessian Blob Detection Suite
Follows the same workflow as described in the Igor Pro tutorial
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import sys

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

        # Initialize data browser
        global data_browser
        self.data_browser = data_browser

        self.setup_ui()
        self.setup_menu()

        # Status variables
        self.current_folder = "root"
        self.loaded_images = []

        # Display welcome message
        self.log_message("Welcome to the Hessian Blob Particle Detection Suite")
        self.log_message("G.M. King Laboratory - University of Missouri-Columbia")
        self.log_message("Coded by: Brendan Marsh - Email: marshbp@stanford.edu")
        self.log_message("Python port maintains 1-1 functionality with Igor Pro version")
        self.log_message("-" * 60)

    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Single Image...", command=self.load_single_image)
        file_menu.add_command(label="Load Image Folder...", command=self.load_image_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Batch Preprocess", command=self.batch_preprocess)
        analysis_menu.add_command(label="Batch Hessian Blobs", command=self.batch_hessian_blobs)
        analysis_menu.add_command(label="Single Image Hessian Blobs", command=self.single_hessian_blobs)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Particles", command=self.view_particles)
        view_menu.add_command(label="View Statistics", command=self.view_statistics)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Tutorial", command=self.show_tutorial)

    def setup_ui(self):
        """Setup the main UI layout"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Data Browser and Controls
        left_frame = ttk.Frame(main_paned, width=300)
        main_paned.add(left_frame, weight=1)

        # Data Browser
        browser_frame = ttk.LabelFrame(left_frame, text="Data Browser", padding="5")
        browser_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Treeview for data browser
        self.tree = ttk.Treeview(browser_frame, selectmode='browse')
        tree_scroll = ttk.Scrollbar(browser_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Initialize tree with root
        self.tree.insert("", "end", iid="root", text="root", values=("folder",))
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Control buttons
        control_frame = ttk.LabelFrame(left_frame, text="Quick Actions", padding="5")
        control_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(control_frame, text="Load Single Image",
                   command=self.load_single_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Load Image Folder",
                   command=self.load_image_folder).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Create Data Folder",
                   command=self.create_data_folder).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Batch Preprocess",
                   command=self.batch_preprocess).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Batch Hessian Blobs",
                   command=self.batch_hessian_blobs).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="View Particles",
                   command=self.view_particles).pack(fill=tk.X, pady=2)

        # Testing function button (as in original Igor code)
        ttk.Button(control_frame, text="Testing Function",
                   command=self.run_testing).pack(fill=tk.X, pady=2)

        # Right panel - Display and Log
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Image display tab
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Display")

        # Create matplotlib figure for image display
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Command Log")

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log_message(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_tree(self):
        """Update the data browser tree"""

        def add_folder_to_tree(folder, parent_id, path=""):
            current_path = f"{path}:{folder.name}" if path else folder.name

            # Add waves
            for wave_name, wave in folder.waves.items():
                wave_id = f"{current_path}:{wave_name}"
                if not self.tree.exists(wave_id):
                    self.tree.insert(parent_id, "end", iid=wave_id,
                                     text=wave_name, values=("wave", str(wave.data.shape)))

            # Add subfolders
            for subfolder_name, subfolder in folder.subfolders.items():
                folder_id = f"{current_path}:{subfolder_name}"
                if not self.tree.exists(folder_id):
                    self.tree.insert(parent_id, "end", iid=folder_id,
                                     text=subfolder_name, values=("folder",))
                add_folder_to_tree(subfolder, folder_id, current_path)

        # Clear existing items except root
        for item in self.tree.get_children("root"):
            self.tree.delete(item)

        # Add all folders and waves
        add_folder_to_tree(self.data_browser, "root")

        # Expand root
        self.tree.item("root", open=True)

    def on_tree_select(self, event):
        """Handle tree selection"""
        selection = self.tree.selection()
        if selection:
            item_id = selection[0]
            self.current_folder = item_id
            SetBrowserSelection(item_id)

            # If it's a wave, try to display it
            item_type = self.tree.item(item_id, "values")[0] if self.tree.item(item_id, "values") else ""
            if item_type == "wave":
                self.display_wave(item_id)

    def display_wave(self, wave_path):
        """Display a wave in the image panel"""
        try:
            wave = self.data_browser.get_wave(wave_path.replace("root:", ""))
            if wave is not None and len(wave.data.shape) >= 2:
                self.ax.clear()

                # Handle 2D or 3D data (show first layer of 3D)
                data_to_show = wave.data[:, :, 0] if len(wave.data.shape) > 2 else wave.data

                # Set up proper extent based on wave scaling
                extent = [
                    DimOffset(wave, 0),
                    DimOffset(wave, 0) + wave.data.shape[1] * DimDelta(wave, 0),
                    DimOffset(wave, 1),
                    DimOffset(wave, 1) + wave.data.shape[0] * DimDelta(wave, 1)
                ]

                im = self.ax.imshow(data_to_show, cmap='gray', origin='lower', extent=extent)
                self.ax.set_title(f"Wave: {wave.name}")
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")

                # Add colorbar
                if hasattr(self, 'colorbar'):
                    self.colorbar.remove()
                self.colorbar = self.fig.colorbar(im, ax=self.ax)

                self.canvas.draw()

        except Exception as e:
            self.log_message(f"Error displaying wave: {str(e)}")

    def load_single_image(self):
        """Load a single image file"""
        file_path = filedialog.askopenfilename(
            title="Select Single Image File",
            filetypes=[("Igor Binary Wave", "*.ibw"), ("All Files", "*.*")]
        )

        if file_path:
            self.update_status("Loading image...")
            self.log_message(f"Loading image: {os.path.basename(file_path)}")

            # Ask for destination folder
            folder_name = self.ask_folder_name("Enter folder name for image:", "Images")
            if not folder_name:
                return

            # Create folder if it doesn't exist
            if not DataFolderExists(folder_name):
                NewDataFolder(folder_name)

            # Load image
            try:
                loaded_waves = LoadWaves([file_path], folder_name)
                self.loaded_images.extend(loaded_waves)

                self.log_message(f"Successfully loaded image to folder '{folder_name}'")
                self.update_tree()
                self.update_status("Image loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.log_message(f"Error loading image: {str(e)}")
                self.update_status("Error loading image")

    def load_image_folder(self):
        """Load all images from a folder"""
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Images"
        )

        if folder_path:
            # Find all .ibw files in the folder
            ibw_files = []
            for file in os.listdir(folder_path):
                if file.lower().endswith('.ibw'):
                    ibw_files.append(os.path.join(folder_path, file))

            if not ibw_files:
                messagebox.showwarning("Warning", "No .ibw files found in selected folder")
                return

            self.update_status("Loading images from folder...")
            self.log_message(f"Loading {len(ibw_files)} image(s) from folder...")

            # Ask for destination folder
            folder_name = self.ask_folder_name("Enter folder name for images:", "Images")
            if not folder_name:
                return

            # Create folder if it doesn't exist
            if not DataFolderExists(folder_name):
                NewDataFolder(folder_name)

            # Load images
            try:
                loaded_waves = LoadWaves(ibw_files, folder_name)
                self.loaded_images.extend(loaded_waves)

                self.log_message(f"Successfully loaded {len(loaded_waves)} images to folder '{folder_name}'")
                self.update_tree()
                self.update_status("Images loaded successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load images: {str(e)}")
                self.log_message(f"Error loading images: {str(e)}")
                self.update_status("Error loading images")

    def create_data_folder(self):
        """Create a new data folder"""
        folder_name = self.ask_folder_name("Enter new folder name:", "NewFolder")
        if folder_name:
            try:
                NewDataFolder(folder_name)
                self.update_tree()
                self.log_message(f"Created data folder: {folder_name}")
                self.update_status(f"Created folder: {folder_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create folder: {str(e)}")

    def ask_folder_name(self, prompt, default=""):
        """Ask user for folder name"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Folder Name")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt).pack(pady=10)

        entry_var = tk.StringVar(value=default)
        entry = tk.Entry(dialog, textvariable=entry_var, width=30)
        entry.pack(pady=5)
        entry.focus_set()
        entry.select_range(0, tk.END)

        result = {"name": None}

        def ok():
            result["name"] = entry_var.get().strip()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="OK", command=ok).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        # Bind Enter key
        dialog.bind('<Return>', lambda e: ok())

        dialog.wait_window()
        return result["name"]

    def batch_preprocess(self):
        """Run batch preprocessing"""
        if not self.current_folder or self.current_folder == "root":
            messagebox.showwarning("Warning", "Please select a folder containing images")
            return

        self.update_status("Running batch preprocessing...")
        self.log_message("Starting batch preprocessing...")

        try:
            # Set the selected folder as current
            SetBrowserSelection(self.current_folder)

            # Run preprocessing
            result = BatchPreprocess()

            if result == 0:
                self.log_message("Batch preprocessing completed successfully")
                self.update_status("Preprocessing completed")
                self.update_tree()
            else:
                self.log_message("Batch preprocessing was cancelled or failed")
                self.update_status("Preprocessing cancelled")

        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.log_message(f"Preprocessing error: {str(e)}")
            self.update_status("Preprocessing failed")

    def batch_hessian_blobs(self):
        """Run batch Hessian blob detection"""
        if not self.current_folder or self.current_folder == "root":
            messagebox.showwarning("Warning", "Please select a folder containing images")
            return

        self.update_status("Running Hessian blob detection...")
        self.log_message("Starting batch Hessian blob detection...")

        try:
            # Set the selected folder as current
            SetBrowserSelection(self.current_folder)

            # Run blob detection
            result = BatchHessianBlobs()

            if result:
                self.log_message(f"Batch Hessian blob detection completed successfully")
                self.log_message(f"Results stored in: {result}")
                self.update_status("Blob detection completed")
                self.update_tree()
            else:
                self.log_message("Batch Hessian blob detection was cancelled")
                self.update_status("Blob detection cancelled")

        except Exception as e:
            messagebox.showerror("Error", f"Blob detection failed: {str(e)}")
            self.log_message(f"Blob detection error: {str(e)}")
            self.update_status("Blob detection failed")

    def single_hessian_blobs(self):
        """Run Hessian blob detection on single image"""
        # Get selected wave
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an image wave")
            return

        item_id = selection[0]
        item_type = self.tree.item(item_id, "values")[0] if self.tree.item(item_id, "values") else ""

        if item_type != "wave":
            messagebox.showwarning("Warning", "Please select an image wave")
            return

        try:
            wave = self.data_browser.get_wave(item_id.replace("root:", ""))
            if wave is None:
                messagebox.showerror("Error", "Could not find selected wave")
                return

            self.update_status("Running single image Hessian blob detection...")
            self.log_message(f"Starting Hessian blob detection on: {wave.name}")

            # Run blob detection
            result = HessianBlobs(wave)

            if result:
                self.log_message(f"Hessian blob detection completed successfully")
                self.log_message(f"Results stored in: {result}")
                self.update_status("Blob detection completed")
                self.update_tree()
            else:
                self.log_message("Hessian blob detection was cancelled")
                self.update_status("Blob detection cancelled")

        except Exception as e:
            messagebox.showerror("Error", f"Blob detection failed: {str(e)}")
            self.log_message(f"Blob detection error: {str(e)}")
            self.update_status("Blob detection failed")

    def view_particles(self):
        """View detected particles"""
        try:
            SetBrowserSelection(self.current_folder)
            result = ViewParticles()
            if result == 0:
                self.log_message("Particle viewer launched successfully")
            else:
                self.log_message("Could not launch particle viewer - check folder selection")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch particle viewer: {str(e)}")
            self.log_message(f"Particle viewer error: {str(e)}")

    def view_statistics(self):
        """View statistics of detected particles"""
        # Simple statistics viewer
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a data folder or wave")
            return

        try:
            # Try to find measurement waves in current selection
            item_id = selection[0]
            folder = self.data_browser.get_folder(item_id.replace("root:", ""))

            if folder and "AllHeights" in folder.waves:
                heights = folder.waves["AllHeights"]
                stats = WaveStats(heights)

                # Create statistics window
                stats_window = tk.Toplevel(self.root)
                stats_window.title("Particle Statistics")
                stats_window.geometry("400x300")

                stats_text = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD)
                stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                stats_info = f"""Particle Height Statistics:

Number of particles: {stats['V_npnts']}
Average height: {stats['V_avg']:.6f}
Standard deviation: {stats['V_sdev']:.6f}
Minimum height: {stats['V_min']:.6f}
Maximum height: {stats['V_max']:.6f}
Total sum: {stats['V_sum']:.6f}
"""

                # Add volume and area stats if available
                if "AllVolumes" in folder.waves:
                    volumes = folder.waves["AllVolumes"]
                    vol_stats = WaveStats(volumes)
                    stats_info += f"""
Volume Statistics:
Average volume: {vol_stats['V_avg']:.6e}
Standard deviation: {vol_stats['V_sdev']:.6e}
"""

                if "AllAreas" in folder.waves:
                    areas = folder.waves["AllAreas"]
                    area_stats = WaveStats(areas)
                    stats_info += f"""
Area Statistics:
Average area: {area_stats['V_avg']:.6e}
Standard deviation: {area_stats['V_sdev']:.6e}
"""

                stats_text.insert(tk.END, stats_info)
                stats_text.config(state=tk.DISABLED)

            else:
                messagebox.showinfo("Info", "No particle measurement data found in selected folder")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate statistics: {str(e)}")

    def run_testing(self):
        """Run the testing function (as in original Igor code)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Testing Function")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Enter a string:").pack(pady=5)
        string_var = tk.StringVar(value="Some string")
        tk.Entry(dialog, textvariable=string_var, width=30).pack(pady=5)

        tk.Label(dialog, text="Enter a number:").pack(pady=5)
        number_var = tk.DoubleVar(value=5.0)
        tk.Entry(dialog, textvariable=number_var, width=30).pack(pady=5)

        def run_test():
            try:
                Testing(string_var.get(), number_var.get())
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Testing function failed: {str(e)}")

        tk.Button(dialog, text="Run Test", command=run_test).pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite

Python Port of Igor Pro Implementation

Original Igor Pro Code:
Copyright 2019 by The Curators of the University of Missouri
G.M. King Laboratory
University of Missouri-Columbia
Coded by: Brendan Marsh
Email: marshbp@stanford.edu

Python Port maintains 1-1 functionality with the original Igor Pro version.

The Hessian blob algorithm is a general-purpose particle detection algorithm,
designed to detect, isolate, and draw the boundaries of roughly "blob-like" 
particles in an image.

For more information, see:
"The Hessian Blob Algorithm: Precise Particle Detection in Atomic Force 
Microscopy Imagery" - Scientific Reports
doi:10.1038/s41598-018-19379-x
"""

        messagebox.showinfo("About", about_text)

    def show_tutorial(self):
        """Show tutorial information"""
        tutorial_text = """Hessian Blob Detection Tutorial

1. PRELIMINARIES:
   - Load images using File -> Load Images or the Load Images button
   - Create data folders to organize your analysis
   - Images should be in Igor Binary Wave (.ibw) format

2. IMAGE PREPROCESSING (Optional):
   - Select folder containing images
   - Click "Batch Preprocess" to run flattening and streak removal
   - Flattening removes background variations
   - Streak removal corrects scanning artifacts

3. HESSIAN BLOB PARTICLE DETECTION:
   - Select folder containing preprocessed images
   - Click "Batch Hessian Blobs" for multiple images
   - Or select single image and use Analysis -> Single Image Hessian Blobs
   - Set parameters in the dialog:
     * Minimum/Maximum Size: particle size range in pixels
     * Scaling Factor: precision of scale-space (1.2-2.0 recommended)
     * Blob Strength: threshold (-2 for interactive, -1 for auto)
     * Particle Type: +1 for bumps, -1 for holes, 0 for both
     * Subpixel Ratio: subpixel precision multiplier
     * Overlap: whether to allow overlapping detections

4. DATA ANALYSIS:
   - Results are stored in Series_X folders
   - Use View -> View Particles to examine individual particles
   - Use View -> View Statistics for summary statistics
   - AllHeights, AllVolumes, AllAreas contain measurements

The algorithm maintains the same functionality and parameters as the 
original Igor Pro implementation.
"""

        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title("Tutorial")
        tutorial_window.geometry("800x600")

        text_widget = scrolledtext.ScrolledText(tutorial_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, tutorial_text)
        text_widget.config(state=tk.DISABLED)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = HessianBlobGUI(root)

    # Handle closing
    def on_closing():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()