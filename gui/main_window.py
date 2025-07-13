"""Contains the main application window class and its UI layout."""

# #######################################################################
#                     GUI: MAIN APPLICATION WINDOW
#
#   CONTENTS:
#       - class HessianBlobGUI: The main Tkinter application class that
#         builds the UI, connects buttons to functions, and manages
#         threading to keep the application responsive.
#
# #######################################################################

import tkinter as tk
from tkinter import simpledialog, filedialog
import os
import queue
import threading
import warnings
import numpy as np
import matplotlib.pyplot as plt
from core.analysis import BatchHessianBlobs, HessianBlobs, WaveStats, Testing
from core.preprocessing import BatchPreprocess
from utils.data_manager import DataManager
from utils.error_handler import handle_error, safe_print
from utils.igor_compat import SetDataFolder, get_script_directory
from .particle_viewer import ViewParticles

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Agg.*')

class HessianBlobGUI:
    """Main GUI application for Hessian blob detection - matching Igor Pro tutorial style."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hessian Blob Particle Detection Suite")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Set up the GUI to match Igor Pro style
        self.setup_main_gui()

        # Output capture
        self.output_queue = queue.Queue()

        # Initialize global folder to script directory
        SetDataFolder(get_script_directory())
        safe_print(f"Working directory set to: {get_script_directory()}")

    def setup_main_gui(self):
        """Set up the main GUI similar to Igor Pro interface."""

        # Main title frame
        title_frame = tk.Frame(self.root, bg='#34495e', pady=15)
        title_frame.pack(fill='x')

        title_label = tk.Label(title_frame,
                               text="Hessian Blob Particle Detection Suite",
                               font=('Arial', 18, 'bold'),
                               fg='white', bg='#34495e')
        title_label.pack()

        subtitle_label = tk.Label(title_frame,
                                  text="G.M. King Laboratory - University of Missouri-Columbia",
                                  font=('Arial', 11),
                                  fg='#ecf0f1', bg='#34495e')
        subtitle_label.pack(pady=(5, 0))

        author_label = tk.Label(title_frame,
                                text="Python Port - Coded by: Brendan Marsh (marshbp@stanford.edu)",
                                font=('Arial', 9),
                                fg='#bdc3c7', bg='#34495e')
        author_label.pack()

        # Create main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)

        # Left panel for buttons (similar to Igor Pro menus)
        left_panel = tk.Frame(main_frame, bg='#f0f0f0', width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)

        # Right panel for output
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side='right', fill='both', expand=True)

        # Setup left panel buttons
        self.setup_button_panel(left_panel)

        # Setup right panel output
        self.setup_output_panel(right_panel)

    def setup_button_panel(self, parent):
        """Setup the button panel similar to Igor Pro menus."""

        # Section 1: Main Analysis Functions
        analysis_frame = tk.LabelFrame(parent, text="I. Main Analysis Functions",
                                       font=('Arial', 12, 'bold'),
                                       bg='#f0f0f0', pady=10)
        analysis_frame.pack(fill='x', pady=(0, 15))

        btn_batch = tk.Button(analysis_frame,
                              text="BatchHessianBlobs()",
                              font=('Arial', 11, 'bold'),
                              width=35, height=2,
                              bg='#3498db', fg='white',
                              command=self.run_batch_hessian_blobs,
                              cursor='hand2')
        btn_batch.pack(pady=5, padx=10, fill='x')

        batch_desc = tk.Label(analysis_frame,
                              text="Detect Hessian blobs in a series of images\nin a chosen data folder.",
                              font=('Arial', 9),
                              fg='#7f8c8d', bg='#f0f0f0')
        batch_desc.pack(pady=(0, 10))

        btn_single = tk.Button(analysis_frame,
                               text="HessianBlobs(image)",
                               font=('Arial', 11, 'bold'),
                               width=35, height=2,
                               bg='#2ecc71', fg='white',
                               command=self.run_single_hessian_blobs,
                               cursor='hand2')
        btn_single.pack(pady=5, padx=10, fill='x')

        single_desc = tk.Label(analysis_frame,
                               text="Execute the Hessian blob algorithm\non a single image.",
                               font=('Arial', 9),
                               fg='#7f8c8d', bg='#f0f0f0')
        single_desc.pack(pady=(0, 5))

        # Section 2: Preprocessing Functions
        preprocess_frame = tk.LabelFrame(parent, text="II. Preprocessing Functions",
                                         font=('Arial', 12, 'bold'),
                                         bg='#f0f0f0', pady=10)
        preprocess_frame.pack(fill='x', pady=(0, 15))

        btn_preprocess = tk.Button(preprocess_frame,
                                   text="BatchPreprocess()",
                                   font=('Arial', 11, 'bold'),
                                   width=35, height=2,
                                   bg='#e67e22', fg='white',
                                   command=self.run_batch_preprocess,
                                   cursor='hand2')
        btn_preprocess.pack(pady=5, padx=10, fill='x')

        preprocess_desc = tk.Label(preprocess_frame,
                                   text="Preprocess multiple images in a data folder\nsuccessively using flattening and streak removal.",
                                   font=('Arial', 9),
                                   fg='#7f8c8d', bg='#f0f0f0')
        preprocess_desc.pack(pady=(0, 5))

        # Section 3: Data Analysis and Visualization
        analysis_viz_frame = tk.LabelFrame(parent, text="III. Data Analysis & Visualization",
                                           font=('Arial', 12, 'bold'),
                                           bg='#f0f0f0', pady=10)
        analysis_viz_frame.pack(fill='x', pady=(0, 15))

        btn_view_particles = tk.Button(analysis_viz_frame,
                                       text="ViewParticles()",
                                       font=('Arial', 11, 'bold'),
                                       width=35, height=2,
                                       bg='#9b59b6', fg='white',
                                       command=self.run_view_particles,
                                       cursor='hand2')
        btn_view_particles.pack(pady=5, padx=10, fill='x')

        view_desc = tk.Label(analysis_viz_frame,
                             text="Convenient method to view and examine\nindividual detected particles.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        view_desc.pack(pady=(0, 10))

        btn_wave_stats = tk.Button(analysis_viz_frame,
                                   text="WaveStats(data)",
                                   font=('Arial', 11, 'bold'),
                                   width=35, height=2,
                                   bg='#34495e', fg='white',
                                   command=self.run_wave_stats,
                                   cursor='hand2')
        btn_wave_stats.pack(pady=5, padx=10, fill='x')

        stats_desc = tk.Label(analysis_viz_frame,
                              text="Compute basic statistics of particle\nmeasurements (heights, areas, volumes).",
                              font=('Arial', 9),
                              fg='#7f8c8d', bg='#f0f0f0')
        stats_desc.pack(pady=(0, 10))

        btn_create_histogram = tk.Button(analysis_viz_frame,
                                         text="Create Histogram",
                                         font=('Arial', 11, 'bold'),
                                         width=35, height=2,
                                         bg='#8e44ad', fg='white',
                                         command=self.create_histogram,
                                         cursor='hand2')
        btn_create_histogram.pack(pady=5, padx=10, fill='x')

        hist_desc = tk.Label(analysis_viz_frame,
                             text="Generate histograms depicting distributions\nof particle measurements.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        hist_desc.pack(pady=(0, 5))

        # Section 4: Testing and Demo
        test_frame = tk.LabelFrame(parent, text="IV. Testing & Demo",
                                   font=('Arial', 12, 'bold'),
                                   bg='#f0f0f0', pady=10)
        test_frame.pack(fill='x', pady=(0, 15))

        btn_demo = tk.Button(test_frame,
                             text="Run Synthetic Demo",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#1abc9c', fg='white',
                             command=self.run_synthetic_demo,
                             cursor='hand2')
        btn_demo.pack(pady=5, padx=10, fill='x')

        demo_desc = tk.Label(test_frame,
                             text="Create synthetic blob image and run\nHessian blob detection demonstration.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        demo_desc.pack(pady=(0, 10))

        btn_test = tk.Button(test_frame,
                             text="Testing(string, number)",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#95a5a6', fg='white',
                             command=self.run_test_function,
                             cursor='hand2')
        btn_test.pack(pady=5, padx=10, fill='x')

        test_desc = tk.Label(test_frame,
                             text="Demonstrate how user-defined functions\nwork with input parameters.",
                             font=('Arial', 9),
                             fg='#7f8c8d', bg='#f0f0f0')
        test_desc.pack(pady=(0, 5))

        # Exit button
        btn_exit = tk.Button(parent,
                             text="Exit Application",
                             font=('Arial', 11, 'bold'),
                             width=35, height=2,
                             bg='#e74c3c', fg='white',
                             command=self.root.quit,
                             cursor='hand2')
        btn_exit.pack(side='bottom', pady=10, padx=10, fill='x')

    def setup_output_panel(self, parent):
        """Setup the output panel for displaying results."""

        output_frame = tk.LabelFrame(parent, text="Command History / Output",
                                     font=('Arial', 12, 'bold'),
                                     bg='#f0f0f0')
        output_frame.pack(fill='both', expand=True)

        # Create text widget with scrollbar
        text_frame = tk.Frame(output_frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.output_text = tk.Text(text_frame,
                                   wrap='word',
                                   font=('Consolas', 10),
                                   bg='#2c3e50',
                                   fg='#ecf0f1',
                                   insertbackground='white')

        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self.output_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Clear button
        btn_clear = tk.Button(output_frame,
                              text="Clear Output",
                              command=self.clear_output,
                              bg='#7f8c8d', fg='white')
        btn_clear.pack(pady=5)

        # Initial welcome message
        self.print_to_output("=" * 80)
        self.print_to_output("Hessian Blob Particle Detection Suite - Python Port")
        self.print_to_output("=" * 80)
        self.print_to_output("Based on Igor Pro code by Brendan Marsh")
        self.print_to_output("G.M. King Laboratory, University of Missouri-Columbia")
        self.print_to_output("Python Port Email: marshbp@stanford.edu")
        self.print_to_output("=" * 80)
        self.print_to_output("")
        self.print_to_output("INSTRUCTIONS:")
        self.print_to_output("1. Select an analysis function from the left panel to begin")
        self.print_to_output("2. Follow the Igor Pro tutorial structure")
        self.print_to_output("3. Results are saved to folders matching Igor Pro format")
        self.print_to_output("")
        self.print_to_output("Current working directory: " + os.getcwd())
        self.print_to_output("")

    def print_to_output(self, text):
        """Print text to the output panel."""
        try:
            self.output_text.insert('end', text + '\n')
            self.output_text.see('end')
            self.root.update_idletasks()
        except Exception as e:
            handle_error("print_to_output", e)

    def clear_output(self):
        """Clear the output panel."""
        try:
            self.output_text.delete(1.0, 'end')
        except Exception as e:
            handle_error("clear_output", e)

    def run_in_thread(self, func, *args, **kwargs):
        """Run a function in a separate thread - FIXED VERSION."""

        def worker():
            try:
                result = func(*args, **kwargs)
                # Schedule GUI update on main thread
                self.root.after(0, lambda: self._handle_worker_success(result))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                import traceback
                full_traceback = traceback.format_exc()
                # Schedule error handling on main thread
                self.root.after(0, lambda: self._handle_worker_error(error_msg, full_traceback))

        # Only start thread if we're not already processing
        if not hasattr(self, '_is_processing') or not self._is_processing:
            self._is_processing = True
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

    def _handle_worker_success(self, result):
        """Handle successful worker completion on main thread"""
        self._is_processing = False
        # Result handling is done by individual methods

    def _handle_worker_error(self, error_msg, full_traceback):
        """Handle worker error on main thread"""
        self._is_processing = False
        self.print_to_output(error_msg)
        self.print_to_output(full_traceback)

    def run_batch_hessian_blobs(self):
        """Run batch Hessian blob analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: BatchHessianBlobs()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select folder containing images...")

        def batch_analysis():
            try:
                result_folder = BatchHessianBlobs()

                if result_folder:
                    # Use root.after to safely update GUI from worker thread
                    self.root.after(0, lambda: self._show_batch_results(result_folder))
                else:
                    self.root.after(0, lambda: self.print_to_output("Analysis cancelled or failed."))

            except Exception as e:
                raise e  # Let the thread handler deal with it

        self.run_in_thread(batch_analysis)

    def _show_batch_results(self, result_folder):
        """Show batch analysis results - called from main thread"""
        self.print_to_output(f"\n✓ Batch analysis complete!")
        self.print_to_output(f"Results saved to: {result_folder}")
        self.print_to_output("Series data folder created with:")
        self.print_to_output("  - AllHeights.npy")
        self.print_to_output("  - AllVolumes.npy")
        self.print_to_output("  - AllAreas.npy")
        self.print_to_output("  - AllAvgHeights.npy")
        self.print_to_output("  - Parameters.npy")
        self.print_to_output("  - Individual image particle folders")

        # Show summary
        try:
            heights = np.load(os.path.join(result_folder, "AllHeights.npy"))
            volumes = np.load(os.path.join(result_folder, "AllVolumes.npy"))
            areas = np.load(os.path.join(result_folder, "AllAreas.npy"))

            self.print_to_output(f"\nSERIES ANALYSIS SUMMARY:")
            self.print_to_output(f"  Series complete. Total particles detected: {len(heights)}")
            if len(heights) > 0:
                self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                self.print_to_output(f"  Average area: {np.mean(areas):.3e}")
        except Exception as e:
            self.print_to_output(f"Could not load summary data: {e}")

    def run_single_hessian_blobs(self):
        """Run single image Hessian blob analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: HessianBlobs(image)")
        self.print_to_output("=" * 60)
        self.print_to_output("Select an image file...")

        def single_analysis():
            try:
                # File dialog
                image_path = filedialog.askopenfilename(
                    title="Select Image File",
                    filetypes=[
                        ("Igor Binary Wave", "*.ibw"),
                        ("TIFF files", "*.tiff *.tif"),
                        ("PNG files", "*.png"),
                        ("JPEG files", "*.jpg *.jpeg"),
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not image_path:
                    self.print_to_output("No file selected.")
                    return

                self.print_to_output(f"Loading: {os.path.basename(image_path)}")
                image = DataManager.load_image_file(image_path)

                if image is None:
                    self.print_to_output("Failed to load image.")
                    return

                self.print_to_output(f"Image loaded. Shape: {image.shape}")

                # Run analysis
                result_folder = HessianBlobs(image)

                if result_folder:
                    self.print_to_output(f"\n✓ Analysis complete!")
                    self.print_to_output(f"Results saved to: {result_folder}")
                    self.print_to_output("Image_Particles folder created with:")
                    self.print_to_output("  - Heights.npy")
                    self.print_to_output("  - Volumes.npy")
                    self.print_to_output("  - Areas.npy")
                    self.print_to_output("  - AvgHeights.npy")
                    self.print_to_output("  - COM.npy")
                    self.print_to_output("  - Original.npy")
                    self.print_to_output("  - Individual Particle_X folders")

                    # Show summary
                    try:
                        heights = np.load(os.path.join(result_folder, "Heights.npy"))
                        volumes = np.load(os.path.join(result_folder, "Volumes.npy"))
                        areas = np.load(os.path.join(result_folder, "Areas.npy"))

                        self.print_to_output(f"\nSINGLE IMAGE ANALYSIS SUMMARY:")
                        self.print_to_output(f"  Particles detected: {len(heights)}")
                        if len(heights) > 0:
                            self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                            self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                            self.print_to_output(f"  Average area: {np.mean(areas):.3e}")
                    except Exception as e:
                        self.print_to_output(f"Could not load summary data: {e}")
                else:
                    self.print_to_output("Analysis cancelled or failed.")

            except Exception as e:
                self.print_to_output(f"Error during analysis: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(single_analysis)

    def run_batch_preprocess(self):
        """Run batch preprocessing - IGOR PRO STYLE."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: BatchPreprocess()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select folder containing images to preprocess...")
        self.print_to_output("Note: Preprocessed images will be saved to a new folder")

        def preprocess():
            try:
                result = BatchPreprocess()
                if result == 0:
                    self.print_to_output("✓ Preprocessing completed successfully.")
                    self.print_to_output("✓ Preprocessed images saved to new folder.")
                else:
                    self.print_to_output("Preprocessing failed or was cancelled.")
            except Exception as e:
                self.print_to_output(f"Error during preprocessing: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(preprocess)

    def run_view_particles(self):
        """Run particle viewer - FIXED THREADING VERSION."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: ViewParticles()")
        self.print_to_output("=" * 60)
        self.print_to_output("Select particles folder...")

        # CRITICAL: Run ViewParticles in main thread, not worker thread
        try:
            ViewParticles()
        except Exception as e:
            error_msg = handle_error("ViewParticles", e)
            self.print_to_output(error_msg)
            messagebox.showerror("Viewer Error", error_msg)

    def run_wave_stats(self):
        """Run WaveStats analysis."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: WaveStats(data)")
        self.print_to_output("=" * 60)
        self.print_to_output("Select data file (.npy)...")

        def wave_stats_analysis():
            try:
                data_file = filedialog.askopenfilename(
                    title="Select Data File",
                    filetypes=[
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not data_file:
                    self.print_to_output("No file selected.")
                    return

                self.print_to_output(f"WaveStats {os.path.basename(data_file)}")
                stats = WaveStats(data_file)

                if stats:
                    self.print_to_output("Statistics calculated successfully.")

            except Exception as e:
                self.print_to_output(f"Error running WaveStats: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(wave_stats_analysis)

    def create_histogram(self):
        """Create histogram from data."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("CREATING HISTOGRAM")
        self.print_to_output("=" * 60)
        self.print_to_output("Select data file (.npy)...")

        def make_histogram():
            try:
                data_file = filedialog.askopenfilename(
                    title="Select Data File",
                    filetypes=[
                        ("NumPy files", "*.npy"),
                        ("All files", "*.*")
                    ]
                )

                if not data_file:
                    self.print_to_output("No file selected.")
                    return

                data = np.load(data_file)
                base_name = os.path.splitext(os.path.basename(data_file))[0]

                self.print_to_output(f"Creating histogram for {base_name}...")

                # Create histogram
                plt.figure(figsize=(12, 8))
                clean_data = data[~np.isnan(data)]
                counts, bins, patches = plt.hist(clean_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f"Histogram of {base_name}", fontsize=14, fontweight='bold')
                plt.xlabel("Value", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.grid(True, alpha=0.3)

                # Add statistics text box
                stats_text = f"Count: {len(clean_data)}\n"
                stats_text += f"Mean: {np.mean(clean_data):.3e}\n"
                stats_text += f"Std Dev: {np.std(clean_data):.3e}\n"
                stats_text += f"Min: {np.min(clean_data):.3e}\n"
                stats_text += f"Max: {np.max(clean_data):.3e}"

                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                plt.tight_layout()
                plt.show()

                # Print statistics Igor Pro style
                self.print_to_output(f"\nHistogram created for {base_name}")
                self.print_to_output(f"Statistics:")
                self.print_to_output(f"  V_npnts= {len(clean_data)}; V_numNaNs= {np.sum(np.isnan(data))};")
                self.print_to_output(f"  V_avg= {np.mean(clean_data):.6g}; V_sum= {np.sum(clean_data):.6g};")
                self.print_to_output(
                    f"  V_sdev= {np.std(clean_data, ddof=1):.6g}; V_rms= {np.sqrt(np.mean(clean_data ** 2)):.6g};")
                self.print_to_output(f"  V_min= {np.min(clean_data):.6g}; V_max= {np.max(clean_data):.6g};")

            except Exception as e:
                self.print_to_output(f"Error creating histogram: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(make_histogram)

    def run_synthetic_demo(self):
        """Run synthetic data demo."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("RUNNING SYNTHETIC DEMO")
        self.print_to_output("=" * 60)
        self.print_to_output("Creating synthetic blob image...")

        def demo():
            try:
                # Create synthetic image with blob-like features
                size = 100
                x, y = np.meshgrid(np.arange(size), np.arange(size))

                # Create multiple Gaussian blobs
                image = np.zeros((size, size))

                # Add some blobs of different sizes
                blob_params = [
                    (25, 25, 5, 1.0),  # x, y, sigma, amplitude
                    (75, 30, 8, 0.8),
                    (40, 70, 6, 1.2),
                    (80, 80, 4, 0.9),
                    (15, 60, 3, 0.7),
                    (60, 15, 4, 1.1),
                ]

                for bx, by, sigma, amp in blob_params:
                    blob = amp * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * sigma ** 2))
                    image += blob

                # Add some noise
                image += np.random.normal(0, 0.05, (size, size))

                self.print_to_output(f"Synthetic image created with {len(blob_params)} blobs.")
                self.print_to_output("Running Hessian blob detection...")

                # Run analysis
                result_folder = HessianBlobs(image)

                if result_folder:
                    self.print_to_output(f"\n✓ Demo analysis complete!")
                    self.print_to_output(f"Results saved to: {result_folder}")

                    # Load and show results
                    heights = np.load(os.path.join(result_folder, "Heights.npy"))
                    volumes = np.load(os.path.join(result_folder, "Volumes.npy"))
                    areas = np.load(os.path.join(result_folder, "Areas.npy"))

                    self.print_to_output(f"\nDEMO ANALYSIS SUMMARY:")
                    self.print_to_output(f"  Expected blobs: {len(blob_params)}")
                    self.print_to_output(f"  Detected particles: {len(heights)}")
                    if len(heights) > 0:
                        self.print_to_output(f"  Average height: {np.mean(heights):.3e}")
                        self.print_to_output(f"  Average volume: {np.mean(volumes):.3e}")
                        self.print_to_output(f"  Average area: {np.mean(areas):.3e}")

                    # Show the synthetic image
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='hot', interpolation='bilinear')
                    plt.title('Synthetic Blob Image')
                    plt.colorbar()

                    # Show detection results if available
                    if len(heights) > 0:
                        plt.subplot(1, 2, 2)
                        plt.hist(heights, bins=20, alpha=0.7, color='green', edgecolor='black')
                        plt.title('Detected Height Distribution')
                        plt.xlabel('Height')
                        plt.ylabel('Count')
                        plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                else:
                    self.print_to_output("Demo analysis failed.")

            except Exception as e:
                self.print_to_output(f"Error during demo: {e}")
                import traceback
                self.print_to_output(traceback.format_exc())

        self.run_in_thread(demo)

    def run_test_function(self):
        """Run test function."""
        self.print_to_output("\n" + "=" * 60)
        self.print_to_output("EXECUTING: Testing(string, number)")
        self.print_to_output("=" * 60)

        def test_function():
            try:
                # Get input from user
                test_string = simpledialog.askstring("Testing Function", "Enter a test string:")
                if test_string is None:
                    self.print_to_output("Test cancelled.")
                    return

                test_number = simpledialog.askfloat("Testing Function", "Enter a test number:")
                if test_number is None:
                    self.print_to_output("Test cancelled.")
                    return

                # Run the testing function
                Testing(test_string, test_number)

            except Exception as e:
                self.print_to_output(f"Error running test function: {e}")

        self.run_in_thread(test_function)

    def run(self):
        """Start the GUI application."""
        try:
            self.root.mainloop()
        except Exception as e:
            handle_error("GUI.run", e)

    def _get_image_file_threadsafe(self):
        """Get image file using thread-safe file dialog"""
        return filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Igor Binary Wave", "*.ibw"),
                ("TIFF files", "*.tiff *.tif"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )
