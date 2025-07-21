import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path


class HessianBlobGUI:
    """Main GUI for Hessian Blob Detection"""

    def __init__(self, root):
        self.root = root
        self.root.title("Hessian Blob Particle Detection")
        self.root.geometry("1200x800")

        # Variables
        self.data_folder = tk.StringVar()
        self.images = {}
        self.results = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets"""

        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # Data folder selection
        ttk.Label(control_frame, text="Data Folder:").grid(row=0, column=0, padx=5)
        ttk.Entry(control_frame, textvariable=self.data_folder, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Load Images", command=self.load_images).grid(row=0, column=3, padx=5)

        # Parameters frame
        param_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        param_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)

        # Parameter inputs
        self.params = {}
        param_names = [
            ("min_size", "Minimum Size (pixels):", "1"),
            ("max_size", "Maximum Size (pixels):", "256"),
            ("scale_factor", "Scaling Factor:", "1.5"),
            ("blob_strength", "Blob Strength Threshold:", "-2"),
            ("particle_type", "Particle Type (-1/0/1):", "1"),
            ("subpixel_ratio", "Subpixel Ratio:", "1"),
            ("allow_overlap", "Allow Overlap (0/1):", "0")
        ]

        for i, (key, label, default) in enumerate(param_names):
            ttk.Label(param_frame, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky=tk.W, padx=5, pady=2)
            self.params[key] = tk.StringVar(value=default)
            ttk.Entry(param_frame, textvariable=self.params[key], width=15).grid(row=i // 2, column=(i % 2) * 2 + 1,
                                                                                 padx=5, pady=2)

        # Preprocessing options
        preproc_frame = ttk.LabelFrame(self.root, text="Preprocessing", padding="10")
        preproc_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=5)

        self.streak_removal = tk.StringVar(value="0")
        self.flatten_order = tk.StringVar(value="0")

        ttk.Label(preproc_frame, text="Streak Removal (std devs):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(preproc_frame, textvariable=self.streak_removal, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(preproc_frame, text="Flatten Order:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(preproc_frame, textvariable=self.flatten_order, width=10).grid(row=0, column=3, padx=5)

        # Action buttons
        action_frame = ttk.Frame(self.root, padding="10")
        action_frame.grid(row=3, column=0)

        ttk.Button(action_frame, text="Run Analysis", command=self.run_analysis).grid(row=0, column=0, padx=5)
        ttk.Button(action_frame, text="View Results", command=self.view_results).grid(row=0, column=1, padx=5)
        ttk.Button(action_frame, text="Export Results", command=self.export_results).grid(row=0, column=2, padx=5)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN).grid(row=4, column=0, sticky=(tk.W, tk.E))

    def browse_folder(self):
        """Browse for data folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.data_folder.set(folder)

    def load_images(self):
        """Load images from selected folder"""
        if not self.data_folder.get():
            messagebox.showerror("Error", "Please select a data folder")
            return

        try:
            from igor_io import load_ibw_file
            folder = Path(self.data_folder.get())
            self.images = {}

            for file_path in folder.glob('*.ibw'):
                image_data, wave_info = load_ibw_file(file_path)
                self.images[file_path.stem] = image_data

            self.status.set(f"Loaded {len(self.images)} images")

            if not self.images:
                messagebox.showwarning("Warning", "No IBW files found in selected folder")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")

    def run_analysis(self):
        """Run Hessian blob analysis"""
        if not self.images:
            messagebox.showerror("Error", "Please load images first")
            return

        try:
            from hessian_blobs import batch_hessian_blobs

            # Collect parameters
            params = np.array([
                float(self.params["min_size"].get()),
                float(self.params["max_size"].get()),
                float(self.params["scale_factor"].get()),
                float(self.params["blob_strength"].get()),
                int(self.params["particle_type"].get()),
                int(self.params["subpixel_ratio"].get()),
                int(self.params["allow_overlap"].get()),
                -np.inf, np.inf,  # height constraints
                -np.inf, np.inf,  # area constraints
                -np.inf, np.inf  # volume constraints
            ])

            # Preprocessing
            images_to_analyze = self.images.copy()

            streak_sdevs = float(self.streak_removal.get())
            flatten_ord = int(self.flatten_order.get())

            if streak_sdevs > 0 or flatten_ord > 0:
                from preprocessing import batch_preprocess
                images_to_analyze = batch_preprocess(images_to_analyze, streak_sdevs, flatten_ord)

            # Run analysis
            self.status.set("Running analysis...")
            self.root.update()

            self.results = batch_hessian_blobs(images_to_analyze, params)

            if self.results:
                num_particles = len(self.results['all_heights'])
                self.status.set(f"Analysis complete. Found {num_particles} particles")
                messagebox.showinfo("Success", f"Analysis complete!\nFound {num_particles} particles")
            else:
                self.status.set("Analysis failed")

        except Exception as e:
            self.status.set("Analysis failed")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def view_results(self):
        """View analysis results"""
        if not self.results:
            messagebox.showerror("Error", "No results to view. Run analysis first.")
            return

        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title("Analysis Results")
        results_window.geometry("800x600")

        # Create matplotlib figure
        fig = plt.Figure(figsize=(8, 6))

        # Height histogram
        ax = fig.add_subplot(111)
        ax.hist(self.results['all_heights'], bins=30, alpha=0.7)
        ax.set_xlabel('Height')
        ax.set_ylabel('Count')
        ax.set_title('Particle Height Distribution')

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, results_window)
        toolbar.update()

    def export_results(self):
        """Export results to file"""
        if not self.results:
            messagebox.showerror("Error", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                import pandas as pd

                # Create dataframe
                df = pd.DataFrame({
                    'Height': self.results['all_heights'],
                    'Area': self.results['all_areas'],
                    'Volume': self.results['all_volumes'],
                    'Avg_Height': self.results['all_avg_heights']
                })

                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Results exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")


def launch_gui():
    """Launch the GUI application"""
    root = tk.Tk()
    app = HessianBlobGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()