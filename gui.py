# gui.py - Complete Igor Pro-style GUI implementation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
import threading


class HessianBlobGUI:
    """Main GUI for Hessian Blob Detection matching Igor Pro interface"""

    def __init__(self, root):
        self.root = root
        self.root.title("Hessian Blob Particle Detection - Igor Pro Style")
        self.root.geometry("1400x900")

        # Variables
        self.data_folder = tk.StringVar()
        self.images = {}
        self.coord_systems = {}
        self.current_image = None
        self.results = None
        self.preprocessing_done = False

        # Create menu bar
        self.create_menu()

        # Create main interface
        self.create_widgets()

    def create_menu(self):
        """Create Igor Pro style menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Images", command=self.load_images)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Batch Hessian Blobs", command=self.run_batch_analysis)
        analysis_menu.add_command(label="Single Image Analysis", command=self.run_single_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Preprocessing", command=self.show_preprocessing)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="View Particles", command=self.view_particles)
        view_menu.add_command(label="Show Histograms", command=self.show_histograms)

    def create_widgets(self):
        """Create main GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        # Data Browser tab
        self.create_data_browser_tab()

        # Image Display tab
        self.create_image_display_tab()

        # Results tab
        self.create_results_tab()

    def create_data_browser_tab(self):
        """Create data browser similar to Igor Pro"""
        browser_frame = ttk.Frame(self.notebook)
        self.notebook.add(browser_frame, text="Data Browser")

        # Tree view for data folders
        tree_frame = ttk.Frame(browser_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create treeview with scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side='right', fill='y')

        self.data_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set)
        tree_scroll.config(command=self.data_tree.yview)

        # Define columns
        self.data_tree['columns'] = ('Type', 'Size', 'Values')
        self.data_tree.column('#0', width=200, minwidth=100)
        self.data_tree.column('Type', width=100)
        self.data_tree.column('Size', width=100)
        self.data_tree.column('Values', width=200)

        # Headings
        self.data_tree.heading('#0', text='Name')
        self.data_tree.heading('Type', text='Type')
        self.data_tree.heading('Size', text='Size')
        self.data_tree.heading('Values', text='Values')

        self.data_tree.pack(fill='both', expand=True)

        # Buttons
        button_frame = ttk.Frame(browser_frame)
        button_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(button_frame, text="Load Folder", command=self.load_images).pack(side='left', padx=5)
        ttk.Button(button_frame, text="New Image", command=self.display_selected_image).pack(side='left', padx=5)

    def create_image_display_tab(self):
        """Create image display tab"""
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Image Display")

        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.image_frame)
        toolbar.update()

    def create_results_tab(self):
        """Create results display tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")

        # Text widget for results
        self.results_text = tk.Text(results_frame, wrap='word')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)

    def load_images(self):
        """Load images from folder"""
        folder = filedialog.askdirectory(title="Select folder containing IBW files")
        if not folder:
            return

        self.data_folder.set(folder)

        # Clear tree
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # Add root folder
        root_id = self.data_tree.insert('', 'end', text='root', values=('Folder', '', ''))

        # Add Images folder
        images_id = self.data_tree.insert(root_id, 'end', text='Images', values=('Folder', '', ''))

        # Load images
        from igor_io import load_ibw_file
        from utilities import CoordinateSystem

        folder_path = Path(folder)
        self.images = {}
        self.coord_systems = {}

        for file_path in folder_path.glob('*.ibw'):
            try:
                image_data, wave_info = load_ibw_file(str(file_path))

                # Create coordinate system
                coord_system = CoordinateSystem(
                    image_data.shape,
                    x_start=wave_info.get('x_start', 0),
                    x_delta=wave_info.get('x_delta', 1),
                    y_start=wave_info.get('y_start', 0),
                    y_delta=wave_info.get('y_delta', 1)
                )

                self.images[file_path.stem] = image_data
                self.coord_systems[file_path.stem] = coord_system

                # Add to tree
                self.data_tree.insert(images_id, 'end', text=file_path.stem,
                                      values=('Image', f'{image_data.shape}', f'{image_data.dtype}'))

            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

        self.data_tree.item(root_id, open=True)
        self.data_tree.item(images_id, open=True)

        # Update status
        self.results_text.insert('end', f"Loaded {len(self.images)} images from {folder}\n")

    def display_selected_image(self):
        """Display selected image from tree"""
        selection = self.data_tree.selection()
        if not selection:
            return

        item = self.data_tree.item(selection[0])
        image_name = item['text']

        if image_name in self.images:
            self.current_image = image_name
            self.display_image(self.images[image_name])

    def display_image(self, image):
        """Display image in matplotlib canvas"""
        self.ax.clear()
        im = self.ax.imshow(image, cmap='gray', origin='lower')
        self.ax.set_title(f'Image: {self.current_image}')
        self.fig.colorbar(im, ax=self.ax)
        self.canvas.draw()

        # Switch to image display tab
        self.notebook.select(1)

    def run_batch_analysis(self):
        """Run batch Hessian blob analysis"""
        if not self.images:
            messagebox.showerror("Error", "Please load images first")
            return

        # Show parameters dialog
        params = self.show_parameters_dialog()
        if params is None:
            return

        # Show constraints dialog
        constraints = self.show_constraints_dialog()
        if constraints is None:
            return

        # Run analysis in thread to keep GUI responsive
        self.results_text.insert('end', "\n" + "=" * 60 + "\n")
        self.results_text.insert('end', "Running Batch Hessian Blob Analysis\n")
        self.results_text.insert('end', "=" * 60 + "\n")

        # If interactive threshold selected, show threshold GUI
        if params[3] == -2:
            self.show_interactive_threshold(params, constraints)
        else:
            self.run_analysis_thread(params, constraints)

    def show_parameters_dialog(self):
        """Show Hessian blob parameters dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Hessian Blob Parameters")
        dialog.geometry("400x400")

        # Parameter variables
        vars = {
            'min_size': tk.StringVar(value="1"),
            'max_size': tk.StringVar(value="256"),
            'scale_factor': tk.StringVar(value="1.5"),
            'blob_strength': tk.StringVar(value="-2"),
            'particle_type': tk.StringVar(value="1"),
            'subpixel_ratio': tk.StringVar(value="1"),
            'allow_overlap': tk.StringVar(value="0")
        }

        # Create input fields
        labels = [
            ("Minimum Size in Pixels", 'min_size'),
            ("Maximum Size in Pixels", 'max_size'),
            ("Scaling Factor", 'scale_factor'),
            ("Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)", 'blob_strength'),
            ("Particle Type (-1 for negative, +1 for positive, 0 for both)", 'particle_type'),
            ("Subpixel Ratio", 'subpixel_ratio'),
            ("Allow Hessian Blobs to Overlap? (1=yes 0=no)", 'allow_overlap')
        ]

        for i, (label, key) in enumerate(labels):
            ttk.Label(dialog, text=label).grid(row=i, column=0, sticky='w', padx=10, pady=5)
            ttk.Entry(dialog, textvariable=vars[key], width=20).grid(row=i, column=1, padx=10, pady=5)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=len(labels), column=0, columnspan=2, pady=20)

        result = {'ok': False}

        def on_continue():
            result['ok'] = True
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Continue", command=on_continue).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left', padx=5)

        dialog.wait_window()

        if result['ok']:
            return np.array([
                float(vars['min_size'].get()),
                float(vars['max_size'].get()),
                float(vars['scale_factor'].get()),
                float(vars['blob_strength'].get()),
                int(vars['particle_type'].get()),
                int(vars['subpixel_ratio'].get()),
                int(vars['allow_overlap'].get()),
                -np.inf, np.inf,  # height constraints
                -np.inf, np.inf,  # area constraints
                -np.inf, np.inf  # volume constraints
            ])
        return None

    def show_constraints_dialog(self):
        """Show constraints dialog"""
        response = messagebox.askyesno("Constraints",
                                       "Would you like to limit the analysis to particles of certain height, volume, or area?")

        if not response:
            return np.array([-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf])

        # Show constraints input dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Constraints")
        dialog.geometry("350x250")

        vars = {
            'min_h': tk.StringVar(value="-inf"),
            'max_h': tk.StringVar(value="inf"),
            'min_a': tk.StringVar(value="-inf"),
            'max_a': tk.StringVar(value="inf"),
            'min_v': tk.StringVar(value="-inf"),
            'max_v': tk.StringVar(value="inf")
        }

        labels = [
            ("Minimum height", 'min_h'),
            ("Maximum height", 'max_h'),
            ("Minimum area", 'min_a'),
            ("Maximum area", 'max_a'),
            ("Minimum volume", 'min_v'),
            ("Maximum volume", 'max_v')
        ]

        for i, (label, key) in enumerate(labels):
            ttk.Label(dialog, text=label).grid(row=i, column=0, sticky='w', padx=10, pady=5)
            ttk.Entry(dialog, textvariable=vars[key], width=15).grid(row=i, column=1, padx=10, pady=5)

        result = {'ok': False}

        def on_continue():
            result['ok'] = True
            dialog.destroy()

        ttk.Button(dialog, text="Continue", command=on_continue).grid(row=len(labels), column=0, columnspan=2, pady=20)

        dialog.wait_window()

        if result['ok']:
            def parse_value(s):
                if s.lower() in ['-inf', 'inf']:
                    return float(s)
                return float(s)

            return np.array([
                parse_value(vars['min_h'].get()),
                parse_value(vars['max_h'].get()),
                parse_value(vars['min_a'].get()),
                parse_value(vars['max_a'].get()),
                parse_value(vars['min_v'].get()),
                parse_value(vars['max_v'].get())
            ])
        return None

    def show_interactive_threshold(self, params, constraints):
        """Show interactive threshold selection window"""
        # Create new window
        threshold_window = tk.Toplevel(self.root)
        threshold_window.title("Interactive Blob Strength Selection")
        threshold_window.geometry("1000x700")

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Get first image for display
        first_image_name = list(self.images.keys())[0]
        image = self.images[first_image_name]

        # Display image
        im_display = ax1.imshow(image, cmap='gray', origin='lower')
        ax1.set_title(f'Image: {first_image_name}')

        # Calculate scale space and blob detectors for this image
        from scale_space import scale_space_representation, blob_detectors, find_scale_space_maxima
        from utilities import CoordinateSystem

        coord_system = self.coord_systems[first_image_name]

        # Calculate scale-space
        scale_start = (params[0] * coord_system.x_delta) ** 2 / 2
        layers = int(np.ceil(np.log((params[1] * coord_system.x_delta) ** 2 / (2 * scale_start)) / np.log(params[2])))

        L, scale_coords = scale_space_representation(
            image, layers,
            np.sqrt(scale_start) / coord_system.x_delta,
            params[2], coord_system
        )

        detH, LapG = blob_detectors(L, 1, coord_system)

        # Find maxima
        maxes, map_data, scale_map = find_scale_space_maxima(
            detH, LapG, int(params[4]), 10
        )

        # Convert to image units
        maxes_sqrt = np.sqrt(maxes[maxes > 0])

        # Histogram
        ax2.hist(maxes_sqrt, bins=50, alpha=0.7)
        ax2.set_xlabel('Blob Strength')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Blob Strengths')

        # Initial threshold
        initial_thresh = np.median(maxes_sqrt) if len(maxes_sqrt) > 0 else 1e-10
        thresh_line = ax2.axvline(initial_thresh, color='r', linestyle='--', label='Threshold')

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=threshold_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Control frame
        control_frame = ttk.Frame(threshold_window)
        control_frame.pack(fill='x', padx=10, pady=10)

        # Threshold value display and input
        thresh_var = tk.StringVar(value=f"{initial_thresh:.2e}")
        ttk.Label(control_frame, text="Blob Strength:").pack(side='left', padx=5)
        thresh_entry = ttk.Entry(control_frame, textvariable=thresh_var, width=15)
        thresh_entry.pack(side='left', padx=5)

        # Slider
        slider_var = tk.DoubleVar(value=initial_thresh)
        slider = ttk.Scale(control_frame, from_=0, to=maxes_sqrt.max() * 1.1 if len(maxes_sqrt) > 0 else 1,
                           variable=slider_var, orient='horizontal', length=300)
        slider.pack(side='left', padx=20)

        # Circles for detected blobs
        circles = []

        def update_display(*args):
            """Update display with new threshold"""
            try:
                thresh = float(thresh_var.get())
            except:
                return

            # Update threshold line
            thresh_line.set_xdata([thresh, thresh])

            # Clear previous circles
            for circle in circles:
                circle.remove()
            circles.clear()

            # Find blobs above threshold
            for i in range(map_data.shape[0]):
                for j in range(map_data.shape[1]):
                    if map_data[i, j] > thresh ** 2:
                        # Calculate radius from scale
                        k = int(scale_map[i, j])
                        scale = scale_coords['scales'][k] if k < len(scale_coords['scales']) else \
                        scale_coords['scales'][-1]
                        radius = np.sqrt(2 * scale) / coord_system.x_delta

                        circle = Circle((j, i), radius, fill=False, color='red', linewidth=2)
                        ax1.add_patch(circle)
                        circles.append(circle)

            canvas.draw()

        def on_slider_change(val):
            thresh_var.set(f"{val:.2e}")

        slider.config(command=on_slider_change)
        thresh_var.trace('w', update_display)

        # Buttons
        button_frame = ttk.Frame(threshold_window)
        button_frame.pack(pady=10)

        result = {'threshold': initial_thresh, 'accepted': False}

        def on_accept():
            result['threshold'] = float(thresh_var.get())
            result['accepted'] = True
            threshold_window.destroy()

        def on_quit():
            threshold_window.destroy()

        ttk.Button(button_frame, text="Accept", command=on_accept).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Quit", command=on_quit).pack(side='left', padx=5)

        # Initial display
        update_display()

        threshold_window.wait_window()

        if result['accepted']:
            # Update parameters with chosen threshold
            params[3] = result['threshold']
            self.run_analysis_thread(params, constraints)

    def run_analysis_thread(self, params, constraints):
        """Run analysis in separate thread"""

        def run():
            from hessian_blobs import batch_hessian_blobs

            # Update constraints in params
            params[7:13] = constraints

            try:
                # Preprocess if needed
                images_to_analyze = self.images.copy()

                # Run analysis
                self.results = batch_hessian_blobs(images_to_analyze, params)

                # Update GUI in main thread
                self.root.after(0, self.display_results)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))

        thread = threading.Thread(target=run)
        thread.start()

    def display_results(self):
        """Display analysis results"""
        if not self.results:
            return

        # Update results text
        self.results_text.insert('end', f"\nAnalysis complete!\n")
        self.results_text.insert('end', f"Total particles detected: {len(self.results['all_heights'])}\n")

        if len(self.results['all_heights']) > 0:
            self.results_text.insert('end', f"\nHeight statistics:\n")
            self.results_text.insert('end', f"  Mean: {np.mean(self.results['all_heights']):.4f}\n")
            self.results_text.insert('end', f"  Std: {np.std(self.results['all_heights']):.4f}\n")
            self.results_text.insert('end', f"  Min: {np.min(self.results['all_heights']):.4f}\n")
            self.results_text.insert('end', f"  Max: {np.max(self.results['all_heights']):.4f}\n")

        # Update data tree with results
        root_items = self.data_tree.get_children()
        if root_items:
            root_id = root_items[0]

            # Add Series folder
            series_id = self.data_tree.insert(root_id, 'end', text=f"Series_0", values=('Folder', '', ''))

            # Add result waves
            self.data_tree.insert(series_id, 'end', text='AllHeights',
                                  values=('Wave', f'{len(self.results["all_heights"])}', ''))
            self.data_tree.insert(series_id, 'end', text='AllAreas',
                                  values=('Wave', f'{len(self.results["all_areas"])}', ''))
            self.data_tree.insert(series_id, 'end', text='AllVolumes',
                                  values=('Wave', f'{len(self.results["all_volumes"])}', ''))

            # Add image results
            for image_name, image_result in self.results['image_results'].items():
                image_folder_id = self.data_tree.insert(series_id, 'end',
                                                        text=f"{image_name}_Particles",
                                                        values=('Folder', '', ''))

                if 'particles' in image_result:
                    for i, particle in enumerate(image_result['particles']):
                        self.data_tree.insert(image_folder_id, 'end',
                                              text=f"Particle_{i}",
                                              values=('Particle', '', f"H:{particle['height']:.3f}"))

        # Show results with overlays
        self.show_results_overlay()

    def show_results_overlay(self):
        """Show results with blob overlays"""
        if not self.results or 'image_results' not in self.results:
            return

        # Create new window for results display
        results_window = tk.Toplevel(self.root)
        results_window.title("Detection Results")
        results_window.geometry("1200x800")

        # Create matplotlib figure with subplots for each image
        n_images = len(self.results['image_results'])
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_images == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Display each image with detected blobs
        for idx, (image_name, image_result) in enumerate(self.results['image_results'].items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            # Display image
            image = self.images[image_name]
            ax.imshow(image, cmap='gray', origin='lower')
            ax.set_title(f'{image_name} - {len(image_result.get("particles", []))} particles')

            # Add blob overlays
            if 'particles' in image_result:
                for particle in image_result['particles']:
                    if 'info' in particle:
                        info = particle['info']
                        # Draw bounding box or circle
                        if 'scale' in info:
                            x_center = (info['p_start'] + info['p_stop']) / 2
                            y_center = (info['q_start'] + info['q_stop']) / 2
                            radius = np.sqrt(2 * info['scale']) / self.coord_systems[image_name].x_delta

                            circle = Circle((y_center, x_center), radius,
                                            fill=False, color='red', linewidth=2)
                            ax.add_patch(circle)

        # Remove empty subplots
        if n_images < n_rows * n_cols:
            for idx in range(n_images, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].remove() if n_rows > 1 else axes[col].remove()

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=results_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_histograms(self):
        """Show measurement histograms"""
        if not self.results or len(self.results['all_heights']) == 0:
            messagebox.showinfo("Info", "No results to display. Run analysis first.")
            return

        # Create histogram window
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Measurement Histograms")
        hist_window.geometry("1200x800")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Height histogram
        axes[0, 0].hist(self.results['all_heights'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Height')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Particle Height Distribution')

        # Area histogram
        axes[0, 1].hist(self.results['all_areas'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Area')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Particle Area Distribution')

        # Volume histogram
        axes[1, 0].hist(self.results['all_volumes'], bins=30, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Volume')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Particle Volume Distribution')

        # Height vs Area scatter
        axes[1, 1].scatter(self.results['all_areas'], self.results['all_heights'], alpha=0.5)
        axes[1, 1].set_xlabel('Area')
        axes[1, 1].set_ylabel('Height')
        axes[1, 1].set_title('Height vs Area')

        plt.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def view_particles(self):
        """View individual particles"""
        if not self.results:
            messagebox.showinfo("Info", "No particles to view. Run analysis first.")
            return

        # Create particle viewer window
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title("Particle Viewer")
        viewer_window.geometry("800x800")

        # Get all particles from results
        all_particles = []
        for image_name, image_result in self.results['image_results'].items():
            if 'particles' in image_result:
                for i, particle in enumerate(image_result['particles']):
                    all_particles.append({
                        'image_name': image_name,
                        'particle_index': i,
                        'data': particle
                    })

        if not all_particles:
            messagebox.showinfo("Info", "No particles found.")
            viewer_window.destroy()
            return

            # Current particle index
        current_idx = [0]

        # Create main frame with controls
        main_frame = ttk.Frame(viewer_window)
        main_frame.pack(fill='both', expand=True)

        # Control panel (right side, similar to Igor Pro)
        control_panel = ttk.Frame(main_frame, width=200)
        control_panel.pack(side='right', fill='y', padx=10, pady=10)

        # Particle info
        info_frame = ttk.LabelFrame(control_panel, text="Particle Info")
        info_frame.pack(fill='x', pady=5)

        particle_label = ttk.Label(info_frame, text="Particle 0", font=('Arial', 14, 'bold'))
        particle_label.pack(pady=5)

        # Navigation buttons
        nav_frame = ttk.Frame(control_panel)
        nav_frame.pack(fill='x', pady=10)

        prev_btn = ttk.Button(nav_frame, text="Prev", width=10)
        prev_btn.pack(side='left', padx=2)

        next_btn = ttk.Button(nav_frame, text="Next", width=10)
        next_btn.pack(side='right', padx=2)

        # Go To control
        goto_frame = ttk.Frame(control_panel)
        goto_frame.pack(fill='x', pady=5)

        ttk.Label(goto_frame, text="Go To:").pack(side='left')
        goto_var = tk.StringVar(value="0")
        goto_entry = ttk.Entry(goto_frame, textvariable=goto_var, width=10)
        goto_entry.pack(side='left', padx=5)

        # Display options
        options_frame = ttk.LabelFrame(control_panel, text="Display Options")
        options_frame.pack(fill='x', pady=10)

        # Color table selection
        ttk.Label(options_frame, text="Color Table:").pack(anchor='w')
        color_var = tk.StringVar(value="hot")
        color_combo = ttk.Combobox(options_frame, textvariable=color_var,
                                   values=['hot', 'gray', 'viridis', 'plasma', 'inferno', 'magma'])
        color_combo.pack(fill='x', pady=2)

        # Color range
        ttk.Label(options_frame, text="Color Range:").pack(anchor='w')
        range_var = tk.StringVar(value="-1")
        range_entry = ttk.Entry(options_frame, textvariable=range_var)
        range_entry.pack(fill='x', pady=2)

        # Checkboxes
        interp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Interpolate", variable=interp_var).pack(anchor='w')

        perim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Perimeter", variable=perim_var).pack(anchor='w')

        # Measurements display
        meas_frame = ttk.LabelFrame(control_panel, text="Measurements")
        meas_frame.pack(fill='x', pady=10)

        height_label = ttk.Label(meas_frame, text="Height: 0.0000")
        height_label.pack(anchor='w', pady=2)

        area_label = ttk.Label(meas_frame, text="Area: 0.0000")
        area_label.pack(anchor='w', pady=2)

        volume_label = ttk.Label(meas_frame, text="Volume: 0.0000")
        volume_label.pack(anchor='w', pady=2)

        # Delete button
        delete_btn = ttk.Button(control_panel, text="DELETE",
                                command=lambda: delete_particle())
        delete_btn.pack(pady=20)

        # Image display (left side)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side='left', fill='both', expand=True)

        # Create matplotlib figure
        fig = plt.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        def show_particle(idx):
            """Display particle at given index"""
            if idx < 0 or idx >= len(all_particles):
                return

            current_idx[0] = idx
            particle_info = all_particles[idx]
            particle = particle_info['data']

            # Update labels
            particle_label.config(text=f"Particle {idx}")
            goto_var.set(str(idx))

            # Update measurements
            height_label.config(text=f"Height: {particle.get('height', 0):.4f}")
            area_label.config(text=f"Area: {particle.get('area', 0):.4f}")
            volume_label.config(text=f"Volume: {particle.get('volume', 0):.4f}")

            # Clear and display particle
            ax.clear()

            # Display particle image
            if 'image' in particle:
                im = ax.imshow(particle['image'], cmap=color_var.get(),
                               interpolation='bilinear' if interp_var.get() else 'nearest',
                               origin='lower')

                # Show perimeter if requested
                if perim_var.get() and 'perimeter' in particle:
                    perimeter = particle['perimeter']
                    # Create colored overlay for perimeter
                    overlay = np.zeros((*perimeter.shape, 4))
                    overlay[perimeter > 0] = [0, 1, 0, 1]  # Green perimeter
                    ax.imshow(overlay, origin='lower')

                ax.set_title(f"{particle_info['image_name']} - Particle {particle_info['particle_index']}")

                # Set color range
                try:
                    range_val = float(range_var.get())
                    if range_val > 0:
                        vmin = particle['image'].min()
                        im.set_clim(vmin, vmin + range_val)
                except:
                    pass

            canvas.draw()

        def next_particle():
            if current_idx[0] < len(all_particles) - 1:
                show_particle(current_idx[0] + 1)

        def prev_particle():
            if current_idx[0] > 0:
                show_particle(current_idx[0] - 1)

        def goto_particle(*args):
            try:
                idx = int(goto_var.get())
                if 0 <= idx < len(all_particles):
                    show_particle(idx)
            except:
                pass

        def delete_particle():
            response = messagebox.askyesno("Delete Particle",
                                           f"Are you sure you want to delete Particle {current_idx[0]}?")
            if response:
                # Remove from list
                all_particles.pop(current_idx[0])

                if all_particles:
                    # Show next or previous
                    if current_idx[0] >= len(all_particles):
                        show_particle(len(all_particles) - 1)
                    else:
                        show_particle(current_idx[0])
                else:
                    viewer_window.destroy()

        def update_display(*args):
            """Update display when settings change"""
            show_particle(current_idx[0])

        # Connect controls
        next_btn.config(command=next_particle)
        prev_btn.config(command=prev_particle)
        goto_var.trace('w', goto_particle)
        color_var.trace('w', update_display)
        range_var.trace('w', update_display)
        interp_var.trace('w', update_display)
        perim_var.trace('w', update_display)

        # Keyboard shortcuts
        def on_key(event):
            if event.keysym == 'Right':
                next_particle()
            elif event.keysym == 'Left':
                prev_particle()
            elif event.keysym == 'space' or event.keysym == 'Down':
                delete_particle()

        viewer_window.bind('<KeyPress>', on_key)

        # Show first particle
        if all_particles:
            show_particle(0)

    def show_preprocessing(self):
        """Show preprocessing dialog"""
        if not self.images:
            messagebox.showinfo("Info", "Please load images first")
            return

        # Create preprocessing window
        preproc_window = tk.Toplevel(self.root)
        preproc_window.title("Image Preprocessing")
        preproc_window.geometry("400x300")

        # Options
        ttk.Label(preproc_window, text="Preprocessing Options",
                  font=('Arial', 12, 'bold')).pack(pady=10)

        # Streak removal
        streak_frame = ttk.Frame(preproc_window)
        streak_frame.pack(fill='x', padx=20, pady=10)

        ttk.Label(streak_frame, text="Streak Removal (std devs):").pack(side='left')
        streak_var = tk.StringVar(value="0")
        ttk.Entry(streak_frame, textvariable=streak_var, width=10).pack(side='left', padx=10)

        # Flattening
        flatten_frame = ttk.Frame(preproc_window)
        flatten_frame.pack(fill='x', padx=20, pady=10)

        ttk.Label(flatten_frame, text="Polynomial Order for Flattening:").pack(side='left')
        flatten_var = tk.StringVar(value="0")
        ttk.Entry(flatten_frame, textvariable=flatten_var, width=10).pack(side='left', padx=10)

        # Buttons
        button_frame = ttk.Frame(preproc_window)
        button_frame.pack(pady=20)

        def apply_preprocessing():
            streak_sdevs = float(streak_var.get())
            flatten_order = int(flatten_var.get())

            if streak_sdevs > 0 or flatten_order > 0:
                from preprocessing import batch_preprocess

                # Apply preprocessing
                self.results_text.insert('end', f"\nApplying preprocessing...\n")
                self.results_text.insert('end', f"  Streak removal: {streak_sdevs} std devs\n")
                self.results_text.insert('end', f"  Flattening order: {flatten_order}\n")

                self.images = batch_preprocess(self.images, streak_sdevs, flatten_order)
                self.preprocessing_done = True

                # Update display if current image is shown
                if self.current_image:
                    self.display_image(self.images[self.current_image])

                messagebox.showinfo("Success", "Preprocessing complete!")

            preproc_window.destroy()

        ttk.Button(button_frame, text="Apply", command=apply_preprocessing).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=preproc_window.destroy).pack(side='left', padx=5)

    def run_single_analysis(self):
        """Run analysis on single selected image"""
        selection = self.data_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Please select an image from the data browser")
            return

        item = self.data_tree.item(selection[0])
        image_name = item['text']

        if image_name not in self.images:
            messagebox.showinfo("Info", "Please select a valid image")
            return

        # Show parameters dialog
        params = self.show_parameters_dialog()
        if params is None:
            return

        # Show constraints dialog
        constraints = self.show_constraints_dialog()
        if constraints is None:
            return

        # Update params with constraints
        params[7:13] = constraints

        # Run analysis on single image
        from hessian_blobs import hessian_blobs

        self.results_text.insert('end', f"\nRunning analysis on {image_name}...\n")

        try:
            result = hessian_blobs(self.images[image_name], params=params)

            if result:
                self.results_text.insert('end', f"Analysis complete for {image_name}\n")
                self.results_text.insert('end', f"  Particles found: {len(result['particles'])}\n")

                # Add to data tree
                root_items = self.data_tree.get_children()
                if root_items:
                    root_id = root_items[0]

                    # Add results folder
                    results_id = self.data_tree.insert(root_id, 'end',
                                                       text=f"{image_name}_Particles",
                                                       values=('Folder', '', ''))

                    # Add particles
                    for i, particle in enumerate(result['particles']):
                        self.data_tree.insert(results_id, 'end',
                                              text=f"Particle_{i}",
                                              values=('Particle', '', f"H:{particle['height']:.3f}"))

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

def launch_gui():
    """Launch the GUI application"""
    root = tk.Tk()
    app = HessianBlobGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()