# gui.py - Complete Igor Pro-style GUI implementation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
import threading
import queue
import os


class HessianBlobGUI:
    """Main GUI for Hessian Blob Detection matching Igor Pro interface"""

    def __init__(self, root):
        self.root = root
        self.root.title("Hessian Blob Particle Detection Suite - Igor Pro Style")
        self.root.geometry("1400x900")

        # Set Igor Pro-like styling
        self.setup_igor_style()

        # Variables
        self.data_folder = tk.StringVar()
        self.images = {}
        self.coord_systems = {}
        self.current_image = None
        self.results = None
        self.preprocessing_done = False
        self.progress_queue = queue.Queue()

        # Create menu bar
        self.create_menu()

        # Create main interface
        self.create_widgets()

        # Start progress monitoring
        self.monitor_progress()

    def setup_igor_style(self):
        """Setup Igor Pro-like styling"""
        style = ttk.Style()

        # Configure Igor Pro-like colors
        igor_bg = '#f0f0f0'
        igor_fg = '#000000'
        igor_select = '#0078d7'

        style.configure('Igor.TFrame', background=igor_bg)
        style.configure('Igor.TLabel', background=igor_bg, foreground=igor_fg)
        style.configure('Igor.TButton', relief='raised', borderwidth=2)
        style.configure('Igor.Treeview', background='white', fieldbackground='white')

        # Configure notebook tabs to look like Igor Pro
        style.configure('Igor.TNotebook', background=igor_bg)
        style.configure('Igor.TNotebook.Tab', padding=[12, 8], background=igor_bg)
        style.map('Igor.TNotebook.Tab', background=[('selected', 'white')])

    def create_menu(self):
        """Create Igor Pro style menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Images", command=self.load_images)
        file_menu.add_command(label="Load IBW Files", command=self.load_ibw_files)
        file_menu.add_separator()
        file_menu.add_command(label="Save Experiment", state='disabled')
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Batch Hessian Blobs", command=self.run_batch_analysis)
        analysis_menu.add_command(label="Single Image Analysis", command=self.run_single_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Batch Preprocessing", command=self.show_preprocessing)

        # Windows menu
        windows_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Windows", menu=windows_menu)
        windows_menu.add_command(label="Command Window", command=self.show_command_window)
        windows_menu.add_command(label="New Graph", command=self.new_graph)
        windows_menu.add_separator()
        windows_menu.add_command(label="View Particles", command=self.view_particles)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Igor Pro Manual", state='disabled')
        help_menu.add_command(label="About Hessian Blobs", command=self.show_about)

    def create_widgets(self):
        """Create main GUI widgets in Igor Pro style"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient='horizontal')
        main_paned.pack(fill='both', expand=True)

        # Left panel - Data Browser
        self.create_data_browser(main_paned)

        # Right panel - Notebook
        right_frame = ttk.Frame(main_paned, style='Igor.TFrame')
        main_paned.add(right_frame, weight=3)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame, style='Igor.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Command Line tab
        self.create_command_line_tab()

        # Image Display tab
        self.create_image_display_tab()

        # Results tab
        self.create_results_tab()

        # Status bar
        self.create_status_bar()

    def create_data_browser(self, parent):
        """Create data browser similar to Igor Pro"""
        browser_frame = ttk.Frame(parent, style='Igor.TFrame')
        parent.add(browser_frame, weight=1)

        # Title
        title_frame = ttk.Frame(browser_frame, style='Igor.TFrame')
        title_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(title_frame, text="Data Browser", font=('Arial', 12, 'bold'),
                  style='Igor.TLabel').pack(side='left')

        # Current folder display
        folder_frame = ttk.Frame(browser_frame, style='Igor.TFrame')
        folder_frame.pack(fill='x', padx=5)

        ttk.Label(folder_frame, text="Current Data Folder: root:",
                  style='Igor.TLabel').pack(side='left')

        # Tree frame
        tree_frame = ttk.Frame(browser_frame, style='Igor.TFrame')
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create treeview with scrollbar
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side='right', fill='y')

        self.data_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set,
                                      style='Igor.Treeview')
        tree_scroll.config(command=self.data_tree.yview)

        # Configure tree columns
        self.data_tree['columns'] = ('Type', 'Size')
        self.data_tree.column('#0', width=200, minwidth=100)
        self.data_tree.column('Type', width=80)
        self.data_tree.column('Size', width=80)

        # Configure headings
        self.data_tree.heading('#0', text='Name')
        self.data_tree.heading('Type', text='Type')
        self.data_tree.heading('Size', text='Size')

        self.data_tree.pack(fill='both', expand=True)

        # Add root folder
        self.root_id = self.data_tree.insert('', 'end', text='root',
                                             values=('Folder', ''), open=True)

        # Bind double-click
        self.data_tree.bind('<Double-Button-1>', self.on_tree_double_click)

        # Button frame
        button_frame = ttk.Frame(browser_frame, style='Igor.TFrame')
        button_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(button_frame, text="New Folder",
                   command=self.new_folder, style='Igor.TButton').pack(side='left', padx=2)
        ttk.Button(button_frame, text="New Image",
                   command=self.display_selected_image, style='Igor.TButton').pack(side='left', padx=2)

    def create_command_line_tab(self):
        """Create command line tab"""
        cmd_frame = ttk.Frame(self.notebook, style='Igor.TFrame')
        self.notebook.add(cmd_frame, text="Command")

        # History area
        history_frame = ttk.Frame(cmd_frame, style='Igor.TFrame')
        history_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.cmd_history = tk.Text(history_frame, wrap='word', height=20,
                                   bg='white', fg='black', font=('Courier', 10))
        self.cmd_history.pack(side='left', fill='both', expand=True)

        history_scroll = ttk.Scrollbar(history_frame, command=self.cmd_history.yview)
        history_scroll.pack(side='right', fill='y')
        self.cmd_history.config(yscrollcommand=history_scroll.set)

        # Command input
        input_frame = ttk.Frame(cmd_frame, style='Igor.TFrame')
        input_frame.pack(fill='x', padx=5, pady=5)

        self.cmd_entry = ttk.Entry(input_frame, font=('Courier', 10))
        self.cmd_entry.pack(side='left', fill='x', expand=True)
        self.cmd_entry.bind('<Return>', self.execute_command)

        ttk.Button(input_frame, text="Execute", command=self.execute_command,
                   style='Igor.TButton').pack(side='right', padx=5)

        # Add welcome message
        self.cmd_history.insert('end', "Igor Pro - Hessian Blob Detection Suite\n")
        self.cmd_history.insert('end', "=" * 50 + "\n")
        self.cmd_history.insert('end', "Type 'help' for available commands\n\n")
        self.cmd_history.insert('end', "•")
        self.cmd_history.config(state='disabled')

    def create_image_display_tab(self):
        """Create image display tab"""
        self.image_frame = ttk.Frame(self.notebook, style='Igor.TFrame')
        self.notebook.add(self.image_frame, text="Graph0")

        # Create matplotlib figure with Igor Pro styling
        self.fig = plt.Figure(figsize=(10, 8), facecolor='#f0f0f0')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('white')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(self.image_frame, style='Igor.TFrame')
        toolbar_frame.pack(fill='x')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def create_results_tab(self):
        """Create results display tab"""
        results_frame = ttk.Frame(self.notebook, style='Igor.TFrame')
        self.notebook.add(results_frame, text="Table0")

        # Create table-like display
        self.results_tree = ttk.Treeview(results_frame, style='Igor.Treeview')
        self.results_tree.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure columns for results
        self.results_tree['columns'] = ('Height', 'Area', 'Volume', 'X', 'Y')
        self.results_tree.column('#0', width=100)
        self.results_tree.column('Height', width=100)
        self.results_tree.column('Area', width=100)
        self.results_tree.column('Volume', width=100)
        self.results_tree.column('X', width=100)
        self.results_tree.column('Y', width=100)

        # Set headings
        self.results_tree.heading('#0', text='Particle')
        self.results_tree.heading('Height', text='Height')
        self.results_tree.heading('Area', text='Area')
        self.results_tree.heading('Volume', text='Volume')
        self.results_tree.heading('X', text='X Center')
        self.results_tree.heading('Y', text='Y Center')

    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.root, style='Igor.TFrame', relief='sunken')
        self.status_frame.pack(fill='x', side='bottom')

        self.status_label = ttk.Label(self.status_frame, text="Ready",
                                      style='Igor.TLabel')
        self.status_label.pack(side='left', padx=5)

        # Progress bar (initially hidden)
        self.progress_bar = ttk.Progressbar(self.status_frame, mode='indeterminate',
                                            length=200)

    def load_images(self):
        """Load images from folder"""
        folder = filedialog.askdirectory(title="Select folder containing images")
        if not folder:
            return

        self.status_label.config(text="Loading images...")
        self.data_folder.set(folder)

        # Clear existing images folder if exists
        for item in self.data_tree.get_children(self.root_id):
            if self.data_tree.item(item)['text'] == 'Images':
                self.data_tree.delete(item)
                break

        # Add Images folder
        images_id = self.data_tree.insert(self.root_id, 'end', text='Images',
                                          values=('Folder', ''), open=True)

        # Load images
        from utilities import CoordinateSystem

        folder_path = Path(folder)
        self.images = {}
        self.coord_systems = {}

        # Support various image formats
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(ext))

        for file_path in image_files:
            try:
                # Load image
                import matplotlib.pyplot as plt
                image_data = plt.imread(str(file_path))

                # Convert to grayscale if needed
                if image_data.ndim == 3:
                    image_data = np.mean(image_data, axis=2)

                # Create coordinate system
                coord_system = CoordinateSystem(
                    image_data.shape,
                    x_start=0,
                    x_delta=1,
                    y_start=0,
                    y_delta=1
                )

                self.images[file_path.stem] = image_data
                self.coord_systems[file_path.stem] = coord_system

                # Add to tree
                self.data_tree.insert(images_id, 'end', text=file_path.stem,
                                      values=('Image', f'{image_data.shape}'))

            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

        # Update command history
        self.add_to_history(f"Loaded {len(self.images)} images from {folder}")
        self.status_label.config(text=f"Loaded {len(self.images)} images")

    def load_ibw_files(self):
        """Load IBW files"""
        folder = filedialog.askdirectory(title="Select folder containing IBW files")
        if not folder:
            return

        self.status_label.config(text="Loading IBW files...")
        self.data_folder.set(folder)

        # Clear existing images folder if exists
        for item in self.data_tree.get_children(self.root_id):
            if self.data_tree.item(item)['text'] == 'Images':
                self.data_tree.delete(item)
                break

        # Add Images folder
        images_id = self.data_tree.insert(self.root_id, 'end', text='Images',
                                          values=('Folder', ''), open=True)

        # Load IBW files
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
                                      values=('Wave', f'{image_data.shape}'))

            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

        self.add_to_history(f"Loaded {len(self.images)} IBW files from {folder}")
        self.status_label.config(text=f"Loaded {len(self.images)} IBW files")

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

        # Update status
        self.status_label.config(text="Running Batch Hessian Blob Analysis...")
        self.progress_bar.pack(side='right', padx=5)
        self.progress_bar.start()

        # Add to command history
        self.add_to_history("\nBatchHessianBlobs()")
        self.add_to_history("-------------------------------------------------------")
        self.add_to_history("Running Batch Hessian Blob Analysis")
        self.add_to_history("-------------------------------------------------------")

        # If interactive threshold selected, show threshold GUI
        if params[3] == -2:
            self.show_interactive_threshold(params, constraints)
        else:
            self.run_analysis_thread(params, constraints)

    def show_parameters_dialog(self):
        """Show Hessian blob parameters dialog matching Igor Pro exactly"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Hessian Blob Parameters")
        dialog.geometry("550x350")
        dialog.configure(bg='#f0f0f0')

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        # Parameter variables with Igor Pro defaults
        vars = {
            'min_size': tk.StringVar(value="1"),
            'max_size': tk.StringVar(value="256"),
            'scale_factor': tk.StringVar(value="1.5"),
            'blob_strength': tk.StringVar(value="-2"),
            'particle_type': tk.StringVar(value="1"),
            'subpixel_ratio': tk.StringVar(value="1"),
            'allow_overlap': tk.StringVar(value="0")
        }

        # Main frame
        main_frame = ttk.Frame(dialog, style='Igor.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create parameter inputs matching Igor Pro layout
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
            ttk.Label(main_frame, text=label, style='Igor.TLabel').grid(
                row=i, column=0, sticky='w', padx=5, pady=5)
            ttk.Entry(main_frame, textvariable=vars[key], width=15).grid(
                row=i, column=1, sticky='w', padx=5, pady=5)

        # Button frame
        button_frame = ttk.Frame(dialog, style='Igor.TFrame')
        button_frame.pack(fill='x', pady=10)

        result = {'ok': False}

        def on_continue():
            result['ok'] = True
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        def on_help():
            messagebox.showinfo("Help", "See Igor Pro manual for parameter details")

        ttk.Button(button_frame, text="Continue", command=on_continue,
                   style='Igor.TButton').pack(side='left', padx=20)
        ttk.Button(button_frame, text="Help", command=on_help,
                   style='Igor.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel,
                   style='Igor.TButton').pack(side='left', padx=5)

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

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
        """Show constraints dialog matching Igor Pro"""
        # First dialog
        response = messagebox.askyesnocancel(
            "Igor Pro wants to know...",
            "Would you like to limit the analysis to particles of certain height, volume, or area?",
            icon='question'
        )

        if response is None:  # Cancel
            return None
        elif not response:  # No
            return np.array([-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf])

        # Show constraints input dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Constraints")
        dialog.geometry("400x300")
        dialog.configure(bg='#f0f0f0')

        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()

        vars = {
            'min_h': tk.StringVar(value="-inf"),
            'max_h': tk.StringVar(value="inf"),
            'min_a': tk.StringVar(value="-inf"),
            'max_a': tk.StringVar(value="inf"),
            'min_v': tk.StringVar(value="-inf"),
            'max_v': tk.StringVar(value="inf")
        }

        # Main frame
        main_frame = ttk.Frame(dialog, style='Igor.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        labels = [
            ("Minimum height", 'min_h'),
            ("Maximum height", 'max_h'),
            ("Minimum area", 'min_a'),
            ("Maximum area", 'max_a'),
            ("Minimum volume", 'min_v'),
            ("Maximum volume", 'max_v')
        ]

        for i, (label, key) in enumerate(labels):
            ttk.Label(main_frame, text=label, style='Igor.TLabel').grid(
                row=i, column=0, sticky='w', padx=5, pady=5)
            ttk.Entry(main_frame, textvariable=vars[key], width=15).grid(
                row=i, column=1, sticky='w', padx=5, pady=5)

        # Button frame
        button_frame = ttk.Frame(dialog, style='Igor.TFrame')
        button_frame.pack(fill='x', pady=10)

        result = {'ok': False}

        def on_continue():
            result['ok'] = True
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        ttk.Button(button_frame, text="Continue", command=on_continue,
                   style='Igor.TButton').pack(side='left', padx=20)
        ttk.Button(button_frame, text="Cancel", command=on_cancel,
                   style='Igor.TButton').pack(side='left', padx=5)

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

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
        """Show interactive threshold selection window matching Igor Pro"""
        # Create new window
        threshold_window = tk.Toplevel(self.root)
        threshold_window.title("IMAGE:Original")
        threshold_window.geometry("800x700")

        # Make window modal
        threshold_window.transient(self.root)
        threshold_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(threshold_window)
        main_frame.pack(fill='both', expand=True)

        # Create matplotlib figure
        fig = plt.Figure(figsize=(6, 6), facecolor='#f0f0f0')
        ax = fig.add_subplot(111)

        # Get first image for display
        first_image_name = list(self.images.keys())[0]
        image = self.images[first_image_name]

        # Display image
        im_display = ax.imshow(image, cmap='gray', origin='lower')
        ax.set_title(f'{first_image_name}')
        ax.set_xlabel('nm')
        ax.set_ylabel('nm')

        # Calculate scale space for this image
        self.add_to_history("Calculating scale-space representation..")

        from scale_space import scale_space_representation, blob_detectors, find_scale_space_maxima

        coord_system = self.coord_systems[first_image_name]

        # Calculate scale-space
        scale_start = (params[0] * coord_system.x_delta) ** 2 / 2
        layers = int(np.ceil(np.log((params[1] * coord_system.x_delta) ** 2 / (2 * scale_start)) / np.log(params[2])))

        L, scale_coords = scale_space_representation(
            image, layers,
            np.sqrt(scale_start) / coord_system.x_delta,
            params[2], coord_system
        )

        self.add_to_history("Calculating scale-space derivatives..")
        detH, LapG = blob_detectors(L, 1, coord_system)

        # Find maxima
        maxes, map_data, scale_map = find_scale_space_maxima(
            detH, LapG, int(params[4]), 10
        )

        # Convert to image units
        maxes_sqrt = np.sqrt(maxes[maxes > 0])

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        # Control panel (right side)
        control_frame = ttk.Frame(main_frame, style='Igor.TFrame', width=250)
        control_frame.pack(side='right', fill='y', padx=10, pady=10)
        control_frame.pack_propagate(False)

        # Buttons at top
        button_frame = ttk.Frame(control_frame, style='Igor.TFrame')
        button_frame.pack(fill='x', pady=(0, 20))

        ttk.Button(button_frame, text="Accept", command=lambda: on_accept(),
                   style='Igor.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Quit", command=lambda: on_quit(),
                   style='Igor.TButton').pack(side='left', padx=5)

        # Blob Strength control
        strength_frame = ttk.LabelFrame(control_frame, text="Blob Strength",
                                        style='Igor.TFrame')
        strength_frame.pack(fill='x', pady=10)

        # Initial threshold
        initial_thresh = np.median(maxes_sqrt) if len(maxes_sqrt) > 0 else 1e-10
        thresh_var = tk.StringVar(value=f"{initial_thresh:.4e}")

        ttk.Entry(strength_frame, textvariable=thresh_var, width=15).pack(padx=10, pady=5)

        # Slider
        slider_var = tk.DoubleVar(value=initial_thresh)
        max_val = maxes_sqrt.max() * 1.1 if len(maxes_sqrt) > 0 else 1

        slider = ttk.Scale(strength_frame, from_=0, to=max_val,
                           variable=slider_var, orient='vertical', length=300)
        slider.pack(padx=10, pady=10)

        # Circles for detected blobs
        circles = []

        def update_display(*args):
            """Update display with new threshold"""
            try:
                thresh = float(thresh_var.get())
                slider_var.set(thresh)
            except:
                return

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
                        ax.add_patch(circle)
                        circles.append(circle)

            canvas.draw()

        def on_slider_change(val):
            thresh_var.set(f"{val:.4e}")

        slider.config(command=on_slider_change)
        thresh_var.trace('w', update_display)

        # Result handling
        result = {'threshold': initial_thresh, 'accepted': False}

        def on_accept():
            result['threshold'] = float(thresh_var.get())
            result['accepted'] = True
            self.add_to_history(f"Chosen Det H Response Threshold: {result['threshold']}")
            threshold_window.destroy()

        def on_quit():
            threshold_window.destroy()

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
                self.root.after(0, lambda: self.progress_bar.stop())
                self.root.after(0, lambda: self.progress_bar.pack_forget())

        thread = threading.Thread(target=run)
        thread.start()

    def display_results(self):
        """Display analysis results"""
        if not self.results:
            return

        # Stop progress bar
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        # Update command history
        self.add_to_history(f"  Series complete. Total particles detected: {len(self.results['all_heights'])}")

        # Update data tree with results
        # Add Series folder
        series_count = len([item for item in self.data_tree.get_children(self.root_id)
                            if self.data_tree.item(item)['text'].startswith('Series_')])
        series_id = self.data_tree.insert(self.root_id, 'end',
                                          text=f"Series_{series_count}",
                                          values=('Folder', ''), open=True)

        # Add result waves
        self.data_tree.insert(series_id, 'end', text='AllHeights',
                              values=('Wave', f'({len(self.results["all_heights"])})'))
        self.data_tree.insert(series_id, 'end', text='AllAreas',
                              values=('Wave', f'({len(self.results["all_areas"])})'))
        self.data_tree.insert(series_id, 'end', text='AllVolumes',
                              values=('Wave', f'({len(self.results["all_volumes"])})'))
        self.data_tree.insert(series_id, 'end', text='AllAvgHeights',
                              values=('Wave', f'({len(self.results["all_avg_heights"])})'))
        self.data_tree.insert(series_id, 'end', text='Parameters',
                              values=('Wave', '(13)'))

        # Add image results folders
        for image_name, image_result in self.results['image_results'].items():
            image_folder_id = self.data_tree.insert(series_id, 'end',
                                                    text=f"{image_name}_Particles",
                                                    values=('Folder', ''), open=False)

            # Add waves for this image
            if 'particles' in image_result:
                n_particles = len(image_result['particles'])
                self.data_tree.insert(image_folder_id, 'end', text='Heights',
                                      values=('Wave', f'({n_particles})'))
                self.data_tree.insert(image_folder_id, 'end', text='Areas',
                                      values=('Wave', f'({n_particles})'))
                self.data_tree.insert(image_folder_id, 'end', text='Volumes',
                                      values=('Wave', f'({n_particles})'))

                # Add particle folders
                for i, particle in enumerate(image_result['particles']):
                    particle_folder_id = self.data_tree.insert(image_folder_id, 'end',
                                                               text=f"Particle_{i}",
                                                               values=('Folder', ''))

                    # Add particle waves
                    self.data_tree.insert(particle_folder_id, 'end',
                                          text=f'Particle_{i}',
                                          values=('Wave', f'{particle["image"].shape}'))
                    self.data_tree.insert(particle_folder_id, 'end',
                                          text=f'Mask_{i}',
                                          values=('Wave', f'{particle["mask"].shape}'))
                    self.data_tree.insert(particle_folder_id, 'end',
                                          text=f'Perimeter_{i}',
                                          values=('Wave', f'{particle["perimeter"].shape}'))

        # Update results table
        self.update_results_table()

        # Show results with overlays
        self.show_results_overlay()

        # Update status
        self.status_label.config(text=f"Analysis complete - {len(self.results['all_heights'])} particles detected")

    def update_results_table(self):
        """Update the results table"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Add particles to table
        if self.results and 'image_results' in self.results:
            particle_num = 0
            for image_name, image_result in self.results['image_results'].items():
                if 'particles' in image_result:
                    for i, particle in enumerate(image_result['particles']):
                        self.results_tree.insert('', 'end', text=f'P{particle_num}',
                                                 values=(
                                                     f"{particle.get('height', 0):.4f}",
                                                     f"{particle.get('area', 0):.4f}",
                                                     f"{particle.get('volume', 0):.4f}",
                                                     f"{particle.get('center', (0, 0))[0]:.2f}",
                                                     f"{particle.get('center', (0, 0))[1]:.2f}"
                                                 ))
                        particle_num += 1

    def show_results_overlay(self):
        """Show results with blob overlays"""
        if not self.results or 'image_results' not in self.results:
            return

        # Clear current plot
        self.ax.clear()

        # If multiple images, create subplots
        n_images = len(self.results['image_results'])

        if n_images == 1:
            # Single image
            image_name = list(self.results['image_results'].keys())[0]
            image = self.images[image_name]
            image_result = self.results['image_results'][image_name]

            # Display image
            self.ax.imshow(image, cmap='gray', origin='lower')
            self.ax.set_title(f'{image_name} - {len(image_result.get("particles", []))} particles')

            # Add blob overlays
            if 'particles' in image_result:
                for particle in image_result['particles']:
                    if 'info' in particle:
                        info = particle['info']
                        # Draw perimeter overlay
                        if 'perimeter' in particle:
                            perimeter = particle['perimeter']
                            # Create colored overlay
                            overlay = np.zeros((*perimeter.shape, 4))
                            overlay[perimeter > 0] = [1, 0, 0, 1]  # Red perimeter

                            # Calculate position in main image
                            p_start = info.get('p_start', 0)
                            q_start = info.get('q_start', 0)

                            extent = [q_start, q_start + perimeter.shape[1],
                                      p_start, p_start + perimeter.shape[0]]

                            self.ax.imshow(overlay, extent=extent, origin='lower')
        else:
            # Multiple images - show first one with note
            first_name = list(self.results['image_results'].keys())[0]
            image = self.images[first_name]
            image_result = self.results['image_results'][first_name]

            self.ax.imshow(image, cmap='gray', origin='lower')
            self.ax.set_title(
                f'{first_name} - {len(image_result.get("particles", []))} particles\n(Showing 1 of {n_images} images)')

            # Add overlays for first image
            if 'particles' in image_result:
                for particle in image_result['particles']:
                    if 'info' in particle and 'perimeter' in particle:
                        info = particle['info']
                        perimeter = particle['perimeter']
                        overlay = np.zeros((*perimeter.shape, 4))
                        overlay[perimeter > 0] = [1, 0, 0, 1]

                        p_start = info.get('p_start', 0)
                        q_start = info.get('q_start', 0)
                        extent = [q_start, q_start + perimeter.shape[1],
                                  p_start, p_start + perimeter.shape[0]]

                        self.ax.imshow(overlay, extent=extent, origin='lower')

        self.canvas.draw()

    def view_particles(self):
        """View individual particles"""
        if not self.results:
            messagebox.showinfo("Info", "No particles to view. Run analysis first.")
            return

        # Create particle viewer window
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title("Particle Viewer")
        viewer_window.geometry("900x700")

        # Make it look like Igor Pro
        viewer_window.configure(bg='#f0f0f0')

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

        # Main frame
        main_frame = ttk.Frame(viewer_window, style='Igor.TFrame')
        main_frame.pack(fill='both', expand=True)

        # Image display (left side)
        image_frame = ttk.Frame(main_frame, style='Igor.TFrame')
        image_frame.pack(side='left', fill='both', expand=True)

        # Create matplotlib figure
        fig = plt.Figure(figsize=(6, 6), facecolor='#f0f0f0')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Control panel (right side, Igor Pro style)
        control_panel = ttk.Frame(main_frame, style='Igor.TFrame', width=250)
        control_panel.pack(side='right', fill='y', padx=10, pady=10)
        control_panel.pack_propagate(False)

        # Title
        title_frame = ttk.Frame(control_panel, style='Igor.TFrame')
        title_frame.pack(fill='x', pady=(0, 10))

        particle_label = ttk.Label(title_frame, text="Particle 0",
                                   font=('Arial', 14, 'bold'), style='Igor.TLabel')
        particle_label.pack()

        # Navigation buttons
        nav_frame = ttk.Frame(control_panel, style='Igor.TFrame')
        nav_frame.pack(fill='x', pady=10)

        prev_btn = ttk.Button(nav_frame, text="Prev", width=10, style='Igor.TButton')
        prev_btn.pack(side='left', padx=2)

        next_btn = ttk.Button(nav_frame, text="Next", width=10, style='Igor.TButton')
        next_btn.pack(side='right', padx=2)

        # Go To control
        goto_frame = ttk.Frame(control_panel, style='Igor.TFrame')
        goto_frame.pack(fill='x', pady=5)

        ttk.Label(goto_frame, text="Go To:", style='Igor.TLabel').pack(side='left')
        goto_var = tk.StringVar(value="0")
        goto_entry = ttk.Entry(goto_frame, textvariable=goto_var, width=10)
        goto_entry.pack(side='left', padx=5)

        # Display options
        options_frame = ttk.LabelFrame(control_panel, text="Display Options",
                                       style='Igor.TFrame')
        options_frame.pack(fill='x', pady=10)

        # Color table selection
        color_var = tk.StringVar(value="Mud")
        ttk.Label(options_frame, text="Color Table:", style='Igor.TLabel').pack(anchor='w', padx=5)
        color_combo = ttk.Combobox(options_frame, textvariable=color_var,
                                   values=['Mud', 'hot', 'gray', 'viridis', 'plasma'])
        color_combo.pack(fill='x', padx=5, pady=2)

        # Color range
        range_var = tk.StringVar(value="-1")
        ttk.Label(options_frame, text="Color Range:", style='Igor.TLabel').pack(anchor='w', padx=5)
        range_entry = ttk.Entry(options_frame, textvariable=range_var)
        range_entry.pack(fill='x', padx=5, pady=2)

        # Checkboxes
        interp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Interpolate", variable=interp_var,
                        style='Igor.TLabel').pack(anchor='w', padx=5)

        perim_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Perimeter", variable=perim_var,
                        style='Igor.TLabel').pack(anchor='w', padx=5)

        # Range controls
        x_range_var = tk.StringVar(value="-1")
        ttk.Label(options_frame, text="X-Range:", style='Igor.TLabel').pack(anchor='w', padx=5)
        ttk.Entry(options_frame, textvariable=x_range_var).pack(fill='x', padx=5, pady=2)

        y_range_var = tk.StringVar(value="-1")
        ttk.Label(options_frame, text="Y-Range:", style='Igor.TLabel').pack(anchor='w', padx=5)
        ttk.Entry(options_frame, textvariable=y_range_var).pack(fill='x', padx=5, pady=2)

        # Measurements display
        meas_frame = ttk.Frame(control_panel, style='Igor.TFrame')
        meas_frame.pack(fill='x', pady=10)

        ttk.Label(meas_frame, text="Height", style='Igor.TLabel').pack(anchor='w')
        height_display = ttk.Label(meas_frame, text="0.0000",
                                   font=('Courier', 12), relief='sunken',
                                   background='white', foreground='black')
        height_display.pack(fill='x', pady=2)

        ttk.Label(meas_frame, text="Volume", style='Igor.TLabel').pack(anchor='w')
        volume_display = ttk.Label(meas_frame, text="0.0000",
                                   font=('Courier', 12), relief='sunken',
                                   background='white', foreground='black')
        volume_display.pack(fill='x', pady=2)

        # Delete button
        delete_btn = ttk.Button(control_panel, text="DELETE",
                                command=lambda: delete_particle(),
                                style='Igor.TButton')
        delete_btn.pack(pady=20)

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
            height_display.config(text=f"{particle.get('height', 0):.4f}")
            volume_display.config(text=f"{particle.get('volume', 0):.4f}")

            # Clear and display particle
            ax.clear()

            # Display particle image
            if 'image' in particle:
                # Determine colormap
                cmap = color_var.get()
                if cmap == 'Mud':
                    cmap = 'hot'  # Use hot as substitute for Mud

                im = ax.imshow(particle['image'], cmap=cmap,
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
                ax.set_xlabel('nm')
                ax.set_ylabel('nm')

                # Set color range
                try:
                    range_val = float(range_var.get())
                    if range_val > 0:
                        vmin = particle['image'].min()
                        im.set_clim(vmin, vmin + range_val)
                except:
                    pass

                # Set axis ranges
                try:
                    x_range = float(x_range_var.get())
                    if x_range > 0:
                        x_center = particle['image'].shape[1] / 2
                        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
                except:
                    pass

                try:
                    y_range = float(y_range_var.get())
                    if y_range > 0:
                        y_center = particle['image'].shape[0] / 2
                        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
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
        x_range_var.trace('w', update_display)
        y_range_var.trace('w', update_display)

        # Keyboard shortcuts
        def on_key(event):
            if event.keysym == 'Right':
                next_particle()
            elif event.keysym == 'Left':
                prev_particle()
            elif event.keysym == 'space' or event.keysym == 'Down':
                delete_particle()

        viewer_window.bind('<KeyPress>', on_key)
        viewer_window.focus_set()

        # Show first particle
        if all_particles:
            show_particle(0)

    def show_preprocessing(self):
        """Show preprocessing dialog"""
        if not self.images:
            messagebox.showinfo("Info", "Please load images first")
            return

        # Create preprocessing dialog
        preproc_window = tk.Toplevel(self.root)
        preproc_window.title("Preprocessing Parameters")
        preproc_window.geometry("400x250")
        preproc_window.configure(bg='#f0f0f0')

        # Make modal
        preproc_window.transient(self.root)
        preproc_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(preproc_window, style='Igor.TFrame')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Options
        ttk.Label(main_frame, text="Std. Deviations for streak removal?",
                  style='Igor.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        streak_var = tk.StringVar(value="3")
        ttk.Entry(main_frame, textvariable=streak_var, width=10).grid(row=0, column=1, pady=5)

        ttk.Label(main_frame, text="Polynomial order for flattening?",
                  style='Igor.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        flatten_var = tk.StringVar(value="2")
        ttk.Entry(main_frame, textvariable=flatten_var, width=10).grid(row=1, column=1, pady=5)

        # Buttons
        button_frame = ttk.Frame(preproc_window, style='Igor.TFrame')
        button_frame.pack(fill='x', pady=10)

        def apply_preprocessing():
            streak_sdevs = float(streak_var.get())
            flatten_order = int(flatten_var.get())

            if streak_sdevs > 0 or flatten_order > 0:
                from preprocessing import batch_preprocess

                # Apply preprocessing
                self.add_to_history("\nBatchPreprocess()")
                self.add_to_history(f"  Streak removal: {streak_sdevs} std devs")
                self.add_to_history(f"  Flattening order: {flatten_order}")

                self.images = batch_preprocess(self.images, streak_sdevs, flatten_order)
                self.preprocessing_done = True

                # Update display if current image is shown
                if self.current_image:
                    self.display_image(self.images[self.current_image])

                messagebox.showinfo("Success", "Preprocessing complete!")

            preproc_window.destroy()

        ttk.Button(button_frame, text="Continue", command=apply_preprocessing,
                   style='Igor.TButton').pack(side='left', padx=20)
        ttk.Button(button_frame, text="Cancel", command=preproc_window.destroy,
                   style='Igor.TButton').pack(side='left', padx=5)

    def show_command_window(self):
        """Show/focus command window"""
        self.notebook.select(0)  # Select command tab

    def new_graph(self):
        """Create new graph window"""
        graph_window = tk.Toplevel(self.root)
        graph_window.title(f"Graph{len(self.root.winfo_children())}")
        graph_window.geometry("600x500")

        # Create matplotlib figure
        fig = plt.Figure(figsize=(8, 6), facecolor='#f0f0f0')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')

        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()

    def new_folder(self):
        """Create new data folder"""
        # Get selected item
        selection = self.data_tree.selection()
        parent = selection[0] if selection else self.root_id

        # Ask for folder name
        dialog = tk.Toplevel(self.root)
        dialog.title("New Folder")
        dialog.geometry("300x100")
        dialog.configure(bg='#f0f0f0')

        ttk.Label(dialog, text="Folder name:", style='Igor.TLabel').pack(pady=5)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var).pack(pady=5)

        def create():
            name = name_var.get()
            if name:
                self.data_tree.insert(parent, 'end', text=name, values=('Folder', ''))
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=create, style='Igor.TButton').pack()

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
        self.ax.set_title(f'{self.current_image}')
        self.ax.set_xlabel('pixels')
        self.ax.set_ylabel('pixels')
        self.fig.colorbar(im, ax=self.ax)
        self.canvas.draw()

        # Switch to image display tab
        self.notebook.select(1)

    def on_tree_double_click(self, event):
        """Handle double-click on tree item"""
        selection = self.data_tree.selection()
        if not selection:
            return

        item = self.data_tree.item(selection[0])
        item_type = item['values'][0] if item['values'] else ''

        if item_type == 'Image' or item_type == 'Wave':
            self.display_selected_image()

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

        self.add_to_history(f"\nHessianBlobs(root:Images:{image_name})")
        self.add_to_history(f"Running analysis on {image_name}...")

        try:
            result = hessian_blobs(self.images[image_name], params=params)

            if result:
                self.add_to_history(f"Analysis complete for {image_name}")
                self.add_to_history(f"  Particles found: {len(result['particles'])}")

                # Add to data tree
                results_id = self.data_tree.insert(self.root_id, 'end',
                                                   text=f"{image_name}_Particles",
                                                   values=('Folder', ''), open=True)

                # Add result waves
                n_particles = len(result['particles'])
                self.data_tree.insert(results_id, 'end', text='Heights',
                                      values=('Wave', f'({n_particles})'))
                self.data_tree.insert(results_id, 'end', text='Areas',
                                      values=('Wave', f'({n_particles})'))
                self.data_tree.insert(results_id, 'end', text='Volumes',
                                      values=('Wave', f'({n_particles})'))

                # Add particles
                for i, particle in enumerate(result['particles']):
                    particle_folder = self.data_tree.insert(results_id, 'end',
                                                            text=f"Particle_{i}",
                                                            values=('Folder', ''))

                    self.data_tree.insert(particle_folder, 'end',
                                          text=f"Particle_{i}",
                                          values=('Wave', f"{particle['image'].shape}"))
                    self.data_tree.insert(particle_folder, 'end',
                                          text=f"Mask_{i}",
                                          values=('Wave', f"{particle['mask'].shape}"))
                    self.data_tree.insert(particle_folder, 'end',
                                          text=f"Perimeter_{i}",
                                          values=('Wave', f"{particle['perimeter'].shape}"))

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def execute_command(self, event=None):
        """Execute command from command line"""
        command = self.cmd_entry.get().strip()
        if not command:
            return

        # Add to history
        self.cmd_history.config(state='normal')
        self.cmd_history.insert('end', f"{command}\n")

        # Process command
        if command.lower() == 'help':
            self.cmd_history.insert('end', "Available commands:\n")
            self.cmd_history.insert('end', "  BatchHessianBlobs() - Run batch analysis\n")
            self.cmd_history.insert('end', "  HessianBlobs(image) - Run on single image\n")
            self.cmd_history.insert('end', "  ViewParticles() - View detected particles\n")
            self.cmd_history.insert('end', "  BatchPreprocess() - Preprocess images\n")
            self.cmd_history.insert('end', "  WaveStats(wave) - Calculate wave statistics\n")
        elif command == 'BatchHessianBlobs()':
            self.run_batch_analysis()
        elif command == 'ViewParticles()':
            self.view_particles()
        elif command == 'BatchPreprocess()':
            self.show_preprocessing()
        else:
            self.cmd_history.insert('end', f"Unknown command: {command}\n")

        self.cmd_history.insert('end', "•")
        self.cmd_history.see('end')
        self.cmd_history.config(state='disabled')

        # Clear entry
        self.cmd_entry.delete(0, 'end')

    def add_to_history(self, text):
        """Add text to command history"""
        self.cmd_history.config(state='normal')
        self.cmd_history.insert('end', f"{text}\n")
        self.cmd_history.see('end')
        self.cmd_history.config(state='disabled')

    def monitor_progress(self):
        """Monitor progress queue"""
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                self.add_to_history(msg)
        except:
            pass

        # Schedule next check
        self.root.after(100, self.monitor_progress)

    def show_about(self):
        """Show about dialog"""
        about_text = """Hessian Blob Particle Detection Suite

Based on the Igor Pro implementation by:
B.P. Marsh, G.M. King Laboratory
University of Missouri

The Hessian Blob Algorithm: Precise Particle Detection 
in Atomic Force Microscopy Imagery

Scientific Reports (2018)
doi:10.1038/s41598-018-19379-x"""

        messagebox.showinfo("About Hessian Blobs", about_text)


def launch_gui():
    """Launch the GUI application"""
    root = tk.Tk()

    # Set application icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass

    app = HessianBlobGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()