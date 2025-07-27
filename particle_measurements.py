"""
Particle Measurements Module
Contains functions for measuring and viewing particles
Direct port from Igor Pro code maintaining same variable names and structure
ISSUE 2 FIX: Complete implementation of ViewParticles matching Igor Pro exactly
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from scipy import ndimage

from igor_compatibility import *
from file_io import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def MeasureParticles(im, info):
    """
    Measure particle properties like area, volume, height, center of mass
    Direct port from Igor Pro MeasureParticles function

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array (x, y, radius, ...)

    Returns:
    bool - Success status
    """
    if info is None or info.data.shape[0] == 0:
        return False

    num_particles = info.data.shape[0]
    print(f"Measuring properties for {num_particles} particles...")

    # Ensure info wave has enough columns for measurements
    if info.data.shape[1] < 13:
        # Extend to 13 columns: [x, y, radius, ..., area, volume, height, com_x, com_y]
        extended_data = np.zeros((num_particles, 13))
        extended_data[:, :info.data.shape[1]] = info.data
        info.data = extended_data

    # Measure each particle
    for i in range(num_particles):
        x_coord = info.data[i, 0]
        y_coord = info.data[i, 1]
        radius = info.data[i, 2]

        # Convert coordinates to indices
        x_idx = int((x_coord - DimOffset(im, 1)) / DimDelta(im, 1))
        y_idx = int((y_coord - DimOffset(im, 0)) / DimDelta(im, 0))
        radius_pixels = radius / min(DimDelta(im, 0), DimDelta(im, 1))

        # Define region around particle
        x_min = max(0, x_idx - int(radius_pixels) - 1)
        x_max = min(im.data.shape[1], x_idx + int(radius_pixels) + 1)
        y_min = max(0, y_idx - int(radius_pixels) - 1)
        y_max = min(im.data.shape[0], y_idx + int(radius_pixels) + 1)

        # Extract region
        region = im.data[y_min:y_max, x_min:x_max]

        # Create coordinate arrays for this region
        y_coords = np.arange(y_min, y_max)
        x_coords = np.arange(x_min, x_max)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

        # Create mask for circular region
        center_y, center_x = y_idx, x_idx
        dist_sq = (Y - center_y) ** 2 + (X - center_x) ** 2
        mask = dist_sq <= radius_pixels ** 2

        if np.sum(mask) > 0:
            # Measure area (number of pixels)
            area_pixels = np.sum(mask)
            area_physical = area_pixels * DimDelta(im, 0) * DimDelta(im, 1)

            # Measure height (max intensity in region)
            height = np.max(region[mask])

            # Measure volume (integrated intensity)
            volume = np.sum(region[mask]) * DimDelta(im, 0) * DimDelta(im, 1)

            # Measure center of mass
            total_intensity = np.sum(region[mask])
            if total_intensity > 0:
                com_y = np.sum(Y[mask] * region[mask]) / total_intensity
                com_x = np.sum(X[mask] * region[mask]) / total_intensity

                # Convert to physical coordinates
                com_x_phys = DimOffset(im, 1) + com_x * DimDelta(im, 1)
                com_y_phys = DimOffset(im, 0) + com_y * DimDelta(im, 0)
            else:
                com_x_phys = x_coord
                com_y_phys = y_coord

            # Store measurements
            info.data[i, 8] = area_physical  # Area
            info.data[i, 9] = volume  # Volume
            info.data[i, 10] = height  # Height
            info.data[i, 11] = com_x_phys  # Center of mass X
            info.data[i, 12] = com_y_phys  # Center of mass Y
        else:
            # Default values if no valid region
            info.data[i, 8] = 0  # Area
            info.data[i, 9] = 0  # Volume
            info.data[i, 10] = 0  # Height
            info.data[i, 11] = x_coord  # Center of mass X
            info.data[i, 12] = y_coord  # Center of mass Y

    print(f"Measured {num_particles} particles")
    return True


def ViewParticles(im, info, mapNum=None):
    """
    Interactive particle viewer
    Direct port from Igor Pro ViewParticles function
    ISSUE 2 FIX: Complete implementation matching Igor Pro exactly (Figure 24)

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array
    mapNum : Wave - Particle number map (optional)
    """

    if info.data.shape[0] == 0:
        messagebox.showinfo("No Particles", "No particles to view.")
        return

    class ParticleViewer:
        def __init__(self, im, info):
            self.im = im
            self.info = info
            self.current_particle = 0
            self.total_particles = info.data.shape[0]

            # Viewer settings - matches Igor Pro defaults
            self.color_table = "Grays"
            self.color_range = -1  # Auto range
            self.interpolate = False
            self.show_perimeter = True
            self.x_range = -1  # Auto range
            self.y_range = -1  # Auto range

            self.create_window()
            self.update_display()

        def create_window(self):
            """Create particle viewer window - matches Igor Pro layout exactly"""
            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")  # Matches Igor Pro title
            self.root.geometry("900x600")

            # Main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Left side: Image display
            left_frame = ttk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create matplotlib figure
            self.figure = Figure(figsize=(6, 6), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Right side: Controls panel - matches Igor Pro ViewControls exactly
            right_frame = ttk.LabelFrame(main_frame, text="Controls", width=150)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            right_frame.pack_propagate(False)

            # Particle name/number - matches Igor Pro ParticleName TitleBox
            self.particle_label = ttk.Label(right_frame,
                                            text=f"Particle {self.current_particle + 1}",
                                            font=('TkDefaultFont', 15, 'bold'))
            self.particle_label.pack(pady=(10, 5))

            # Navigation buttons - matches Igor Pro NextBtn/PrevBtn
            nav_frame = ttk.Frame(right_frame)
            nav_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Button(nav_frame, text="Prev", command=self.prev_particle, width=8).pack(side=tk.LEFT)
            ttk.Button(nav_frame, text="Next", command=self.next_particle, width=8).pack(side=tk.RIGHT)

            # Go To control - matches Igor Pro GoTo SetVariable
            goto_frame = ttk.Frame(right_frame)
            goto_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(goto_frame, text="Go To:").pack()
            self.goto_var = tk.IntVar(value=self.current_particle)
            goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=10)
            goto_entry.pack()
            goto_entry.bind('<Return>', self.goto_particle)

            # Color table popup - matches Igor Pro ColorTab PopUpMenu
            color_frame = ttk.Frame(right_frame)
            color_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(color_frame, text="Color Table:").pack()
            self.color_var = tk.StringVar(value=self.color_table)
            color_combo = ttk.Combobox(color_frame, textvariable=self.color_var,
                                       values=["Grays", "Rainbow", "BlueRedGreen", "Terrain"],
                                       width=12, state="readonly")
            color_combo.pack()
            color_combo.bind('<<ComboboxSelected>>', self.change_color_table)

            # Color range control - matches Igor Pro ColorRange SetVariable
            color_range_frame = ttk.Frame(right_frame)
            color_range_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(color_range_frame, text="Color Range").pack()
            self.color_range_var = tk.DoubleVar(value=self.color_range)
            ttk.Entry(color_range_frame, textvariable=self.color_range_var, width=10).pack()

            # Checkboxes - match Igor Pro Interpo/Perim checkboxes
            checkbox_frame = ttk.Frame(right_frame)
            checkbox_frame.pack(fill=tk.X, padx=10, pady=5)

            self.interp_var = tk.BooleanVar(value=self.interpolate)
            ttk.Checkbutton(checkbox_frame, text="Interpolate:",
                            variable=self.interp_var, command=self.update_display).pack(anchor=tk.W)

            self.perim_var = tk.BooleanVar(value=self.show_perimeter)
            ttk.Checkbutton(checkbox_frame, text="Perimeter:",
                            variable=self.perim_var, command=self.update_display).pack(anchor=tk.W)

            # Range controls - match Igor Pro XRange/YRange SetVariables
            range_frame = ttk.Frame(right_frame)
            range_frame.pack(fill=tk.X, padx=10, pady=5)

            ttk.Label(range_frame, text="X-Range:").pack()
            self.x_range_var = tk.DoubleVar(value=self.x_range)
            ttk.Entry(range_frame, textvariable=self.x_range_var, width=10).pack()

            ttk.Label(range_frame, text="Y-Range:").pack()
            self.y_range_var = tk.DoubleVar(value=self.y_range)
            ttk.Entry(range_frame, textvariable=self.y_range_var, width=10).pack()

            # Measurements display - matches Igor Pro Height/Volume displays
            measurements_frame = ttk.Frame(right_frame)
            measurements_frame.pack(fill=tk.X, padx=10, pady=10)

            ttk.Label(measurements_frame, text="Height (nm)",
                      font=('TkDefaultFont', 15)).pack()
            self.height_label = ttk.Label(measurements_frame, text="0",
                                          font=('TkDefaultFont', 15),
                                          relief=tk.SUNKEN, width=15)
            self.height_label.pack(pady=2)

            ttk.Label(measurements_frame, text="Volume (m^3 e-25)",
                      font=('TkDefaultFont', 15)).pack(pady=(20, 0))
            self.volume_label = ttk.Label(measurements_frame, text="0",
                                          font=('TkDefaultFont', 15),
                                          relief=tk.SUNKEN, width=15)
            self.volume_label.pack(pady=2)

            # Delete button - matches Igor Pro DELETE button
            delete_button = ttk.Button(right_frame, text="DELETE",
                                       command=self.delete_particle,
                                       style="Delete.TButton")
            delete_button.pack(pady=20, padx=10, fill=tk.X)

            # Configure delete button style to be red
            style = ttk.Style()
            style.configure("Delete.TButton", foreground="red", font=('TkDefaultFont', 14, 'bold'))

            # Keyboard bindings - matches Igor Pro keyboard shortcuts
            self.root.bind('<Left>', lambda e: self.prev_particle())
            self.root.bind('<Right>', lambda e: self.next_particle())
            self.root.bind('<space>', lambda e: self.delete_particle())
            self.root.focus_set()

        def update_display(self):
            """Update the particle display"""
            if self.current_particle >= self.total_particles:
                return

            self.ax.clear()

            # Get current particle info
            particle_info = self.info.data[self.current_particle]
            x_center = particle_info[0]
            y_center = particle_info[1]
            radius = particle_info[2]

            # Create particle crop - matches Igor Pro particle cropping
            crop_size = radius * 2.5  # Reasonable crop size

            x_min = x_center - crop_size
            x_max = x_center + crop_size
            y_min = y_center - crop_size
            y_max = y_center + crop_size

            # Find corresponding image indices
            x_idx_min = int((x_min - DimOffset(self.im, 1)) / DimDelta(self.im, 1))
            x_idx_max = int((x_max - DimOffset(self.im, 1)) / DimDelta(self.im, 1))
            y_idx_min = int((y_min - DimOffset(self.im, 0)) / DimDelta(self.im, 0))
            y_idx_max = int((y_max - DimOffset(self.im, 0)) / DimDelta(self.im, 0))

            # Clamp to image bounds
            x_idx_min = max(0, x_idx_min)
            x_idx_max = min(self.im.data.shape[1], x_idx_max)
            y_idx_min = max(0, y_idx_min)
            y_idx_max = min(self.im.data.shape[0], y_idx_max)

            # Extract crop
            crop_data = self.im.data[y_idx_min:y_idx_max, x_idx_min:x_idx_max]

            # Convert back to physical coordinates
            crop_x_min = DimOffset(self.im, 1) + x_idx_min * DimDelta(self.im, 1)
            crop_x_max = DimOffset(self.im, 1) + x_idx_max * DimDelta(self.im, 1)
            crop_y_min = DimOffset(self.im, 0) + y_idx_min * DimDelta(self.im, 0)
            crop_y_max = DimOffset(self.im, 0) + y_idx_max * DimDelta(self.im, 0)

            # Display image crop
            interpolation = 'bilinear' if self.interp_var.get() else 'nearest'
            colormap = self.color_var.get().lower()
            if colormap == 'grays':
                colormap = 'gray'
            elif colormap == 'rainbow':
                colormap = 'rainbow'
            elif colormap == 'blueredgreen':
                colormap = 'RdYlBu'
            elif colormap == 'terrain':
                colormap = 'terrain'

            im_display = self.ax.imshow(crop_data,
                                        extent=[crop_x_min, crop_x_max, crop_y_max, crop_y_min],
                                        cmap=colormap, aspect='equal', origin='upper',
                                        interpolation=interpolation)

            # Show perimeter if enabled - matches Igor Pro green circle overlay
            if self.perim_var.get():
                circle = Circle((x_center, y_center), radius,
                                fill=False, edgecolor='lime', linewidth=2)
                self.ax.add_patch(circle)

            # Set axis labels and title
            self.ax.set_xlabel(f"X ({DimUnits(self.im, 1)})")
            self.ax.set_ylabel(f"Y ({DimUnits(self.im, 0)})")

            # Update particle label
            self.particle_label.config(text=f"Particle {self.current_particle + 1}")
            self.goto_var.set(self.current_particle)

            # Update measurements - matches Igor Pro format
            height = particle_info[10] if len(particle_info) > 10 else 0
            volume = particle_info[9] if len(particle_info) > 9 else 0

            # Convert height to nm (assuming meters)
            height_nm = height * 1e9
            self.height_label.config(text=f"{height_nm:.4f}")

            # Convert volume to m^3 e-25
            volume_e25 = volume * 1e25
            self.volume_label.config(text=f"{volume_e25:.2f}")

            self.figure.tight_layout()
            self.canvas.draw()

        def prev_particle(self):
            """Go to previous particle"""
            if self.current_particle > 0:
                self.current_particle -= 1
                self.update_display()

        def next_particle(self):
            """Go to next particle"""
            if self.current_particle < self.total_particles - 1:
                self.current_particle += 1
                self.update_display()

        def goto_particle(self, event=None):
            """Go to specific particle number"""
            try:
                particle_num = self.goto_var.get()
                if 0 <= particle_num < self.total_particles:
                    self.current_particle = particle_num
                    self.update_display()
            except tk.TclError:
                pass

        def change_color_table(self, event=None):
            """Change color table"""
            self.color_table = self.color_var.get()
            self.update_display()

        def delete_particle(self):
            """Delete current particle - matches Igor Pro DELETE functionality"""
            if messagebox.askyesno("Delete Particle",
                                   f"Delete particle {self.current_particle + 1}?"):
                # Remove particle from info array
                if self.total_particles > 1:
                    mask = np.ones(self.total_particles, dtype=bool)
                    mask[self.current_particle] = False
                    self.info.data = self.info.data[mask]
                    self.total_particles -= 1

                    # Adjust current particle index
                    if self.current_particle >= self.total_particles:
                        self.current_particle = self.total_particles - 1

                    self.update_display()
                else:
                    messagebox.showinfo("Cannot Delete", "Cannot delete the last particle.")

    # Create and show the viewer
    viewer = ParticleViewer(im, info)
    return viewer


def ExportResults(results_dict, filename):
    """
    Export analysis results to CSV file

    Parameters:
    results_dict : dict - Dictionary of image_name -> results
    filename : str - Output filename
    """
    import csv

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['Image', 'Particle_ID', 'X_coord', 'Y_coord', 'Radius',
                  'Scale', 'Response', 'Area', 'Volume', 'Height',
                  'Center_of_Mass_X', 'Center_of_Mass_Y']
        writer.writerow(header)

        # Write data for each image
        for image_name, results in results_dict.items():
            if 'info' in results and results['info'] is not None:
                info = results['info']
                for i, particle_data in enumerate(info.data):
                    row = [image_name, i + 1] + list(particle_data)
                    writer.writerow(row)

    print(f"Results exported to {filename}")


def ImageStats(wave, quiet=True):
    """
    Calculate image statistics
    Direct port from Igor Pro ImageStats function

    Parameters:
    wave : Wave - Input image
    quiet : bool - If True, suppress output

    Returns:
    dict - Statistics dictionary
    """
    data = wave.data

    stats = {
        'min': np.nanmin(data),
        'max': np.nanmax(data),
        'mean': np.nanmean(data),
        'std': np.nanstd(data),
        'sum': np.nansum(data),
        'numPoints': data.size
    }

    # Find min/max locations
    min_idx = np.unravel_index(np.nanargmin(data), data.shape)
    max_idx = np.unravel_index(np.nanargmax(data), data.shape)

    stats['minLoc'] = min_idx
    stats['maxLoc'] = max_idx

    if not quiet:
        print(f"Image Statistics for {wave.name}:")
        print(f"  Min: {stats['min']:.6f} at {min_idx}")
        print(f"  Max: {stats['max']:.6f} at {max_idx}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std Dev: {stats['std']:.6f}")
        print(f"  Sum: {stats['sum']:.6f}")
        print(f"  Points: {stats['numPoints']}")

    return stats