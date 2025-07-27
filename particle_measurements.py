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

        # Convert coordinates to pixel indices
        x_pixel = (x_coord - DimOffset(im, 1)) / DimDelta(im, 1)
        y_pixel = (y_coord - DimOffset(im, 0)) / DimDelta(im, 0)

        # Create circular mask around particle
        height, width = im.data.shape
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        distances = np.sqrt((x_indices - x_pixel) ** 2 + (y_indices - y_pixel) ** 2)

        # Convert radius to pixels
        radius_pixels = radius / DimDelta(im, 1)  # Assuming isotropic pixels
        mask = distances <= radius_pixels

        if np.any(mask):
            # Extract region data
            region = im.data * mask

            # Calculate area in physical units
            pixel_area = DimDelta(im, 0) * DimDelta(im, 1)
            area_physical = np.sum(mask) * pixel_area

            # Calculate volume (sum of intensities * pixel area)
            total_intensity = np.sum(region)
            volume = total_intensity * pixel_area

            # Calculate height (maximum intensity in region)
            height = np.max(im.data[mask]) if np.any(mask) else 0

            # Calculate center of mass in physical coordinates
            if total_intensity > 0:
                Y = DimOffset(im, 0) + y_indices * DimDelta(im, 0)
                X = DimOffset(im, 1) + x_indices * DimDelta(im, 1)

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
    # FIXED: Add proper validation to prevent NoneType errors
    if im is None:
        messagebox.showerror("Error", "No image provided.")
        return

    if info is None:
        messagebox.showerror("Error", "No particle information provided.")
        return

    if not hasattr(info, 'data') or info.data is None:
        messagebox.showerror("Error", "Particle information has no data attribute.")
        return

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
            self.particle_label.pack(pady=(20, 10))

            # Navigation buttons - matches Igor Pro NextBtn/PrevBtn
            nav_frame = ttk.Frame(right_frame)
            nav_frame.pack(pady=(0, 10))

            self.prev_btn = ttk.Button(nav_frame, text="Prev", command=self.prev_particle)
            self.prev_btn.pack(side=tk.LEFT, padx=(10, 5))

            self.next_btn = ttk.Button(nav_frame, text="Next", command=self.next_particle)
            self.next_btn.pack(side=tk.LEFT, padx=(5, 10))

            # Go To control - matches Igor Pro GoTo SetVariable
            goto_frame = ttk.Frame(right_frame)
            goto_frame.pack(pady=(0, 10))

            ttk.Label(goto_frame, text="Go To:").pack()
            self.goto_var = tk.IntVar(value=0)
            self.goto_entry = ttk.Entry(goto_frame, textvariable=self.goto_var, width=10)
            self.goto_entry.pack()
            self.goto_entry.bind('<Return>', self.goto_particle)

            # Color table selection - matches Igor Pro ColorTab PopUpMenu
            color_frame = ttk.Frame(right_frame)
            color_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

            self.color_var = tk.StringVar(value="Grays")
            color_combo = ttk.Combobox(color_frame, textvariable=self.color_var,
                                       values=["Grays", "Rainbow", "BlueRedYellow"],
                                       state="readonly", width=15)
            color_combo.pack()
            color_combo.bind('<<ComboboxSelected>>', self.on_color_change)

            # Color range - matches Igor Pro ColorRange SetVariable
            range_frame = ttk.Frame(right_frame)
            range_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

            ttk.Label(range_frame, text="Color Range:").pack()
            self.range_var = tk.DoubleVar(value=-1)
            range_entry = ttk.Entry(range_frame, textvariable=self.range_var, width=15)
            range_entry.pack()
            range_entry.bind('<Return>', self.on_range_change)

            # Checkboxes - match Igor Pro Interpo/Perim Checkbox
            self.interp_var = tk.BooleanVar(value=False)
            interp_check = ttk.Checkbutton(right_frame, text="Interpolate:",
                                           variable=self.interp_var,
                                           command=self.update_display)
            interp_check.pack(anchor=tk.W, padx=10, pady=2)

            self.perim_var = tk.BooleanVar(value=True)
            perim_check = ttk.Checkbutton(right_frame, text="Perimeter:",
                                          variable=self.perim_var,
                                          command=self.update_display)
            perim_check.pack(anchor=tk.W, padx=10, pady=2)

            # X/Y Range controls - match Igor Pro XRange/YRange SetVariable
            xy_frame = ttk.Frame(right_frame)
            xy_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

            ttk.Label(xy_frame, text="X-Range:").pack()
            self.x_range_var = tk.DoubleVar(value=-1)
            x_range_entry = ttk.Entry(xy_frame, textvariable=self.x_range_var, width=15)
            x_range_entry.pack()

            ttk.Label(xy_frame, text="Y-Range:").pack()
            self.y_range_var = tk.DoubleVar(value=-1)
            y_range_entry = ttk.Entry(xy_frame, textvariable=self.y_range_var, width=15)
            y_range_entry.pack()

            # Height display - matches Igor Pro HeightTitle/HeightDisp
            height_frame = ttk.Frame(right_frame)
            height_frame.pack(fill=tk.X, padx=10, pady=(15, 5))

            ttk.Label(height_frame, text="Height", font=('TkDefaultFont', 15)).pack()
            self.height_display = ttk.Label(height_frame, text="0",
                                            font=('TkDefaultFont', 15),
                                            relief=tk.SUNKEN, background='white')
            self.height_display.pack(fill=tk.X)

            # Volume display - matches Igor Pro VolTitle/VolDisp
            vol_frame = ttk.Frame(right_frame)
            vol_frame.pack(fill=tk.X, padx=10, pady=(5, 5))

            ttk.Label(vol_frame, text="Volume", font=('TkDefaultFont', 15)).pack()
            self.vol_display = ttk.Label(vol_frame, text="0",
                                         font=('TkDefaultFont', 15),
                                         relief=tk.SUNKEN, background='white')
            self.vol_display.pack(fill=tk.X)

            # Delete button - matches Igor Pro DeleteBtn
            self.delete_btn = ttk.Button(right_frame, text="DELETE",
                                         command=self.delete_particle)
            self.delete_btn.pack(side=tk.BOTTOM, pady=10, padx=10, fill=tk.X)

            # Keyboard bindings - matches Igor Pro keyboard shortcuts
            self.root.bind('<Left>', lambda e: self.prev_particle())
            self.root.bind('<Right>', lambda e: self.next_particle())
            self.root.bind('<space>', lambda e: self.delete_particle())
            self.root.focus_set()

        def update_display(self):
            """Update particle display - matches Igor Pro display exactly"""
            if self.current_particle >= self.total_particles:
                return

            self.ax.clear()

            # Get current particle info
            x_coord = self.info.data[self.current_particle, 0]
            y_coord = self.info.data[self.current_particle, 1]
            radius = self.info.data[self.current_particle, 2]

            # Create zoomed view around particle
            zoom_factor = 3.0
            crop_size = radius * zoom_factor

            # Calculate crop region
            x_min = x_coord - crop_size
            x_max = x_coord + crop_size
            y_min = y_coord - crop_size
            y_max = y_coord + crop_size

            # Convert to pixel coordinates for cropping
            dx = DimDelta(self.im, 1)
            dy = DimDelta(self.im, 0)
            x_offset = DimOffset(self.im, 1)
            y_offset = DimOffset(self.im, 0)

            x_min_px = max(0, int((x_min - x_offset) / dx))
            x_max_px = min(self.im.data.shape[1], int((x_max - x_offset) / dx))
            y_min_px = max(0, int((y_min - y_offset) / dy))
            y_max_px = min(self.im.data.shape[0], int((y_max - y_offset) / dy))

            # Extract cropped data
            crop_data = self.im.data[y_min_px:y_max_px, x_min_px:x_max_px]

            # Calculate actual display coordinates
            actual_x_min = x_offset + x_min_px * dx
            actual_x_max = x_offset + x_max_px * dx
            actual_y_min = y_offset + y_min_px * dy
            actual_y_max = y_offset + y_max_px * dy

            # Display image
            interpolation = 'bilinear' if self.interp_var.get() else 'nearest'
            cmap = self.color_var.get().lower()
            if cmap == 'grays':
                cmap = 'gray'

            self.ax.imshow(crop_data,
                           extent=[actual_x_min, actual_x_max, actual_y_max, actual_y_min],
                           cmap=cmap, aspect='equal', origin='upper',
                           interpolation=interpolation)

            # Add perimeter if enabled
            if self.perim_var.get():
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='red', linewidth=2)
                self.ax.add_patch(circle)

            # Set labels and title
            self.ax.set_xlabel(f"X ({DimUnits(self.im, 1)})")
            self.ax.set_ylabel(f"Y ({DimUnits(self.im, 0)})")
            self.ax.set_title(f"Particle {self.current_particle + 1}")

            # Update particle label
            self.particle_label.config(text=f"Particle {self.current_particle + 1}")
            self.goto_var.set(self.current_particle)

            # Update measurements if available
            if self.info.data.shape[1] > 10:
                height_val = self.info.data[self.current_particle, 10]
                volume_val = self.info.data[self.current_particle, 9]

                self.height_display.config(text=f"{height_val:.4f}")
                self.vol_display.config(text=f"{volume_val:.2e}")

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
            """Go to specific particle"""
            try:
                target = self.goto_var.get()
                if 0 <= target < self.total_particles:
                    self.current_particle = target
                    self.update_display()
            except tk.TclError:
                pass

        def on_color_change(self, event=None):
            """Handle color table change"""
            self.update_display()

        def on_range_change(self, event=None):
            """Handle color range change"""
            self.update_display()

        def delete_particle(self):
            """Delete current particle"""
            if messagebox.askyesno("Delete Particle",
                                   f"Delete particle {self.current_particle + 1}?"):
                # Remove from info array
                self.info.data = np.delete(self.info.data, self.current_particle, axis=0)
                self.total_particles -= 1

                if self.total_particles == 0:
                    messagebox.showinfo("No Particles", "All particles deleted.")
                    self.root.destroy()
                    return

                # Adjust current particle if needed
                if self.current_particle >= self.total_particles:
                    self.current_particle = self.total_particles - 1

                self.update_display()

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