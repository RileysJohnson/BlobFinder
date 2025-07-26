"""
Particle Measurements Module
Contains particle measurement and analysis functions
Direct port from Igor Pro code maintaining same variable names and structure
Complete implementation with all measurement and viewing functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from igor_compatibility import *
from file_io import *
from utilities import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def MeasureParticles(im, mapNum, info, particleType=1):
    """
    Measure properties of detected particles
    Direct port from Igor Pro MeasureParticles function

    Parameters:
    im : Wave - Original image
    mapNum : Wave - Particle number map
    info : Wave - Particle information array (will be updated)
    particleType : int - Type of particles being measured

    Returns:
    bool - Success status
    """
    print("Measuring particle properties...")

    if info.data.shape[0] == 0:
        print("No particles to measure.")
        return False

    num_particles = info.data.shape[0]

    # Expand info array to include all measurements
    # Columns: X, Y, Scale, DetH, LG, XIndex, YIndex, ScaleIndex, Area, Volume, Height, COM_X, COM_Y
    if info.data.shape[1] < 13:
        new_info = np.zeros((num_particles, 13))
        new_info[:, :info.data.shape[1]] = info.data
        info.data = new_info

    for i in range(num_particles):
        x_coord = info.data[i, 0]
        y_coord = info.data[i, 1]
        scale = info.data[i, 2]

        # Convert coordinates to indices
        x_idx = int((x_coord - DimOffset(im, 0)) / DimDelta(im, 0))
        y_idx = int((y_coord - DimOffset(im, 1)) / DimDelta(im, 1))

        # Ensure indices are within bounds
        x_idx = max(0, min(im.data.shape[1] - 1, x_idx))
        y_idx = max(0, min(im.data.shape[0] - 1, y_idx))

        # Measure particle properties in a region around the center
        radius_pixels = int(scale / DimDelta(im, 0))
        radius_pixels = max(3, radius_pixels)  # Minimum radius

        # Define measurement region
        y_min = max(0, y_idx - radius_pixels)
        y_max = min(im.data.shape[0], y_idx + radius_pixels + 1)
        x_min = max(0, x_idx - radius_pixels)
        x_max = min(im.data.shape[1], x_idx + radius_pixels + 1)

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
                com_x_phys = DimOffset(im, 0) + com_x * DimDelta(im, 0)
                com_y_phys = DimOffset(im, 1) + com_y * DimDelta(im, 1)
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
    Interactive particle viewer with enhanced zoom and information display.
    """
    if info.data.shape[0] == 0:
        messagebox.showinfo("No Particles", "No particles to view.")
        return

    class ParticleViewer:
        def __init__(self, image, particle_info, particle_map=None):
            self.image = image
            self.info = particle_info  # This is a Wave object, we use its .data attribute
            self.map = particle_map
            self.current_particle_idx = 0
            self.num_particles = self.info.data.shape[0]

            self.root = tk.Toplevel()
            self.root.title("Interactive Particle Viewer")
            self.root.geometry("1200x800")

            self.setup_ui()
            self.update_view()

        def setup_ui(self):
            # Main layout
            main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
            main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Left panel for particle list and info
            left_frame = ttk.Frame(main_paned, width=350)
            main_paned.add(left_frame, weight=0)

            # Right panel for image display
            right_frame = ttk.Frame(main_paned)
            main_paned.add(right_frame, weight=1)

            # --- Left Panel ---
            # Particle list
            list_frame = ttk.LabelFrame(left_frame, text="Particles", padding=5)
            list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

            self.particle_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, exportselection=False)
            for i in range(self.num_particles):
                self.particle_listbox.insert(tk.END, f"Particle {i}")
            self.particle_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.particle_listbox.yview)
            self.particle_listbox.configure(yscrollcommand=list_scrollbar.set)
            list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.particle_listbox.bind('<<ListboxSelect>>', self.on_list_select)

            # Info display
            info_frame = ttk.LabelFrame(left_frame, text="Particle Details", padding=5)
            info_frame.pack(fill=tk.X, pady=(5, 0))
            self.info_text = scrolledtext.ScrolledText(info_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
            self.info_text.pack(fill=tk.BOTH, expand=True)

            # --- Right Panel ---
            # Controls
            controls_frame = ttk.Frame(right_frame)
            controls_frame.pack(fill=tk.X, pady=(0, 5))
            ttk.Button(controls_frame, text="<< Prev", command=self.prev_particle).pack(side=tk.LEFT, padx=2)
            ttk.Button(controls_frame, text="Next >>", command=self.next_particle).pack(side=tk.LEFT, padx=2)
            ttk.Button(controls_frame, text="Delete", command=self.delete_particle).pack(side=tk.RIGHT, padx=2)

            # Matplotlib figure for zoomed view
            self.fig = Figure(figsize=(8, 8), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def on_list_select(self, event):
            selection = self.particle_listbox.curselection()
            if selection:
                self.current_particle_idx = selection[0]
                self.update_view()

        def update_view(self):
            if not (0 <= self.current_particle_idx < self.num_particles):
                return

            # Update listbox selection
            self.particle_listbox.selection_clear(0, tk.END)
            self.particle_listbox.selection_set(self.current_particle_idx)
            self.particle_listbox.see(self.current_particle_idx)

            # Update particle info display
            self.update_info_text()

            # Update zoomed image display
            self.update_image_zoom()

        def update_info_text(self):
            p_info = self.info.data[self.current_particle_idx]

            info_str = f"--- Particle {self.current_particle_idx} ---\n"
            info_str += f"Position (X, Y): ({p_info[0]:.2f}, {p_info[1]:.2f})\n"
            info_str += f"Scale (Radius): {p_info[2]:.3f}\n"
            info_str += f"DetH Response: {p_info[3]:.4e}\n"

            if self.info.data.shape[1] > 4:
                info_str += f"Laplacian of Gaussian: {p_info[4]:.4f}\n"

            if self.info.data.shape[1] > 8:
                info_str += "\n--- Measurements ---\n"
                info_str += f"Area: {p_info[8]:.2f}\n"
                info_str += f"Volume: {p_info[9]:.3e}\n"
                info_str += f"Height: {p_info[10]:.4f}\n"

            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info_str)
            self.info_text.config(state=tk.DISABLED)

        def update_image_zoom(self):
            p_info = self.info.data[self.current_particle_idx]
            x_coord, y_coord, scale = p_info[0], p_info[1], p_info[2]

            x_pixel = (x_coord - DimOffset(self.image, 0)) / DimDelta(self.image, 0)
            y_pixel = (y_coord - DimOffset(self.image, 1)) / DimDelta(self.image, 1)

            zoom_radius = max(15, int(3 * scale / DimDelta(self.image, 0)))
            x_min = max(0, int(x_pixel - zoom_radius))
            x_max = min(self.image.data.shape[1], int(x_pixel + zoom_radius) + 1)
            y_min = max(0, int(y_pixel - zoom_radius))
            y_max = min(self.image.data.shape[0], int(y_pixel + zoom_radius) + 1)

            region = self.image.data[y_min:y_max, x_min:x_max]
            extent = [
                DimOffset(self.image, 0) + x_min * DimDelta(self.image, 0),
                DimOffset(self.image, 0) + (x_max - 1) * DimDelta(self.image, 0),
                DimOffset(self.image, 1) + y_min * DimDelta(self.image, 1),
                DimOffset(self.image, 1) + (y_max - 1) * DimDelta(self.image, 1)
            ]

            self.ax.clear()
            self.ax.imshow(region, cmap='gray', origin='lower', extent=extent, aspect='equal')

            # Draw circle for the blob
            circle = Circle((x_coord, y_coord), scale, fill=False, color='yellow', linewidth=1.5)
            self.ax.add_patch(circle)

            # Mark center
            self.ax.plot(x_coord, y_coord, 'r+', markersize=12, markeredgewidth=1.5)

            self.ax.set_title(f"Zoomed View: Particle {self.current_particle_idx}")
            self.ax.set_xlabel("X Coordinate")
            self.ax.set_ylabel("Y Coordinate")
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw()

        def next_particle(self):
            if self.current_particle_idx < self.num_particles - 1:
                self.current_particle_idx += 1
                self.update_view()

        def prev_particle(self):
            if self.current_particle_idx > 0:
                self.current_particle_idx -= 1
                self.update_view()

        def delete_particle(self):
            if self.num_particles == 0:
                return

            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete Particle {self.current_particle_idx}?"):
                # Remove from info wave
                self.info.data = np.delete(self.info.data, self.current_particle_idx, axis=0)

                # Update particle count
                old_num_particles = self.num_particles
                self.num_particles = self.info.data.shape[0]

                # Re-populate the listbox
                self.particle_listbox.delete(0, tk.END)
                for i in range(self.num_particles):
                    self.particle_listbox.insert(tk.END, f"Particle {i}")

                if self.num_particles == 0:
                    messagebox.showinfo("Empty", "All particles have been deleted.")
                    self.root.destroy()
                    return

                # Adjust current index if we deleted the last one
                if self.current_particle_idx >= self.num_particles:
                    self.current_particle_idx = self.num_particles - 1

                self.update_view()

    viewer = ParticleViewer(im, info, mapNum)


def AnalyzeParticleDistribution(info):
    """
    Analyze the spatial and size distribution of particles

    Parameters:
    info : Wave - Particle information array

    Returns:
    dict - Dictionary containing distribution statistics
    """
    if info.data.shape[0] == 0:
        return {}

    # Extract data
    x_coords = info.data[:, 0]
    y_coords = info.data[:, 1]
    scales = info.data[:, 2]

    # Spatial statistics
    spatial_stats = {
        'x_mean': np.mean(x_coords),
        'x_std': np.std(x_coords),
        'x_min': np.min(x_coords),
        'x_max': np.max(x_coords),
        'y_mean': np.mean(y_coords),
        'y_std': np.std(y_coords),
        'y_min': np.min(y_coords),
        'y_max': np.max(y_coords)
    }

    # Size statistics
    size_stats = {
        'scale_mean': np.mean(scales),
        'scale_std': np.std(scales),
        'scale_min': np.min(scales),
        'scale_max': np.max(scales),
        'scale_median': np.median(scales)
    }

    # Additional measurements if available
    measurements = {}
    if info.data.shape[1] > 8:
        if np.any(info.data[:, 8] > 0):  # Area
            areas = info.data[:, 8]
            measurements['area_mean'] = np.mean(areas)
            measurements['area_std'] = np.std(areas)
            measurements['area_median'] = np.median(areas)

        if np.any(info.data[:, 9] > 0):  # Volume
            volumes = info.data[:, 9]
            measurements['volume_mean'] = np.mean(volumes)
            measurements['volume_std'] = np.std(volumes)
            measurements['volume_median'] = np.median(volumes)

        if np.any(info.data[:, 10] > 0):  # Height
            heights = info.data[:, 10]
            measurements['height_mean'] = np.mean(heights)
            measurements['height_std'] = np.std(heights)
            measurements['height_median'] = np.median(heights)

    return {
        'num_particles': info.data.shape[0],
        'spatial': spatial_stats,
        'size': size_stats,
        'measurements': measurements
    }


def ExportParticleData(info, filename):
    """
    Export particle data to CSV file

    Parameters:
    info : Wave - Particle information array
    filename : str - Output filename
    """
    if info.data.shape[0] == 0:
        print("No particle data to export")
        return False

    try:
        # Create header
        headers = ["Particle_ID", "X", "Y", "Scale", "DetH", "LG", "X_Index", "Y_Index", "Scale_Index"]

        if info.data.shape[1] > 8:
            headers.extend(["Area", "Volume", "Height"])

        if info.data.shape[1] > 11:
            headers.extend(["COM_X", "COM_Y"])

        # Prepare data
        data_to_save = []
        for i in range(info.data.shape[0]):
            row = [i] + list(info.data[i, :len(headers) - 1])
            data_to_save.append(row)

        # Save to CSV
        import csv
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(data_to_save)

        print(f"Exported {info.data.shape[0]} particles to {filename}")
        return True

    except Exception as e:
        print(f"Error exporting data: {e}")
        return False


def PlotParticleDistribution(info):
    """
    Create plots showing particle distribution

    Parameters:
    info : Wave - Particle information array
    """
    if info.data.shape[0] == 0:
        messagebox.showinfo("No Data", "No particle data to plot.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Particle Distribution Analysis ({info.data.shape[0]} particles)")

    # Spatial distribution
    axes[0, 0].scatter(info.data[:, 0], info.data[:, 1], alpha=0.6)
    axes[0, 0].set_xlabel("X Position")
    axes[0, 0].set_ylabel("Y Position")
    axes[0, 0].set_title("Spatial Distribution")
    axes[0, 0].grid(True, alpha=0.3)

    # Size distribution
    axes[0, 1].hist(info.data[:, 2], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("Scale (radius)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Size Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    # Detection response
    axes[1, 0].scatter(info.data[:, 3], info.data[:, 4], alpha=0.6)
    axes[1, 0].set_xlabel("DetH Response")
    axes[1, 0].set_ylabel("Laplacian of Gaussian")
    axes[1, 0].set_title("Detection Response")
    axes[1, 0].grid(True, alpha=0.3)

    # Size vs intensity (if height data available)
    if info.data.shape[1] > 10 and np.any(info.data[:, 10] > 0):
        axes[1, 1].scatter(info.data[:, 2], info.data[:, 10], alpha=0.6)
        axes[1, 1].set_xlabel("Scale (radius)")
        axes[1, 1].set_ylabel("Height")
        axes[1, 1].set_title("Size vs Height")
    else:
        axes[1, 1].scatter(info.data[:, 2], info.data[:, 3], alpha=0.6)
        axes[1, 1].set_xlabel("Scale (radius)")
        axes[1, 1].set_ylabel("DetH Response")
        axes[1, 1].set_title("Size vs Response")

    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def FilterParticles(info, min_scale=None, max_scale=None, min_height=None, max_height=None):
    """
    Filter particles based on various criteria

    Parameters:
    info : Wave - Particle information array
    min_scale, max_scale : float - Scale (radius) filtering
    min_height, max_height : float - Height filtering

    Returns:
    Wave - Filtered particle information
    """
    if info.data.shape[0] == 0:
        return info

    # Start with all particles
    mask = np.ones(info.data.shape[0], dtype=bool)

    # Apply scale filtering
    if min_scale is not None:
        mask &= (info.data[:, 2] >= min_scale)
    if max_scale is not None:
        mask &= (info.data[:, 2] <= max_scale)

    # Apply height filtering (if height data available)
    if info.data.shape[1] > 10:
        if min_height is not None:
            mask &= (info.data[:, 10] >= min_height)
        if max_height is not None:
            mask &= (info.data[:, 10] <= max_height)

    # Create filtered array
    filtered_data = info.data[mask]
    filtered_info = Wave(filtered_data, f"{info.name}_filtered")

    print(f"Filtered {np.sum(~mask)} particles, {np.sum(mask)} remaining")

    return filtered_info


def Testing(string_input, number_input):
    """Testing function for particle_measurements module"""
    print(f"Particle measurements testing: {string_input}, {number_input}")
    return len(string_input) + number_input