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
    Interactive particle viewer
    Direct port from Igor Pro ViewParticles function

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array
    mapNum : Wave - Particle number map (optional)
    """

    if info.data.shape[0] == 0:
        messagebox.showinfo("No Particles", "No particles to view.")
        return

    class ParticleViewer:
        def __init__(self, image, particle_info, particle_map=None):
            self.image = image
            self.info = particle_info
            self.map = particle_map
            self.current_particle = 0
            self.num_particles = particle_info.data.shape[0]

            self.root = tk.Toplevel()
            self.root.title("Particle Viewer")
            self.root.geometry("800x600")

            self.setup_ui()
            self.update_display()

        def setup_ui(self):
            """Setup the particle viewer UI"""
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Controls frame
            controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            controls_frame.pack(fill=tk.X, pady=(0, 10))

            # Navigation controls
            nav_frame = ttk.Frame(controls_frame)
            nav_frame.pack(fill=tk.X)

            ttk.Button(nav_frame, text="Previous",
                       command=self.prev_particle).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(nav_frame, text="Next",
                       command=self.next_particle).pack(side=tk.LEFT, padx=(0, 5))

            # Particle selection
            ttk.Label(nav_frame, text="Particle:").pack(side=tk.LEFT, padx=(20, 5))
            self.particle_var = tk.IntVar(value=0)
            self.particle_spinbox = ttk.Spinbox(nav_frame, from_=0, to=self.num_particles - 1,
                                                textvariable=self.particle_var, width=10,
                                                command=self.on_particle_change)
            self.particle_spinbox.pack(side=tk.LEFT, padx=(0, 5))

            ttk.Label(nav_frame, text=f"of {self.num_particles}").pack(side=tk.LEFT, padx=(0, 20))

            # Delete button
            ttk.Button(nav_frame, text="Delete Particle",
                       command=self.delete_particle).pack(side=tk.RIGHT)

            # Display frame
            display_frame = ttk.LabelFrame(main_frame, text="Particle Display", padding="10")
            display_frame.pack(fill=tk.BOTH, expand=True)

            # Create matplotlib figure
            self.fig = Figure(figsize=(8, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Info frame
            info_frame = ttk.LabelFrame(main_frame, text="Particle Information", padding="10")
            info_frame.pack(fill=tk.X, pady=(10, 0))

            self.info_text = scrolledtext.ScrolledText(info_frame, height=6, width=80)
            self.info_text.pack(fill=tk.BOTH, expand=True)

        def update_display(self):
            """Update the particle display"""
            if self.current_particle >= self.num_particles:
                self.current_particle = self.num_particles - 1
            if self.current_particle < 0:
                self.current_particle = 0

            # Update spinbox
            self.particle_var.set(self.current_particle)

            # Get particle info
            p_info = self.info.data[self.current_particle]
            x_coord = p_info[0]
            y_coord = p_info[1]
            scale = p_info[2]

            # Convert to pixel coordinates
            x_pixel = (x_coord - DimOffset(self.image, 0)) / DimDelta(self.image, 0)
            y_pixel = (y_coord - DimOffset(self.image, 1)) / DimDelta(self.image, 1)

            # Define view region (4x the particle scale)
            view_radius = max(20, int(4 * scale / DimDelta(self.image, 0)))

            x_min = max(0, int(x_pixel - view_radius))
            x_max = min(self.image.data.shape[1], int(x_pixel + view_radius))
            y_min = max(0, int(y_pixel - view_radius))
            y_max = min(self.image.data.shape[0], int(y_pixel + view_radius))

            # Extract region
            region = self.image.data[y_min:y_max, x_min:x_max]

            # Create coordinate arrays for display
            x_coords = np.arange(x_min, x_max) * DimDelta(self.image, 0) + DimOffset(self.image, 0)
            y_coords = np.arange(y_min, y_max) * DimDelta(self.image, 1) + DimOffset(self.image, 1)

            extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]

            # Clear and plot
            self.ax.clear()
            self.ax.imshow(region, extent=extent, cmap='gray', aspect='auto')

            # Draw particle circle
            circle = Circle((x_coord, y_coord), scale, fill=False, color='red', linewidth=2)
            self.ax.add_patch(circle)

            # Mark center
            self.ax.plot(x_coord, y_coord, 'r+', markersize=10, markeredgewidth=2)

            # Mark center of mass if available
            if self.info.data.shape[1] > 11:
                com_x = p_info[11]
                com_y = p_info[12]
                self.ax.plot(com_x, com_y, 'bx', markersize=8, markeredgewidth=2)

            self.ax.set_title(f"Particle {self.current_particle}")
            self.ax.set_xlabel("X (pixels)")
            self.ax.set_ylabel("Y (pixels)")

            self.canvas.draw()

            # Update info text
            self.update_info_text()

        def update_info_text(self):
            """Update the information text"""
            self.info_text.delete(1.0, tk.END)

            p_info = self.info.data[self.current_particle]

            info_str = f"Particle {self.current_particle}\n"
            info_str += "=" * 40 + "\n\n"
            info_str += f"Position:\n"
            info_str += f"  X: {p_info[0]:.3f} pixels\n"
            info_str += f"  Y: {p_info[1]:.3f} pixels\n"
            info_str += f"  Scale: {p_info[2]:.3f} pixels\n\n"

            info_str += f"Detection Response:\n"
            info_str += f"  DetH: {p_info[3]:.6f}\n"
            info_str += f"  LaplacianG: {p_info[4]:.6f}\n\n"

            if self.info.data.shape[1] > 8:
                info_str += f"Measurements:\n"
                info_str += f"  Area: {p_info[8]:.3f}\n"
                info_str += f"  Volume: {p_info[9]:.3f}\n"
                info_str += f"  Height: {p_info[10]:.6f}\n\n"

                if self.info.data.shape[1] > 11:
                    info_str += f"Center of Mass:\n"
                    info_str += f"  COM X: {p_info[11]:.3f} pixels\n"
                    info_str += f"  COM Y: {p_info[12]:.3f} pixels\n"

            self.info_text.insert(1.0, info_str)

        def prev_particle(self):
            """Go to previous particle"""
            if self.current_particle > 0:
                self.current_particle -= 1
                self.update_display()

        def next_particle(self):
            """Go to next particle"""
            if self.current_particle < self.num_particles - 1:
                self.current_particle += 1
                self.update_display()

        def on_particle_change(self):
            """Handle particle selection change"""
            try:
                new_particle = self.particle_var.get()
                if 0 <= new_particle < self.num_particles:
                    self.current_particle = new_particle
                    self.update_display()
            except:
                pass

        def delete_particle(self):
            """Delete current particle"""
            if self.num_particles == 0:
                return

            result = messagebox.askyesno("Delete Particle",
                                         f"Delete particle {self.current_particle}?")
            if result:
                # Remove from info array
                self.info.data = np.delete(self.info.data, self.current_particle, axis=0)
                self.num_particles -= 1

                # Update map if provided
                if self.map is not None:
                    # Set deleted particle locations to -1
                    self.map.data[self.map.data == self.current_particle] = -1
                    # Renumber remaining particles
                    for i in range(self.current_particle, self.num_particles):
                        self.map.data[self.map.data == i + 1] = i

                # Update display
                if self.num_particles == 0:
                    messagebox.showinfo("No Particles", "No more particles to view.")
                    self.root.destroy()
                    return

                if self.current_particle >= self.num_particles:
                    self.current_particle = self.num_particles - 1

                # Update spinbox range
                self.particle_spinbox.config(to=self.num_particles - 1)

                self.update_display()

    # Launch the particle viewer
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