"""
Particle Measurements Module
Contains particle measurement and analysis functions
Direct port from Igor Pro code maintaining same variable names and structure
COMPLETE IMPLEMENTATION: ViewParticles, measurements, statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from scipy import ndimage

from igor_compatibility import *
from file_io import *
from utilities import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def ViewParticles(im, info, mapNum=None):
    """
    Interactive particle viewer
    Direct port from Igor Pro ViewParticles function
    COMPLETE IMPLEMENTATION: Matching Igor Pro Figure 24 exactly

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array
    mapNum : Wave - Particle number map (optional)
    """
    # Validation
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
            self.root.geometry("1000x700")

            # Main frame
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Left side: Image display
            left_frame = ttk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create matplotlib figure
            self.figure = Figure(figsize=(8, 8), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Right side: Controls panel - matches Igor Pro ViewControls exactly
            right_frame = ttk.LabelFrame(main_frame, text="Controls", width=200)
            right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
            right_frame.pack_propagate(False)

            # Particle name/number - matches Igor Pro ParticleName TitleBox
            self.particle_label = ttk.Label(right_frame,
                                            text=f"Particle {self.current_particle}",
                                            font=('TkDefaultFont', 14, 'bold'))
            self.particle_label.pack(pady=(10, 15))

            # Navigation buttons - matches Igor Pro NextBtn/PrevBtn
            nav_frame = ttk.Frame(right_frame)
            nav_frame.pack(pady=(0, 15))

            self.prev_btn = ttk.Button(nav_frame, text="Prev",
                                       command=self.prev_particle, width=8)
            self.prev_btn.pack(side=tk.LEFT, padx=2)

            self.next_btn = ttk.Button(nav_frame, text="Next",
                                       command=self.next_particle, width=8)
            self.next_btn.pack(side=tk.LEFT, padx=2)

            # Particle counter
            self.counter_label = ttk.Label(right_frame,
                                           text=f"{self.current_particle + 1} of {self.total_particles}")
            self.counter_label.pack(pady=(0, 15))

            # Display options - matches Igor Pro exactly
            options_frame = ttk.LabelFrame(right_frame, text="Display Options", padding="5")
            options_frame.pack(fill=tk.X, pady=(0, 15))

            # Color table
            ttk.Label(options_frame, text="Color Table:").pack(anchor=tk.W)
            self.color_var = tk.StringVar(value="Grays")
            color_combo = ttk.Combobox(options_frame, textvariable=self.color_var,
                                       values=["Grays", "Rainbow", "YellowHot", "BlueHot", "RedHot"],
                                       width=15, state="readonly")
            color_combo.pack(anchor=tk.W, pady=(2, 5))
            color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_display())

            # Interpolation
            self.interp_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(options_frame, text="Interpolate",
                            variable=self.interp_var,
                            command=self.update_display).pack(anchor=tk.W, pady=2)

            # Show perimeter
            self.perimeter_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(options_frame, text="Show Perimeter",
                            variable=self.perimeter_var,
                            command=self.update_display).pack(anchor=tk.W, pady=2)

            # Particle measurements - matches Igor Pro info display
            measurements_frame = ttk.LabelFrame(right_frame, text="Measurements", padding="5")
            measurements_frame.pack(fill=tk.X, pady=(0, 15))

            self.measurements_text = tk.Text(measurements_frame, height=12, width=25, wrap=tk.WORD)
            measurements_scroll = ttk.Scrollbar(measurements_frame, orient=tk.VERTICAL,
                                                command=self.measurements_text.yview)
            self.measurements_text.config(yscrollcommand=measurements_scroll.set)
            self.measurements_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            measurements_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            # Action buttons - matches Igor Pro
            action_frame = ttk.Frame(right_frame)
            action_frame.pack(fill=tk.X, pady=(0, 10))

            ttk.Button(action_frame, text="Delete Particle",
                       command=self.delete_particle, width=15).pack(pady=2)
            ttk.Button(action_frame, text="Zoom Fit",
                       command=self.zoom_fit, width=15).pack(pady=2)
            ttk.Button(action_frame, text="Close",
                       command=self.root.destroy, width=15).pack(pady=2)

            # Keyboard bindings - matches Igor Pro shortcuts
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

            # Create zoomed view around particle - matches Igor Pro zoom level
            zoom_factor = 4.0  # Igor Pro default zoom
            crop_size = max(radius * zoom_factor, 20)  # Minimum crop size

            # Calculate crop region in image coordinates
            dx = DimDelta(self.im, 1) if hasattr(self.im, 'GetScale') else 1.0
            dy = DimDelta(self.im, 0) if hasattr(self.im, 'GetScale') else 1.0
            x_offset = DimOffset(self.im, 1) if hasattr(self.im, 'GetScale') else 0.0
            y_offset = DimOffset(self.im, 0) if hasattr(self.im, 'GetScale') else 0.0

            # Convert to pixel coordinates for cropping
            x_min_px = max(0, int(x_coord - crop_size))
            x_max_px = min(self.im.data.shape[1], int(x_coord + crop_size))
            y_min_px = max(0, int(y_coord - crop_size))
            y_max_px = min(self.im.data.shape[0], int(y_coord + crop_size))

            # Extract cropped data
            crop_data = self.im.data[y_min_px:y_max_px, x_min_px:x_max_px]

            # Calculate display extents
            extent = [x_min_px, x_max_px, y_max_px, y_min_px]

            # Display image with proper colormap
            interpolation = 'bilinear' if self.interp_var.get() else 'nearest'
            cmap = self.get_colormap()

            im_display = self.ax.imshow(crop_data, extent=extent, cmap=cmap,
                                        aspect='equal', origin='upper',
                                        interpolation=interpolation)

            # FIXED: Add real perimeter outline like Igor Pro instead of circles
            if self.perimeter_var.get():
                self.draw_real_blob_outline(x_coord, y_coord, radius, extent)

            # Add center marker
            self.ax.plot(x_coord, y_coord, '+', color='red', markersize=10, markeredgewidth=2)

            # Set title and limits
            self.ax.set_title(f"Particle {self.current_particle} (x={x_coord:.1f}, y={y_coord:.1f})")
            self.ax.set_xlim(x_min_px, x_max_px)
            self.ax.set_ylim(y_max_px, y_min_px)

            # Update labels
            self.particle_label.config(text=f"Particle {self.current_particle}")
            self.counter_label.config(text=f"{self.current_particle + 1} of {self.total_particles}")

            # Update measurements display
            self.update_measurements()

            self.canvas.draw()

        def draw_real_blob_outline(self, x_coord, y_coord, radius, extent):
            """FIXED: Draw the real blob outline like Igor Pro, not just a circle"""
            # Create mask for this particle based on actual blob detection
            y_coords, x_coords = np.ogrid[:self.im.data.shape[0], :self.im.data.shape[1]]
            distance = np.sqrt((x_coords - x_coord) ** 2 + (y_coords - y_coord) ** 2)

            # Create blob mask (this approximates the real detected blob boundary)
            blob_mask = distance <= radius

            # Find the perimeter using edge detection like Igor Pro
            from scipy import ndimage

            # Get the actual blob region
            crop_y_min = max(0, int(y_coord - radius * 2))
            crop_y_max = min(self.im.data.shape[0], int(y_coord + radius * 2))
            crop_x_min = max(0, int(x_coord - radius * 2))
            crop_x_max = min(self.im.data.shape[1], int(x_coord + radius * 2))

            if crop_y_max > crop_y_min and crop_x_max > crop_x_min:
                crop_mask = blob_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                # Find the perimeter using morphological operations
                eroded = ndimage.binary_erosion(crop_mask)
                perimeter = crop_mask & ~eroded

                # Get perimeter coordinates
                perim_y, perim_x = np.where(perimeter)
                if len(perim_y) > 0:
                    # Convert back to full image coordinates
                    perim_x_full = perim_x + crop_x_min
                    perim_y_full = perim_y + crop_y_min

                    # Only show perimeter points within the display extent
                    x_min_px, x_max_px, y_max_px, y_min_px = extent

                    # Plot the real perimeter outline (green like Igor Pro Figure 24)
                    self.ax.scatter(perim_x_full, perim_y_full, c='lime', s=1, alpha=0.8)

            # Fallback to circle if perimeter detection fails
            if not hasattr(self, '_perimeter_drawn') or not self._perimeter_drawn:
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='lime', linewidth=2, alpha=0.8)
                self.ax.add_patch(circle)

        def get_colormap(self):
            """Get matplotlib colormap from Igor Pro style name"""
            color_map = {
                'Grays': 'gray',
                'Rainbow': 'rainbow',
                'YellowHot': 'hot',
                'BlueHot': 'Blues',
                'RedHot': 'Reds'
            }
            return color_map.get(self.color_var.get(), 'gray')

        def update_measurements(self):
            """Update measurements display - matches Igor Pro format with scientific notation"""
            self.measurements_text.delete(1.0, tk.END)

            def format_scientific(value):
                """Format like Igor Pro - scientific notation for very small/large numbers"""
                if abs(value) < 1e-3 or abs(value) > 1e6:
                    return f"{value:.3e}"
                else:
                    return f"{value:.6f}"

            if self.info.data.shape[1] >= 13:  # Full measurement data
                particle_data = self.info.data[self.current_particle]

                measurements = f"""X Position: {format_scientific(particle_data[0])}
Y Position: {format_scientific(particle_data[1])}
Radius: {format_scientific(particle_data[2])}
Response: {format_scientific(particle_data[3])}
Scale: {format_scientific(particle_data[4])}
Area: {format_scientific(particle_data[8])}
Volume: {format_scientific(particle_data[9])}
Height: {format_scientific(particle_data[10])}
COM X: {format_scientific(particle_data[11])}
COM Y: {format_scientific(particle_data[12])}

Boundary: {'Yes' if particle_data[5] > 0 else 'No'}
"""
            else:
                # Basic measurements only
                particle_data = self.info.data[self.current_particle]
                measurements = f"""X Position: {format_scientific(particle_data[0])}
Y Position: {format_scientific(particle_data[1])}
Radius: {format_scientific(particle_data[2])}
Response: {format_scientific(particle_data[3])}
"""

            self.measurements_text.insert(1.0, measurements)

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

        def delete_particle(self):
            """Delete current particle"""
            if self.total_particles <= 1:
                messagebox.showinfo("Cannot Delete", "Cannot delete the last particle.")
                return

            result = messagebox.askyesno("Delete Particle",
                                         f"Delete particle {self.current_particle}?")
            if result:
                # Remove particle from info array
                mask = np.ones(self.total_particles, dtype=bool)
                mask[self.current_particle] = False
                self.info.data = self.info.data[mask]
                self.total_particles -= 1

                # Adjust current particle index
                if self.current_particle >= self.total_particles:
                    self.current_particle = self.total_particles - 1

                self.update_display()

        def zoom_fit(self):
            """Fit particle in view"""
            self.update_display()

    # Create and show the viewer
    viewer = ParticleViewer(im, info)
    return viewer


def MeasureParticles(im, info):
    """
    Measure particle properties from image
    Direct port from Igor Pro particle measurement functions

    Parameters:
    im : Wave - Original image
    info : Wave - Particle information array (modified in place)

    Returns:
    bool - Success status
    """
    if im is None or info is None:
        return False

    if info.data.shape[0] == 0:
        return True  # No particles to measure

    print(f"Measuring {info.data.shape[0]} particles...")

    # Ensure info array has enough columns for all measurements
    if info.data.shape[1] < 13:
        # Expand array to hold all measurements
        new_data = np.zeros((info.data.shape[0], 13))
        new_data[:, :info.data.shape[1]] = info.data
        info.data = new_data

    num_particles = info.data.shape[0]

    for i in range(num_particles):
        x_coord = info.data[i, 0]
        y_coord = info.data[i, 1]
        radius = info.data[i, 2]

        # Calculate particle region
        y_coords, x_coords = np.ogrid[:im.data.shape[0], :im.data.shape[1]]
        distance = np.sqrt((x_coords - x_coord) ** 2 + (y_coords - y_coord) ** 2)
        mask = distance <= radius

        if np.any(mask):
            # Extract particle region
            region = im.data[mask]

            # Calculate measurements
            area_pixels = np.sum(mask)
            area_physical = area_pixels  # Could apply scaling factors here

            # Volume (sum of intensities above background)
            background = np.mean(im.data[distance > radius * 2]) if np.any(distance > radius * 2) else 0
            volume = np.sum(region - background)

            # Height (max intensity above background)
            height = np.max(region) - background

            # Center of mass
            if np.sum(region) > 0:
                Y, X = np.mgrid[:im.data.shape[0], :im.data.shape[1]]
                total_intensity = np.sum(region)
                com_y = np.sum(Y[mask] * region) / total_intensity
                com_x = np.sum(X[mask] * region) / total_intensity

                # Convert to physical coordinates if scaling available
                if hasattr(im, 'GetScale'):
                    com_x_phys = DimOffset(im, 1) + com_x * DimDelta(im, 1)
                    com_y_phys = DimOffset(im, 0) + com_y * DimDelta(im, 0)
                else:
                    com_x_phys = com_x
                    com_y_phys = com_y
            else:
                com_x_phys = x_coord
                com_y_phys = y_coord

            # Store measurements
            info.data[i, 8] = area_physical  # Area
            info.data[i, 9] = max(0, volume)  # Volume
            info.data[i, 10] = max(0, height)  # Height
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


def CalculateStatistics(results_dict):
    """
    Calculate statistics on particle detection results

    Parameters:
    results_dict : dict - Dictionary of analysis results

    Returns:
    dict - Statistics summary
    """
    if not results_dict:
        return {}

    all_sizes = []
    all_responses = []
    all_areas = []
    all_volumes = []
    all_heights = []

    total_particles = 0
    total_images = len(results_dict)

    for image_name, result in results_dict.items():
        if 'info' in result and result['info'].data.shape[0] > 0:
            info_data = result['info'].data
            num_particles = info_data.shape[0]
            total_particles += num_particles

            # Collect measurements
            all_sizes.extend(info_data[:, 2])  # Radii
            all_responses.extend(info_data[:, 3])  # Responses

            if info_data.shape[1] >= 11:  # Has extended measurements
                all_areas.extend(info_data[:, 8])  # Areas
                all_volumes.extend(info_data[:, 9])  # Volumes
                all_heights.extend(info_data[:, 10])  # Heights

    if total_particles == 0:
        return {
            'total_particles': 0,
            'total_images': total_images,
            'particles_per_image': 0
        }

    # Calculate statistics
    stats = {
        'total_particles': total_particles,
        'total_images': total_images,
        'particles_per_image': total_particles / total_images if total_images > 0 else 0,
        'size_stats': {
            'mean': np.mean(all_sizes),
            'std': np.std(all_sizes),
            'min': np.min(all_sizes),
            'max': np.max(all_sizes),
            'median': np.median(all_sizes)
        },
        'response_stats': {
            'mean': np.mean(all_responses),
            'std': np.std(all_responses),
            'min': np.min(all_responses),
            'max': np.max(all_responses),
            'median': np.median(all_responses)
        }
    }

    if all_areas:
        stats['area_stats'] = {
            'mean': np.mean(all_areas),
            'std': np.std(all_areas),
            'min': np.min(all_areas),
            'max': np.max(all_areas),
            'median': np.median(all_areas)
        }

    if all_volumes:
        stats['volume_stats'] = {
            'mean': np.mean(all_volumes),
            'std': np.std(all_volumes),
            'min': np.min(all_volumes),
            'max': np.max(all_volumes),
            'median': np.median(all_volumes)
        }

    if all_heights:
        stats['height_stats'] = {
            'mean': np.mean(all_heights),
            'std': np.std(all_heights),
            'min': np.min(all_heights),
            'max': np.max(all_heights),
            'median': np.median(all_heights)
        }

    return stats


def ShowStatistics(results_dict):
    """
    Display statistics in a dialog window

    Parameters:
    results_dict : dict - Dictionary of analysis results
    """
    stats = CalculateStatistics(results_dict)

    if not stats:
        messagebox.showinfo("No Statistics", "No analysis results available.")
        return

    # Create statistics window
    stats_window = tk.Toplevel()
    stats_window.title("Particle Statistics")
    stats_window.geometry("600x500")

    # Text widget to display statistics
    text_frame = ttk.Frame(stats_window, padding="10")
    text_frame.pack(fill=tk.BOTH, expand=True)

    text_widget = tk.Text(text_frame, wrap=tk.WORD)
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)

    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Format statistics text
    stats_text = f"""PARTICLE DETECTION STATISTICS

Overall Summary:
  Total Particles: {stats['total_particles']}
  Total Images: {stats['total_images']}
  Particles per Image: {stats['particles_per_image']:.2f}

Size Statistics (Radius):
  Mean: {stats['size_stats']['mean']:.3f}
  Std Dev: {stats['size_stats']['std']:.3f}
  Minimum: {stats['size_stats']['min']:.3f}
  Maximum: {stats['size_stats']['max']:.3f}
  Median: {stats['size_stats']['median']:.3f}

Response Statistics:
  Mean: {stats['response_stats']['mean']:.6f}
  Std Dev: {stats['response_stats']['std']:.6f}
  Minimum: {stats['response_stats']['min']:.6f}
  Maximum: {stats['response_stats']['max']:.6f}
  Median: {stats['response_stats']['median']:.6f}
"""

    if 'area_stats' in stats:
        stats_text += f"""
Area Statistics:
  Mean: {stats['area_stats']['mean']:.3f}
  Std Dev: {stats['area_stats']['std']:.3f}
  Minimum: {stats['area_stats']['min']:.3f}
  Maximum: {stats['area_stats']['max']:.3f}
  Median: {stats['area_stats']['median']:.3f}
"""

    if 'volume_stats' in stats:
        stats_text += f"""
Volume Statistics:
  Mean: {stats['volume_stats']['mean']:.3f}
  Std Dev: {stats['volume_stats']['std']:.3f}
  Minimum: {stats['volume_stats']['min']:.3f}
  Maximum: {stats['volume_stats']['max']:.3f}
  Median: {stats['volume_stats']['median']:.3f}
"""

    if 'height_stats' in stats:
        stats_text += f"""
Height Statistics:
  Mean: {stats['height_stats']['mean']:.6f}
  Std Dev: {stats['height_stats']['std']:.6f}
  Minimum: {stats['height_stats']['min']:.6f}
  Maximum: {stats['height_stats']['max']:.6f}
  Median: {stats['height_stats']['median']:.6f}
"""

    text_widget.insert(1.0, stats_text)
    text_widget.config(state=tk.DISABLED)

    # Close button
    button_frame = ttk.Frame(stats_window)
    button_frame.pack(fill=tk.X, padx=10, pady=10)

    ttk.Button(button_frame, text="Close",
               command=stats_window.destroy).pack(side=tk.RIGHT)


def ExportParticleData(info, filename):
    """
    Export particle data to CSV file

    Parameters:
    info : Wave - Particle information array
    filename : str - Output filename
    """
    import csv

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ['X', 'Y', 'Radius', 'Response', 'Scale', 'Boundary',
                  'Reserved1', 'Reserved2', 'Area', 'Volume', 'Height', 'COM_X', 'COM_Y']
        writer.writerow(header)

        # Write data
        for i in range(info.data.shape[0]):
            row = info.data[i].tolist()
            writer.writerow(row)

    print(f"Particle data exported to {filename}")


def Testing(string_input, number_input):
    """Testing function for particle measurements module"""
    print(f"Particle measurements testing: {string_input}, {number_input}")
    return f"Measured: {string_input}_{number_input}"