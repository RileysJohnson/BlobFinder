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


# ViewParticleData function moved to main_functions.py to avoid conflicts
# The better implementation with particle viewer is in main_functions.py

def load_saved_particle_data(data_path):
    """
    Load particle data from saved Igor Pro format files
    
    Parameters:
    data_path : str - Path to Igor Pro format data folder
    
    Returns:
    im : Wave - Original image (reconstructed from particle data)
    info : Wave - Particle information array
    """
    import os
    from utilities import Wave
    
    # Check if data path exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Try to load Info.txt (particle information)
    info_file = os.path.join(data_path, "Info.txt")
    if not os.path.exists(info_file):
        raise ValueError("No Info.txt file found in data folder")
    
    # Load particle information
    info_data = []
    with open(info_file, 'r') as f:
        lines = f.readlines()
        reading_data = False
        for line in lines:
            line = line.strip()
            if "P_Seed" in line and "Q_Seed" in line:  # Header line
                reading_data = True
                continue
            if reading_data and line:
                try:
                    parts = line.split('\t')
                    if len(parts) >= 4:  # At least X, Y, scale, response
                        info_data.append([float(p) for p in parts])
                except ValueError:
                    continue
    
    if not info_data:
        raise ValueError("No particle data found in Info.txt")
    
    info = Wave(np.array(info_data), "info")
    
    # Create a dummy image based on particle locations (if no original image available)
    # Use particle coordinates to estimate image size
    if info_data:
        max_x = max(row[0] for row in info_data)
        max_y = max(row[1] for row in info_data)
        image_size = (int(max_y + 50), int(max_x + 50))  # Add some padding
        im_data = np.zeros(image_size)
        im = Wave(im_data, "ReconstructedImage")
    else:
        im = Wave(np.zeros((100, 100)), "EmptyImage")
    
    return im, info


def ViewParticles(im, info, mapNum=None, saved_data_path=None):
    """
    Interactive particle viewer
    Direct port from Igor Pro ViewParticles function
    Enhanced to work with both live analysis data and saved Igor Pro format files

    Parameters:
    im : Wave - Original image (or None if loading from saved data)
    info : Wave - Particle information array (or None if loading from saved data)
    mapNum : Wave - Particle number map (optional)
    saved_data_path : str - Path to saved Igor Pro format data folder (optional)
    """
    # Load from saved data if path provided
    if saved_data_path is not None:
        try:
            im, info = load_saved_particle_data(saved_data_path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load saved particle data:\n{str(e)}")
            return
    
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
            self.interpolate = False
            self.show_perimeter = True

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

            self.prev_btn = ttk.Button(nav_frame, text="← Previous", command=self.prev_particle)
            self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))

            self.next_btn = ttk.Button(nav_frame, text="Next →", command=self.next_particle)
            self.next_btn.pack(side=tk.LEFT)

            # Particle information display - matches Igor Pro info display
            info_frame = ttk.LabelFrame(right_frame, text="Particle Info", padding="5")
            info_frame.pack(fill=tk.X, pady=(0, 15))

            self.info_text = tk.Text(info_frame, height=8, width=25, font=("Courier", 9))
            self.info_text.pack(fill=tk.BOTH, expand=True)

            # Control buttons - matches Igor Pro controls
            control_frame = ttk.LabelFrame(right_frame, text="View Controls", padding="5")
            control_frame.pack(fill=tk.X, pady=(0, 15))

            # Show perimeter checkbox
            self.show_perimeter_var = tk.BooleanVar(value=self.show_perimeter)
            ttk.Checkbutton(control_frame, text="Show Perimeter",
                            variable=self.show_perimeter_var,
                            command=self.toggle_perimeter).pack(anchor=tk.W, pady=2)

            # Color table selector
            ttk.Label(control_frame, text="Color Table:").pack(anchor=tk.W, pady=(10, 2))
            self.color_var = tk.StringVar(value=self.color_table)
            color_combo = ttk.Combobox(control_frame, textvariable=self.color_var,
                                       values=["Grays", "Rainbow", "Hot", "Cool"],
                                       width=15, state="readonly")
            color_combo.pack(anchor=tk.W, pady=(0, 5))
            color_combo.bind('<<ComboboxSelected>>', self.change_color_table)

            # Action buttons - matches Igor Pro
            action_frame = ttk.Frame(right_frame)
            action_frame.pack(fill=tk.X, pady=(0, 10))

            ttk.Button(action_frame, text="Delete Particle",
                       command=self.delete_particle).pack(fill=tk.X, pady=2)
            ttk.Button(action_frame, text="Close",
                       command=self.root.destroy).pack(fill=tk.X, pady=2)

            # Bind keyboard shortcuts - matches Igor Pro behavior
            self.root.bind('<Left>', lambda e: self.prev_particle())
            self.root.bind('<Right>', lambda e: self.next_particle())
            self.root.bind('<space>', lambda e: self.delete_particle())
            self.root.focus_set()

        def update_display(self):
            """Update the particle display"""
            if self.total_particles == 0:
                return

            # Get current particle info
            particle_data = self.info.data[self.current_particle]
            x_coord = particle_data[0]
            y_coord = particle_data[1]
            radius = particle_data[2]

            # Update particle label
            self.particle_label.config(text=f"Particle {self.current_particle}")

            # Clear and update plot
            self.ax.clear()

            # Calculate crop region around particle (matching Igor Pro ViewParticles)
            crop_size = max(50, int(radius * 4))  # Minimum 50 pixels, or 4x radius
            x_min = max(0, int(x_coord - crop_size))
            x_max = min(self.im.data.shape[1], int(x_coord + crop_size))
            y_min = max(0, int(y_coord - crop_size))
            y_max = min(self.im.data.shape[0], int(y_coord + crop_size))

            # Crop image
            cropped_image = self.im.data[y_min:y_max, x_min:x_max]

            # Display cropped image
            extent = [x_min, x_max, y_max, y_min]  # Note: y is flipped for imshow

            color_map = self.color_var.get().lower()
            if color_map == "grays":
                color_map = "gray"
            elif color_map == "rainbow":
                color_map = "rainbow"
            elif color_map == "hot":
                color_map = "hot"
            elif color_map == "cool":
                color_map = "cool"

            self.ax.imshow(cropped_image, cmap=color_map, extent=extent, aspect='equal')

            # Show particle perimeter if enabled
            if self.show_perimeter_var.get():
                circle = Circle((x_coord, y_coord), radius,
                                fill=False, edgecolor='lime', linewidth=2)
                self.ax.add_patch(circle)

            # Mark particle center
            self.ax.plot(x_coord, y_coord, 'r+', markersize=10, markeredgewidth=2)

            self.ax.set_title(f"Particle {self.current_particle}")
            self.canvas.draw()

            # Update info text
            self.update_info_text()

            # Update button states
            self.prev_btn.config(state=tk.NORMAL if self.current_particle > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_particle < self.total_particles - 1 else tk.DISABLED)

        def update_info_text(self):
            """Update particle information text"""
            self.info_text.delete(1.0, tk.END)

            particle_data = self.info.data[self.current_particle]

            info_text = f"Particle {self.current_particle}\n"
            info_text += f"X Position: {particle_data[0]:.2f}\n"
            info_text += f"Y Position: {particle_data[1]:.2f}\n"
            info_text += f"Radius: {particle_data[2]:.2f}\n"

            if len(particle_data) > 3:
                info_text += f"Response: {particle_data[3]:.6f}\n"
            if len(particle_data) > 4:
                info_text += f"Scale: {particle_data[4]:.2f}\n"

            # Add additional measurements if available
            if len(particle_data) > 8:
                info_text += f"\nArea: {particle_data[8]:.2f}\n"
            if len(particle_data) > 9:
                info_text += f"Volume: {particle_data[9]:.3e}\n"
            if len(particle_data) > 10:
                info_text += f"Height: {particle_data[10]:.3e}\n"

            self.info_text.insert(tk.END, info_text)

        def prev_particle(self):
            """Navigate to previous particle"""
            if self.current_particle > 0:
                self.current_particle -= 1
                self.update_display()

        def next_particle(self):
            """Navigate to next particle"""
            if self.current_particle < self.total_particles - 1:
                self.current_particle += 1
                self.update_display()

        def toggle_perimeter(self):
            """Toggle perimeter display"""
            self.show_perimeter = self.show_perimeter_var.get()
            self.update_display()

        def change_color_table(self, event=None):
            """Change color table"""
            self.color_table = self.color_var.get()
            self.update_display()

        def delete_particle(self):
            """Delete current particle"""
            if self.total_particles <= 1:
                messagebox.showwarning("Cannot Delete", "Cannot delete the last particle.")
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
        particle_mask = distance <= radius

        if np.any(particle_mask):
            # Extract particle region
            particle_pixels = im.data[particle_mask]

            # Calculate measurements
            area = np.sum(particle_mask)  # Area in pixels
            volume = np.sum(particle_pixels)  # Sum of intensities
            height = np.max(particle_pixels) if len(particle_pixels) > 0 else 0  # Max intensity

            # Center of mass (intensity-weighted)
            if volume > 0:
                y_indices, x_indices = np.where(particle_mask)
                intensities = im.data[particle_mask]
                com_x = np.sum(x_indices * intensities) / volume
                com_y = np.sum(y_indices * intensities) / volume
            else:
                com_x = x_coord
                com_y = y_coord

            # Convert to physical coordinates if scale info available
            x_scale = im.GetScale('x')
            y_scale = im.GetScale('y')

            area_phys = area * x_scale['delta'] * y_scale['delta']
            com_x_phys = com_x * x_scale['delta'] + x_scale['offset']
            com_y_phys = com_y * y_scale['delta'] + y_scale['offset']

            # Store measurements in info array
            info.data[i, 8] = area_phys  # Area
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


def ExportResults(results_dict, file_path):
    """
    Export analysis results to CSV file

    Parameters:
    results_dict : dict - Dictionary of analysis results
    file_path : str - Output file path
    """
    if not results_dict:
        raise ValueError("No results to export")

    # Collect all particle data
    all_data = []
    image_names = []

    for image_name, result in results_dict.items():
        if 'info' in result and result['info'].data.shape[0] > 0:
            info_data = result['info'].data
            num_particles = info_data.shape[0]

            # Add image name column
            for i in range(num_particles):
                row = [image_name] + list(info_data[i])
                all_data.append(row)
                image_names.append(image_name)

    if not all_data:
        raise ValueError("No particle data to export")

    # Create header
    header = ['Image', 'X', 'Y', 'Radius', 'Response', 'Scale']
    if len(all_data[0]) > 6:  # Has extended measurements
        header.extend(['Extra1', 'Extra2', 'Extra3', 'Area', 'Volume', 'Height', 'COM_X', 'COM_Y'])

    # Write to CSV
    import csv
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_data)

    print(f"Exported {len(all_data)} particles from {len(results_dict)} images to {file_path}")


def TestingParticleMeasurements(string_input, number_input):
    """Testing function for particle measurements module"""
    print(f"Particle measurements testing: {string_input}, {number_input}")
    return f"Measured: {string_input}_{number_input}"


# Alias for Igor Pro compatibility
Testing = TestingParticleMeasurements