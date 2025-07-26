"""
Particle Measurements Module
Contains particle measurement and analysis functions
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
from scipy import ndimage
from scipy.optimize import curve_fit

from igor_compatibility import *
from file_io import *

# Monkey patch for numpy complex deprecation
if not hasattr(np, 'complex'):
    np.complex = complex


def MeasureParticles():
    """
    Perform detailed measurements on detected particles
    Matches Igor Pro MeasureParticles functionality
    """
    selection = GetBrowserSelection(0)

    if not DataFolderExists(selection):
        messagebox.showerror("Error", "Please select a particle analysis folder in the data browser.")
        return False

    folder = data_browser.get_folder(selection.rstrip(':'))

    # Look for required waves
    required_waves = ['Original', 'Info', 'ParticleMap']
    missing_waves = []

    for wave_name in required_waves:
        if wave_name not in folder.waves:
            missing_waves.append(wave_name)

    if missing_waves:
        messagebox.showerror("Error", f"Missing required waves: {', '.join(missing_waves)}")
        return False

    # Get the waves
    original = folder.waves['Original']
    info = folder.waves['Info']
    particle_map = folder.waves['ParticleMap']

    num_particles = info.data.shape[0]

    if num_particles == 0:
        messagebox.showinfo("No Particles", "No particles found to measure.")
        return False

    print(f"Measuring {num_particles} particles...")

    # Perform detailed measurements
    measurements = PerformDetailedMeasurements(original, info, particle_map)

    if measurements is None:
        messagebox.showerror("Error", "Failed to perform particle measurements.")
        return False

    # Store measurement results
    StoreMeasurementResults(folder, measurements)

    print("Particle measurements completed successfully!")
    messagebox.showinfo("Success", f"Measurements completed for {num_particles} particles.")

    return True


def PerformDetailedMeasurements(original, info, particle_map):
    """
    Perform detailed measurements on all particles

    Parameters:
    original : Wave - Original image
    info : Wave - Particle information
    particle_map : Wave - Map of particle locations

    Returns:
    dict - Dictionary containing all measurements
    """
    try:
        num_particles = info.data.shape[0]

        # Initialize measurement arrays
        measurements = {
            'heights': np.zeros(num_particles),
            'volumes': np.zeros(num_particles),
            'areas': np.zeros(num_particles),
            'perimeters': np.zeros(num_particles),
            'centroids_x': np.zeros(num_particles),
            'centroids_y': np.zeros(num_particles),
            'equivalent_diameters': np.zeros(num_particles),
            'aspect_ratios': np.zeros(num_particles),
            'circularities': np.zeros(num_particles),
            'orientations': np.zeros(num_particles),
            'major_axes': np.zeros(num_particles),
            'minor_axes': np.zeros(num_particles),
            'mean_intensities': np.zeros(num_particles),
            'integrated_intensities': np.zeros(num_particles),
            'background_levels': np.zeros(num_particles)
        }

        # Process each particle
        for p in range(num_particles):
            # Get particle mask
            particle_mask = (particle_map.data == p)

            if not np.any(particle_mask):
                continue

            # Basic measurements
            particle_data = original.data[particle_mask]
            measurements['heights'][p] = np.max(particle_data)
            measurements['mean_intensities'][p] = np.mean(particle_data)
            measurements['integrated_intensities'][p] = np.sum(particle_data)

            # Area (convert to real units)
            pixel_area = DimDelta(original, 0) * DimDelta(original, 1)
            measurements['areas'][p] = np.sum(particle_mask) * pixel_area

            # Volume (integrated intensity times pixel area)
            measurements['volumes'][p] = measurements['integrated_intensities'][p] * pixel_area

            # Geometric measurements
            geom_props = MeasureParticleGeometry(particle_mask, original)
            measurements['centroids_x'][p] = geom_props['centroid_x']
            measurements['centroids_y'][p] = geom_props['centroid_y']
            measurements['perimeters'][p] = geom_props['perimeter']
            measurements['equivalent_diameters'][p] = geom_props['equivalent_diameter']
            measurements['aspect_ratios'][p] = geom_props['aspect_ratio']
            measurements['circularities'][p] = geom_props['circularity']
            measurements['orientations'][p] = geom_props['orientation']
            measurements['major_axes'][p] = geom_props['major_axis']
            measurements['minor_axes'][p] = geom_props['minor_axis']

            # Background level (estimate from particle boundary)
            measurements['background_levels'][p] = EstimateBackground(original, particle_mask)

            if (p + 1) % 50 == 0:  # Progress indicator
                print(f"  Measured {p + 1}/{num_particles} particles...")

        return measurements

    except Exception as e:
        print(f"Error in detailed measurements: {e}")
        return None


def MeasureParticleGeometry(particle_mask, original):
    """
    Measure geometric properties of a particle

    Parameters:
    particle_mask : ndarray - Boolean mask of particle
    original : Wave - Original image for coordinate conversion

    Returns:
    dict - Geometric properties
    """
    try:
        # Get particle coordinates
        y_coords, x_coords = np.where(particle_mask)

        if len(y_coords) == 0:
            return create_empty_geometry()

        # Convert to real coordinates
        real_x = DimOffset(original, 0) + x_coords * DimDelta(original, 0)
        real_y = DimOffset(original, 1) + y_coords * DimDelta(original, 1)

        # Centroid
        weights = original.data[y_coords, x_coords]
        total_weight = np.sum(weights)

        if total_weight > 0:
            centroid_x = np.sum(real_x * weights) / total_weight
            centroid_y = np.sum(real_y * weights) / total_weight
        else:
            centroid_x = np.mean(real_x)
            centroid_y = np.mean(real_y)

        # Area and equivalent diameter
        area = len(y_coords) * DimDelta(original, 0) * DimDelta(original, 1)
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        # Perimeter (approximate)
        perimeter = calculate_perimeter(particle_mask) * DimDelta(original, 0)

        # Circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # Second moments for aspect ratio and orientation
        x_mean = np.mean(real_x)
        y_mean = np.mean(real_y)

        # Calculate covariance matrix
        mu20 = np.mean((real_x - x_mean) ** 2)
        mu02 = np.mean((real_y - y_mean) ** 2)
        mu11 = np.mean((real_x - x_mean) * (real_y - y_mean))

        # Eigenvalues and eigenvectors
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 ** 2

        if det > 0 and trace > 0:
            lambda1 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2
            lambda2 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2

            if lambda2 > 0:
                aspect_ratio = np.sqrt(lambda1 / lambda2)
                major_axis = 2 * np.sqrt(lambda1)
                minor_axis = 2 * np.sqrt(lambda2)
            else:
                aspect_ratio = 1.0
                major_axis = equivalent_diameter
                minor_axis = equivalent_diameter

            # Orientation angle
            if mu11 != 0 or mu20 != mu02:
                orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            else:
                orientation = 0.0
        else:
            aspect_ratio = 1.0
            major_axis = equivalent_diameter
            minor_axis = equivalent_diameter
            orientation = 0.0

        return {
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'area': area,
            'perimeter': perimeter,
            'equivalent_diameter': equivalent_diameter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'orientation': orientation,
            'major_axis': major_axis,
            'minor_axis': minor_axis
        }

    except Exception as e:
        print(f"Error measuring particle geometry: {e}")
        return create_empty_geometry()


def create_empty_geometry():
    """Create empty geometry dictionary for error cases"""
    return {
        'centroid_x': 0, 'centroid_y': 0, 'area': 0, 'perimeter': 0,
        'equivalent_diameter': 0, 'circularity': 0, 'aspect_ratio': 1,
        'orientation': 0, 'major_axis': 0, 'minor_axis': 0
    }


def calculate_perimeter(mask):
    """Calculate perimeter of a binary mask"""
    try:
        # Simple perimeter calculation using edge detection
        edge_mask = mask.astype(np.uint8)
        edge_mask = edge_mask - ndimage.binary_erosion(edge_mask).astype(np.uint8)
        return np.sum(edge_mask)
    except:
        return 0


def EstimateBackground(original, particle_mask):
    """
    Estimate background level around a particle

    Parameters:
    original : Wave - Original image
    particle_mask : ndarray - Particle mask

    Returns:
    float - Estimated background level
    """
    try:
        # Dilate particle mask to get surrounding region
        dilated_mask = ndimage.binary_dilation(particle_mask, iterations=3)

        # Background region is dilated - original
        background_mask = dilated_mask & ~particle_mask

        if np.any(background_mask):
            return np.mean(original.data[background_mask])
        else:
            # Fallback: use image percentile
            return np.percentile(original.data, 10)

    except Exception as e:
        print(f"Error estimating background: {e}")
        return 0.0


def StoreMeasurementResults(folder, measurements):
    """
    Store measurement results as waves in the folder

    Parameters:
    folder : DataFolder - Folder to store results
    measurements : dict - Measurement results
    """
    try:
        # Store each measurement type as a separate wave
        for key, values in measurements.items():
            if len(values) > 0:
                wave_name = f"All{key.title()}"  # e.g., AllHeights, AllVolumes
                wave = Wave(values, wave_name)
                folder.add_wave(wave)

        # Create summary statistics
        CreateMeasurementSummary(folder, measurements)

        print("Measurement results stored successfully")

    except Exception as e:
        print(f"Error storing measurement results: {e}")


def CreateMeasurementSummary(folder, measurements):
    """
    Create summary statistics for measurements

    Parameters:
    folder : DataFolder - Target folder
    measurements : dict - Measurement data
    """
    try:
        summary_data = []
        summary_labels = []

        key_mappings = {
            'heights': 'Height',
            'volumes': 'Volume',
            'areas': 'Area',
            'perimeters': 'Perimeter',
            'equivalent_diameters': 'Equiv. Diameter',
            'aspect_ratios': 'Aspect Ratio',
            'circularities': 'Circularity'
        }

        for key, label in key_mappings.items():
            if key in measurements and len(measurements[key]) > 0:
                data = measurements[key]
                valid_data = data[~np.isnan(data)]

                if len(valid_data) > 0:
                    summary_data.extend([
                        len(valid_data),  # Count
                        np.mean(valid_data),  # Mean
                        np.std(valid_data),  # Std Dev
                        np.min(valid_data),  # Min
                        np.max(valid_data)  # Max
                    ])

                    summary_labels.extend([
                        f"{label} Count",
                        f"{label} Mean",
                        f"{label} StdDev",
                        f"{label} Min",
                        f"{label} Max"
                    ])

        if summary_data:
            summary_wave = Wave(np.array(summary_data), "MeasurementSummary")
            folder.add_wave(summary_wave)

            # Store labels as wave note
            summary_wave.note = '; '.join(summary_labels)

    except Exception as e:
        print(f"Error creating measurement summary: {e}")


def ViewParticles():
    """
    View detected particles in an interactive viewer
    Matches Igor Pro ViewParticles functionality
    """
    selection = GetBrowserSelection(0)

    if not DataFolderExists(selection):
        messagebox.showerror("Error", "Please select a particle analysis folder in the data browser.")
        return False

    folder = data_browser.get_folder(selection.rstrip(':'))

    # Check for required waves
    if 'Original' not in folder.waves or 'Info' not in folder.waves:
        messagebox.showerror("Error", "Missing required waves (Original, Info) for particle viewing.")
        return False

    original = folder.waves['Original']
    info = folder.waves['Info']
    particle_map = folder.waves.get('ParticleMap')

    num_particles = info.data.shape[0]

    if num_particles == 0:
        messagebox.showinfo("No Particles", "No particles found to view.")
        return False

    # Launch particle viewer
    viewer = ParticleViewer(original, info, particle_map, folder.name)
    viewer.show()

    return True


class ParticleViewer:
    """
    Interactive particle viewer window
    Similar to Igor Pro's particle viewer
    """

    def __init__(self, original, info, particle_map, folder_name):
        self.original = original
        self.info = info
        self.particle_map = particle_map
        self.folder_name = folder_name
        self.current_particle = 0
        self.num_particles = info.data.shape[0]

        self.window = None
        self.fig = None
        self.ax = None
        self.canvas = None

    def show(self):
        """Display the particle viewer window"""
        try:
            # Create tkinter window
            self.window = tk.Toplevel()
            self.window.title(f"Particle Viewer - {self.folder_name}")
            self.window.geometry("800x600")

            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(8, 6))

            # Embed in tkinter
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            self.canvas = FigureCanvasTkAgg(self.fig, self.window)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Navigation toolbar
            toolbar = NavigationToolbar2Tk(self.canvas, self.window)
            toolbar.update()

            # Controls frame
            controls_frame = tk.Frame(self.window)
            controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

            # Navigation controls
            tk.Button(controls_frame, text="<< First", command=self.first_particle).pack(side=tk.LEFT, padx=2)
            tk.Button(controls_frame, text="< Prev", command=self.prev_particle).pack(side=tk.LEFT, padx=2)

            self.particle_var = tk.StringVar(value=f"Particle 1 of {self.num_particles}")
            tk.Label(controls_frame, textvariable=self.particle_var).pack(side=tk.LEFT, padx=10)

            tk.Button(controls_frame, text="Next >", command=self.next_particle).pack(side=tk.LEFT, padx=2)
            tk.Button(controls_frame, text="Last >>", command=self.last_particle).pack(side=tk.LEFT, padx=2)

            # Go to particle
            tk.Label(controls_frame, text="Go to:").pack(side=tk.LEFT, padx=(20, 5))
            self.goto_var = tk.IntVar(value=1)
            goto_entry = tk.Entry(controls_frame, textvariable=self.goto_var, width=8)
            goto_entry.pack(side=tk.LEFT, padx=2)
            tk.Button(controls_frame, text="Go", command=self.goto_particle).pack(side=tk.LEFT, padx=2)

            # Delete particle button
            tk.Button(controls_frame, text="Delete Particle",
                      command=self.delete_particle, bg='red', fg='white').pack(side=tk.RIGHT, padx=2)

            # Display first particle
            self.display_current_particle()

        except Exception as e:
            print(f"Error creating particle viewer: {e}")
            messagebox.showerror("Error", f"Failed to create particle viewer: {str(e)}")

    def display_current_particle(self):
        """Display the current particle"""
        try:
            if self.current_particle >= self.num_particles:
                return

            self.ax.clear()

            # Get particle info
            x_coord = self.info.data[self.current_particle, 0]
            y_coord = self.info.data[self.current_particle, 1]
            height = self.info.data[self.current_particle, 4]
            area = self.info.data[self.current_particle, 5] if self.info.data.shape[1] > 5 else 0

            # Display original image
            extent = [
                DimOffset(self.original, 0),
                DimOffset(self.original, 0) + self.original.data.shape[1] * DimDelta(self.original, 0),
                DimOffset(self.original, 1),
                DimOffset(self.original, 1) + self.original.data.shape[0] * DimDelta(self.original, 1)
            ]

            self.ax.imshow(self.original.data, cmap='gray', origin='lower', extent=extent)

            # Highlight current particle
            if self.particle_map is not None:
                particle_mask = (self.particle_map.data == self.current_particle)
                if np.any(particle_mask):
                    # Create colored overlay for particle
                    overlay = np.zeros((*self.original.data.shape, 4))
                    overlay[particle_mask] = [1, 0, 0, 0.3]  # Semi-transparent red
                    self.ax.imshow(overlay, origin='lower', extent=extent)

            # Add marker at particle center
            self.ax.plot(x_coord, y_coord, 'r+', markersize=15, markeredgewidth=2)

            # Add circle showing approximate size
            if area > 0:
                radius = np.sqrt(area / np.pi)
                circle = Circle((x_coord, y_coord), radius, fill=False, color='red', linewidth=2)
                self.ax.add_patch(circle)

            # Set title with particle info
            title = f"Particle {self.current_particle + 1}: Height={height:.3f}, Area={area:.3e}"
            self.ax.set_title(title)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")

            # Update particle counter
            self.particle_var.set(f"Particle {self.current_particle + 1} of {self.num_particles}")

            # Set view to focus on particle
            window_size = max(20, np.sqrt(area) * 2) if area > 0 else 20
            self.ax.set_xlim(x_coord - window_size, x_coord + window_size)
            self.ax.set_ylim(y_coord - window_size, y_coord + window_size)

            self.canvas.draw()

        except Exception as e:
            print(f"Error displaying particle: {e}")

    def next_particle(self):
        """Go to next particle"""
        if self.current_particle < self.num_particles - 1:
            self.current_particle += 1
            self.display_current_particle()

    def prev_particle(self):
        """Go to previous particle"""
        if self.current_particle > 0:
            self.current_particle -= 1
            self.display_current_particle()

    def first_particle(self):
        """Go to first particle"""
        self.current_particle = 0
        self.display_current_particle()

    def last_particle(self):
        """Go to last particle"""
        self.current_particle = self.num_particles - 1
        self.display_current_particle()

    def goto_particle(self):
        """Go to specified particle"""
        try:
            particle_num = self.goto_var.get()
            if 1 <= particle_num <= self.num_particles:
                self.current_particle = particle_num - 1  # Convert to 0-based index
                self.display_current_particle()
            else:
                messagebox.showwarning("Invalid Particle",
                                       f"Particle number must be between 1 and {self.num_particles}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid particle number: {str(e)}")

    def delete_particle(self):
        """Delete current particle"""
        if messagebox.askyesno("Delete Particle", f"Delete particle {self.current_particle + 1}?"):
            try:
                # This would require modifying the original data
                # For now, just show a message
                messagebox.showinfo("Delete Particle",
                                    "Particle deletion not yet implemented.\nThis would require modifying the original analysis results.")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting particle: {str(e)}")


def ExportMeasurements(folder_path, filename):
    """
    Export particle measurements to a text file

    Parameters:
    folder_path : str - Path to particle analysis folder
    filename : str - Output filename

    Returns:
    bool - Success flag
    """
    try:
        folder = data_browser.get_folder(folder_path.rstrip(':'))

        if 'Info' not in folder.waves:
            print("No measurement data found")
            return False

        info = folder.waves['Info']

        # Create header
        header = ["Particle", "X_Coord", "Y_Coord", "Scale_Index", "Detector_Response",
                  "Max_Height", "Area", "Volume", "Refined_X", "Refined_Y"]

        # Prepare data
        data_rows = []
        for i in range(info.data.shape[0]):
            row = [i + 1]  # Particle number (1-based)
            row.extend(info.data[i, :min(9, info.data.shape[1])])
            data_rows.append(row)

        # Write to file
        with open(filename, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for row in data_rows:
                f.write('\t'.join(map(str, row)) + '\n')

        print(f"Measurements exported to {filename}")
        return True

    except Exception as e:
        print(f"Error exporting measurements: {e}")
        return False


def FitParticleProfiles():
    """
    Fit analytical profiles to particles
    Advanced measurement function
    """
    messagebox.showinfo("Profile Fitting",
                        "Particle profile fitting functionality would be implemented here.\n\nThis would fit Gaussian or other analytical functions to particle profiles for sub-pixel measurements.")
    return True


def TestParticleMeasurements():
    """Test function for particle measurements module"""
    print("Testing particle measurements module...")

    # Create test data
    test_data = np.zeros((100, 100))

    # Add some fake particles
    test_data[30:40, 30:40] = 100  # Square particle
    test_data[60:70, 60:70] = 80  # Another particle

    test_image = Wave(test_data, "TestImage")
    test_image.SetScale('x', 0, 1)
    test_image.SetScale('y', 0, 1)

    # Create fake particle map
    particle_map_data = np.full((100, 100), -1, dtype=np.int32)
    particle_map_data[30:40, 30:40] = 0
    particle_map_data[60:70, 60:70] = 1
    particle_map = Wave(particle_map_data, "ParticleMap")

    # Create fake info
    info_data = np.array([
        [35, 35, 5, 1000, 100, 100, 10000, 35, 35],  # Particle 0
        [65, 65, 5, 800, 80, 100, 8000, 65, 65]  # Particle 1
    ])
    info = Wave(info_data, "Info")

    # Test measurements
    measurements = PerformDetailedMeasurements(test_image, info, particle_map)

    if measurements is not None:
        print("✓ Detailed measurements working")
        print(f"  Heights: {measurements['heights']}")
        print(f"  Areas: {measurements['areas']}")
    else:
        print("✗ Detailed measurements failed")

    print("Particle measurements test completed")
    return True


if __name__ == "__main__":
    TestParticleMeasurements()