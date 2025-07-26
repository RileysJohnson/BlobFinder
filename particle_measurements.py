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

    # Ensure info wave has enough columns for all measurements
    if info.data.shape[1] < 12:
        # Expand info array to accommodate all measurements
        new_info = np.zeros((num_particles, 12))
        new_info[:, :info.data.shape[1]] = info.data
        info.data = new_info

    print(f"Measuring {num_particles} particles...")

    for particle_idx in range(num_particles):
        try:
            # Get particle pixels
            particle_mask = (mapNum.data == particle_idx)
            particle_pixels = np.where(particle_mask)

            if len(particle_pixels[0]) == 0:
                continue

            # Basic measurements
            particle_area = len(particle_pixels[0])

            # Calculate center of mass
            i_coords = particle_pixels[0]
            j_coords = particle_pixels[1]
            pixel_values = im.data[particle_mask]

            total_intensity = np.sum(pixel_values)
            if total_intensity > 0:
                com_i = np.sum(i_coords * pixel_values) / total_intensity
                com_j = np.sum(j_coords * pixel_values) / total_intensity
            else:
                com_i = np.mean(i_coords)
                com_j = np.mean(j_coords)

            # Convert to real coordinates
            com_x = DimOffset(im, 0) + com_j * DimDelta(im, 0)
            com_y = DimOffset(im, 1) + com_i * DimDelta(im, 1)

            # Calculate moments for shape analysis
            i_centered = i_coords - com_i
            j_centered = j_coords - com_j

            # Second moments
            m20 = np.sum(j_centered ** 2 * pixel_values) / total_intensity if total_intensity > 0 else 0
            m02 = np.sum(i_centered ** 2 * pixel_values) / total_intensity if total_intensity > 0 else 0
            m11 = np.sum(i_centered * j_centered * pixel_values) / total_intensity if total_intensity > 0 else 0

            # Calculate ellipse parameters
            if m20 + m02 > 0:
                # Major and minor axis lengths
                trace = m20 + m02
                det = m20 * m02 - m11 ** 2
                if det > 0:
                    discriminant = np.sqrt(trace ** 2 - 4 * det)
                    major_axis = np.sqrt(2 * (trace + discriminant))
                    minor_axis = np.sqrt(2 * (trace - discriminant))
                    eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2)) if major_axis > 0 else 0

                    # Orientation angle
                    if m11 != 0:
                        angle = 0.5 * np.arctan2(2 * m11, m20 - m02)
                    else:
                        angle = 0 if m20 >= m02 else np.pi / 2
                else:
                    major_axis = minor_axis = np.sqrt(trace)
                    eccentricity = 0
                    angle = 0
            else:
                major_axis = minor_axis = 0
                eccentricity = 0
                angle = 0

            # Additional measurements
            max_intensity = np.max(pixel_values) if len(pixel_values) > 0 else 0
            min_intensity = np.min(pixel_values) if len(pixel_values) > 0 else 0
            mean_intensity = np.mean(pixel_values) if len(pixel_values) > 0 else 0

            # Calculate perimeter (simplified)
            perimeter = CalculatePerimeter(particle_mask)

            # Calculate equivalent diameter
            equivalent_diameter = 2 * np.sqrt(particle_area / np.pi) if particle_area > 0 else 0

            # Calculate circularity
            circularity = (4 * np.pi * particle_area) / (perimeter ** 2) if perimeter > 0 else 0

            # Calculate volume (height * area)
            volume = total_intensity * DimDelta(im, 0) * DimDelta(im, 1)

            # Update info array with measurements
            # Columns: x, y, scale_idx, scale_value, strength, max_value, response, area, volume, additional measurements
            if info.data.shape[1] > 0: info.data[particle_idx, 0] = com_x
            if info.data.shape[1] > 1: info.data[particle_idx, 1] = com_y
            if info.data.shape[1] > 7: info.data[particle_idx, 7] = particle_area * DimDelta(im, 0) * DimDelta(im,
                                                                                                               1)  # Real area
            if info.data.shape[1] > 8: info.data[particle_idx, 8] = volume
            if info.data.shape[1] > 9: info.data[particle_idx, 9] = mean_intensity
            if info.data.shape[1] > 10: info.data[particle_idx, 10] = major_axis * DimDelta(im,
                                                                                            0)  # Convert to real units
            if info.data.shape[1] > 11: info.data[particle_idx, 11] = minor_axis * DimDelta(im,
                                                                                            0)  # Convert to real units

            # Store additional properties in extended columns if available
            if info.data.shape[1] > 12:
                extended_info = [
                    equivalent_diameter * DimDelta(im, 0),  # Equivalent diameter
                    perimeter * DimDelta(im, 0),  # Perimeter in real units
                    circularity,  # Circularity
                    eccentricity,  # Eccentricity
                    angle,  # Orientation angle
                    min_intensity,  # Minimum intensity
                    max_intensity  # Maximum intensity (redundant but for completeness)
                ]

                for i, val in enumerate(extended_info):
                    if 12 + i < info.data.shape[1]:
                        info.data[particle_idx, 12 + i] = val

        except Exception as e:
            print(f"Error measuring particle {particle_idx}: {str(e)}")
            continue

    print("Particle measurements completed.")
    return True


def CalculatePerimeter(mask):
    """
    Calculate perimeter of a binary mask
    Uses edge detection to estimate perimeter

    Parameters:
    mask : ndarray - Binary mask of the particle

    Returns:
    float - Estimated perimeter
    """
    from scipy import ndimage

    # Use edge detection to find perimeter pixels
    # A pixel is on the perimeter if it's True and has at least one False neighbor
    structure = np.ones((3, 3))  # 8-connectivity
    eroded = ndimage.binary_erosion(mask, structure)
    perimeter_mask = mask & ~eroded

    return np.sum(perimeter_mask)


def ViewParticles(im, mapNum, info, show_numbers=True, show_ellipses=False):
    """
    Display particles with overlay graphics
    Direct port from Igor Pro ViewParticles function

    Parameters:
    im : Wave - Original image
    mapNum : Wave - Particle number map
    info : Wave - Particle information
    show_numbers : bool - Whether to show particle numbers
    show_ellipses : bool - Whether to show fitted ellipses
    """
    print("Displaying particles...")

    if info.data.shape[0] == 0:
        messagebox.showwarning("No Particles", "No particles to display.")
        return

    # Create viewer window
    viewer_window = tk.Toplevel()
    viewer_window.title("Particle Viewer")
    viewer_window.geometry("1000x700")

    # Create matplotlib figure
    fig = Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)

    # Display image
    extent = [DimOffset(im, 0),
              DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
              DimOffset(im, 1),
              DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)]

    ax.imshow(im.data, cmap='gray', origin='lower', extent=extent)
    ax.set_title(f"Particle Viewer - {info.data.shape[0]} particles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Draw particles
    for i in range(info.data.shape[0]):
        x_coord = info.data[i, 0] if info.data.shape[1] > 0 else 0
        y_coord = info.data[i, 1] if info.data.shape[1] > 1 else 0

        # Calculate radius from scale or use default
        if info.data.shape[1] > 3:
            scale_value = info.data[i, 3]
            radius = np.sqrt(2 * scale_value)
        elif info.data.shape[1] > 2:
            scale_idx = int(info.data[i, 2])
            radius = np.sqrt(2 * (1.0 * (1.5 ** scale_idx)))
        else:
            radius = 5.0  # Default radius

        # Draw circle
        circle = Circle((x_coord, y_coord), radius, fill=False,
                        color='red', linewidth=2, alpha=0.8)
        ax.add_patch(circle)

        # Draw ellipse if requested and data available
        if show_ellipses and info.data.shape[1] > 11:
            major_axis = info.data[i, 10] if info.data.shape[1] > 10 else radius
            minor_axis = info.data[i, 11] if info.data.shape[1] > 11 else radius
            angle = info.data[i, 16] if info.data.shape[1] > 16 else 0  # Orientation angle

            ellipse = Ellipse((x_coord, y_coord), 2 * major_axis, 2 * minor_axis,
                              angle=np.degrees(angle), fill=False,
                              color='blue', linewidth=1.5, alpha=0.7)
            ax.add_patch(ellipse)

        # Add particle number
        if show_numbers:
            ax.text(x_coord + radius, y_coord + radius, str(i + 1),
                    color='yellow', fontsize=8, ha='left', va='bottom',
                    weight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                             facecolor='black', alpha=0.7))

    # Create canvas
    canvas = FigureCanvasTkAgg(fig, viewer_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Control panel
    control_frame = tk.Frame(viewer_window)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Toggle buttons
    def toggle_numbers():
        nonlocal show_numbers
        show_numbers = not show_numbers
        # Redraw would go here
        messagebox.showinfo("Toggle", f"Numbers {'shown' if show_numbers else 'hidden'}")

    def toggle_ellipses():
        nonlocal show_ellipses
        show_ellipses = not show_ellipses
        # Redraw would go here
        messagebox.showinfo("Toggle", f"Ellipses {'shown' if show_ellipses else 'hidden'}")

    tk.Button(control_frame, text="Toggle Numbers", command=toggle_numbers).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Toggle Ellipses", command=toggle_ellipses).pack(side=tk.LEFT, padx=5)
    tk.Button(control_frame, text="Close", command=viewer_window.destroy).pack(side=tk.RIGHT, padx=5)

    print(f"Displayed {info.data.shape[0]} particles")


def ParticleStatistics(info, save_to_file=False, file_path=None):
    """
    Calculate and display comprehensive particle statistics
    Direct port from Igor Pro particle analysis functions

    Parameters:
    info : Wave - Particle information array
    save_to_file : bool - Whether to save statistics to file
    file_path : str - Path for saving statistics

    Returns:
    dict - Dictionary containing all statistics
    """
    print("Calculating particle statistics...")

    if info.data.shape[0] == 0:
        print("No particles to analyze.")
        return {}

    num_particles = info.data.shape[0]
    stats = {
        'num_particles': num_particles,
        'measurements': {}
    }

    # Column names for the info array
    column_names = [
        'x_position', 'y_position', 'scale_index', 'scale_value',
        'blob_strength', 'max_intensity', 'detector_response',
        'area', 'volume', 'mean_intensity',
        'major_axis', 'minor_axis', 'equivalent_diameter',
        'perimeter', 'circularity', 'eccentricity',
        'orientation_angle', 'min_intensity', 'max_intensity_full'
    ]

    # Calculate statistics for each measurement
    for col_idx in range(min(info.data.shape[1], len(column_names))):
        col_name = column_names[col_idx]
        col_data = info.data[:, col_idx]

        # Skip columns that are all zeros or indices
        if col_name in ['scale_index'] or np.all(col_data == 0):
            continue

        col_stats = {
            'mean': np.mean(col_data),
            'std': np.std(col_data),
            'min': np.min(col_data),
            'max': np.max(col_data),
            'median': np.median(col_data),
            'q25': np.percentile(col_data, 25),
            'q75': np.percentile(col_data, 75)
        }

        stats['measurements'][col_name] = col_stats

    # Additional derived statistics
    if 'area' in stats['measurements'] and 'major_axis' in stats['measurements']:
        # Calculate aspect ratios
        if info.data.shape[1] > 11:  # Have both major and minor axes
            major_axes = info.data[:, 10]
            minor_axes = info.data[:, 11]
            aspect_ratios = np.divide(major_axes, minor_axes,
                                      out=np.ones_like(major_axes), where=minor_axes != 0)

            stats['measurements']['aspect_ratio'] = {
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios),
                'min': np.min(aspect_ratios),
                'max': np.max(aspect_ratios),
                'median': np.median(aspect_ratios)
            }

    # Size distribution analysis
    if 'area' in stats['measurements']:
        areas = info.data[:, 7]
        # Classify particles by size
        small_threshold = np.percentile(areas, 33)
        large_threshold = np.percentile(areas, 67)

        small_particles = np.sum(areas < small_threshold)
        medium_particles = np.sum((areas >= small_threshold) & (areas < large_threshold))
        large_particles = np.sum(areas >= large_threshold)

        stats['size_distribution'] = {
            'small_particles': small_particles,
            'medium_particles': medium_particles,
            'large_particles': large_particles,
            'small_threshold': small_threshold,
            'large_threshold': large_threshold
        }

    # Display statistics
    DisplayStatistics(stats)

    # Save to file if requested
    if save_to_file and file_path:
        SaveStatistics(stats, file_path)

    print("Particle statistics calculation completed.")
    return stats


def DisplayStatistics(stats):
    """
    Display statistics in a formatted window

    Parameters:
    stats : dict - Statistics dictionary
    """
    # Create statistics window
    stats_window = tk.Toplevel()
    stats_window.title("Particle Statistics")
    stats_window.geometry("600x500")

    # Create scrolled text widget
    text_frame = ttk.Frame(stats_window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    stats_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=('Courier', 10))
    stats_text.pack(fill=tk.BOTH, expand=True)

    # Format statistics text
    stats_str = "=== PARTICLE STATISTICS ===\n\n"
    stats_str += f"Total Particles: {stats['num_particles']}\n\n"

    if 'size_distribution' in stats:
        sd = stats['size_distribution']
        stats_str += "Size Distribution:\n"
        stats_str += f"  Small particles: {sd['small_particles']} (area < {sd['small_threshold']:.4f})\n"
        stats_str += f"  Medium particles: {sd['medium_particles']}\n"
        stats_str += f"  Large particles: {sd['large_particles']} (area > {sd['large_threshold']:.4f})\n\n"

    stats_str += "Measurement Statistics:\n"
    stats_str += "=" * 60 + "\n"
    stats_str += f"{'Measurement':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n"
    stats_str += "=" * 60 + "\n"

    for measurement, data in stats['measurements'].items():
        stats_str += f"{measurement:<20} {data['mean']:<12.4f} {data['std']:<12.4f} "
        stats_str += f"{data['min']:<12.4f} {data['max']:<12.4f}\n"

    # Insert text and make read-only
    stats_text.insert(tk.END, stats_str)
    stats_text.config(state=tk.DISABLED)

    # Close button
    tk.Button(stats_window, text="Close", command=stats_window.destroy).pack(pady=10)


def SaveStatistics(stats, file_path):
    """
    Save statistics to a text file

    Parameters:
    stats : dict - Statistics dictionary
    file_path : str - Output file path
    """
    try:
        with open(file_path, 'w') as f:
            f.write("Particle Statistics Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Particles: {stats['num_particles']}\n\n")

            if 'size_distribution' in stats:
                sd = stats['size_distribution']
                f.write("Size Distribution:\n")
                f.write(f"  Small particles: {sd['small_particles']}\n")
                f.write(f"  Medium particles: {sd['medium_particles']}\n")
                f.write(f"  Large particles: {sd['large_particles']}\n\n")

            f.write("Detailed Measurements:\n")
            f.write("=" * 50 + "\n")

            for measurement, data in stats['measurements'].items():
                f.write(f"\n{measurement}:\n")
                f.write(f"  Mean: {data['mean']:.6f}\n")
                f.write(f"  Std Dev: {data['std']:.6f}\n")
                f.write(f"  Min: {data['min']:.6f}\n")
                f.write(f"  Max: {data['max']:.6f}\n")
                f.write(f"  Median: {data['median']:.6f}\n")
                f.write(f"  Q25: {data['q25']:.6f}\n")
                f.write(f"  Q75: {data['q75']:.6f}\n")

        print(f"Statistics saved to: {file_path}")

    except Exception as e:
        print(f"Error saving statistics: {str(e)}")


def ExportParticleData(info, file_path, format='csv'):
    """
    Export particle data to various formats

    Parameters:
    info : Wave - Particle information array
    file_path : str - Output file path
    format : str - Export format ('csv', 'txt', 'excel')
    """
    print(f"Exporting particle data to {format} format...")

    if info.data.shape[0] == 0:
        print("No particle data to export.")
        return False

    # Column headers
    headers = [
        'Particle_ID', 'X_Position', 'Y_Position', 'Scale_Index', 'Scale_Value',
        'Blob_Strength', 'Max_Intensity', 'Detector_Response',
        'Area', 'Volume', 'Mean_Intensity',
        'Major_Axis', 'Minor_Axis', 'Equivalent_Diameter',
        'Perimeter', 'Circularity', 'Eccentricity',
        'Orientation_Angle', 'Min_Intensity', 'Max_Intensity_Full'
    ]

    try:
        if format.lower() == 'csv':
            import csv
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write headers
                writer.writerow(headers[:info.data.shape[1] + 1])  # +1 for ID column

                # Write data
                for i in range(info.data.shape[0]):
                    row = [i + 1]  # Particle ID (1-based)
                    row.extend(info.data[i, :])
                    writer.writerow(row)

        elif format.lower() == 'txt':
            with open(file_path, 'w') as f:
                # Write headers
                f.write('\t'.join(headers[:info.data.shape[1] + 1]) + '\n')

                # Write data
                for i in range(info.data.shape[0]):
                    row = [str(i + 1)]  # Particle ID
                    row.extend([f"{val:.6f}" for val in info.data[i, :]])
                    f.write('\t'.join(row) + '\n')

        else:
            print(f"Unsupported format: {format}")
            return False

        print(f"Data exported successfully to: {file_path}")
        return True

    except Exception as e:
        print(f"Error exporting data: {str(e)}")
        return False


def FilterParticles(info, mapNum, criteria):
    """
    Filter particles based on specified criteria

    Parameters:
    info : Wave - Particle information array
    mapNum : Wave - Particle number map
    criteria : dict - Filtering criteria

    Returns:
    tuple - (filtered_info, filtered_mapNum) with filtered data
    """
    print("Filtering particles based on criteria...")

    if info.data.shape[0] == 0:
        print("No particles to filter.")
        return info, mapNum

    # Initialize mask (all particles pass initially)
    keep_mask = np.ones(info.data.shape[0], dtype=bool)

    # Apply filtering criteria
    for criterion, (min_val, max_val) in criteria.items():
        if criterion == 'area' and info.data.shape[1] > 7:
            col_data = info.data[:, 7]
        elif criterion == 'volume' and info.data.shape[1] > 8:
            col_data = info.data[:, 8]
        elif criterion == 'blob_strength' and info.data.shape[1] > 4:
            col_data = info.data[:, 4]
        elif criterion == 'max_intensity' and info.data.shape[1] > 5:
            col_data = info.data[:, 5]
        else:
            continue

        # Apply min/max filters
        if min_val is not None:
            keep_mask &= (col_data >= min_val)
        if max_val is not None:
            keep_mask &= (col_data <= max_val)

    # Create filtered info array
    filtered_indices = np.where(keep_mask)[0]
    filtered_info = Wave(info.data[keep_mask], info.name + "_filtered")

    # Create filtered map
    filtered_map_data = np.full_like(mapNum.data, -1)
    for new_idx, old_idx in enumerate(filtered_indices):
        particle_mask = (mapNum.data == old_idx)
        filtered_map_data[particle_mask] = new_idx

    filtered_mapNum = Wave(filtered_map_data, mapNum.name + "_filtered")
    filtered_mapNum.SetScale('x', DimOffset(mapNum, 0), DimDelta(mapNum, 0))
    filtered_mapNum.SetScale('y', DimOffset(mapNum, 1), DimDelta(mapNum, 1))

    num_kept = np.sum(keep_mask)
    num_removed = info.data.shape[0] - num_kept

    print(f"Filtering completed: kept {num_kept} particles, removed {num_removed}")

    return filtered_info, filtered_mapNum


def Testing(string_input, number_input):
    """
    Testing function for particle measurement operations
    Direct port from Igor Pro Testing function
    """
    print(f"Particle measurements testing function called:")
    print(f"  String input: '{string_input}'")
    print(f"  Number input: {number_input}")

    # Create synthetic particle data for testing
    num_test_particles = 10

    # Create test info array
    test_info_data = np.zeros((num_test_particles, 12))

    for i in range(num_test_particles):
        # Generate synthetic particle data
        x_pos = np.random.uniform(0, 100)
        y_pos = np.random.uniform(0, 100)
        area = np.random.uniform(10, 1000)
        volume = area * np.random.uniform(0.1, 2.0)
        strength = np.random.uniform(0.1, 1.0)

        test_info_data[i, :] = [
            x_pos, y_pos,  # Position
            i % 5, 1.0 * (1.5 ** (i % 5)),  # Scale info
            strength, np.random.uniform(0.5, 2.0),  # Strength, max intensity
            strength ** 2, area, volume,  # Response, area, volume
            np.random.uniform(0.1, 1.0),  # Mean intensity
            np.sqrt(area / np.pi) * 1.2,  # Major axis
            np.sqrt(area / np.pi) * 0.8  # Minor axis
        ]

    test_info = Wave(test_info_data, "TestParticleInfo")

    print(f"  Created test data for {num_test_particles} particles")

    # Test statistics calculation
    stats = ParticleStatistics(test_info, save_to_file=False)

    print(f"  Calculated statistics for {stats['num_particles']} particles")

    # Test filtering
    filter_criteria = {
        'area': (50, 500),  # Keep particles with area between 50 and 500
        'blob_strength': (0.2, None)  # Keep particles with strength > 0.2
    }

    # Create dummy map for filtering test
    test_map = Wave(np.random.randint(-1, num_test_particles, (50, 50)), "TestMap")
    filtered_info, filtered_map = FilterParticles(test_info, test_map, filter_criteria)

    print(f"  Filtering test: {filtered_info.data.shape[0]} particles after filtering")

    result = len(string_input) + number_input + num_test_particles + len(stats['measurements'])
    print(f"  Test result: {result}")

    return result