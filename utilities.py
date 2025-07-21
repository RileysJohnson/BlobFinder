import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os
from datetime import datetime
import json


class WaveNote:
    """Class to handle wave note functionality similar to Igor Pro"""

    def __init__(self):
        self.notes = {}

    def add_note(self, key, value):
        """Add a note entry"""
        self.notes[key] = value

    def get_note(self, key):
        """Get a note entry"""
        return self.notes.get(key, None)

    def to_string(self):
        """Convert notes to string format similar to Igor Pro"""
        note_str = ""
        for key, value in self.notes.items():
            note_str += f"{key}:{value}\n"
        return note_str

    def from_string(self, note_str):
        """Parse notes from string format"""
        self.notes = {}
        lines = note_str.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                self.notes[key.strip()] = value.strip()

    def to_dict(self):
        """Convert to dictionary"""
        return self.notes.copy()


class DataFolder:
    """Class to handle Igor Pro-like data folder structure"""

    def __init__(self, base_path, folder_name):
        self.base_path = base_path
        self.folder_name = folder_name
        self.full_path = os.path.join(base_path, folder_name)
        os.makedirs(self.full_path, exist_ok=True)
        self.waves = {}
        self.subfolders = {}

    def create_subfolder(self, name):
        """Create a subfolder"""
        subfolder = DataFolder(self.full_path, name)
        self.subfolders[name] = subfolder
        return subfolder

    def save_wave(self, name, data, note=None):
        """Save a wave (numpy array) with optional note"""
        file_path = os.path.join(self.full_path, f"{name}.npy")
        np.save(file_path, data)

        if note:
            note_path = os.path.join(self.full_path, f"{name}_note.json")
            with open(note_path, 'w') as f:
                json.dump(note.to_dict() if isinstance(note, WaveNote) else note, f, indent=2)

        self.waves[name] = data

    def load_wave(self, name):
        """Load a wave and its note"""
        file_path = os.path.join(self.full_path, f"{name}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)

            # Try to load note
            note_path = os.path.join(self.full_path, f"{name}_note.json")
            note = None
            if os.path.exists(note_path):
                with open(note_path, 'r') as f:
                    note_dict = json.load(f)
                    note = WaveNote()
                    note.notes = note_dict

            return data, note
        return None, None

    def get_path(self):
        """Get full path of this folder"""
        return self.full_path


class CoordinateSystem:
    """Handle coordinate transformations similar to Igor Pro's SetScale"""

    def __init__(self, shape, x_start=0, x_delta=1, y_start=0, y_delta=1):
        self.shape = shape
        self.x_start = x_start
        self.x_delta = x_delta
        self.y_start = y_start
        self.y_delta = y_delta

    def index_to_scale(self, p, q):
        """Convert pixel indices to scaled coordinates"""
        x = self.x_start + p * self.x_delta
        y = self.y_start + q * self.y_delta
        return x, y

    def scale_to_index(self, x, y):
        """Convert scaled coordinates to pixel indices"""
        p = int((x - self.x_start) / self.x_delta)
        q = int((y - self.y_start) / self.y_delta)
        return p, q

    def get_extent(self):
        """Get extent for matplotlib imshow"""
        return [self.x_start,
                self.x_start + self.shape[1] * self.x_delta,
                self.y_start,
                self.y_start + self.shape[0] * self.y_delta]


def interactive_threshold(im, detH, LG, particleType, maxCurvatureRatio):
    """
    Interactive threshold selection similar to Igor Pro's InteractiveThreshold

    Returns:
    float: Selected threshold value
    """
    # First identify the maxes
    from scale_space import find_scale_space_maxima
    maxes, map_data, scale_map = find_scale_space_maxima(detH, LG, particleType, maxCurvatureRatio)

    # Convert to image units (square root)
    maxes = np.sqrt(maxes[maxes > 0])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display image
    im_display = ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Image')

    # Histogram of maxima
    ax2.hist(maxes, bins=50, alpha=0.7)
    ax2.set_xlabel('Blob Strength')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Blob Strengths')

    # Initial threshold
    initial_thresh = np.median(maxes)
    thresh_line = ax2.axvline(initial_thresh, color='r', linestyle='--', label='Threshold')

    # Circles for detected blobs
    circles = []

    def update_display(thresh):
        """Update the display with new threshold"""
        # Clear previous circles
        for circle in circles:
            circle.remove()
        circles.clear()

        # Find blobs above threshold
        for i in range(map_data.shape[0]):
            for j in range(map_data.shape[1]):
                if map_data[i, j] > thresh ** 2:
                    radius = np.sqrt(2 * scale_map[i, j])
                    circle = plt.Circle((j, i), radius, fill=False, color='red', linewidth=2)
                    ax1.add_patch(circle)
                    circles.append(circle)

        # Update threshold line
        thresh_line.set_xdata([thresh, thresh])
        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Blob Strength', 0, maxes.max() * 1.1,
                    valinit=initial_thresh, valstep=maxes.max() / 200)

    # Store threshold value
    threshold_value = [initial_thresh]

    def update(val):
        threshold_value[0] = slider.val
        update_display(slider.val)

    slider.on_changed(update)

    # Buttons
    ax_accept = plt.axes([0.7, 0.92, 0.1, 0.04])
    btn_accept = Button(ax_accept, 'Accept')

    ax_quit = plt.axes([0.85, 0.92, 0.1, 0.04])
    btn_quit = Button(ax_quit, 'Quit')

    def accept(event):
        plt.close(fig)

    def quit_func(event):
        threshold_value[0] = -1
        plt.close(fig)

    btn_accept.on_clicked(accept)
    btn_quit.on_clicked(quit_func)

    # Initial display
    update_display(initial_thresh)

    plt.show()

    return threshold_value[0] if threshold_value[0] != -1 else None


def flattening_interactive_threshold(im):
    """
    Interactive threshold selection for flattening mask

    Returns:
    float: Selected threshold value
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display image
    im_display = ax.imshow(im, cmap='gray')
    ax.set_title('Set Height Threshold for Flattening')

    # Mask overlay
    mask = np.zeros_like(im)
    mask_display = ax.imshow(mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)

    # Initial threshold
    initial_thresh = np.mean(im)

    # Slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Threshold', im.min(), im.max(),
                    valinit=initial_thresh, valstep=(im.max() - im.min()) / 300)

    # Store threshold value
    threshold_value = [initial_thresh]

    def update(val):
        threshold_value[0] = slider.val
        # Update mask
        mask = (im > slider.val).astype(float)
        mask_display.set_data(mask)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Button
    ax_accept = plt.axes([0.45, 0.92, 0.1, 0.04])
    btn_accept = Button(ax_accept, 'Accept')

    def accept(event):
        plt.close(fig)

    btn_accept.on_clicked(accept)

    # Initial display
    update(initial_thresh)

    plt.show()

    return threshold_value[0]


def create_series_folder(base_path="./results"):
    """Create a series folder similar to Igor Pro's Series_X naming"""
    os.makedirs(base_path, exist_ok=True)

    # Find next available series number
    series_num = 0
    while os.path.exists(os.path.join(base_path, f"Series_{series_num}")):
        series_num += 1

    series_folder = DataFolder(base_path, f"Series_{series_num}")
    return series_folder


def bilinear_interpolate(im, x0, y0, r0=0):
    """
    Bilinear interpolation matching Igor Pro implementation
    """
    # Get dimensions
    if im.ndim == 2:
        limP, limQ = im.shape
    else:
        limP, limQ, _ = im.shape

    # Calculate positions
    pMid = x0
    p0 = max(0, int(np.floor(pMid)))
    p1 = min(limP - 1, int(np.ceil(pMid)))

    qMid = y0
    q0 = max(0, int(np.floor(qMid)))
    q1 = min(limQ - 1, int(np.ceil(qMid)))

    # Interpolate
    if im.ndim == 2:
        pInterp0 = im[p0, q0] + (im[p1, q0] - im[p0, q0]) * (pMid - p0)
        pInterp1 = im[p0, q1] + (im[p1, q1] - im[p0, q1]) * (pMid - p0)
    else:
        pInterp0 = im[p0, q0, r0] + (im[p1, q0, r0] - im[p0, q0, r0]) * (pMid - p0)
        pInterp1 = im[p0, q1, r0] + (im[p1, q1, r0] - im[p0, q1, r0]) * (pMid - p0)

    return pInterp0 + (pInterp1 - pInterp0) * (qMid - q0)


def expand_boundary_8(mask):
    """Expand boundary by 8-connectivity"""
    mask_expanded = np.zeros_like(mask)

    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 0:
                # Check 8-connected neighbors
                if (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                        mask[i, j + 1] == 1 or mask[i, j - 1] == 1 or
                        mask[i + 1, j + 1] == 1 or mask[i - 1, j + 1] == 1 or
                        mask[i + 1, j - 1] == 1 or mask[i - 1, j - 1] == 1):
                    mask_expanded[i, j] = 2
            else:
                mask_expanded[i, j] = mask[i, j]

    return (mask_expanded > 0).astype(float)


def expand_boundary_4(mask):
    """Expand boundary by 4-connectivity"""
    mask_expanded = np.zeros_like(mask)

    for i in range(1, mask.shape[0] - 1):
        for j in range(1, mask.shape[1] - 1):
            if mask[i, j] == 0:
                # Check 4-connected neighbors
                if (mask[i + 1, j] == 1 or mask[i - 1, j] == 1 or
                        mask[i, j + 1] == 1 or mask[i, j - 1] == 1):
                    mask_expanded[i, j] = 2
            else:
                mask_expanded[i, j] = mask[i, j]

    return (mask_expanded > 0).astype(float)


def view_particles(particle_results):
    """
    View particles with navigation similar to Igor Pro's ViewParticles
    """
    if not particle_results or 'particles' not in particle_results:
        print("No particles to view")
        return

    particles = particle_results['particles']
    if not particles:
        print("No particles found")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Navigation state
    current_idx = [0]

    def show_particle(idx):
        """Display particle at given index"""
        ax.clear()

        particle = particles[idx]
        ax.imshow(particle['image'], cmap='hot')

        # Show perimeter if available
        if 'perimeter' in particle:
            perimeter = particle['perimeter']
            # Create colored overlay for perimeter
            overlay = np.zeros((*perimeter.shape, 4))
            overlay[perimeter > 0] = [0, 1, 0, 1]  # Green perimeter
            ax.imshow(overlay)

        # Display info
        info_text = f"Particle {idx}\n"
        if 'height' in particle:
            info_text += f"Height: {particle['height']:.4f}\n"
        if 'volume' in particle:
            info_text += f"Volume: {particle['volume']:.4f}\n"
        if 'area' in particle:
            info_text += f"Area: {particle['area']:.4f}"

        ax.set_title(info_text)
        fig.canvas.draw()

    def on_key(event):
        """Handle keyboard navigation"""
        if event.key == 'right' and current_idx[0] < len(particles) - 1:
            current_idx[0] += 1
            show_particle(current_idx[0])
        elif event.key == 'left' and current_idx[0] > 0:
            current_idx[0] -= 1
            show_particle(current_idx[0])
        elif event.key == ' ' or event.key == 'down':
            # Delete particle
            response = input(f"Delete particle {current_idx[0]}? (y/n): ")
            if response.lower() == 'y':
                particles.pop(current_idx[0])
                if current_idx[0] >= len(particles) and current_idx[0] > 0:
                    current_idx[0] -= 1
                if particles:
                    show_particle(current_idx[0])
                else:
                    plt.close(fig)

    # Connect keyboard handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show first particle
    show_particle(0)

    plt.show()