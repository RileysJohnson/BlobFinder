"""Contains the GUI for interactively viewing detected particles."""

# #######################################################################
#                        GUI: PARTICLE VIEWER
#
#   CONTENTS:
#       - def ViewParticles: The main entry point to launch the particle
#         viewer window. It finds particle folders and initializes the
#         viewer class, which handles the UI and user interactions.
#
# #######################################################################

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from utils.error_handler import handle_error, safe_print

def ViewParticles():
    """View and examine individual detected particles - EXACT IGOR PRO TUTORIAL MATCH."""
    try:
        # Get the folder containing particles - matching Igor Pro GetBrowserSelection
        particles_folder = filedialog.askdirectory(title="Select particles folder (e.g., YEG_1_Particles)")
        if not particles_folder:
            safe_print("No folder selected.")
            return

        # Look for Particle_X folders - matching Igor Pro structure exactly
        particle_folders = []
        try:
            for item in os.listdir(particles_folder):
                item_path = os.path.join(particles_folder, item)
                if os.path.isdir(item_path) and item.startswith("Particle_"):
                    try:
                        # Verify it's a valid particle number
                        particle_num = int(item.split("_")[-1])
                        particle_folders.append(item_path)
                    except ValueError:
                        continue
        except Exception as e:
            safe_print(f"Error reading folder contents: {e}")
            return

        if len(particle_folders) == 0:
            safe_print("No particle folders found in selected directory.")
            safe_print("Make sure you selected a folder like 'YEG_1_Particles' that contains 'Particle_X' subfolders.")
            return

        # Sort by particle number - matching Igor Pro order
        particle_folders.sort(key=lambda x: int(x.split("_")[-1]))

        safe_print(f"Found {len(particle_folders)} particles to view.")

        # Import required for image display
        try:
            from PIL import ImageTk
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
        except ImportError:
            safe_print("PIL/Pillow or matplotlib not found. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'Pillow'])
            from PIL import ImageTk
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize

        # Create Igor Pro-style particle viewer
        class IgorProParticleViewer:
            def __init__(self, folders):
                self.folders = folders
                self.current_index = 0

                # Create main window - matching Igor Pro Particle Viewer size
                self.root = tk.Toplevel()
                self.root.title("Particle Viewer")
                self.root.geometry("1200x900")
                self.root.configure(bg='#e6e6e6')  # Igor Pro gray background

                # Set up the interface matching Figure 24 from tutorial
                self.setup_igor_interface()

                # Show first particle
                self.show_particle()

                # Bind keyboard events - matching Igor Pro shortcuts
                self.root.bind('<Key>', self.on_key_press)
                self.root.focus_set()

            def setup_igor_interface(self):
                """Set up interface matching Igor Pro tutorial Figure 24"""

                # Main content frame
                main_frame = tk.Frame(self.root, bg='#e6e6e6')
                main_frame.pack(fill='both', expand=True, padx=10, pady=10)

                # Left side - image display (larger, matching Igor Pro)
                left_frame = tk.Frame(main_frame, bg='#e6e6e6', width=800)
                left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
                left_frame.pack_propagate(False)

                # Image frame with border (matching Igor Pro image window)
                image_frame = tk.Frame(left_frame, bg='white', relief='sunken', bd=2)
                image_frame.pack(pady=10, padx=10, fill='both', expand=True)

                # Image canvas - matching Igor Pro image size
                self.canvas = tk.Canvas(image_frame, bg='black', width=700, height=700)
                self.canvas.pack(padx=5, pady=5)

                # Right side - controls panel (matching Igor Pro Controls panel)
                right_frame = tk.Frame(main_frame, bg='#e6e6e6', width=350)
                right_frame.pack(side='right', fill='y')
                right_frame.pack_propagate(False)

                # Controls title - matching Igor Pro "Controls" panel
                controls_title = tk.Label(right_frame, text="Controls",
                                          font=('Arial', 14, 'bold'), bg='#e6e6e6')
                controls_title.pack(pady=(0, 10))

                # Particle title - matching Igor Pro "Particle 21" style
                self.particle_title = tk.Label(right_frame, text="Particle 0",
                                               font=('Arial', 16, 'bold'), bg='#e6e6e6')
                self.particle_title.pack(pady=5)

                # Navigation buttons - matching Igor Pro layout
                nav_frame = tk.Frame(right_frame, bg='#e6e6e6')
                nav_frame.pack(pady=10)

                # Prev and Next buttons side by side
                btn_frame = tk.Frame(nav_frame, bg='#e6e6e6')
                btn_frame.pack()

                self.prev_btn = tk.Button(btn_frame, text="Prev",
                                          command=self.prev_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.prev_btn.pack(side='left', padx=5)

                self.next_btn = tk.Button(btn_frame, text="Next",
                                          command=self.next_particle,
                                          bg='white', relief='raised', bd=2,
                                          font=('Arial', 12), width=8, height=1)
                self.next_btn.pack(side='left', padx=5)

                # Go To field - matching Igor Pro "Go To: 0"
                goto_frame = tk.Frame(right_frame, bg='#e6e6e6')
                goto_frame.pack(pady=10)

                tk.Label(goto_frame, text="Go To:", font=('Arial', 12), bg='#e6e6e6').pack(side='left')
                self.goto_entry = tk.Entry(goto_frame, width=10, font=('Arial', 12))
                self.goto_entry.pack(side='left', padx=5)
                self.goto_entry.bind('<Return>', self.goto_particle)

                # Counter display - matching Igor Pro "1/10" style
                self.counter_label = tk.Label(right_frame, text="1/1",
                                              font=('Arial', 12), bg='#e6e6e6')
                self.counter_label.pack(pady=5)

                # Measurements section - matching Igor Pro Figure 24
                measurements_frame = tk.LabelFrame(right_frame, text="Measurements",
                                                   font=('Arial', 12, 'bold'), bg='#e6e6e6')
                measurements_frame.pack(fill='x', pady=10, padx=10)

                # Height display - matching Igor Pro "Height (nm)"
                height_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                height_frame.pack(fill='x', pady=5)
                tk.Label(height_frame, text="Height (nm)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.height_label = tk.Label(height_frame, text="0.0000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.height_label.pack(fill='x', padx=5)

                # Volume display - matching Igor Pro "Volume (m^3 e-25)"
                volume_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                volume_frame.pack(fill='x', pady=5)
                tk.Label(volume_frame, text="Volume (m^3 e-25)", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.volume_label = tk.Label(volume_frame, text="0.000",
                                             font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.volume_label.pack(fill='x', padx=5)

                # Area display - additional measurement
                area_frame = tk.Frame(measurements_frame, bg='#e6e6e6')
                area_frame.pack(fill='x', pady=5)
                tk.Label(area_frame, text="Area", font=('Arial', 11, 'bold'),
                         bg='#e6e6e6').pack()
                self.area_label = tk.Label(area_frame, text="0.0",
                                           font=('Arial', 11), bg='white', relief='sunken', bd=1)
                self.area_label.pack(fill='x', padx=5)

                # DELETE button - matching Igor Pro red DELETE button
                self.delete_btn = tk.Button(right_frame, text="DELETE",
                                            command=self.delete_particle,
                                            bg='#ff6b6b', fg='white',
                                            font=('Arial', 12, 'bold'),
                                            width=20, height=2, relief='raised', bd=2)
                self.delete_btn.pack(pady=20)

            def show_particle(self):
                """Display current particle - matching Igor Pro tutorial visualization"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Load particle files
                    particle_file = os.path.join(current_folder, f"Particle_{particle_num}.npy")
                    mask_file = os.path.join(current_folder, f"Mask_{particle_num}.npy")
                    info_file = os.path.join(current_folder, f"Particle_{particle_num}_info.txt")

                    if not os.path.exists(particle_file):
                        safe_print(f"Particle file not found: {particle_file}")
                        return

                    particle = np.load(particle_file)

                    # Update labels - matching Igor Pro style
                    self.particle_title.config(text=f"Particle {particle_num}")
                    self.counter_label.config(text=f"{self.current_index + 1}/{len(self.folders)}")
                    self.goto_entry.delete(0, tk.END)
                    self.goto_entry.insert(0, str(self.current_index))

                    # Display particle image with hot colormap and green contour
                    self.display_igor_image(particle, mask_file)

                    # Update measurements - matching Igor Pro format
                    self.update_measurements(info_file)

                    # Update window title
                    self.root.title(f"Particle Viewer - Particle {particle_num}")

                except Exception as e:
                    handle_error("show_particle", e)

            def display_igor_image(self, particle, mask_file):
                """Display particle image matching Igor Pro tutorial Figure 24 exactly"""
                try:
                    # Clear canvas
                    self.canvas.delete("all")

                    # Apply hot colormap exactly like Igor Pro
                    # Normalize data
                    vmin, vmax = np.min(particle), np.max(particle)
                    norm = Normalize(vmin=vmin, vmax=vmax)

                    hot_cmap = cm.get_cmap('hot')
                    colored = hot_cmap(norm(particle))
                    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

                    # Convert to PIL Image
                    img = Image.fromarray(colored_rgb)

                    # Resize to fit canvas (matching Igor Pro size)
                    canvas_size = 650
                    img_resized = img.resize((canvas_size, canvas_size), Image.NEAREST)

                    # Convert to PhotoImage
                    self.photo = ImageTk.PhotoImage(img_resized)

                    # Display on canvas
                    self.canvas.create_image(350, 350, image=self.photo)

                    # Add GREEN contour exactly like Igor Pro Figure 24
                    if os.path.exists(mask_file):
                        try:
                            mask = np.load(mask_file)

                            # Resize mask to match image
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_resized = np.array(mask_img.resize((canvas_size, canvas_size), Image.NEAREST))

                            # Create GREEN contour - matching Igor Pro exactly
                            contour_points = []
                            for i in range(1, canvas_size - 1):
                                for j in range(1, canvas_size - 1):
                                    if mask_resized[i, j] > 127:  # Inside mask
                                        # Check if it's an edge pixel
                                        neighbors = [
                                            mask_resized[i - 1, j], mask_resized[i + 1, j],
                                            mask_resized[i, j - 1], mask_resized[i, j + 1]
                                        ]
                                        if any(n < 127 for n in neighbors):
                                            x, y = j + 25, i + 25  # Offset for centering
                                            # Draw GREEN pixels - matching Igor Pro green contour
                                            self.canvas.create_rectangle(x, y, x + 2, y + 2,
                                                                         fill='#00ff00', outline='#00ff00')

                        except Exception as e:
                            safe_print(f"Could not display mask contour: {e}")

                    # Add coordinate axes labels if needed (matching Igor Pro)
                    self.canvas.create_text(350, 680, text="Pixels", font=('Arial', 10))

                except Exception as e:
                    handle_error("display_igor_image", e)

            def update_measurements(self, info_file):
                """Update measurements display - matching Igor Pro format"""
                try:
                    height_val = "0.0000"
                    volume_val = "0.000"
                    area_val = "0.0"

                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.startswith('Height:'):
                                    height_val = line.split(':')[1].strip()
                                elif line.startswith('Volume:'):
                                    volume_val = line.split(':')[1].strip()
                                elif line.startswith('Area:'):
                                    area_val = line.split(':')[1].strip()

                    # Update labels with proper formatting
                    self.height_label.config(text=height_val)
                    self.volume_label.config(text=volume_val)
                    self.area_label.config(text=area_val)

                except Exception as e:
                    handle_error("update_measurements", e)

            def prev_particle(self):
                """Navigate to previous particle - matching Igor Pro Prev button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index - 1) % len(self.folders)
                    self.show_particle()

            def next_particle(self):
                """Navigate to next particle - matching Igor Pro Next button"""
                if len(self.folders) > 0:
                    self.current_index = (self.current_index + 1) % len(self.folders)
                    self.show_particle()

            def goto_particle(self, event=None):
                """Go to specific particle - matching Igor Pro Go To functionality"""
                try:
                    particle_index = int(self.goto_entry.get())
                    if 0 <= particle_index < len(self.folders):
                        self.current_index = particle_index
                        self.show_particle()
                except ValueError:
                    pass  # Invalid input, ignore

            def delete_particle(self):
                """Delete current particle - matching Igor Pro DELETE button"""
                try:
                    if len(self.folders) == 0:
                        return

                    current_folder = self.folders[self.current_index]
                    particle_num = int(current_folder.split("_")[-1])

                    # Matching Igor Pro delete dialog exactly
                    result = messagebox.askyesno(
                        f"Deleting Particle {particle_num}..",
                        f"Are you sure you want to delete Particle {particle_num}?",
                        parent=self.root
                    )

                    if result:
                        shutil.rmtree(current_folder)
                        self.folders.pop(self.current_index)

                        if len(self.folders) == 0:
                            safe_print("No more particles to view.")
                            self.root.destroy()
                            return

                        if self.current_index >= len(self.folders):
                            self.current_index = len(self.folders) - 1

                        self.show_particle()

                except Exception as e:
                    handle_error("delete_particle", e)

            def on_key_press(self, event):
                """Handle keyboard shortcuts - matching Igor Pro tutorial"""
                if event.keysym == 'Left':
                    self.prev_particle()
                elif event.keysym == 'Right':
                    self.next_particle()
                elif event.keysym == 'space' or event.keysym == 'Down':
                    self.delete_particle()
                elif event.keysym == 'Return':
                    self.goto_particle()

        # Create and run viewer
        viewer = IgorProParticleViewer(particle_folders)

        # Print Igor Pro-style instructions
        safe_print("=" * 60)
        safe_print("PARTICLE VIEWER CONTROLS:")
        safe_print("- Left/Right arrows: Navigate between particles")
        safe_print("- Space bar or Down arrow: Delete current particle")
        safe_print("- Enter in Go To field: Jump to particle")
        safe_print("- Close window when finished")
        safe_print("=" * 60)

    except Exception as e:
        error_msg = handle_error("ViewParticles", e)
        try:
            messagebox.showerror("Viewer Error", error_msg)
        except:
            safe_print(error_msg)
