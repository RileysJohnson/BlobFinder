import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from utils.validators import validate_hessian_parameters, validate_constraints
from utils.error_handler import handle_error, safe_print
import matplotlib


class ParameterDialog:
    """Parameter dialog with validation"""

    @staticmethod
    def get_hessian_parameters() -> Optional[Tuple]:
        """Get Hessian blob parameters"""
        root = tk.Tk()
        root.title("Hessian Blob Parameters")
        root.geometry("700x550")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (700 // 2)
        y = (root.winfo_screenheight() // 2) - (550 // 2)
        root.geometry(f"700x550+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Hessian Blob Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Default values
        default_values = {
            'scaleStart': 1,
            'layers': 256,
            'scaleFactor': 1.5,
            'particleType': 1,
            'subPixelMult': 1,
            'allowOverlap': 0
        }

        vars_dict = {}
        for key, default in default_values.items():
            if isinstance(default, int):
                vars_dict[key] = tk.IntVar(value=default)
            else:
                vars_dict[key] = tk.DoubleVar(value=default)

        labels = [
            "Minimum Size in Pixels",
            "Maximum Size in Pixels",
            "Scaling Factor",
            "Particle Type (-1 for negative, +1 for positive, 0 for both)",
            "Subpixel Ratio",
            "Allow Hessian Blobs to Overlap? (1=yes 0=no)"
        ]

        # Create parameter input fields
        entries = {}
        for i, (key, var) in enumerate(vars_dict.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=5)

            label = tk.Label(frame, text=labels[i], width=50, anchor='w',
                             font=('Arial', 10), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=15, font=('Arial', 10))
            entry.pack(side='right', padx=(10, 0))
            entries[key] = entry

        # Special handling for blob strength threshold
        threshold_frame = tk.Frame(main_frame, bg='#f0f0f0')
        threshold_frame.pack(fill='x', pady=5)

        threshold_label = tk.Label(threshold_frame, text="Minimum Blob Strength", width=50, anchor='w',
                                   font=('Arial', 10), bg='#f0f0f0')
        threshold_label.pack(side='left')

        # Right side container for dropdown and entry
        right_container = tk.Frame(threshold_frame, bg='#f0f0f0')
        right_container.pack(side='right', padx=(10, 0))

        # Dropdown for threshold method
        threshold_method = tk.StringVar(value="Interactive")
        threshold_dropdown = ttk.Combobox(right_container, textvariable=threshold_method,
                                          values=["Interactive", "Otsu's Method", "Manual"],
                                          state="readonly", width=12)
        threshold_dropdown.pack(side='left', padx=(0, 5))

        # Entry for manual threshold value
        manual_threshold = tk.DoubleVar(value=0.001)
        threshold_entry = tk.Entry(right_container, textvariable=manual_threshold, width=15,
                                   font=('Arial', 10), state='disabled')
        threshold_entry.pack(side='left')

        def on_threshold_method_change(*args):
            if threshold_method.get() == "Manual":
                threshold_entry.config(state='normal')
            else:
                threshold_entry.config(state='disabled')

        threshold_method.trace('w', on_threshold_method_change)

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def validate_and_continue():
            try:
                # Determine threshold value based on method
                method = threshold_method.get()
                if method == "Interactive":
                    detHResponseThresh = -2
                elif method == "Otsu's Method":
                    detHResponseThresh = -1
                else:  # Manual
                    detHResponseThresh = manual_threshold.get()
                    if detHResponseThresh <= 0:
                        raise ValueError("Manual threshold must be positive")

                params = [
                    vars_dict['scaleStart'].get(),
                    vars_dict['layers'].get(),
                    vars_dict['scaleFactor'].get(),
                    detHResponseThresh,
                    vars_dict['particleType'].get(),
                    vars_dict['subPixelMult'].get(),
                    vars_dict['allowOverlap'].get()
                ]

                validate_hessian_parameters(params)
                result[0] = tuple(params)
                root.destroy()

            except Exception as e:
                error_label.config(text=str(e))

        def on_cancel():
            root.destroy()

        def show_help():
            help_text = """
Hessian Blob Parameters Help:

1. Minimum Size: Minimum radius of particles to detect (pixels)
2. Maximum Size: Maximum radius of particles to detect (pixels)  
3. Scaling Factor: Scale-space precision (1.2-2.0, default 1.5)
4. Blob Strength: 
   - Interactive: Select threshold visually
   - Otsu's Method: Automatic threshold calculation
   - Manual: Enter specific threshold value
5. Particle Type: +1=positive blobs, -1=negative, 0=both
6. Subpixel Ratio: Subpixel precision multiplier (1=pixel accuracy)
7. Allow Overlap: 1=allow overlapping particles, 0=no overlap
"""
            messagebox.showinfo("Parameter Help", help_text)

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=validate_and_continue,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Help", command=show_help,
                  bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]

    @staticmethod
    def get_constraints_dialog() -> Optional[Tuple]:
        """Get particle constraints"""
        root = tk.Tk()
        root.title("Constraints")
        root.geometry("500x400")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (500 // 2)
        y = (root.winfo_screenheight() // 2) - (400 // 2)
        root.geometry(f"500x400+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Particle Constraints",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        subtitle_label = tk.Label(root,
                                  text="Limit analysis to particles within certain bounds\n(use -inf and inf for no constraints)",
                                  font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 15))

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20)

        # Default constraint values
        vars_dict = {
            'minHeight': tk.StringVar(value="-inf"),
            'maxHeight': tk.StringVar(value="inf"),
            'minArea': tk.StringVar(value="-inf"),
            'maxArea': tk.StringVar(value="inf"),
            'minVolume': tk.StringVar(value="-inf"),
            'maxVolume': tk.StringVar(value="inf")
        }

        labels = [
            "Minimum Height",
            "Maximum Height",
            "Minimum Area",
            "Maximum Area",
            "Minimum Volume",
            "Maximum Volume"
        ]

        # Create constraint input fields
        for i, (key, var) in enumerate(vars_dict.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=5)

            label = tk.Label(frame, text=labels[i], width=20, anchor='w',
                             font=('Arial', 10), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=20, font=('Arial', 10))
            entry.pack(side='right', padx=(10, 0))

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def parse_value(val_str):
            val_str = val_str.strip()
            if val_str.lower() in ['-inf', '-infinity']:
                return -np.inf
            elif val_str.lower() in ['inf', 'infinity']:
                return np.inf
            else:
                return float(val_str)

        def validate_and_continue():
            try:
                constraints = [
                    parse_value(vars_dict['minHeight'].get()),
                    parse_value(vars_dict['maxHeight'].get()),
                    parse_value(vars_dict['minArea'].get()),
                    parse_value(vars_dict['maxArea'].get()),
                    parse_value(vars_dict['minVolume'].get()),
                    parse_value(vars_dict['maxVolume'].get())
                ]

                validate_constraints(constraints)
                result[0] = tuple(constraints)
                root.destroy()

            except Exception as e:
                error_label.config(text=str(e))

        def on_cancel():
            root.destroy()

        def show_help():
            help_text = """
Particle Constraints Help:

Set bounds to filter particles by their measurements:
- Height: vertical extent above background
- Area: 2D projected area in image
- Volume: integrated intensity above background

Use "-inf" for no lower bound
Use "inf" for no upper bound
Use numbers for specific limits

Example: minHeight=0, maxHeight=5e-9 
(particles between 0 and 5 nanometers tall)
"""
            messagebox.showinfo("Constraints Help", help_text)

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=validate_and_continue,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Help", command=show_help,
                  bg='#95a5a6', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """Lets the user interactively choose a blob strength for the determinant of Hessian."""
    try:
        from core.blob_detection import GetMaxes

        # First identify the maxes
        SS_MAXMAP = np.full_like(im, -1.0)
        SS_MAXSCALEMAP = np.zeros_like(im)
        Maxes = GetMaxes(detH, LG, particleType, maxCurvatureRatio, map_wave=SS_MAXMAP, scaleMap=SS_MAXSCALEMAP)

        # Put it into image units - Igor Pro applies Sqrt() to maxes values
        Maxes_sqrt = np.sqrt(np.maximum(Maxes, 0))

        if len(Maxes_sqrt) == 0:
            safe_print("No maxima found for interactive threshold selection.")
            return 0.0

        # Ensure thread-safe matplotlib backend
        import matplotlib
        matplotlib.use('TkAgg')

        # Close any existing plots
        plt.close('all')

        # Create interactive plot exactly matching Igor Pro Figure 17
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, right=0.8)

        im_display = ax.imshow(im, cmap='gray', interpolation='bilinear')
        ax.set_title('Interactive Blob Strength Selection\nAdjust slider to see detected blobs',
                     fontsize=14, fontweight='bold')

        # Igor Pro: SS_THRESH = WaveMax(Maxes)/2
        SS_THRESH = np.max(Maxes_sqrt) / 2

        # Create slider panel exactly like Igor Pro
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Blob Strength', 0, np.max(Maxes_sqrt) * 1.1,
                        valinit=SS_THRESH, valfmt='%.3e')

        # Create text display for current threshold value
        ax_text = plt.axes([0.82, 0.5, 0.15, 0.3])
        ax_text.axis('off')
        threshold_text = ax_text.text(0.1, 0.9, f'Blob Strength:\n{SS_THRESH:.3e}',
                                      fontsize=10, transform=ax_text.transAxes)

        circles = []

        def update_display(thresh):
            # Clear previous circles
            for circle in circles:
                try:
                    circle.remove()
                except:
                    pass
            circles.clear()

            # Igor Pro compares Map[i][j]>S_Struct.curval^2
            # Since SS_MAXMAP contains the original detH values (not sqrt),
            # we need to square the threshold for comparison
            thresh_squared = thresh ** 2
            count = 0
            for i in range(SS_MAXMAP.shape[0]):
                for j in range(SS_MAXMAP.shape[1]):
                    if SS_MAXMAP[i, j] > thresh_squared:
                        xc = j  # Column is x-coordinate
                        yc = i  # Row is y-coordinate

                        # Igor Pro: rad = sqrt(2*SS_MAXSCALEMAP[i,j])
                        rad = max(2, np.sqrt(2 * SS_MAXSCALEMAP[i, j]))

                        # Create RED circles exactly like Igor Pro Figure 17
                        circle = plt.Circle((xc, yc), rad, color='red', fill=False,
                                            linewidth=2.5, alpha=0.9)
                        ax.add_patch(circle)
                        circles.append(circle)
                        count += 1

            # Update title and text display exactly like Igor Pro
            ax.set_title(f'Interactive Blob Strength Selection\n'
                         f'Blob Strength: {thresh:.3e}, Particles: {count}',
                         fontsize=14, fontweight='bold')

            threshold_text.set_text(f'Blob Strength:\n{thresh:.3e}\n\nParticles: {count}')

            # Thread-safe canvas update
            try:
                fig.canvas.draw_idle()
            except:
                pass

        slider.on_changed(update_display)
        update_display(SS_THRESH)

        # Create Accept and Quit buttons exactly like Igor Pro
        ax_accept = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_quit = plt.axes([0.81, 0.02, 0.1, 0.04])
        button_accept = Button(ax_accept, 'Accept')
        button_quit = Button(ax_quit, 'Quit')

        result = [SS_THRESH]

        def accept_threshold(event):
            result[0] = slider.val
            plt.close(fig)

        def quit_threshold(event):
            result[0] = SS_THRESH
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)
        button_quit.on_clicked(quit_threshold)

        # Add instructions exactly like Igor Pro
        instructions_text = ('Use the slider to adjust blob strength threshold.\n'
                             'Red circles show detected particles.\n'
                             'Click "Accept" when satisfied with detection.')

        ax_text.text(0.1, 0.3, instructions_text, fontsize=9,
                     transform=ax_text.transAxes, style='italic')

        safe_print("Interactive threshold selection:")
        safe_print("- Use slider to adjust blob strength threshold")
        safe_print("- Red circles show detected particles")
        safe_print("- Click 'Accept' when satisfied with detection")

        # Use blocking show() to prevent threading issues
        plt.show(block=True)

        return result[0]

    except Exception as e:
        handle_error("InteractiveThreshold", e)
        return 0.0