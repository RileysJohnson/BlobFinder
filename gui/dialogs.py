"""Dialog windows for parameter input and interactive threshold selection."""

# #######################################################################
#                           GUI: DIALOGS
#
#   CONTENTS:
#       - class ParameterDialog: Methods for parameter input dialogs
#       - InteractiveThreshold(): Interactive blob strength selection
#       - validate_hessian_parameters(): Parameter validation
#
# #######################################################################

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import threading
from typing import Optional, Tuple
from utils.error_handler import handle_error, safe_print
from core.blob_detection import GetMaxes


def validate_hessian_parameters(params):
    """Validate Hessian blob parameters - EXACT IGOR PRO VALIDATION."""
    try:
        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params

        if scaleStart <= 0:
            raise ValueError("Minimum size must be positive")
        if layers <= 0:
            raise ValueError("Maximum size must be positive")
        if scaleFactor <= 1.0:
            raise ValueError("Scaling factor must be greater than 1.0")

        # Igor Pro threshold validation - exact logic
        # -1 = Otsu's method, -2 = Interactive, positive number = Manual
        if detHResponseThresh != -1 and detHResponseThresh != -2 and detHResponseThresh <= 0:
            raise ValueError("Manual threshold must be positive (use -1 for Otsu's, -2 for Interactive)")

        if particleType not in [-1, 0, 1]:
            raise ValueError("Particle type must be -1, 0, or 1")
        if subPixelMult < 1:
            raise ValueError("Subpixel ratio must be >= 1")
        if allowOverlap not in [0, 1]:
            raise ValueError("Allow overlap must be 0 or 1")

    except Exception as e:
        raise ValueError(f"Parameter validation failed: {e}")


class ParameterDialog:
    """Dialog for getting Hessian blob parameters - EXACT IGOR PRO INTERFACE."""

    @staticmethod
    def get_hessian_parameters() -> Optional[Tuple]:
        """Get Hessian blob parameters - matches Igor Pro DoPrompt exactly"""
        root = tk.Tk()
        root.title("Hessian Blob Parameters")
        root.geometry("600x550")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (600 // 2)
        y = (root.winfo_screenheight() // 2) - (550 // 2)
        root.geometry(f"600x550+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Hessian Blob Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        subtitle_label = tk.Label(root, text="Configure parameters for blob detection",
                                  font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 15))

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20)

        # Parameter variables - Igor Pro defaults
        vars_dict = {
            'scaleStart': tk.DoubleVar(value=1.0),  # Minimum Size in Pixels
            'layers': tk.DoubleVar(value=15.0),  # Maximum Size in Pixels
            'scaleFactor': tk.DoubleVar(value=1.5),  # Scaling Factor
            'particleType': tk.IntVar(value=1),  # Particle Type
            'subPixelMult': tk.IntVar(value=1),  # Subpixel Ratio
            'allowOverlap': tk.IntVar(value=0)  # Allow Overlap
        }

        # Igor Pro parameter labels - exact wording
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

        # Special handling for blob strength threshold - EXACT IGOR PRO LOGIC
        threshold_frame = tk.Frame(main_frame, bg='#f0f0f0')
        threshold_frame.pack(fill='x', pady=5)

        threshold_label = tk.Label(threshold_frame, text="Minimum Blob Strength", width=50, anchor='w',
                                   font=('Arial', 10), bg='#f0f0f0')
        threshold_label.pack(side='left')

        # Right side container for dropdown and entry
        right_container = tk.Frame(threshold_frame, bg='#f0f0f0')
        right_container.pack(side='right', padx=(10, 0))

        # Dropdown for threshold method - Igor Pro exact options
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
            """Handle threshold method changes - EXACT IGOR PRO BEHAVIOR."""
            method = threshold_method.get()
            if method == "Manual":
                threshold_entry.config(state='normal')
            else:
                threshold_entry.config(state='disabled')

        threshold_method.trace('w', on_threshold_method_change)

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def validate_and_continue():
            """Validate and continue - EXACT IGOR PRO LOGIC."""
            try:
                # Determine threshold value based on method - EXACT IGOR PRO LOGIC
                method = threshold_method.get()
                if method == "Interactive":
                    detHResponseThresh = -2  # Igor Pro: -2 for interactive
                elif method == "Otsu's Method":
                    detHResponseThresh = -1  # Igor Pro: -1 for Otsu's method
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
            # Igor Pro-style help text
            help_text = """
Hessian Blob Parameters Help:

1. Minimum Size: Minimum radius of particles to detect (pixels)
2. Maximum Size: Maximum radius of particles to detect (pixels)  
3. Scaling Factor: Scale-space precision (1.2-2.0, default 1.5)
4. Blob Strength: 
   - Interactive: Select threshold visually with slider
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
        """Get particle constraints - EXACT IGOR PRO INTERFACE."""
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

        # Igor Pro constraint variables with defaults
        constraint_vars = {
            'minH': tk.StringVar(value="-inf"),  # Minimum height
            'maxH': tk.StringVar(value="inf"),  # Maximum height
            'minA': tk.StringVar(value="-inf"),  # Minimum area
            'maxA': tk.StringVar(value="inf"),  # Maximum area
            'minV': tk.StringVar(value="-inf"),  # Minimum volume
            'maxV': tk.StringVar(value="inf")  # Maximum volume
        }

        # Igor Pro constraint labels - exact wording
        constraint_labels = [
            "Minimum height",
            "Maximum height",
            "Minimum area",
            "Maximum area",
            "Minimum volume",
            "Maximum volume"
        ]

        # Create constraint input fields
        for i, (key, var) in enumerate(constraint_vars.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=8)

            label = tk.Label(frame, text=constraint_labels[i], width=20, anchor='w',
                             font=('Arial', 11), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=15, font=('Arial', 11))
            entry.pack(side='right', padx=(10, 0))

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=10)

        def validate_and_accept():
            """Validate constraints and accept - EXACT IGOR PRO LOGIC."""
            try:
                constraints = []
                for key in ['minH', 'maxH', 'minA', 'maxA', 'minV', 'maxV']:
                    value = constraint_vars[key].get().strip()
                    if value == "-inf":
                        constraints.append(-np.inf)
                    elif value == "inf":
                        constraints.append(np.inf)
                    else:
                        constraints.append(float(value))

                # Validate ranges
                if constraints[0] > constraints[1]:  # minH > maxH
                    raise ValueError("Minimum height cannot be greater than maximum height")
                if constraints[2] > constraints[3]:  # minA > maxA
                    raise ValueError("Minimum area cannot be greater than maximum area")
                if constraints[4] > constraints[5]:  # minV > maxV
                    raise ValueError("Minimum volume cannot be greater than maximum volume")

                result[0] = tuple(constraints)
                root.destroy()

            except Exception as e:
                error_label.config(text=str(e))

        def on_cancel():
            root.destroy()

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="OK", command=validate_and_accept,
                  bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]


def InteractiveThreshold(im, detH, LG, particleType, maxCurvatureRatio):
    """Interactive threshold selection - EXACT IGOR PRO IMPLEMENTATION."""
    try:
        # Ensure we're in main thread for GUI operations
        if threading.current_thread() != threading.main_thread():
            safe_print("Warning: Interactive threshold requires main thread. Using Otsu method.")
            from core.blob_detection import OtsuThreshold
            return np.sqrt(OtsuThreshold(detH, LG, particleType, maxCurvatureRatio))

        # Close any existing plots
        plt.close('all')

        import matplotlib
        matplotlib.use('TkAgg')

        # Igor Pro: Duplicate/O detH SS_MAXMAP
        # Create maxima map and scale map
        maxes_result = GetMaxes(detH, LG, particleType, maxCurvatureRatio, create_maps=True)
        if maxes_result is None:
            safe_print("No maxima found for interactive threshold.")
            return 0.0

        SS_MAXMAP, SS_MAXSCALEMAP = maxes_result

        # Igor Pro: WaveStats/Q SS_MAXMAP
        # Get statistics for threshold range
        max_val = np.max(SS_MAXMAP)
        min_val = np.min(SS_MAXMAP[SS_MAXMAP > 0])  # Only positive values

        if max_val <= 0:
            safe_print("No positive maxima found.")
            return 0.0

        # Igor Pro: SS_THRESH = sqrt(max_val * 0.1)
        SS_THRESH = np.sqrt(max_val * 0.1)

        # Create display - EXACT IGOR PRO INTERFACE
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.25, right=0.75)

        # Igor Pro: NewImage/K=1 /F im
        # Display image exactly like Igor Pro
        im_display = ax.imshow(im, cmap='gray', aspect='equal')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        # Igor Pro slider setup
        ax_slider = plt.axes([0.2, 0.1, 0.4, 0.03])
        slider = Slider(ax_slider, '', np.sqrt(min_val), np.sqrt(max_val),
                        valinit=SS_THRESH, valfmt='%.3e')

        # Igor Pro text panel - exact layout
        ax_text = plt.axes([0.77, 0.3, 0.2, 0.4])
        ax_text.axis('off')
        threshold_text = ax_text.text(0.1, 0.9, f'Blob Strength:\n{SS_THRESH:.3e}',
                                      fontsize=11, transform=ax_text.transAxes,
                                      verticalalignment='top')

        # Store circles for redrawing
        circles = []

        def update_display(thresh):
            """Update display with new threshold - EXACT IGOR PRO LOGIC."""
            # Igor Pro: SetDrawLayer/K /W=IMAGE overlay
            for circle in circles:
                circle.remove()
            circles.clear()

            count = 0
            limI, limJ = SS_MAXMAP.shape

            # Igor Pro: For(i=0;i<limI;i+=1) For(j=0;j<limJ;j+=1)
            for i in range(limI):
                for j in range(limJ):
                    # Igor Pro: If(Map[i][j]>S_Struct.curval^2)
                    if SS_MAXMAP[i, j] > thresh ** 2:
                        # Igor Pro: xc = DimOffset(map,0)+i*DimDelta(map,0)
                        # Igor Pro: yc = DimOffset(map,1)+j*DimDelta(map,1)
                        xc = j  # Column coordinate
                        yc = i  # Row coordinate

                        # Igor Pro: rad = ScaleMap[i][j]
                        rad = max(1.0, SS_MAXSCALEMAP[i, j])

                        # Igor Pro: SetDrawEnv xCoord= prel,yCoord= prel,linethick= 2,linefgc= (65535,16385,16385)
                        circle = plt.Circle((xc, yc), rad, fill=False, color='red', linewidth=2)
                        ax.add_patch(circle)
                        circles.append(circle)
                        count += 1

            # Update display
            ax.set_title(f'IMAGE:Original\nBlob Strength: {thresh:.3e}, Particles: {count}',
                         fontsize=12, loc='left')
            threshold_text.set_text(f'Blob Strength:\n{thresh:.3e}\n\n{count} particles')
            fig.canvas.draw_idle()

        slider.on_changed(update_display)
        update_display(SS_THRESH)

        # Igor Pro: Button btn title="Accept"
        # Igor Pro: Button btnQuit title="Quit"
        ax_accept = plt.axes([0.77, 0.15, 0.08, 0.05])
        ax_quit = plt.axes([0.86, 0.15, 0.08, 0.05])
        button_accept = Button(ax_accept, 'Accept')
        button_quit = Button(ax_quit, 'Quit')

        result = [SS_THRESH]

        def accept_threshold(event):
            """Igor Pro InteractiveContinue function equivalent."""
            # Igor Pro: If( B_Struct.eventCode==2 ) KillWindow/Z IMAGE
            result[0] = slider.val
            plt.close(fig)

        def quit_threshold(event):
            """Igor Pro InteractiveQuit function equivalent."""
            # Igor Pro: KillWindow/Z IMAGE, Abort
            result[0] = SS_THRESH
            plt.close(fig)

        button_accept.on_clicked(accept_threshold)
        button_quit.on_clicked(quit_threshold)

        # Igor Pro: PauseForUser IMAGE
        plt.show()

        # Igor Pro: Variable returnVal = SS_THRESH
        returnVal = result[0]

        # Igor Pro: KillVariables/Z SS_THRESH, KillWaves/Z Map
        return returnVal

    except Exception as e:
        handle_error("InteractiveThreshold", e)
        return 0.0