"""
Parameter Dialog Windows

Provides Igor Pro-style parameter input dialogs:
- ParameterDialog: Main parameter dialog class
- get_hessian_parameters: Hessian blob parameters dialog
- get_constraints_dialog: Particle constraints dialog
- get_preprocess_parameters: Preprocessing parameters dialog

These dialogs match the Igor Pro interface exactly.
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from core.validation import validate_hessian_parameters, validate_constraints


class ParameterDialog:
    """Parameter dialog with validation matching Igor Pro exactly"""

    @staticmethod
    def get_hessian_parameters():
        """
        Get Hessian blob parameters - EXACT IGOR PRO DIALOG

        Returns:
            Tuple of parameters or None if cancelled
        """
        root = tk.Tk()
        root.title("Hessian Blob Parameters")
        root.geometry("600x500")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (600 // 2)
        y = (root.winfo_screenheight() // 2) - (500 // 2)
        root.geometry(f"600x500+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Hessian Blob Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)

        # Default values exactly like Igor Pro
        default_values = {
            'scaleStart': 1,
            'layers': 256,
            'scaleFactor': 1.5,
            'detHResponseThresh': -2,
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

        # Labels exactly like Igor Pro
        labels = [
            "Minimum Size in Pixels",
            "Maximum Size in Pixels",
            "Scaling Factor",
            "Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)",
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

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def validate_and_continue():
            try:
                params = [
                    vars_dict['scaleStart'].get(),
                    vars_dict['layers'].get(),
                    vars_dict['scaleFactor'].get(),
                    vars_dict['detHResponseThresh'].get(),
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
4. Blob Strength: Threshold (-2=interactive, -1=Otsu, >0=manual)
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
    def get_constraints_dialog():
        """
        Get particle constraints - EXACT IGOR PRO DIALOG

        Returns:
            Tuple of constraints or None if cancelled
        """
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
            "Minimum height",
            "Maximum height",
            "Minimum area",
            "Maximum area",
            "Minimum volume",
            "Maximum volume"
        ]

        # Create constraint input fields
        for i, (key, var) in enumerate(vars_dict.items()):
            frame = tk.Frame(main_frame, bg='#f0f0f0')
            frame.pack(fill='x', pady=8)

            label = tk.Label(frame, text=labels[i], width=20, anchor='w',
                             font=('Arial', 11), bg='#f0f0f0')
            label.pack(side='left')

            entry = tk.Entry(frame, textvariable=var, width=15, font=('Arial', 11))
            entry.pack(side='right', padx=(10, 0))

        result = [None]
        error_label = tk.Label(main_frame, text="", fg='red', bg='#f0f0f0')
        error_label.pack(pady=5)

        def parse_value(val_str):
            """Parse constraint values, handling inf/-inf."""
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

    @staticmethod
    def get_preprocess_parameters():
        """
        Get preprocessing parameters - EXACT IGOR PRO DIALOG

        Returns:
            Tuple of parameters or None if cancelled
        """
        root = tk.Tk()
        root.title("Preprocessing Parameters")
        root.geometry("500x300")
        root.configure(bg='#f0f0f0')

        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (500 // 2)
        y = (root.winfo_screenheight() // 2) - (300 // 2)
        root.geometry(f"500x300+{x}+{y}")

        # Title
        title_label = tk.Label(root, text="Preprocessing Parameters",
                               font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=15)

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=30)

        # Default preprocessing values
        streak_var = tk.DoubleVar(value=3)  # 3 standard deviations
        flatten_var = tk.IntVar(value=2)  # 2nd order polynomial

        # Streak removal parameter
        frame1 = tk.Frame(main_frame, bg='#f0f0f0')
        frame1.pack(fill='x', pady=10)
        tk.Label(frame1, text="Std. Deviations for streak removal:",
                 width=35, anchor='w', font=('Arial', 11), bg='#f0f0f0').pack(side='left')
        tk.Entry(frame1, textvariable=streak_var, width=15, font=('Arial', 11)).pack(side='right')

        # Flattening parameter
        frame2 = tk.Frame(main_frame, bg='#f0f0f0')
        frame2.pack(fill='x', pady=10)
        tk.Label(frame2, text="Polynomial order for flattening:",
                 width=35, anchor='w', font=('Arial', 11), bg='#f0f0f0').pack(side='left')
        tk.Entry(frame2, textvariable=flatten_var, width=15, font=('Arial', 11)).pack(side='right')

        # Help text
        help_text = tk.Label(main_frame,
                             text="Note: Enter 0 to skip either preprocessing step\n" +
                                  "Streak removal: removes horizontal artifacts\n" +
                                  "Flattening: removes background slope/curvature",
                             font=('Arial', 9), fg='#7f8c8d', bg='#f0f0f0')
        help_text.pack(pady=15)

        result = [None]

        def on_ok():
            try:
                result[0] = (streak_var.get(), flatten_var.get())
                root.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

        def on_cancel():
            root.destroy()

        # Button frame
        button_frame = tk.Frame(root, bg='#f0f0f0')
        button_frame.pack(pady=20)

        tk.Button(button_frame, text="Continue", command=on_ok,
                  bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=on_cancel,
                  bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                  width=12, height=2).pack(side='left', padx=5)

        root.mainloop()
        return result[0]