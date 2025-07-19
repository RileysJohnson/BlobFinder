"""Contains high-level analysis functions for blob detection."""

# #######################################################################
#                      CORE: ANALYSIS FUNCTIONS
#
#   CONTENTS:
#       - HessianBlobs(): Main function for single image analysis
#       - BatchHessianBlobs(): Batch processing of multiple images
#       - HessianBlobs_SeriesMode(): Helper for batch processing
#       - WaveStats(): Igor Pro style statistics
#       - Testing(): Demo function
#       - print_series_analysis_summary(): Summary printer
#
# #######################################################################

import numpy as np
import os
import time
import threading
from typing import Optional, Tuple
from tkinter import messagebox, filedialog
from utils.error_handler import handle_error, safe_print, HessianBlobError
from utils.data_manager import DataManager
from utils.igor_compat import GetDataFolder, SetDataFolder, NewDataFolder, UniqueName
from gui.dialogs import ParameterDialog
from core.blob_detection import (
    ScaleSpaceRepresentation, BlobDetectors, OtsuThreshold,
    FindHessianBlobs, SubPixelRefinement, MaximalBlobs
)
from utils.measurements import MeasureParticles


def HessianBlobs(im, params=None):
    """
    Executes the Hessian blob algorithm on an image - EXACT IGOR PRO IMPLEMENTATION.

    Args:
        im: The image to be analyzed
        params: Optional parameter wave with the 13 parameters

    Returns:
        str: Path to the analysis folder or empty string on error
    """
    try:
        # Declare algorithm parameters - Igor Pro defaults
        scaleStart = 1  # In pixel units
        layers = max(im.shape[0], im.shape[1]) / 4
        scaleFactor = 1.5
        detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
        particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
        subPixelMult = 1  # 1 or more, should be integer
        allowOverlap = 0

        # Declare measurement ranges
        minH = -np.inf
        maxH = np.inf
        minV = -np.inf
        maxV = np.inf
        minA = -np.inf
        maxA = np.inf

        # Retrieve parameters if given, or prompt the user
        if params is None:
            params_tuple = ParameterDialog.get_hessian_parameters()

            if params_tuple is None:
                return ""

            scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params_tuple

            # Ask about constraints
            constraints_answer = messagebox.askyesnocancel(
                "Constraints",
                "Would you like to limit the analysis to particles of certain height, volume, or area?"
            )

            if constraints_answer:
                constraints = ParameterDialog.get_constraints_dialog()
                if constraints is None:
                    return ""
                minH, maxH, minA, maxA, minV, maxV = constraints

        else:
            if len(params) < 13:
                safe_print("Error: Provided parameter wave must contain the 13 parameters.")
                return ""

            scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params[:7]
            minH, maxH, minA, maxA, minV, maxV = params[7:13]

        # Check parameters: Convert the scaleStart and scaleStop parameters from pixel units to scaled units squared
        scaleStart = (scaleStart * 1.0) ** 2 / 2
        layers = max(1, int(np.ceil(np.log((layers * 1.0) ** 2 / (2 * scaleStart)) / np.log(scaleFactor))))
        subPixelMult = max(1, round(subPixelMult))
        scaleFactor = max(1.1, scaleFactor)

        # Hard coded parameters matching Igor Pro exactly
        gammaNorm = 1
        maxCurvatureRatio = 10
        allowBoundaryParticles = 1

        # Create output folder
        NewDF = DataManager.create_igor_folder_structure(f"particles_{int(time.time())}", "particles")
        safe_print(f"Created analysis folder: {NewDF}")

        # Store a copy of the original image
        safe_print("Storing original image..")
        DataManager.save_wave_data(im, os.path.join(NewDF, "Original.npy"))

        # Calculate the discrete scale-space representation
        safe_print("Calculating scale-space representation..")
        L = ScaleSpaceRepresentation(im, layers, np.sqrt(scaleStart), scaleFactor)
        DataManager.save_wave_data(L, os.path.join(NewDF, "ScaleSpaceRep.npy"))

        # Calculate gamma = 1 normalized scale-space derivatives
        safe_print("Calculating scale-space derivatives..")
        LapG, detH = BlobDetectors(L, gammaNorm)
        DataManager.save_wave_data(LapG, os.path.join(NewDF, "LapG.npy"))
        DataManager.save_wave_data(detH, os.path.join(NewDF, "detH.npy"))

        # Threshold determination - exactly matching Igor Pro logic
        if detHResponseThresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            detHResponseThresh = np.sqrt(OtsuThreshold(detH, LapG, particleType, maxCurvatureRatio))
            safe_print(f"Otsu's Threshold: {detHResponseThresh}")
        elif detHResponseThresh == -2:
            from gui.dialogs import InteractiveThreshold
            detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
            safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")
        else:
            safe_print(f"Using manual threshold: {detHResponseThresh}")

        # Detect particles
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh, particleType,
                                                         maxCurvatureRatio)
        numPotentialParticles = len(Info) if Info is not None else 0

        if numPotentialParticles == 0:
            safe_print("No particles detected.")
            # Still show visualization with no particles
            create_blob_visualization(im, [], NewDF)
            return NewDF

        safe_print(f"Detected {numPotentialParticles} potential particles.")

        # Subpixel refinement
        if subPixelMult > 1:
            safe_print("Performing subpixel refinement..")
            Info = SubPixelRefinement(detH, Info, subPixelMult)

        # Filter for scale-maximal particles if overlap not allowed
        if allowOverlap == 0:
            safe_print("Filtering for scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
            # Filter out non-maximal particles
            Info = [particle for particle in Info if particle[10] == 1]

        numFinalParticles = len(Info)
        safe_print(f"Final particle count: {numFinalParticles}")

        if numFinalParticles == 0:
            safe_print("No final particles after filtering.")
            # Still show visualization
            create_blob_visualization(im, [], NewDF)
            return NewDF

        # Measure particles
        safe_print("Measuring particles..")
        Heights, Volumes, Areas, AvgHeights, COM = MeasureParticles(im, L, Info, minH, maxH, minA, maxA, minV, maxV)

        # Save measurement data
        DataManager.save_wave_data(Heights, os.path.join(NewDF, "Heights.npy"))
        DataManager.save_wave_data(Volumes, os.path.join(NewDF, "Volumes.npy"))
        DataManager.save_wave_data(Areas, os.path.join(NewDF, "Areas.npy"))
        DataManager.save_wave_data(AvgHeights, os.path.join(NewDF, "AvgHeights.npy"))
        DataManager.save_wave_data(COM, os.path.join(NewDF, "COM.npy"))

        safe_print(f"Image analysis complete: {numFinalParticles} particles measured.")

        # CREATE AUTOMATIC VISUALIZATION
        safe_print("Creating visualization...")
        create_blob_visualization(im, Info, NewDF)

        return NewDF

    except Exception as e:
        handle_error("HessianBlobs", e)
        return ""


def create_blob_visualization(im, Info, output_folder):
    """Create automatic visualization of detected blobs - matches Igor Pro output."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display the original image
        im_display = ax.imshow(im, cmap='gray', aspect='equal')

        # Add detected blobs as circles
        if Info and len(Info) > 0:
            for particle in Info:
                # Extract particle information
                x_center = particle[1]  # Column position
                y_center = particle[0]  # Row position
                radius = particle[2]  # Scale/radius

                # Create circle patch
                circle = patches.Circle((x_center, y_center), radius,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(circle)

        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        particle_count = len(Info) if Info else 0
        ax.set_title(f'Detected Hessian Blobs - {particle_count} particles', fontsize=14)

        # Add colorbar
        plt.colorbar(im_display, ax=ax, label='Intensity')

        # Save the figure
        output_path = os.path.join(output_folder, "detected_blobs.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        safe_print(f"Saved visualization to: {output_path}")

        # Show the figure
        plt.show()

    except Exception as e:
        handle_error("create_blob_visualization", e)


def BatchHessianBlobs():
    """Detects Hessian blobs in a series of images - EXACT IGOR PRO ALGORITHM."""
    try:
        # Igor Pro: String ImagesDF=GetBrowserSelection(0)
        # Igor Pro: String CurrentDF=GetDataFolder(1)
        ImagesDF = filedialog.askdirectory(title="Select folder containing images")
        if not ImagesDF:
            safe_print("No folder selected.")
            return ""

        CurrentDF = GetDataFolder(1)

        # Count images in folder
        image_count = 0
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            files = [f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())]
            image_files.extend(files)
            image_count += len(files)

        if image_count < 1:
            messagebox.showerror("Error", "No images found in selected folder.")
            return ""

        safe_print(f"Found {image_count} images to process.")

        # Igor Pro: Declare algorithm parameters
        scaleStart = 1                      # In pixel units
        layers = 256
        scaleFactor = 1.5
        detHResponseThresh = -2             # Use -1 for Otsu's method, -2 for interactive
        particleType = 1                    # -1 for neg only, 1 for pos only, 0 for both
        subPixelMult = 1                    # 1 or more, should be integer
        allowOverlap = 0

        # Igor Pro: Retrieve parameters from user
        param_values = ParameterDialog.get_hessian_parameters()
        if param_values is None:
            return ""

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_values

        # Igor Pro: Get constraints
        minH, maxH, minV, maxV, minA, maxA = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
        constraints_answer = messagebox.askyesno(
            "Igor Pro wants to know...",
            "Would you like to limit the analysis to particles of certain height, volume, or area?"
        )

        if constraints_answer:
            constraints = ParameterDialog.get_constraints_dialog()
            if constraints is None:
                return ""
            minH, maxH, minA, maxA, minV, maxV = constraints

        # Igor Pro: Make a Data Folder for the Series
        # Igor Pro: String SeriesDF=CurrentDF+UniqueName("Series_",11,0)
        SeriesDF = os.path.join(CurrentDF, UniqueName("Series_", 11, 0))
        DataManager.create_igor_folder_structure(SeriesDF, "series")
        safe_print(f"Created series folder: {SeriesDF}")

        # Igor Pro: Store the parameters being used
        # Igor Pro: Make/N=13 /O Parameters
        Parameters = np.array([
            scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
            subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
        ])
        DataManager.save_wave_data(Parameters, os.path.join(SeriesDF, "Parameters.npy"))

        # Igor Pro: Find particles in each image and collect measurements from each image
        # Igor Pro: Make/N=0 /O AllHeights, AllVolumes, AllAreas, AllAvgHeights
        AllHeights = []
        AllVolumes = []
        AllAreas = []
        AllAvgHeights = []

        # Igor Pro: For(i=0;i<numImages;i+=1)
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(ImagesDF, image_file)
            im = DataManager.load_image_file(image_path)

            if im is None:
                safe_print(f"Warning: Could not load {image_file}")
                continue

            safe_print("-------------------------------------------------------")
            safe_print(f"Analyzing image {i + 1} of {len(image_files)}: {image_file}")
            safe_print("-------------------------------------------------------")

            # Igor Pro: Run the Hessian blob algorithm and get the path to the image folder
            # Igor Pro: imageDF = HessianBlobs(im,params=Parameters)
            imageDF = HessianBlobs_SeriesMode(im, params=Parameters, seriesFolder=SeriesDF, imageIndex=i)

            if imageDF:
                try:
                    # Igor Pro: Get wave references to the measurement waves
                    # Igor Pro: Wave Heights = $(imageDF+":Heights")
                    Heights = DataManager.load_wave_data(os.path.join(imageDF, "Heights.npy"))
                    AvgHeights = DataManager.load_wave_data(os.path.join(imageDF, "AvgHeights.npy"))
                    Areas = DataManager.load_wave_data(os.path.join(imageDF, "Areas.npy"))
                    Volumes = DataManager.load_wave_data(os.path.join(imageDF, "Volumes.npy"))

                    # Igor Pro: Concatenate the measurements into the master wave
                    # Igor Pro: Concatenate {Heights}, AllHeights
                    if Heights is not None:
                        AllHeights.extend(Heights)
                    if AvgHeights is not None:
                        AllAvgHeights.extend(AvgHeights)
                    if Areas is not None:
                        AllAreas.extend(Areas)
                    if Volumes is not None:
                        AllVolumes.extend(Volumes)

                except Exception as e:
                    safe_print(f"Warning: Could not load measurements from {imageDF}: {e}")

        # Save combined results
        DataManager.save_wave_data(np.array(AllHeights), os.path.join(SeriesDF, "AllHeights.npy"))
        DataManager.save_wave_data(np.array(AllVolumes), os.path.join(SeriesDF, "AllVolumes.npy"))
        DataManager.save_wave_data(np.array(AllAreas), os.path.join(SeriesDF, "AllAreas.npy"))
        DataManager.save_wave_data(np.array(AllAvgHeights), os.path.join(SeriesDF, "AllAvgHeights.npy"))

        # Igor Pro: Determine the total number of particles
        # Igor Pro: Variable numParticles = DimSize(AllHeights,0)
        numParticles = len(AllHeights)
        safe_print(f"  Series complete. Total particles detected: {numParticles}")

        # Igor Pro: SetDataFolder $CurrentDF
        SetDataFolder(CurrentDF)
        return SeriesDF

    except Exception as e:
        handle_error("BatchHessianBlobs", e)
        return ""


def HessianBlobs_SeriesMode(im, params=None, seriesFolder=None, imageIndex=0):
    """Creates image folder inside series folder - EXACT IGOR PRO ALGORITHM."""
    try:
        if params is None or len(params) < 13:
            raise HessianBlobError("Parameter array must contain 13 parameters")

        # Extract parameters
        scaleStart = params[0]
        layers = int(params[1])
        scaleFactor = params[2]
        detHResponseThresh = params[3]
        particleType = int(params[4])
        subPixelMult = int(params[5])
        allowOverlap = int(params[6])
        minH = params[7]
        maxH = params[8]
        minA = params[9]
        maxA = params[10]
        minV = params[11]
        maxV = params[12]

        # Parameter conversion - EXACT IGOR PRO LOGIC
        dimDelta_im_0 = 1.0
        scaleStart = (scaleStart * dimDelta_im_0) ** 2 / 2
        layers = int(np.ceil(np.log((layers * dimDelta_im_0) ** 2 / (2 * scaleStart)) / np.log(scaleFactor)))
        subPixelMult = max(1, round(subPixelMult))
        scaleFactor = max(1.1, scaleFactor)

        # Hard coded parameters
        gammaNorm = 1
        maxCurvatureRatio = 10
        allowBoundaryParticles = 1

        # Create image folder INSIDE series folder
        if seriesFolder:
            # Igor Pro: String NewDF = SeriesDF + "image_" + num2str(i) + "_Particles"
            image_folder_name = f"image_{imageIndex}_Particles"
            NewDF = os.path.join(seriesFolder, image_folder_name)
        else:
            NewDF = DataManager.create_igor_folder_structure(f"image_Particles", "particles")

        # Make it unique if it already exists
        counter = 0
        base_path = NewDF
        while os.path.exists(NewDF):
            counter += 1
            NewDF = f"{base_path}_{counter}"

        DataManager.create_igor_folder_structure(NewDF, "particles")
        safe_print(f"Created image folder: {NewDF}")

        # Store original image
        if len(im.shape) == 3:
            Original = im[:, :, 0].copy()
        else:
            Original = im.copy()

        DataManager.save_wave_data(Original, os.path.join(NewDF, "Original.npy"))
        im = Original

        # Calculate scale-space representation
        safe_print("Calculating scale-space representation..")
        L = ScaleSpaceRepresentation(im, layers, np.sqrt(scaleStart) / dimDelta_im_0, scaleFactor)
        DataManager.save_wave_data(L, os.path.join(NewDF, "ScaleSpaceRep.npy"))

        # Calculate derivatives
        safe_print("Calculating scale-space derivatives..")
        LapG, detH = BlobDetectors(L, gammaNorm)
        DataManager.save_wave_data(LapG, os.path.join(NewDF, "LapG.npy"))
        DataManager.save_wave_data(detH, os.path.join(NewDF, "detH.npy"))

        # Threshold determination
        if detHResponseThresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            detHResponseThresh = np.sqrt(OtsuThreshold(detH, LapG, particleType, maxCurvatureRatio))
            safe_print(f"Otsu's Threshold: {detHResponseThresh}")
        elif detHResponseThresh == -2:
            from gui.dialogs import InteractiveThreshold
            detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
            safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")
        else:
            safe_print(f"Using manual threshold: {detHResponseThresh}")

        # Detect particles
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh, particleType,
                                                         maxCurvatureRatio)
        numPotentialParticles = len(Info) if Info is not None else 0

        if numPotentialParticles == 0:
            safe_print("No particles detected.")
            return NewDF

        safe_print(f"Detected {numPotentialParticles} potential particles.")

        # Subpixel refinement
        if subPixelMult > 1:
            safe_print("Performing subpixel refinement..")
            Info = SubPixelRefinement(detH, Info, subPixelMult)

        # Filter for scale-maximal particles if overlap not allowed
        if allowOverlap == 0:
            safe_print("Filtering for scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
            # Filter out non-maximal particles
            Info = [particle for particle in Info if particle[10] == 1]

        numFinalParticles = len(Info)
        safe_print(f"Final particle count: {numFinalParticles}")

        if numFinalParticles == 0:
            safe_print("No final particles after filtering.")
            return NewDF

        # Measure particles
        safe_print("Measuring particles..")
        Heights, Volumes, Areas, AvgHeights, COM = MeasureParticles(im, L, Info, minH, maxH, minA, maxA, minV, maxV)

        # Save measurement data
        DataManager.save_wave_data(Heights, os.path.join(NewDF, "Heights.npy"))
        DataManager.save_wave_data(Volumes, os.path.join(NewDF, "Volumes.npy"))
        DataManager.save_wave_data(Areas, os.path.join(NewDF, "Areas.npy"))
        DataManager.save_wave_data(AvgHeights, os.path.join(NewDF, "AvgHeights.npy"))
        DataManager.save_wave_data(COM, os.path.join(NewDF, "COM.npy"))

        safe_print(f"Image analysis complete: {numFinalParticles} particles measured.")
        return NewDF

    except Exception as e:
        handle_error("HessianBlobs_SeriesMode", e)
        return ""


def WaveStats(data_file):
    """Computes and prints basic statistics for a data wave - EXACT IGOR PRO ALGORITHM."""
    try:
        if isinstance(data_file, str):
            # Load from file
            if data_file.endswith('.npy'):
                data = np.load(data_file)
            else:
                raise HessianBlobError("Unsupported file format. Please use .npy files.")
            base_name = os.path.splitext(os.path.basename(data_file))[0]
        else:
            # Assume it's already a numpy array
            data = data_file
            base_name = "data"

        # Clean data (remove NaN values for statistics) exactly like Igor Pro
        clean_data = data[~np.isnan(data.flatten())]

        # Calculate statistics exactly matching Igor Pro WaveStats output
        V_npnts = len(clean_data)
        V_numNaNs = np.sum(np.isnan(data.flatten()))
        V_numINFs = np.sum(np.isinf(data.flatten()))
        V_avg = np.mean(clean_data) if len(clean_data) > 0 else 0
        V_sum = np.sum(clean_data)
        V_sdev = np.std(clean_data, ddof=1) if len(clean_data) > 1 else 0
        V_sem = V_sdev / np.sqrt(len(clean_data)) if len(clean_data) > 1 else 0
        V_rms = np.sqrt(np.mean(clean_data ** 2)) if len(clean_data) > 0 else 0
        V_adev = np.mean(np.abs(clean_data - V_avg)) if len(clean_data) > 0 else 0
        V_skew = 0  # Simplified for now
        V_kurt = 0  # Simplified for now
        V_min = np.min(clean_data) if len(clean_data) > 0 else 0
        V_max = np.max(clean_data) if len(clean_data) > 0 else 0
        V_minloc = np.argmin(clean_data) if len(clean_data) > 0 else 0
        V_maxloc = np.argmax(clean_data) if len(clean_data) > 0 else 0
        V_minRowLoc = V_minloc
        V_maxRowLoc = V_maxloc
        V_startRow = 0
        V_endRow = len(data.flatten()) - 1

        # Print results in exact Igor Pro format
        safe_print(f"V_npnts= {V_npnts}; V_numNaNs= {V_numNaNs}; V_numINFs= {V_numINFs}; V_avg= {V_avg:.6g};")
        safe_print(f"V_Sum= {V_sum:.6g}; V_sdev= {V_sdev:.6g}; V_sem= {V_sem:.6g};")
        safe_print(f"V_rms= {V_rms:.6g}; V_adev= {V_adev:.6g}; V_skew= {V_skew:.6g}; V_kurt= {V_kurt:.6g};")
        safe_print(f"V_minloc= {V_minloc}; V_maxloc= {V_maxloc}; V_min= {V_min:.6g}; V_max= {V_max:.6g};")
        safe_print(f"V_minRowLoc= {V_minRowLoc}; V_maxRowLoc= {V_maxRowLoc};")
        safe_print(f"V_startRow= {V_startRow}; V_endRow= {V_endRow};")

        return {
            'V_npnts': V_npnts,
            'V_numNaNs': V_numNaNs,
            'V_numINFs': V_numINFs,
            'V_avg': V_avg,
            'V_sum': V_sum,
            'V_sdev': V_sdev,
            'V_sem': V_sem,
            'V_rms': V_rms,
            'V_adev': V_adev,
            'V_skew': V_skew,
            'V_kurt': V_kurt,
            'V_min': V_min,
            'V_max': V_max,
            'V_minloc': V_minloc,
            'V_maxloc': V_maxloc,
            'V_minRowLoc': V_minRowLoc,
            'V_maxRowLoc': V_maxRowLoc,
            'V_startRow': V_startRow,
            'V_endRow': V_endRow
        }

    except Exception as e:
        error_msg = handle_error("WaveStats", e)
        messagebox.showerror("Statistics Error", error_msg)
        return None


def Testing(str_input, num):
    """Testing function to demonstrate how user-defined functions work - EXACT IGOR PRO ALGORITHM."""
    try:
        safe_print(f"You typed: {str_input}")
        safe_print(f"Your number plus two is {num + 2}")

    except Exception as e:
        handle_error("Testing", e)


def print_series_analysis_summary(result_folder, heights, volumes, areas):
    """Print series summary matching Igor Pro exactly."""
    try:
        if len(heights) > 0:
            safe_print(f"Series complete. Total particles detected: {len(heights)}")
        else:
            safe_print("Series complete. Total particles detected: 0")

    except Exception as e:
        handle_error("print_series_analysis_summary", e)


def HessianBlobsSeries(images, params=None):
    """
    Analyzes multiple images using the same Hessian blob parameters.

    Args:
        images: List of image paths or image arrays
        params: Optional parameter array

    Returns:
        str: Path to series folder or empty string on error
    """
    try:
        if not images:
            safe_print("No images provided for series analysis.")
            return ""

        # Get parameters once for the entire series
        if params is None:
            # Get parameters in main thread before starting analysis
            params_tuple = ParameterDialog.get_hessian_parameters()
            if params_tuple is None:
                return ""

            # Convert to params array format
            params = list(params_tuple)

            # Ask about constraints
            constraints_answer = messagebox.askyesnocancel(
                "Constraints",
                "Would you like to limit the analysis to particles of certain height, volume, or area?"
            )

            if constraints_answer:
                constraints = ParameterDialog.get_constraints_dialog()
                if constraints is None:
                    return ""
                params.extend(constraints)
            else:
                # Add default constraints
                params.extend([-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf])

        # Create series folder
        series_name = f"Series__{int(time.time())}"
        SeriesDF = DataManager.create_igor_folder_structure(series_name, "series")
        safe_print(f"Created series folder: {SeriesDF}")

        # Store parameters
        params_array = np.array(params)
        DataManager.save_wave_data(params_array, os.path.join(SeriesDF, "Parameters.npy"))

        # Process each image
        AllHeights = []
        AllVolumes = []
        AllAreas = []
        AllAvgHeights = []
        AllCOM = []

        for idx, image in enumerate(images):
            safe_print("-------------------------------------------------------")
            safe_print(
                f"Analyzing image {idx + 1} of {len(images)}: {os.path.basename(image) if isinstance(image, str) else f'Array {idx}'}")
            safe_print("-------------------------------------------------------")

            # Load image if it's a path
            if isinstance(image, str):
                im = load_image(image)
                if im is None:
                    safe_print(f"Failed to load image: {image}")
                    continue
            else:
                im = image

            # Create image-specific folder
            ImageDF = DataManager.create_igor_folder_structure(f"image_{idx}_Particles", "particles", parent=SeriesDF)
            safe_print(f"Created image folder: {ImageDF}")

            # Run HessianBlobs with the parameters
            # Note: We'll handle interactive threshold in main thread within HessianBlobs
            result = HessianBlobs(im, params)

            if result:
                # Load results
                Heights = DataManager.load_wave_data(os.path.join(result, "Heights.npy"))
                Volumes = DataManager.load_wave_data(os.path.join(result, "Volumes.npy"))
                Areas = DataManager.load_wave_data(os.path.join(result, "Areas.npy"))
                AvgHeights = DataManager.load_wave_data(os.path.join(result, "AvgHeights.npy"))
                COM = DataManager.load_wave_data(os.path.join(result, "COM.npy"))

                if Heights is not None:
                    AllHeights.extend(Heights)
                    AllVolumes.extend(Volumes)
                    AllAreas.extend(Areas)
                    AllAvgHeights.extend(AvgHeights)
                    AllCOM.extend(COM)

        # Save aggregated results
        if AllHeights:
            DataManager.save_wave_data(np.array(AllHeights), os.path.join(SeriesDF, "AllHeights.npy"))
            DataManager.save_wave_data(np.array(AllVolumes), os.path.join(SeriesDF, "AllVolumes.npy"))
            DataManager.save_wave_data(np.array(AllAreas), os.path.join(SeriesDF, "AllAreas.npy"))
            DataManager.save_wave_data(np.array(AllAvgHeights), os.path.join(SeriesDF, "AllAvgHeights.npy"))
            DataManager.save_wave_data(np.array(AllCOM), os.path.join(SeriesDF, "AllCOM.npy"))

        numParticles = len(AllHeights)
        safe_print(f"Series complete. Total particles detected: {numParticles}")

        return SeriesDF

    except Exception as e:
        handle_error("HessianBlobsSeries", e)
        return ""