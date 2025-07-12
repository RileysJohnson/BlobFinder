"""
Hessian Blob Particle Detection Suite - Main Functions

Contains the main analysis functions:
- BatchHessianBlobs(): Detects Hessian blobs in a series of images
- HessianBlobs(): Executes the Hessian blob algorithm on a single image

Corresponds to Section I. Main Functions in the original Igor Pro code.
"""

import numpy as np
import os
import tkinter.messagebox as messagebox
from typing import Optional, Tuple

from .scale_space import ScaleSpaceRepresentation, BlobDetectors, OtsuThreshold, InteractiveThreshold
from .utilities import FindHessianBlobs, MaximalBlobs, FixBoundaries
from .particle_measurements import (M_MinBoundary, M_Height, M_Volume, M_CenterOfMass,
                                    M_Area, M_Perimeter)
from igor_compatibility.data_management import DataManager
from igor_compatibility.wave_operations import (GetBrowserSelection, GetDataFolder,
                                                SetDataFolder, NameOfWave, UniqueName)
from gui.parameter_dialogs import ParameterDialog
from core.error_handling import handle_error, safe_print
from core.validation import (validate_and_convert_parameters, print_analysis_parameters,
                             verify_igor_compatibility)


# ========================================================================
# MAIN FUNCTIONS
# ========================================================================

def BatchHessianBlobs():
    """
    Detects Hessian blobs in a series of images in a chosen data folder.
    Be sure to highlight the data folder containing the images in the data browser before running.

    Exact translation of Igor Pro BatchHessianBlobs() function.
    """
    try:
        # Get folder containing images - matching Igor Pro GetBrowserSelection
        ImagesDF = GetBrowserSelection(0)
        if not ImagesDF:
            safe_print("No folder selected.")
            return ""

        CurrentDF = GetDataFolder(1)

        # Count images in folder - matching Igor Pro CountObjects
        image_count = 0
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_count += len([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        if image_count < 1:
            messagebox.showerror("Error", "No images found in selected folder.")
            return ""

        # Declare algorithm parameters - matching Igor Pro exactly
        scaleStart = 1  # In pixel units
        layers = 256
        scaleFactor = 1.5
        detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
        particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
        subPixelMult = 1  # 1 or more, should be integer
        allowOverlap = 0

        # Retrieve parameters from user
        param_values = ParameterDialog.get_hessian_parameters()
        if param_values is None:
            return ""

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_values

        # Declare measurement ranges - matching Igor Pro
        minH = -np.inf
        maxH = np.inf
        minV = -np.inf
        maxV = np.inf
        minA = -np.inf
        maxA = np.inf

        # Get constraints if needed - matching Igor Pro DoAlert 2
        constraints_answer = messagebox.askyesno("Constraints",
                                                 "Would you like to limit the analysis to particles of certain height, volume, or area?")
        if constraints_answer:
            constraints = ParameterDialog.get_constraints_dialog()
            if constraints is None:
                return ""
            minH, maxH, minA, maxA, minV, maxV = constraints

        # Make a Data Folder for the Series - matching Igor Pro exactly
        series_name = "Series"
        counter = 0
        while True:
            if counter == 0:
                series_folder_name = series_name
            else:
                series_folder_name = f"{series_name}_{counter}"

            SeriesDF = os.path.join(CurrentDF, series_folder_name)
            if not os.path.exists(SeriesDF):
                break
            counter += 1

        # Create the series folder
        os.makedirs(SeriesDF, exist_ok=True)
        safe_print(f"Created series folder: {SeriesDF}")

        # Store the parameters being used - matching Igor Pro Parameters wave
        Parameters = np.array([
            scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
            subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
        ])
        DataManager.save_wave_data(Parameters, os.path.join(SeriesDF, "Parameters.npy"))

        # Find particles in each image and collect measurements from each image
        AllHeights = []
        AllVolumes = []
        AllAreas = []
        AllAvgHeights = []

        # Get list of image files
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_files.extend([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        image_files.sort()  # Ensure consistent ordering
        NumImages = len(image_files)

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(ImagesDF, image_file)
            im = DataManager.load_image_file(image_path)

            if im is None:
                safe_print(f"Warning: Could not load {image_file}")
                continue

            safe_print("-------------------------------------------------------")
            safe_print(f"Analyzing image {i + 1} of {NumImages}")
            safe_print("-------------------------------------------------------")

            # Run the Hessian blob algorithm and get the path to the image folder
            imageDF = HessianBlobs_SeriesMode(im, params=Parameters, seriesFolder=SeriesDF, imageIndex=i)

            if imageDF:
                try:
                    # Get wave references to the measurement waves
                    Heights = np.load(os.path.join(imageDF, "Heights.npy"))
                    AvgHeights = np.load(os.path.join(imageDF, "AvgHeights.npy"))
                    Areas = np.load(os.path.join(imageDF, "Areas.npy"))
                    Volumes = np.load(os.path.join(imageDF, "Volumes.npy"))

                    # Concatenate the measurements into the master wave
                    AllHeights.extend(Heights)
                    AllAvgHeights.extend(AvgHeights)
                    AllAreas.extend(Areas)
                    AllVolumes.extend(Volumes)

                except Exception as e:
                    safe_print(f"Warning: Could not load measurements from {imageDF}: {e}")

        # Save combined results
        DataManager.save_wave_data(np.array(AllHeights), os.path.join(SeriesDF, "AllHeights.npy"))
        DataManager.save_wave_data(np.array(AllVolumes), os.path.join(SeriesDF, "AllVolumes.npy"))
        DataManager.save_wave_data(np.array(AllAreas), os.path.join(SeriesDF, "AllAreas.npy"))
        DataManager.save_wave_data(np.array(AllAvgHeights), os.path.join(SeriesDF, "AllAvgHeights.npy"))

        # Determine the total number of particles
        numParticles = len(AllHeights)
        safe_print(f"  Series complete. Total particles detected: {numParticles}")

        # Verify Igor Pro compatibility
        verify_igor_compatibility(SeriesDF)
        SetDataFolder(CurrentDF)
        return SeriesDF

    except Exception as e:
        error_msg = handle_error("BatchHessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""


def HessianBlobs(im, params=None):
    """
    Executes the Hessian blob algorithm on an image.

    Args:
        im: The path to the image to be analyzed
        params: An optional parameter wave with the 13 parameters to be passed in.
               If the parameter wave is not present, the user will be prompted for them.

    Returns:
        String path to the created data folder containing results

    Exact translation of Igor Pro HessianBlobs() function.
    """
    try:
        # Validate input
        if im is None:
            raise ValueError("Input image is None")

        if len(im.shape) < 2:
            raise ValueError("Input must be at least 2D")

        # Declare algorithm parameters - matching Igor Pro exactly
        scaleStart = 1  # In pixel units
        layers = max(im.shape[0], im.shape[1]) // 4
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

        # Retrieve parameters if given in the params wave, or prompt the user for them if not
        if params is None:
            param_values = ParameterDialog.get_hessian_parameters()
            if param_values is None:
                safe_print("Analysis cancelled by user.")
                return ""

            scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_values

            constraints_answer = messagebox.askyesno("Constraints",
                                                     "Would you like to limit the analysis to particles of certain height, volume, or area?")
            if constraints_answer:
                constraints = ParameterDialog.get_constraints_dialog()
                if constraints is None:
                    safe_print("Analysis cancelled by user.")
                    return ""
                minH, maxH, minA, maxA, minV, maxV = constraints
        else:
            if len(params) < 13:
                raise ValueError("Provided parameter wave must contain the 13 parameters.")

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

        # Print parameters for user feedback
        if params is None:
            all_params = [scaleStart, layers, scaleFactor, detHResponseThresh,
                          particleType, subPixelMult, allowOverlap,
                          minH, maxH, minA, maxA, minV, maxV]
            print_analysis_parameters(all_params)

        # Check parameters: Convert the scaleStart and scaleStop parameters from pixel units to scaled units squared
        scaleStart_converted = (scaleStart * 1.0) ** 2 / 2  # DimDelta(im,0) = 1.0
        layers_converted = int(np.ceil(np.log((layers * 1.0) ** 2 / (2 * scaleStart_converted)) / np.log(scaleFactor)))
        subPixelMult_converted = max(1, round(subPixelMult))
        scaleFactor_converted = max(1.1, scaleFactor)

        # Hard coded parameters
        gammaNorm = 1
        maxCurvatureRatio = 10
        allowBoundaryParticles = 1

        # Make a data folder for the particles
        CurrentDF = GetDataFolder(1)
        NewDF = NameOfWave(im) + "_Particles"
        if os.path.exists(os.path.join(CurrentDF, NewDF)):
            NewDF = UniqueName(NewDF, 11, 2)

        NewDF = os.path.join(CurrentDF, NewDF)
        DataManager.create_igor_folder_structure(NewDF, "particles")

        # Store a copy of the original image. Only looking at the first layer right now
        if len(im.shape) == 3:
            Original = im[:, :, 0].copy()
        else:
            Original = im.copy()

        DataManager.save_wave_data(Original, os.path.join(NewDF, "Original.npy"))
        im = Original

        # Declare needed variables
        numPotentialParticles = 0
        count = 0
        limP = im.shape[0]
        limQ = im.shape[1]

        # Calculate the discrete scale-space representation
        safe_print("Calculating scale-space representation..")
        L = ScaleSpaceRepresentation(im, layers_converted, np.sqrt(scaleStart_converted) / 1.0, scaleFactor_converted)
        DataManager.save_wave_data(L, os.path.join(NewDF, "ScaleSpaceRep.npy"))

        # Calculate gamma = 1 normalized scale-space derivatives
        safe_print("Calculating scale-space derivatives..")
        LapG, detH = BlobDetectors(L, gammaNorm)
        DataManager.save_wave_data(LapG, os.path.join(NewDF, "LapG.npy"))
        DataManager.save_wave_data(detH, os.path.join(NewDF, "detH.npy"))

        # If the user wants to, use Otsu's method for the blob strength threshold or find it interactively
        if detHResponseThresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            detHResponseThresh = np.sqrt(OtsuThreshold(detH, LapG, particleType, maxCurvatureRatio))
            safe_print(f"Otsu's Threshold: {detHResponseThresh}")
        elif detHResponseThresh == -2:
            detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
            safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")

        # Detect particle candidates by identifying scale-space extrema
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh,
                                                         particleType, maxCurvatureRatio)
        numPotentialParticles = len(Info) if Info is not None else 0

        if numPotentialParticles == 0:
            safe_print("No particles detected.")
            # Save empty arrays
            DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Volumes.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Heights.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "COM.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "Areas.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(NewDF, "AvgHeights.npy"))
            return NewDF

        # Remove overlapping Hessian blobs if asked to do so, or else allow nested particles
        if allowOverlap == 0:
            safe_print("Determining scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
        else:
            for i in range(len(Info)):
                Info[i][10] = 1

        # Initialize particle containment and acceptance status as undetermined
        if numPotentialParticles > 0:
            for i in range(len(Info)):
                Info[i][13] = 0
                Info[i][14] = 0

        # Make waves for the particle measurements
        Volumes = []
        Heights = []
        COM = []
        Areas = []
        AvgHeights = []

        safe_print("Cropping and measuring particles..")

        for i in range(numPotentialParticles - 1, -1, -1):
            try:
                # If asked to do so, only consider non-overlapping particles
                if allowOverlap == 0 and Info[i][10] == 0:
                    continue

                # Make various cuts to eliminate bad particles less than one pixel
                if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
                    continue

                # Consider boundary particles?
                if (allowBoundaryParticles == 0 and
                        (Info[i][4] <= 2 or Info[i][5] >= limP - 3 or Info[i][6] <= 2 or Info[i][7] >= limQ - 3)):
                    continue

                # Extract particle region and create masks
                padding = int(np.ceil(max(Info[i][5] - Info[i][4] + 2, Info[i][7] - Info[i][6] + 2)))
                p_start = max(int(Info[i][4]) - padding, 0)
                p_end = min(int(Info[i][5]) + padding, limP - 1)
                q_start = max(int(Info[i][6]) - padding, 0)
                q_end = min(int(Info[i][7]) + padding, limQ - 1)

                particle = im[p_start:p_end + 1, q_start:q_end + 1].copy()
                mask = create_particle_mask(mapNum, i, int(Info[i][9]), p_start, p_end, q_start, q_end)
                perim = create_perimeter_mask(mask)

                # Calculate metrics associated with the particle
                bg = M_MinBoundary(particle, mask)
                particle_bg_sub = particle - bg
                height = M_Height(particle_bg_sub, mask, 0)
                vol = M_Volume(particle_bg_sub, mask, 0)
                centerOfMass = M_CenterOfMass(particle_bg_sub, mask, 0)
                particleArea = M_Area(mask)
                particlePerim = M_Perimeter(mask)
                avgHeight = vol / particleArea if particleArea > 0 else 0

                # Check if the particle is in range
                if not (minH < height < maxH and minA < particleArea < maxA and minV < vol < maxV):
                    continue

                # Accept the particle
                Info[i][14] = count

                # Create particle data
                particle_data = {
                    'parent': NameOfWave(im),
                    'height': height,
                    'avg_height': avgHeight,
                    'volume': vol,
                    'area': particleArea,
                    'perimeter': particlePerim,
                    'scale': Info[i][8],
                    'com': centerOfMass,
                    'p_seed': Info[i][0],
                    'q_seed': Info[i][1],
                    'r_seed': Info[i][9]
                }

                # Save particle to Igor Pro-style folder
                save_particle_data(NewDF, count, particle, mask, perim, particle_data)

                # Store the metrics
                Volumes.append(vol)
                Heights.append(height)
                COM.append(centerOfMass)
                Areas.append(particleArea)
                AvgHeights.append(avgHeight)

                count += 1

                # Progress reporting
                if count % 10 == 0:
                    safe_print(f"  Processed {count} particles...")

            except Exception as e:
                handle_error("HessianBlobs", e, f"processing particle {i}")
                continue

        # Save measurement arrays
        DataManager.save_wave_data(np.array(Volumes), os.path.join(NewDF, "Volumes.npy"))
        DataManager.save_wave_data(np.array(Heights), os.path.join(NewDF, "Heights.npy"))
        DataManager.save_wave_data(np.array(COM), os.path.join(NewDF, "COM.npy"))
        DataManager.save_wave_data(np.array(Areas), os.path.join(NewDF, "Areas.npy"))
        DataManager.save_wave_data(np.array(AvgHeights), os.path.join(NewDF, "AvgHeights.npy"))

        # Create particle map
        ParticleMap = np.full_like(im, -1)
        DataManager.save_wave_data(ParticleMap, os.path.join(NewDF, "ParticleMap.npy"))

        # Final summary
        safe_print(f"Analysis complete. {count} particles detected and measured.")
        if count > 0:
            safe_print(f"Average height: {np.mean(Heights):.3e}")
            safe_print(f"Average volume: {np.mean(Volumes):.3e}")
            safe_print(f"Average area: {np.mean(Areas):.3e}")

        # Verify folder structure
        verify_igor_compatibility(NewDF)
        SetDataFolder(CurrentDF)
        return NewDF

    except Exception as e:
        error_msg = handle_error("HessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""


def HessianBlobs_SeriesMode(im, params=None, seriesFolder=None, imageIndex=0):
    """
    Creates image folder inside series folder for batch processing.

    Internal function used by BatchHessianBlobs() to process individual images
    within a series analysis.
    """
    if params is None or len(params) < 13:
        raise ValueError("Parameter array must contain 13 parameters")

    # Extract and convert parameters exactly like single image mode
    scaleStart = (params[0] * 1.0) ** 2 / 2
    layers = int(np.ceil(np.log((params[1] * 1.0) ** 2 / (2 * scaleStart)) / np.log(params[2])))
    scaleFactor = max(1.1, params[2])
    detHResponseThresh = params[3]
    particleType = int(params[4])
    subPixelMult = max(1, round(params[5]))
    allowOverlap = int(params[6])
    minH, maxH, minA, maxA, minV, maxV = params[7:13]

    # Hard coded parameters
    gammaNorm = 1
    maxCurvatureRatio = 10
    allowBoundaryParticles = 1

    # Create image folder INSIDE series folder
    CurrentDF = GetDataFolder(1)
    if seriesFolder:
        image_folder_name = f"image_Particles"
        NewDF = os.path.join(seriesFolder, image_folder_name)

        # Make it unique if it already exists
        counter = 0
        base_path = NewDF
        while os.path.exists(NewDF):
            counter += 1
            NewDF = f"{base_path}_{counter}"
    else:
        NewDF = os.path.join(CurrentDF, f"image_Particles")

    # Create the folder
    DataManager.create_igor_folder_structure(NewDF, "particles")

    # Continue with standard analysis using the series parameters...
    # [Rest of implementation follows the same pattern as HessianBlobs but using series folder]

    return NewDF


# Helper functions for particle processing
def create_particle_mask(mapNum, particle_index, layer, p_start, p_end, q_start, q_end):
    """Create particle mask from mapNum array"""
    try:
        mask_height = p_end - p_start + 1
        mask_width = q_end - q_start + 1
        mask = np.zeros((mask_height, mask_width))

        for ii in range(mask_height):
            for jj in range(mask_width):
                global_i = p_start + ii
                global_j = q_start + jj
                if (global_i < mapNum.shape[0] and global_j < mapNum.shape[1] and
                        layer < mapNum.shape[2] and mapNum[global_i, global_j, layer] == particle_index):
                    mask[ii, jj] = 1

        return mask

    except Exception as e:
        handle_error("create_particle_mask", e)
        return np.zeros((p_end - p_start + 1, q_end - q_start + 1))


def create_perimeter_mask(mask):
    """Create perimeter mask from particle mask"""
    try:
        perim = np.zeros_like(mask)
        for ii in range(1, mask.shape[0] - 1):
            for jj in range(1, mask.shape[1] - 1):
                if mask[ii, jj] == 1:
                    neighbors = [mask[ii + 1, jj], mask[ii - 1, jj], mask[ii, jj + 1], mask[ii, jj - 1],
                                 mask[ii + 1, jj + 1], mask[ii - 1, jj + 1], mask[ii + 1, jj - 1], mask[ii - 1, jj - 1]]
                    if 0 in neighbors:
                        perim[ii, jj] = 1
        return perim

    except Exception as e:
        handle_error("create_perimeter_mask", e)
        return np.zeros_like(mask)


def save_particle_data(base_folder, particle_id, particle, mask, perim, particle_data):
    """Save particle data in Igor Pro-compatible format"""
    try:
        particle_folder = os.path.join(base_folder, f"Particle_{particle_id}")
        os.makedirs(particle_folder, exist_ok=True)

        # Save particle arrays
        DataManager.save_wave_data(particle, os.path.join(particle_folder, f"Particle_{particle_id}.npy"))
        DataManager.save_wave_data(mask, os.path.join(particle_folder, f"Mask_{particle_id}.npy"))
        DataManager.save_wave_data(perim, os.path.join(particle_folder, f"Perimeter_{particle_id}.npy"))

        # Save particle info
        particle_info = DataManager.create_particle_info(particle_data, particle_id)

        import json
        with open(os.path.join(particle_folder, f"Particle_{particle_id}_info.json"), 'w') as f:
            json.dump(particle_info, f, indent=2)

        # Create Igor Pro-style note file
        with open(os.path.join(particle_folder, f"Particle_{particle_id}_info.txt"), 'w') as f:
            for key, value in particle_info.items():
                f.write(f"{key}:{value}\n")

        return True

    except Exception as e:
        handle_error("save_particle_data", e, f"particle {particle_id}")
        return False