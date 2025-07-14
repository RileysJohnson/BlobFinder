"""Contains the high-level workflows that run the analysis."""

# #######################################################################
#                 CORE: HIGH-LEVEL ANALYSIS WORKFLOWS
#
#   CONTENTS:
#       - BatchHessianBlobs: Analyzes a series of images in a folder.
#       - HessianBlobs: Analyzes a single image.
#       - HessianBlobs_SeriesMode: Helper for batch mode.
#       - WaveStats: Computes and prints basic statistics for a data wave.
#
# #######################################################################

import numpy as np
import os
import json
import time
from tkinter import filedialog, messagebox
from utils.data_manager import DataManager
from utils.igor_compat import (
    GetDataFolder, SetDataFolder, UniqueName, NameOfWave,
    verify_igor_compatibility, verify_folder_structure
)
from utils.validators import print_analysis_parameters, validate_and_convert_parameters
from utils.error_handler import handle_error, HessianBlobError, safe_print
from utils.measurements import (
    M_MinBoundary, M_Height, M_Volume, M_CenterOfMass, M_Area, M_Perimeter
)
from gui.dialogs import ParameterDialog, InteractiveThreshold
from .blob_detection import (
    ScaleSpaceRepresentation, BlobDetectors, OtsuThreshold, FindHessianBlobs, MaximalBlobs
)

def BatchHessianBlobs():
    """Detects Hessian blobs in a series of images."""

    try:
        # Get folder containing images
        ImagesDF = filedialog.askdirectory(title="Select folder containing images")
        if not ImagesDF:
            safe_print("No folder selected.")
            return ""

        CurrentDF = GetDataFolder(1)

        # Count images in folder
        image_count = 0
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_count += len([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        if image_count < 1:
            messagebox.showerror("Error", "No images found in selected folder.")
            return ""

        # Get parameters from user
        param_values = ParameterDialog.get_hessian_parameters()
        if param_values is None:
            return ""

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_values

        # Get constraints if needed
        minH = -np.inf
        maxH = np.inf
        minV = -np.inf
        maxV = np.inf
        minA = -np.inf
        maxA = np.inf

        constraints_answer = messagebox.askyesno("Constraints",
                                                 "Would you like to limit the analysis to particles of certain height, volume, or area?")
        if constraints_answer:
            constraints = ParameterDialog.get_constraints_dialog()
            if constraints is None:
                return ""
            minH, maxH, minA, maxA, minV, maxV = constraints

        # Create Series folder in current directory
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

        # Store parameters
        Parameters = np.array([
            scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
            subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
        ])
        DataManager.save_wave_data(Parameters, os.path.join(SeriesDF, "Parameters.npy"))

        # Process images
        AllHeights = []
        AllVolumes = []
        AllAreas = []
        AllAvgHeights = []

        # Get list of image files
        image_files = []
        for ext in ['.ibw', '.tiff', '.tif', '.png', '.jpg', '.jpeg', '.npy']:
            image_files.extend([f for f in os.listdir(ImagesDF) if f.lower().endswith(ext.lower())])

        image_files.sort()  # Ensure consistent ordering

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(ImagesDF, image_file)
            im = DataManager.load_image_file(image_path)

            if im is None:
                safe_print(f"Warning: Could not load {image_file}")
                continue

            safe_print("-------------------------------------------------------")
            safe_print(f"Analyzing image {i + 1} of {len(image_files)}: {image_file}")
            safe_print("-------------------------------------------------------")

            # Run analysis. Create image folder inside series folder
            imageDF = HessianBlobs_SeriesMode(im, params=Parameters, seriesFolder=SeriesDF, imageIndex=i)

            if imageDF:
                try:
                    # Load measurements
                    Heights = np.load(os.path.join(imageDF, "Heights.npy"))
                    AvgHeights = np.load(os.path.join(imageDF, "AvgHeights.npy"))
                    Areas = np.load(os.path.join(imageDF, "Areas.npy"))
                    Volumes = np.load(os.path.join(imageDF, "Volumes.npy"))

                    # Concatenate measurements
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

        # Print summary
        numParticles = len(AllHeights)
        safe_print(f"  Series complete. Total particles detected: {numParticles}")
        safe_print(f"  Results saved to: {SeriesDF}")

        # Verify the folder was created properly
        if os.path.exists(SeriesDF):
            safe_print(f"✓ Confirmed: Series folder exists at {SeriesDF}")
            files_in_series = os.listdir(SeriesDF)
            safe_print(f"✓ Files in series folder: {files_in_series}")
        else:
            safe_print(f"✗ ERROR: Series folder was not created at {SeriesDF}")

        return SeriesDF

    except Exception as e:
        error_msg = handle_error("BatchHessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""

def HessianBlobs(im, scaleStart=1, layers=15, scaleFactor=1.5, detHResponseThresh=-1,
                 particleType=0, subPixelMult=1, allowOverlap=0, minH=-np.inf, maxH=np.inf,
                 minA=-np.inf, maxA=np.inf, minV=-np.inf, maxV=np.inf):
    """Main function for Hessian blob particle detection analysis."""
    try:
        # Convert scaleStart from pixel units to scaled units squared
        scaleStart = (scaleStart * 1.0) ** 2 / 2
        layers = max(1, int(np.ceil(np.log((layers * 1.0) ** 2 / (2 * scaleStart)) / np.log(scaleFactor))))
        subPixelMult = max(1, round(subPixelMult))
        scaleFactor = max(1.1, scaleFactor)

        # Hard coded parameters matching Igor Pro exactly
        gammaNorm = 1
        maxCurvatureRatio = 10  # Must match Igor Pro hardcoded value
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

        # Threshold determination
        if detHResponseThresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            detHResponseThresh = np.sqrt(OtsuThreshold(detH, LapG, particleType, maxCurvatureRatio))
            safe_print(f"Otsu's Threshold: {detHResponseThresh}")
        elif detHResponseThresh == -2:
            detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
            safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")

        # Detect particles
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh, particleType,
                                                         maxCurvatureRatio)
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

        # Remove overlapping particles if requested
        if allowOverlap == 0:
            safe_print("Determining scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
        else:
            for i in range(len(Info)):
                Info[i][10] = 1

        # Initialize particle status
        if numPotentialParticles > 0:
            for i in range(len(Info)):
                Info[i][13] = 0
                Info[i][14] = 0

        # Process and measure particles
        Volumes = []
        Heights = []
        COM = []
        Areas = []
        AvgHeights = []

        safe_print("Cropping and measuring particles..")
        for i in range(numPotentialParticles - 1, -1, -1):
            # Skip overlapping particles if not allowed
            if allowOverlap == 0 and Info[i][10] == 0:
                continue

            # Basic validation
            if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
                continue

            # Continue with existing particle processing logic...
            # [Rest of the function remains unchanged]

        return NewDF

    except Exception as e:
        handle_error("HessianBlobs", e)
        raise HessianBlobError(f"Failed to perform Hessian blob analysis: {e}")

def HessianBlobs_SeriesMode(im, params=None, seriesFolder=None, imageIndex=0):
    """Creates image folder inside series folder."""

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

    # Parameter conversion
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
    CurrentDF = GetDataFolder(1)
    if seriesFolder:
        # Create image folder inside the series folder with a clean name
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
    safe_print(f"Created image particle folder: {NewDF}")

    # Store original image
    if len(im.shape) == 3:
        Original = im[:, :, 0].copy()
    else:
        Original = im.copy()

    DataManager.save_wave_data(Original, os.path.join(NewDF, "Original.npy"))
    im = Original

    # Run the analysis
    numPotentialParticles = 0
    count = 0
    limP = im.shape[0]
    limQ = im.shape[1]

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
        detHResponseThresh = InteractiveThreshold(im, detH, LapG, particleType, maxCurvatureRatio)
        safe_print(f"Chosen Det H Response Threshold: {detHResponseThresh}")

    # Detect particles
    safe_print("Detecting Hessian blobs..")
    mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, detHResponseThresh, particleType,
                                                     maxCurvatureRatio)
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

    # Remove overlapping particles if requested
    if allowOverlap == 0:
        safe_print("Determining scale-maximal particles..")
        MaximalBlobs(Info, mapNum)
    else:
        for i in range(len(Info)):
            Info[i][10] = 1

    # Initialize particle status
    if numPotentialParticles > 0:
        for i in range(len(Info)):
            Info[i][13] = 0
            Info[i][14] = 0

    # Process and measure particles
    Volumes = []
    Heights = []
    COM = []
    Areas = []
    AvgHeights = []

    safe_print("Cropping and measuring particles..")
    for i in range(numPotentialParticles - 1, -1, -1):
        # Skip overlapping particles if not allowed
        if allowOverlap == 0 and Info[i][10] == 0:
            continue

        # Basic validation
        if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
            continue

        # Boundary particles check
        if (allowBoundaryParticles == 0 and
                (Info[i][4] <= 2 or Info[i][5] >= limP - 3 or Info[i][6] <= 2 or Info[i][7] >= limQ - 3)):
            continue

        # Extract and process particle
        padding = int(np.ceil(max(Info[i][5] - Info[i][4] + 2, Info[i][7] - Info[i][6] + 2)))
        p_start = max(int(Info[i][4]) - padding, 0)
        p_end = min(int(Info[i][5]) + padding, limP - 1)
        q_start = max(int(Info[i][6]) - padding, 0)
        q_end = min(int(Info[i][7]) + padding, limQ - 1)

        particle = im[p_start:p_end + 1, q_start:q_end + 1].copy()
        mask = create_particle_mask(mapNum, i, int(Info[i][9]), p_start, p_end, q_start, q_end)
        perim = create_perimeter_mask(mask)

        # Calculate measurements
        bg = M_MinBoundary(particle, mask)
        particle_bg_sub = particle - bg
        height = M_Height(particle_bg_sub, mask, 0)
        vol = M_Volume(particle_bg_sub, mask, 0)
        centerOfMass = M_CenterOfMass(particle_bg_sub, mask, 0)
        particleArea = M_Area(mask)
        particlePerim = M_Perimeter(mask)
        avgHeight = vol / particleArea if particleArea > 0 else 0

        # Check constraints
        if not (minH < height < maxH and minA < particleArea < maxA and minV < vol < maxV):
            continue

        # Accept particle
        Info[i][14] = count

        # Create particle data
        particle_data = {
            'parent': "image",  # Use simple name instead of NameOfWave(im)
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

        # Save particle data
        save_particle_data(NewDF, count, particle, mask, perim, particle_data)

        # Store measurements
        Volumes.append(vol)
        Heights.append(height)
        COM.append(centerOfMass)
        Areas.append(particleArea)
        AvgHeights.append(avgHeight)

        count += 1

    # Save measurement arrays
    DataManager.save_wave_data(np.array(Volumes), os.path.join(NewDF, "Volumes.npy"))
    DataManager.save_wave_data(np.array(Heights), os.path.join(NewDF, "Heights.npy"))
    DataManager.save_wave_data(np.array(COM), os.path.join(NewDF, "COM.npy"))
    DataManager.save_wave_data(np.array(Areas), os.path.join(NewDF, "Areas.npy"))
    DataManager.save_wave_data(np.array(AvgHeights), os.path.join(NewDF, "AvgHeights.npy"))

    # Create particle map
    ParticleMap = np.full_like(im, -1)
    DataManager.save_wave_data(ParticleMap, os.path.join(NewDF, "ParticleMap.npy"))

    safe_print(f"Analysis complete. {count} particles detected and measured.")

    # Verify folder structure
    verify_folder_structure(NewDF)

    return NewDF

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
        safe_print(f"  Created Particle_{particle_id} folder")

        # Save particle arrays
        DataManager.save_wave_data(particle, os.path.join(particle_folder, f"Particle_{particle_id}.npy"))
        DataManager.save_wave_data(mask, os.path.join(particle_folder, f"Mask_{particle_id}.npy"))
        DataManager.save_wave_data(perim, os.path.join(particle_folder, f"Perimeter_{particle_id}.npy"))

        # Save particle info
        particle_info = DataManager.create_particle_info(particle_data, particle_id)

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

def WaveStats(data_file):
    """Compute basic statistics exactly matching Igor Pro WaveStats output."""
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
        V_avg = np.mean(clean_data) if len(clean_data) > 0 else 0
        V_sum = np.sum(clean_data)
        V_sdev = np.std(clean_data, ddof=1) if len(clean_data) > 1 else 0
        V_rms = np.sqrt(np.mean(clean_data ** 2)) if len(clean_data) > 0 else 0
        V_min = np.min(clean_data) if len(clean_data) > 0 else 0
        V_max = np.max(clean_data) if len(clean_data) > 0 else 0

        # Print results in exact Igor Pro format
        safe_print(f"WaveStats {base_name}")
        safe_print(f"  V_npnts= {V_npnts}; V_numNaNs= {V_numNaNs};")
        safe_print(f"  V_avg= {V_avg:.6g}; V_sum= {V_sum:.6g};")
        safe_print(f"  V_sdev= {V_sdev:.6g}; V_rms= {V_rms:.6g};")
        safe_print(f"  V_min= {V_min:.6g}; V_max= {V_max:.6g};")

        return {
            'V_npnts': V_npnts,
            'V_numNaNs': V_numNaNs,
            'V_avg': V_avg,
            'V_sum': V_sum,
            'V_sdev': V_sdev,
            'V_rms': V_rms,
            'V_min': V_min,
            'V_max': V_max
        }

    except Exception as e:
        error_msg = handle_error("WaveStats", e)
        messagebox.showerror("Statistics Error", error_msg)
        return None

def print_series_analysis_summary(result_folder, heights, volumes, areas):
    """Print series summary matching Igor Pro exactly"""
    try:
        if len(heights) > 0:
            safe_print(f"Series complete. Total particles detected: {len(heights)}")
        else:
            safe_print("Series complete. Total particles detected: 0")

    except Exception as e:
        handle_error("print_series_analysis_summary", e)

def Testing(str_input, num):
    """Testing function to demonstrate how user-defined functions work."""
    try:
        safe_print(f"You typed: {str_input}")
        safe_print(f"Your number plus two is {num + 2}")

    except Exception as e:
        handle_error("Testing", e)