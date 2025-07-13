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

def HessianBlobs(im, params=None):
    """Executes the Hessian blob algorithm on an image."""
    try:
        # Validate input
        if im is None:
            raise HessianBlobError("Input image is None")

        if len(im.shape) < 2:
            raise HessianBlobError("Input must be at least 2D")

        # Measurement ranges
        min_h, max_h, min_v, max_v, min_a, max_a = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf

        # Get parameters
        if params is None:
            param_values = ParameterDialog.get_hessian_parameters()
            if param_values is None:
                safe_print("Analysis cancelled by user.")
                return ""

            scale_start, layers, scale_factor, det_h_response_thresh, particle_type, subpixel_mult, allow_overlap = param_values

            constraints_answer = messagebox.askyesno("Constraints",
                                                     "Would you like to limit the analysis to particles of certain height, volume, or area?")

            if constraints_answer:
                constraints = ParameterDialog.get_constraints_dialog()
                if constraints is None:
                    safe_print("Analysis cancelled by user.")
                    return ""
                min_h, max_h, min_a, max_a, min_v, max_v = constraints
        else:
            if len(params) < 13:
                raise HessianBlobError("Provided parameter array must contain 13 parameters")

            scale_start = params[0]
            layers = int(params[1])
            scale_factor = params[2]
            det_h_response_thresh = params[3]
            particle_type = int(params[4])
            subpixel_mult = int(params[5])
            allow_overlap = int(params[6])
            min_h = params[7]
            max_h = params[8]
            min_a = params[9]
            max_a = params[10]
            min_v = params[11]
            max_v = params[12]

        # Print parameters
        if params is None:
            all_params = [scale_start, layers, scale_factor, det_h_response_thresh,
                          particle_type, subpixel_mult, allow_overlap,
                          min_h, max_h, min_a, max_a, min_v, max_v]
            print_analysis_parameters(all_params)

        # Validate and convert parameters
        converted_params = validate_and_convert_parameters([scale_start, layers, scale_factor,
                                                            det_h_response_thresh, particle_type,
                                                            subpixel_mult, allow_overlap])

        scale_start, layers, scale_factor, det_h_response_thresh, particle_type, subpixel_mult, allow_overlap = converted_params

        # Hard coded parameters
        gamma_norm = 1
        max_curvature_ratio = 10
        allow_boundary_particles = 1

        # Create particle folder
        current_df = GetDataFolder(1)
        new_df = NameOfWave(im) + "_Particles"

        full_path = os.path.join(current_df, new_df)
        if os.path.exists(full_path):
            new_df = UniqueName(new_df, 11, 2)

        new_df = os.path.join(current_df, new_df)
        DataManager.create_igor_folder_structure(new_df, "particles")

        safe_print(f"Created particle analysis folder: {new_df}")

        # Store original image
        if len(im.shape) == 3:
            original = im[:, :, 0].copy()
        else:
            original = im.copy()

        DataManager.save_wave_data(original, os.path.join(new_df, "Original.npy"))
        im = original

        # Scale-space analysis with progress reporting
        safe_print("Calculating scale-space representation..")
        L = ScaleSpaceRepresentation(im, layers, np.sqrt(scale_start) / 1.0, scale_factor)
        DataManager.save_wave_data(L, os.path.join(new_df, "ScaleSpaceRep.npy"))

        safe_print("Calculating scale-space derivatives..")
        LapG, detH = BlobDetectors(L, gamma_norm)
        DataManager.save_wave_data(LapG, os.path.join(new_df, "LapG.npy"))
        DataManager.save_wave_data(detH, os.path.join(new_df, "detH.npy"))

        # Threshold determination
        if det_h_response_thresh == -1:
            safe_print("Calculating Otsu's Threshold..")
            det_h_response_thresh = np.sqrt(OtsuThreshold(detH, LapG, particle_type, max_curvature_ratio))
            safe_print(f"Otsu's Threshold: {det_h_response_thresh}")
        elif det_h_response_thresh == -2:
            det_h_response_thresh = InteractiveThreshold(im, detH, LapG, particle_type, max_curvature_ratio)
            safe_print(f"Chosen Det H Response Threshold: {det_h_response_thresh}")

        # Detect particles
        safe_print("Detecting Hessian blobs..")
        mapNum, mapDetH, mapMax, Info = FindHessianBlobs(im, detH, LapG, det_h_response_thresh,
                                                         particle_type, max_curvature_ratio)

        num_potential_particles = len(Info) if Info is not None else 0

        if num_potential_particles == 0:
            safe_print("No particles detected.")
            # Save empty arrays
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Volumes.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Heights.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "COM.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "Areas.npy"))
            DataManager.save_wave_data(np.array([]), os.path.join(new_df, "AvgHeights.npy"))
            return new_df

        # Remove overlapping particles if requested
        if allow_overlap == 0:
            safe_print("Determining scale-maximal particles..")
            MaximalBlobs(Info, mapNum)
        else:
            if Info is not None:
                for i in range(len(Info)):
                    Info[i][10] = 1

        # Initialize particle status
        if num_potential_particles > 0:
            for i in range(len(Info)):
                Info[i][13] = 0
                Info[i][14] = 0

        # Process particles with enhanced progress reporting
        safe_print("Cropping and measuring particles..")
        volumes = []
        heights = []
        com = []
        areas = []
        avg_heights = []
        count = 0

        for i in range(num_potential_particles - 1, -1, -1):
            try:
                # Skip overlapping particles if not allowed
                if allow_overlap == 0 and Info[i][10] == 0:
                    continue

                # Basic validation
                if Info[i][2] < 1 or (Info[i][5] - Info[i][4]) < 0 or (Info[i][7] - Info[i][6]) < 0:
                    continue

                # Boundary particles check
                if (allow_boundary_particles == 0 and
                        (Info[i][4] <= 2 or Info[i][5] >= im.shape[0] - 3 or
                         Info[i][6] <= 2 or Info[i][7] >= im.shape[1] - 3)):
                    continue

                # Extract particle region
                padding = int(np.ceil(max(Info[i][5] - Info[i][4] + 2, Info[i][7] - Info[i][6] + 2)))
                p_start = max(int(Info[i][4]) - padding, 0)
                p_end = min(int(Info[i][5]) + padding, im.shape[0] - 1)
                q_start = max(int(Info[i][6]) - padding, 0)
                q_end = min(int(Info[i][7]) + padding, im.shape[1] - 1)

                particle = im[p_start:p_end + 1, q_start:q_end + 1].copy()

                # Create mask
                mask = create_particle_mask(mapNum, i, int(Info[i][9]), p_start, p_end, q_start, q_end)

                # Create perimeter
                perim = create_perimeter_mask(mask)

                # Calculate measurements
                bg = M_MinBoundary(particle, mask)
                particle_bg_sub = particle - bg
                height = M_Height(particle_bg_sub, mask, 0)
                vol = M_Volume(particle_bg_sub, mask, 0)
                center_of_mass = M_CenterOfMass(particle_bg_sub, mask, 0)
                particle_area = M_Area(mask)
                particle_perim = M_Perimeter(mask)
                avg_height = vol / particle_area if particle_area > 0 else 0

                # Check constraints
                if not (min_h < height < max_h and min_a < particle_area < max_a and min_v < vol < max_v):
                    continue

                # Accept particle
                Info[i][14] = count

                # Create particle data
                particle_data = {
                    'parent': NameOfWave(im),
                    'height': height,
                    'avg_height': avg_height,
                    'volume': vol,
                    'area': particle_area,
                    'perimeter': particle_perim,
                    'scale': Info[i][8],
                    'com': center_of_mass,
                    'p_seed': Info[i][0],
                    'q_seed': Info[i][1],
                    'r_seed': Info[i][9]
                }

                # Save particle to Igor Pro-style folder
                save_particle_data(new_df, count, particle, mask, perim, particle_data)

                # Store measurements
                volumes.append(vol)
                heights.append(height)
                com.append(center_of_mass)
                areas.append(particle_area)
                avg_heights.append(avg_height)

                count += 1

                # Progress reporting every 10 particles
                if count % 10 == 0:
                    safe_print(f"  Processed {count} particles...")

            except Exception as e:
                handle_error("HessianBlobs", e, f"processing particle {i}")
                continue

        # Save measurement arrays
        DataManager.save_wave_data(np.array(volumes), os.path.join(new_df, "Volumes.npy"))
        DataManager.save_wave_data(np.array(heights), os.path.join(new_df, "Heights.npy"))
        DataManager.save_wave_data(np.array(com), os.path.join(new_df, "COM.npy"))
        DataManager.save_wave_data(np.array(areas), os.path.join(new_df, "Areas.npy"))
        DataManager.save_wave_data(np.array(avg_heights), os.path.join(new_df, "AvgHeights.npy"))

        # Create particle map
        particle_map = np.full_like(im, -1)
        DataManager.save_wave_data(particle_map, os.path.join(new_df, "ParticleMap.npy"))

        # Final summary
        safe_print(f"Analysis complete. {count} particles detected and measured.")
        if count > 0:
            safe_print(f"Average height: {np.mean(heights):.3e}")
            safe_print(f"Average volume: {np.mean(volumes):.3e}")
            safe_print(f"Average area: {np.mean(areas):.3e}")

        # Verify folder structure
        verify_igor_compatibility(new_df)
        SetDataFolder(current_df)
        return new_df

    except Exception as e:
        error_msg = handle_error("HessianBlobs", e)
        messagebox.showerror("Analysis Error", error_msg)
        return ""

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

def print_series_analysis_summary(result_folder):
    """Print comprehensive series analysis summary matching Igor Pro exactly"""
    try:
        safe_print("\n" + "=" * 70)
        safe_print("SERIES ANALYSIS COMPLETE")
        safe_print("=" * 70)

        # Load all measurement arrays
        heights_file = os.path.join(result_folder, "AllHeights.npy")
        volumes_file = os.path.join(result_folder, "AllVolumes.npy")
        areas_file = os.path.join(result_folder, "AllAreas.npy")
        avg_heights_file = os.path.join(result_folder, "AllAvgHeights.npy")

        if not all(os.path.exists(f) for f in [heights_file, volumes_file, areas_file, avg_heights_file]):
            safe_print("Warning: Some measurement files not found")
            return

        heights = np.load(heights_file)
        volumes = np.load(volumes_file)
        areas = np.load(areas_file)
        avg_heights = np.load(avg_heights_file)

        # Print summary statistics exactly like Igor Pro
        safe_print(f"Total particles detected: {len(heights)}")
        safe_print(f"Results saved to: {result_folder}")
        safe_print("")

        if len(heights) > 0:
            safe_print("PARTICLE MEASUREMENT STATISTICS:")
            safe_print("-" * 50)

            # Heights statistics
            safe_print(f"Heights (n={len(heights)}):")
            safe_print(f"  Mean: {np.mean(heights):.6e} m")
            safe_print(f"  Std:  {np.std(heights, ddof=1):.6e} m")
            safe_print(f"  Min:  {np.min(heights):.6e} m")
            safe_print(f"  Max:  {np.max(heights):.6e} m")
            safe_print("")

            # Volumes statistics
            safe_print(f"Volumes (n={len(volumes)}):")
            safe_print(f"  Mean: {np.mean(volumes):.6e} m³")
            safe_print(f"  Std:  {np.std(volumes, ddof=1):.6e} m³")
            safe_print(f"  Min:  {np.min(volumes):.6e} m³")
            safe_print(f"  Max:  {np.max(volumes):.6e} m³")
            safe_print("")

            # Areas statistics
            safe_print(f"Areas (n={len(areas)}):")
            safe_print(f"  Mean: {np.mean(areas):.6e} m²")
            safe_print(f"  Std:  {np.std(areas, ddof=1):.6e} m²")
            safe_print(f"  Min:  {np.min(areas):.6e} m²")
            safe_print(f"  Max:  {np.max(areas):.6e} m²")
            safe_print("")

        # Print file structure created
        safe_print("FILES CREATED:")
        safe_print("-" * 20)
        safe_print("• AllHeights.npy    - Combined particle heights")
        safe_print("• AllVolumes.npy    - Combined particle volumes")
        safe_print("• AllAreas.npy      - Combined particle areas")
        safe_print("• AllAvgHeights.npy - Combined average heights")
        safe_print("• Parameters.npy    - Analysis parameters used")

        # Count individual image folders
        image_folders = [d for d in os.listdir(result_folder)
                         if os.path.isdir(os.path.join(result_folder, d)) and d.endswith('_Particles')]
        safe_print(f"• {len(image_folders)} individual image analysis folders")

        safe_print("\n" + "=" * 70)

    except Exception as e:
        handle_error("print_series_analysis_summary", e)

def Testing(str_input, num):
    """Testing function to demonstrate how user-defined functions work."""
    try:
        safe_print(f"You typed: {str_input}")
        safe_print(f"Your number plus two is {num + 2}")

    except Exception as e:
        handle_error("Testing", e)