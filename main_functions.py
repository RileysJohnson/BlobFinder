"""
Main Functions
Contains the primary Hessian blob detection functions
Direct port from Igor Pro code maintaining same variable names and structure
"""

import numpy as np
from igor_compatibility import *
from file_io import *
from scale_space import *
from particle_measurements import *
from preprocessing import *
from utilities import *
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex


def BatchHessianBlobs():
    """
    Detects Hessian blobs in a series of images in a chosen data folder.
    Be sure to highlight the data folder containing the images in the data browser before running.
    """
    ImagesDF = GetBrowserSelection(0)
    CurrentDF = GetDataFolder(1)

    if not DataFolderExists(ImagesDF) or CountObjects(ImagesDF, 1) < 1:
        messagebox.showerror("Error", "Select the folder with your images in it in the data browser, then try again.")
        return ""

    # Declare algorithm parameters.
    scaleStart = 1  # In pixel units
    layers = 256
    scaleFactor = 1.5
    detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
    particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
    subPixelMult = 1  # 1 or more, should be integer.
    allowOverlap = 0

    # Get parameters from user
    params = GetHessianBlobParameters(scaleStart, layers, scaleFactor,
                                      detHResponseThresh, particleType,
                                      subPixelMult, allowOverlap)

    if params is None:
        return ""

    scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = params

    # Get constraints if requested
    constraints = GetParticleConstraints()
    if constraints is None:
        return ""

    minH, maxH, minV, maxV, minA, maxA = constraints

    # Make a Data Folder for the Series
    folder = data_browser.get_folder(ImagesDF.rstrip(':'))
    NumImages = len(folder.waves)
    SeriesDF = UniqueName("Series_", 11, 0)
    NewDataFolder(SeriesDF)
    series_folder = data_browser.get_folder(SeriesDF)

    # Store the parameters being used
    parameters = Wave(np.array([
        scaleStart, layers, scaleFactor, detHResponseThresh, particleType,
        subPixelMult, allowOverlap, minH, maxH, minA, maxA, minV, maxV
    ]), "Parameters")
    series_folder.add_wave(parameters)

    # Find particles in each image and collect measurements from each image.
    all_heights = Wave(np.array([]), "AllHeights")
    all_volumes = Wave(np.array([]), "AllVolumes")
    all_areas = Wave(np.array([]), "AllAreas")
    all_avg_heights = Wave(np.array([]), "AllAvgHeights")

    for i in range(NumImages):
        wave_name = list(folder.waves.keys())[i]
        im = folder.waves[wave_name]

        print("-------------------------------------------------------")
        print(f"Analyzing image {i + 1} of {NumImages}")
        print("-------------------------------------------------------")

        # Run the Hessian blob algorithm and get the path to the image folder.
        imageDF = HessianBlobs(im, params=parameters)

        if imageDF:
            # Get wave references to the measurement waves.
            image_folder = data_browser.get_folder(imageDF.rstrip(':'))

            if "Heights" in image_folder.waves:
                heights = image_folder.waves["Heights"]
                avg_heights = image_folder.waves["AvgHeights"]
                areas = image_folder.waves["Areas"]
                volumes = image_folder.waves["Volumes"]

                # Concatenate the measurements into the master wave.
                all_heights.data = np.concatenate([all_heights.data, heights.data])
                all_avg_heights.data = np.concatenate([all_avg_heights.data, avg_heights.data])
                all_areas.data = np.concatenate([all_areas.data, areas.data])
                all_volumes.data = np.concatenate([all_volumes.data, volumes.data])

    # Store accumulated results
    series_folder.add_wave(all_heights)
    series_folder.add_wave(all_volumes)
    series_folder.add_wave(all_areas)
    series_folder.add_wave(all_avg_heights)

    # Determine the total number of particles.
    numParticles = len(all_heights.data)
    print(f"  Series complete. Total particles detected: {numParticles}")

    return f"root:{SeriesDF}"


def HessianBlobs(im, params=None):
    """
    Executes the Hessian blob algorithm on an image.
        im : The image to be analyzed.
        params : An optional parameter wave with the 13 parameters to be passed in.
    """
    # Declare algorithm parameters.
    scaleStart = 1  # In pixel units
    layers = max(im.data.shape[0], im.data.shape[1]) // 4
    scaleFactor = 1.5
    detHResponseThresh = -2  # Use -1 for Otsu's method, -2 for interactive
    particleType = 1  # -1 for neg only, 1 for pos only, 0 for both
    subPixelMult = 1  # 1 or more, should be integer.
    allowOverlap = 0

    # Declare measurement ranges.
    minH = -np.inf
    maxH = np.inf
    minV = -np.inf
    maxV = np.inf
    minA = -np.inf
    maxA = np.inf

    # Retrieve parameters if given in the params wave, or prompt the user for them if not.
    if params is None:
        param_result = GetHessianBlobParameters(scaleStart, layers, scaleFactor,
                                                detHResponseThresh, particleType,
                                                subPixelMult, allowOverlap)
        if param_result is None:
            return ""

        scaleStart, layers, scaleFactor, detHResponseThresh, particleType, subPixelMult, allowOverlap = param_result

        constraint_result = GetParticleConstraints()
        if constraint_result is None:
            return ""
        minH, maxH, minV, maxV, minA, maxA = constraint_result

    else:
        if len(params.data) < 13:
            messagebox.showerror("Error", "Provided parameter wave must contain the 13 parameters.")
            return ""

        scaleStart = params.data[0]
        layers = int(params.data[1])
        scaleFactor = params.data[2]
        detHResponseThresh = params.data[3]
        particleType = int(params.data[4])
        subPixelMult = int(params.data[5])
        allowOverlap = int(params.data[6])
        minH = params.data[7]
        maxH = params.data[8]
        minA = params.data[9]
        maxA = params.data[10]
        minV = params.data[11]
        maxV = params.data[12]

    # Check parameters: Convert the scaleStart and layers parameters from pixel units to scaled units squared.
    scaleStart = (scaleStart * DimDelta(im, 0)) ** 2 / 2
    layers = int(np.ceil(np.log((layers * DimDelta(im, 0)) ** 2 / (2 * scaleStart)) / np.log(scaleFactor)))
    subPixelMult = max(1, int(np.round(subPixelMult)))
    scaleFactor = max(1.1, scaleFactor)

    # Hard coded parameters.
    gammaNorm = 1
    maxCurvatureRatio = 10
    allowBoundaryParticles = 1

    # Make a data folder for the particles.
    NewDF = f"{im.name}_Particles"
    if DataFolderExists(NewDF):
        NewDF = UniqueName(NewDF, 11, 2)
    NewDataFolder(NewDF)
    particle_folder = data_browser.get_folder(NewDF)

    # Store a copy of the original image.
    original = Wave(im.data[:, :, 0] if len(im.data.shape) > 2 else im.data, "Original")
    original.note = im.note
    original.SetScale('x', DimOffset(im, 0), DimDelta(im, 0))
    original.SetScale('y', DimOffset(im, 1), DimDelta(im, 1))
    particle_folder.add_wave(original)

    # Use the original copy for processing
    im = original

    # Declare needed variables.
    numPotentialParticles = 0
    count = 0
    limP = im.data.shape[0]
    limQ = im.data.shape[1]

    # Calculate the discrete scale-space representation.
    print("Calculating scale-space representation..")
    L = ScaleSpaceRepresentation(im, layers, np.sqrt(scaleStart) / DimDelta(im, 0), scaleFactor)
    L.name = "ScaleSpaceRep"
    particle_folder.add_wave(L)

    # Calculate gamma = 1 normalized scale-space derivatives
    print("Calculating scale-space derivatives..")
    BlobDetectors(L, gammaNorm)

    # Get the computed blob detectors
    LG = data_browser.waves["LapG"]
    detH = data_browser.waves["detH"]

    # Move to particle folder
    particle_folder.add_wave(LG, "LapG")
    particle_folder.add_wave(detH, "detH")

    # If the user wants to, use Otsu's method for the blob strength threshold or find it interactively.
    if detHResponseThresh == -1:
        print("Calculating Otsu's Threshold..")
        detHResponseThresh = np.sqrt(OtsuThreshold(detH, LG, particleType, maxCurvatureRatio))
        print(f"Otsu's Threshold: {detHResponseThresh}")
    elif detHResponseThresh == -2:
        detHResponseThresh = InteractiveThresholdGUI(im, detH, LG, particleType, maxCurvatureRatio)
        print(f"Chosen Det H Response Threshold: {detHResponseThresh}")

    # Detect particle candidates by identifying scale-space extrema.
    print("Detecting Hessian blobs..")
    mapMax = Wave(np.zeros_like(detH.data), "mapMax")
    mapDetH = Wave(np.zeros_like(detH.data), "mapDetH")
    mapNum = Wave(np.full_like(detH.data, -1), "mapNum")
    info = Wave(np.zeros((limP * limQ * detH.data.shape[2] // 27, 15)), "Info")

    FindHessianBlobs(im, detH, LG, detHResponseThresh, mapNum, mapDetH, mapMax, info,
                     particleType, maxCurvatureRatio)

    numPotentialParticles = info.data.shape[0]

    # Remove overlapping Hessian blobs if asked to do so, or else allow nested particles.
    if allowOverlap == 0:
        print("Determining scale-maximal particles..")
        MaximalBlobs(info, mapNum)
    else:
        info.data[:, 10] = 1

    # Initialize particle containment and acceptance status as undetermined.
    if numPotentialParticles > 0:
        info.data[:, 13] = 0
        info.data[:, 14] = 0

    # Make waves for the particle measurements.
    volumes = Wave(np.zeros(numPotentialParticles), "Volumes")
    heights = Wave(np.zeros(numPotentialParticles), "Heights")
    com = Wave(np.zeros((numPotentialParticles, 2)), "COM")
    areas = Wave(np.zeros(numPotentialParticles), "Areas")
    avg_heights = Wave(np.zeros(numPotentialParticles), "AvgHeights")

    print("Cropping and measuring particles..")
    count = ProcessParticles(im, info, mapNum, LG, detH, allowOverlap, allowBoundaryParticles,
                             maxCurvatureRatio, subPixelMult, minH, maxH, minA, maxA, minV, maxV,
                             volumes, heights, com, areas, avg_heights, particle_folder)

    # Create particle map
    particle_map = Wave(np.full_like(im.data, -1), "ParticleMap")
    CreateParticleMap(particle_map, particle_folder, count)
    particle_folder.add_wave(particle_map)

    # Trim the metric waves of excess points.
    if count < numPotentialParticles:
        volumes.data = volumes.data[:count]
        heights.data = heights.data[:count]
        com.data = com.data[:count, :]
        areas.data = areas.data[:count]
        avg_heights.data = avg_heights.data[:count]

    # Store measurement waves
    particle_folder.add_wave(volumes)
    particle_folder.add_wave(heights)
    particle_folder.add_wave(com)
    particle_folder.add_wave(areas)
    particle_folder.add_wave(avg_heights)
    particle_folder.add_wave(info, "Info")

    # Display results
    DisplayResults(im, particle_folder, count)

    return f"root:{NewDF}"


def ProcessParticles(im, info, mapNum, LG, detH, allowOverlap, allowBoundaryParticles,
                     maxCurvatureRatio, subPixelMult, minH, maxH, minA, maxA, minV, maxV,
                     volumes, heights, com, areas, avg_heights, particle_folder):
    """
    Process and measure individual particles
    """
    count = 0
    limP = im.data.shape[0]
    limQ = im.data.shape[1]
    dx = DimDelta(im, 0)
    dy = DimDelta(im, 1)

    for i in range(info.data.shape[0] - 1, -1, -1):
        # If asked to do so, only consider non-overlapping particles.
        if allowOverlap == 0 and info.data[i, 10] == 0:
            continue

        # Make various cuts to eliminate bad particles less than one pixel.
        if (info.data[i, 2] < 1 or
                (info.data[i, 5] - info.data[i, 4]) < 0 or
                (info.data[i, 7] - info.data[i, 6]) < 0):
            continue

        # Consider boundary particles?
        if (allowBoundaryParticles == 0 and
                (info.data[i, 4] <= 2 or info.data[i, 5] >= limP - 3 or
                 info.data[i, 6] <= 2 or info.data[i, 7] >= limQ - 3)):
            continue

        # Make a crop, mask, and perimeter image for the individual particle.
        padding = int(np.ceil(max(info.data[i, 5] - info.data[i, 4] + 2,
                                  info.data[i, 7] - info.data[i, 6] + 2)))

        x_start = max(int(info.data[i, 4]) - padding, 0)
        x_end = min(int(info.data[i, 5]) + padding, limP - 1)
        y_start = max(int(info.data[i, 6]) - padding, 0)
        y_end = min(int(info.data[i, 7]) + padding, limQ - 1)

        # Create particle crop
        particle_data = im.data[x_start:x_end + 1, y_start:y_end + 1]
        particle = Wave(particle_data, f"Particle_{count}")
        particle.SetScale('x', DimOffset(im, 0) + x_start * dx, dx)
        particle.SetScale('y', DimOffset(im, 1) + y_start * dy, dy)

        # Create mask
        mask_data = np.zeros_like(particle_data)
        layer = int(info.data[i, 9])

        for ii in range(particle_data.shape[0]):
            for jj in range(particle_data.shape[1]):
                global_i = x_start + ii
                global_j = y_start + jj
                if (0 <= global_i < mapNum.data.shape[0] and
                        0 <= global_j < mapNum.data.shape[1] and
                        mapNum.data[global_i, global_j, layer] == i):
                    mask_data[ii, jj] = 1

        mask = Wave(mask_data, f"Mask_{count}")
        mask.SetScale('x', particle.scaling['x']['offset'], particle.scaling['x']['delta'])
        mask.SetScale('y', particle.scaling['y']['offset'], particle.scaling['y']['delta'])

        # Create perimeter
        perim_data = np.zeros_like(mask_data)
        for ii in range(1, mask_data.shape[0] - 1):
            for jj in range(1, mask_data.shape[1] - 1):
                if (mask_data[ii, jj] == 1 and
                        (mask_data[ii + 1, jj] == 0 or mask_data[ii - 1, jj] == 0 or
                         mask_data[ii, jj + 1] == 0 or mask_data[ii, jj - 1] == 0 or
                         mask_data[ii + 1, jj + 1] == 0 or mask_data[ii - 1, jj + 1] == 0 or
                         mask_data[ii + 1, jj - 1] == 0 or mask_data[ii - 1, jj - 1] == 0)):
                    perim_data[ii, jj] = 1

        perim = Wave(perim_data, f"Perimeter_{count}")

        # Calculate subpixel position (simplified)
        p0 = int(info.data[i, 0])
        q0 = int(info.data[i, 1])
        r0 = int(info.data[i, 9])

        # Simplified subpixel calculation
        subPixX = DimOffset(im, 0) + DimDelta(im, 0) * info.data[i, 0]
        subPixY = DimOffset(im, 1) + DimDelta(im, 1) * info.data[i, 1]

        # Calculate metrics associated with the particle.
        bg = M_MinBoundary(particle, mask)
        particle.data -= bg

        height = M_Height(particle, mask, 0)
        vol = M_Volume(particle, mask, 0)
        centerOfMass = M_CenterOfMass(particle, mask, 0)
        particleArea = M_Area(mask)
        particlePerim = M_Perimeter(mask)
        avgHeight = vol / particleArea if particleArea > 0 else 0

        # Check if the particle is in range
        if not (height > minH and height < maxH and
                particleArea > minA and particleArea < maxA and
                vol > minV and vol < maxV):
            continue

        # Accept the particle.
        info.data[i, 14] = count

        # Document the metrics in the wave note of each particle.
        particle.note = f"""Parent:{im.name}
Date:{Date()}
Height:{height}
Avg Height:{avgHeight}
Volume:{vol}
Area:{particleArea}
Perimeter:{particlePerim}
Scale:{info.data[i, 8]}
xCOM:{np.real(centerOfMass)}
yCOM:{np.imag(centerOfMass)}
pSeed:{info.data[i, 0]}
qSeed:{info.data[i, 1]}
rSeed:{info.data[i, 9]}
subPixelXCenter:{subPixX}
subPixelYCenter:{subPixY}"""

        # Make a folder for the particle and move it there
        particle_name = f"Particle_{count}"
        NewDataFolder(f"{particle_folder.name}:{particle_name}")
        subfolder = particle_folder.add_subfolder(particle_name)

        subfolder.add_wave(particle)
        subfolder.add_wave(mask)
        subfolder.add_wave(perim)

        # Store the metrics
        volumes.data[count] = vol
        heights.data[count] = height
        com.data[count, 0] = np.real(centerOfMass)
        com.data[count, 1] = np.imag(centerOfMass)
        areas.data[count] = particleArea
        avg_heights.data[count] = avgHeight

        count += 1

    return count


def CreateParticleMap(particle_map, particle_folder, count):
    """
    Create a map showing where particles are located
    """
    particle_map.data.fill(-1)

    for i in range(count):
        particle_name = f"Particle_{i}"
        if particle_name in particle_folder.subfolders:
            subfolder = particle_folder.subfolders[particle_name]
            if f"Mask_{i}" in subfolder.waves:
                mask = subfolder.waves[f"Mask_{i}"]

                # Map mask coordinates to particle map coordinates
                for ii in range(mask.data.shape[0]):
                    for jj in range(mask.data.shape[1]):
                        if mask.data[ii, jj]:
                            # Convert mask coordinates to particle map coordinates
                            x_coord = mask.scaling['x']['offset'] + ii * mask.scaling['x']['delta']
                            y_coord = mask.scaling['y']['offset'] + jj * mask.scaling['y']['delta']

                            map_i = ScaleToIndex(particle_map, x_coord, 0)
                            map_j = ScaleToIndex(particle_map, y_coord, 1)

                            if (0 <= map_i < particle_map.data.shape[0] and
                                    0 <= map_j < particle_map.data.shape[1]):
                                particle_map.data[map_i, map_j] = i


def DisplayResults(im, particle_folder, count):
    """
    Display the results with particles highlighted
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display original image
    extent = [DimOffset(im, 0),
              DimOffset(im, 0) + im.data.shape[1] * DimDelta(im, 0),
              DimOffset(im, 1),
              DimOffset(im, 1) + im.data.shape[0] * DimDelta(im, 1)]

    ax.imshow(im.data, cmap='gray', origin='lower', extent=extent)

    # Overlay particle boundaries
    for i in range(count):
        particle_name = f"Particle_{i}"
        if particle_name in particle_folder.subfolders:
            subfolder = particle_folder.subfolders[particle_name]
            if f"Mask_{i}" in subfolder.waves:
                mask = subfolder.waves[f"Mask_{i}"]

                # Create contour overlay
                mask_extent = [mask.scaling['x']['offset'],
                               mask.scaling['x']['offset'] + mask.data.shape[1] * mask.scaling['x']['delta'],
                               mask.scaling['y']['offset'],
                               mask.scaling['y']['offset'] + mask.data.shape[0] * mask.scaling['y']['delta']]

                ax.contour(mask.data, levels=[0.5], colors='red', linewidths=2,
                           extent=mask_extent, origin='lower')

    ax.set_title(f"Hessian Blob Detection Results ({count} particles)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


def GetHessianBlobParameters(scaleStart, layers, scaleFactor, detHResponseThresh,
                             particleType, subPixelMult, allowOverlap):
    """
    GUI for getting Hessian blob parameters
    """
    root = tk.Tk()
    root.title("Hessian Blob Parameters")
    root.geometry("500x400")

    # Create variables for parameters
    vars = {
        'scaleStart': tk.DoubleVar(value=scaleStart),
        'layers': tk.IntVar(value=layers),
        'scaleFactor': tk.DoubleVar(value=scaleFactor),
        'detHResponseThresh': tk.DoubleVar(value=detHResponseThresh),
        'particleType': tk.IntVar(value=particleType),
        'subPixelMult': tk.IntVar(value=subPixelMult),
        'allowOverlap': tk.IntVar(value=allowOverlap)
    }

    # Create input fields
    row = 0
    tk.Label(root, text="Minimum Size in Pixels").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['scaleStart'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Maximum Size in Pixels").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['layers'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Scaling Factor").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['scaleFactor'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Minimum Blob Strength (-2 for Interactive, -1 for Otsu's Method)").grid(row=row, column=0,
                                                                                                 sticky='w')
    tk.Entry(root, textvariable=vars['detHResponseThresh'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Particle Type (-1 for negative, +1 for positive, 0 for both)").grid(row=row, column=0,
                                                                                             sticky='w')
    tk.Entry(root, textvariable=vars['particleType'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Subpixel Ratio").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['subPixelMult'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Allow Hessian Blobs to Overlap? (1=yes 0=no)").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['allowOverlap'], width=20).grid(row=row, column=1)
    row += 1

    result = {'confirmed': False}

    def confirm():
        result['params'] = (
            vars['scaleStart'].get(),
            vars['layers'].get(),
            vars['scaleFactor'].get(),
            vars['detHResponseThresh'].get(),
            vars['particleType'].get(),
            vars['subPixelMult'].get(),
            vars['allowOverlap'].get()
        )
        result['confirmed'] = True
        root.destroy()

    def cancel():
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.grid(row=row + 1, column=0, columnspan=2, pady=20)

    tk.Button(button_frame, text="Continue", command=confirm).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=10)

    root.mainloop()

    if result['confirmed']:
        return result['params']
    else:
        return None


def GetParticleConstraints():
    """
    GUI for getting particle constraints
    """
    # Ask if user wants constraints
    root = tk.Tk()
    root.title("Particle Constraints")
    root.geometry("400x150")

    tk.Label(root, text="Would you like to limit the analysis to particles of certain\nheight, volume, or area?").pack(
        pady=20)

    result = {'response': None}

    def yes():
        result['response'] = True
        root.destroy()

    def no():
        result['response'] = False
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Yes", command=yes).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="No", command=no).pack(side=tk.LEFT, padx=10)

    root.mainloop()

    if result['response'] is None:
        return None
    elif not result['response']:
        return (-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)

    # Get constraint values
    root = tk.Tk()
    root.title("Constraints")
    root.geometry("400x300")

    vars = {
        'minH': tk.DoubleVar(value=0),
        'maxH': tk.DoubleVar(value=5e-9),
        'minA': tk.DoubleVar(value=-np.inf),
        'maxA': tk.DoubleVar(value=np.inf),
        'minV': tk.DoubleVar(value=-np.inf),
        'maxV': tk.DoubleVar(value=np.inf)
    }

    row = 0
    tk.Label(root, text="Minimum height in m").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['minH'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Maximum height in m").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['maxH'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Minimum area in m^2").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['minA'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Maximum area in m^2").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['maxA'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Minimum volume in m^3").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['minV'], width=20).grid(row=row, column=1)
    row += 1

    tk.Label(root, text="Maximum volume in m^3").grid(row=row, column=0, sticky='w')
    tk.Entry(root, textvariable=vars['maxV'], width=20).grid(row=row, column=1)
    row += 1

    constraint_result = {'confirmed': False}

    def confirm():
        constraint_result['constraints'] = (
            vars['minH'].get(),
            vars['maxH'].get(),
            vars['minV'].get(),
            vars['maxV'].get(),
            vars['minA'].get(),
            vars['maxA'].get()
        )
        constraint_result['confirmed'] = True
        root.destroy()

    def cancel():
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.grid(row=row + 1, column=0, columnspan=2, pady=20)

    tk.Button(button_frame, text="Continue", command=confirm).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=10)

    root.mainloop()

    if constraint_result['confirmed']:
        return constraint_result['constraints']
    else:
        return None