import numpy as np
import os
from pathlib import Path
from igor_io import load_ibw_file
from hessian_blobs import batch_hessian_blobs, hessian_blobs
from preprocessing import batch_preprocess
from utilities import view_particles, DataFolder, WaveNote, CoordinateSystem
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path):
    """
    Load all IBW files from a folder as images with proper coordinate systems

    Parameters:
    folder_path (str or Path): Path to the folder containing IBW files

    Returns:
    tuple: (images dict, coordinate_systems dict)
    """
    folder_path = Path(folder_path)
    images = {}
    coord_systems = {}

    # Look for IBW files
    for file_path in folder_path.glob('*.ibw'):
        try:
            # Load the image data and info from the IBW file
            image_data, wave_info = load_ibw_file(file_path)

            # Create coordinate system
            coord_system = CoordinateSystem(
                image_data.shape,
                x_start=wave_info.get('x_start', 0),
                x_delta=wave_info.get('x_delta', 1),
                y_start=wave_info.get('y_start', 0),
                y_delta=wave_info.get('y_delta', 1)
            )

            images[file_path.stem] = image_data
            coord_systems[file_path.stem] = coord_system

            print(f"Loaded {file_path.name}: shape={wave_info['shape']}, dtype={wave_info['data_type']}")
            print(
                f"  Coordinate system: x=[{coord_system.x_start}, {coord_system.x_start + coord_system.x_delta * image_data.shape[1]}], y=[{coord_system.y_start}, {coord_system.y_start + coord_system.y_delta * image_data.shape[0]}]")

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    if not images:
        print(f"No IBW files found in {folder_path}")
        # Also check for other image formats as fallback
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            for file_path in folder_path.glob(ext):
                print(f"Found {ext[2:]} file: {file_path.name} (not supported - use IBW format)")

    return images, coord_systems


def display_results(results):
    """Display analysis results with visualization"""
    if not results or 'image_results' not in results:
        print("No results to display")
        return

    # Summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total particles detected: {len(results['all_heights'])}")

    if len(results['all_heights']) > 0:
        print(f"\nHeight statistics:")
        print(f"  Mean: {np.mean(results['all_heights']):.4f}")
        print(f"  Std: {np.std(results['all_heights']):.4f}")
        print(f"  Min: {np.min(results['all_heights']):.4f}")
        print(f"  Max: {np.max(results['all_heights']):.4f}")

        print(f"\nArea statistics:")
        print(f"  Mean: {np.mean(results['all_areas']):.4f}")
        print(f"  Std: {np.std(results['all_areas']):.4f}")

        print(f"\nVolume statistics:")
        print(f"  Mean: {np.mean(results['all_volumes']):.4f}")
        print(f"  Std: {np.std(results['all_volumes']):.4f}")

    # Display images with detected particles
    for image_name, image_result in results['image_results'].items():
        if 'particles' not in image_result:
            continue

        print(f"\nImage: {image_name}")
        print(f"  Particles found: {len(image_result['particles'])}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        data_folder = image_result['data_folder']
        original, _ = data_folder.load_wave("Original")
        if original is not None:
            ax1.imshow(original, cmap='gray')
            ax1.set_title(f'{image_name} - Original')
            ax1.axis('off')

        # Particle map
        particle_map, _ = data_folder.load_wave("ParticleMap")
        if particle_map is not None:
            # Create colored particle map
            masked_map = np.ma.masked_where(particle_map == -1, particle_map)
            ax2.imshow(original, cmap='gray', alpha=0.7)
            im = ax2.imshow(masked_map, cmap='jet', alpha=0.5, interpolation='nearest')
            ax2.set_title(f'{image_name} - Detected Particles')
            ax2.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Particle ID')

        plt.tight_layout()
        plt.show()


def main():
    """Main function demonstrating the Hessian blob detection workflow"""

    print("=" * 60)
    print("HESSIAN BLOB PARTICLE DETECTION")
    print("Python Implementation")
    print("=" * 60)

    # Get data folder from user
    data_folder_path = input("\nEnter path to folder containing IBW files (or press Enter for './data/images'): ")
    if not data_folder_path:
        data_folder_path = "./data/images"

    data_folder = Path(data_folder_path)
    if not data_folder.exists():
        print(f"Error: Folder {data_folder} does not exist!")
        return

    # Load images
    print(f"\nLoading images from {data_folder}...")
    images, coord_systems = load_images_from_folder(data_folder)

    if not images:
        print("No images found. Please ensure you have IBW files in the specified folder.")
        return

    print(f"\nLoaded {len(images)} images")

    # Create a duplicate folder for preprocessing
    duplicated_images = {}
    for name, image in images.items():
        duplicated_images[name + "_dup"] = image.copy()

    # Optional: Preprocess images
    preprocess = input("\nDo you want to preprocess the images? (y/n): ").lower() == 'y'

    if preprocess:
        print("\nPreprocessing Parameters:")
        streak_removal_sdevs = float(input("Streak removal standard deviations (0 to skip): ") or "0")
        flatten_order = int(input("Polynomial order for flattening (0 to skip): ") or "0")

        if streak_removal_sdevs > 0 or flatten_order > 0:
            print("\nPreprocessing images...")
            duplicated_images = batch_preprocess(duplicated_images,
                                                 streak_removal_sdevs,
                                                 flatten_order)
            # Use preprocessed images
            images = duplicated_images

    # Run batch Hessian blob detection
    print("\n" + "-" * 60)
    print("Running Batch Hessian Blob Detection")
    print("-" * 60)

    try:
        results = batch_hessian_blobs(images)

        if results:
            # Display results
            display_results(results)

            # Create histograms
            create_histograms = input("\nCreate histograms of measurements? (y/n): ").lower() == 'y'
            if create_histograms and len(results['all_heights']) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Height histogram
                axes[0, 0].hist(results['all_heights'], bins=30, alpha=0.7, color='blue')
                axes[0, 0].set_xlabel('Height')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Particle Height Distribution')

                # Area histogram
                axes[0, 1].hist(results['all_areas'], bins=30, alpha=0.7, color='green')
                axes[0, 1].set_xlabel('Area')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Particle Area Distribution')

                # Volume histogram
                axes[1, 0].hist(results['all_volumes'], bins=30, alpha=0.7, color='red')
                axes[1, 0].set_xlabel('Volume')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Particle Volume Distribution')

                # Height vs Area scatter
                axes[1, 1].scatter(results['all_areas'], results['all_heights'], alpha=0.5)
                axes[1, 1].set_xlabel('Area')
                axes[1, 1].set_ylabel('Height')
                axes[1, 1].set_title('Height vs Area')

                plt.tight_layout()
                plt.show()

            # View individual particles
            view = input("\nDo you want to view individual particles? (y/n): ").lower() == 'y'
            if view and results['image_results']:
                # Let user choose which image
                image_names = list(results['image_results'].keys())
                print("\nAvailable images:")
                for i, name in enumerate(image_names):
                    print(f"  {i}: {name}")

                choice = input(f"Select image (0-{len(image_names) - 1}): ")
                try:
                    idx = int(choice)
                    if 0 <= idx < len(image_names):
                        selected_name = image_names[idx]
                        selected_results = results['image_results'][selected_name]

                        print(f"\nViewing particles from {selected_name}")
                        print("Use arrow keys to navigate, space/down to delete")
                        view_particles(selected_results)
                except:
                    print("Invalid selection")

            # Save results location
            print(f"\nResults saved to: {results['series_folder'].get_path()}")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()