#!/usr/bin/env python3
"""
Hessian Blob Detector Validation System

This script generates test images and validates your Python implementation
against Igor Pro reference results.

USAGE:
1. Run this script to generate test images
2. Run Igor Pro on the test images (instructions will be provided)
3. Run this script again to validate results

Author: Claude (for Riley's Igor Pro port validation)
"""

# CRITICAL: Apply numpy monkey patch FIRST before any other imports
import numpy as np

if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Now safe to import everything else
import matplotlib.pyplot as plt
import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Try to import your implementation
try:
    import sys

    sys.path.append('.')
    from core.analysis import HessianBlobs  # Fixed: It's in core.analysis, not core.blob_detection
    from utils.data_manager import DataManager

    PYTHON_IMPLEMENTATION_AVAILABLE = True
    print("✓ Found your Python Hessian blob implementation")
except ImportError as e:
    PYTHON_IMPLEMENTATION_AVAILABLE = False
    print(f"⚠ Could not import your Python implementation: {e}")
    print("  This is OK for test generation phase")


class BlobTestGenerator:
    """Generates synthetic test images with known blob properties"""

    def __init__(self, output_dir: str = "validation_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_dir / "test_images").mkdir(exist_ok=True)
        (self.output_dir / "igor_results").mkdir(exist_ok=True)
        (self.output_dir / "python_results").mkdir(exist_ok=True)
        (self.output_dir / "comparison_reports").mkdir(exist_ok=True)

        self.test_metadata = {}

    def generate_synthetic_blob(self, x: float, y: float, sigma: float,
                                amplitude: float, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a single 2D Gaussian blob"""
        height, width = image_shape

        # Create coordinate arrays
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Calculate Gaussian
        blob = amplitude * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))

        return blob

    def create_test_image(self, image_id: str, width: int = 256, height: int = 256,
                          blobs: List[Dict] = None, noise_level: float = 0.0,
                          background_level: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """Create a test image with specified blobs"""

        if blobs is None:
            blobs = []

        # Initialize image
        image = np.full((height, width), background_level, dtype=np.float64)

        # Add each blob
        for i, blob_params in enumerate(blobs):
            blob = self.generate_synthetic_blob(
                x=blob_params['x'],
                y=blob_params['y'],
                sigma=blob_params['sigma'],
                amplitude=blob_params['amplitude'],
                image_shape=(height, width)
            )
            image += blob

        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image += noise

        # Ensure non-negative values
        image = np.maximum(image, 0)

        # Create metadata
        metadata = {
            'image_id': image_id,
            'width': width,
            'height': height,
            'blobs': blobs,
            'noise_level': noise_level,
            'background_level': background_level,
            'created_timestamp': datetime.datetime.now().isoformat(),
            'expected_blob_count': len(blobs)
        }

        return image, metadata

    def generate_test_suite(self) -> None:
        """Generate a comprehensive suite of test images"""

        print("🔧 Generating test image suite...")
        test_cases = []

        # Test Case 1: Single blob - easy case
        print("  📸 Test 1: Single centered blob")
        image, metadata = self.create_test_image(
            image_id="test_01_single_blob",
            blobs=[{'x': 128, 'y': 128, 'sigma': 5.0, 'amplitude': 100.0}]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 2: Multiple separated blobs
        print("  📸 Test 2: Multiple separated blobs")
        image, metadata = self.create_test_image(
            image_id="test_02_multiple_blobs",
            blobs=[
                {'x': 64, 'y': 64, 'sigma': 4.0, 'amplitude': 80.0},
                {'x': 192, 'y': 64, 'sigma': 6.0, 'amplitude': 120.0},
                {'x': 128, 'y': 192, 'sigma': 3.0, 'amplitude': 60.0}
            ]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 3: Boundary blobs (edge cases)
        print("  📸 Test 3: Boundary blobs")
        image, metadata = self.create_test_image(
            image_id="test_03_boundary_blobs",
            blobs=[
                {'x': 5, 'y': 5, 'sigma': 3.0, 'amplitude': 90.0},  # Top-left corner
                {'x': 250, 'y': 5, 'sigma': 4.0, 'amplitude': 85.0},  # Top-right corner
                {'x': 5, 'y': 250, 'sigma': 3.5, 'amplitude': 75.0},  # Bottom-left corner
                {'x': 250, 'y': 250, 'sigma': 4.5, 'amplitude': 95.0},  # Bottom-right corner
                {'x': 128, 'y': 2, 'sigma': 3.0, 'amplitude': 70.0},  # Top edge
                {'x': 2, 'y': 128, 'sigma': 3.0, 'amplitude': 80.0}  # Left edge
            ]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 4: Different scales
        print("  📸 Test 4: Different blob scales")
        image, metadata = self.create_test_image(
            image_id="test_04_different_scales",
            blobs=[
                {'x': 64, 'y': 64, 'sigma': 2.0, 'amplitude': 100.0},  # Small
                {'x': 128, 'y': 64, 'sigma': 5.0, 'amplitude': 100.0},  # Medium
                {'x': 192, 'y': 64, 'sigma': 8.0, 'amplitude': 100.0},  # Large
                {'x': 64, 'y': 128, 'sigma': 1.5, 'amplitude': 100.0},  # Very small
                {'x': 192, 'y': 128, 'sigma': 10.0, 'amplitude': 100.0}  # Very large
            ]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 5: Overlapping blobs
        print("  📸 Test 5: Overlapping blobs")
        image, metadata = self.create_test_image(
            image_id="test_05_overlapping_blobs",
            blobs=[
                {'x': 100, 'y': 100, 'sigma': 8.0, 'amplitude': 80.0},
                {'x': 110, 'y': 105, 'sigma': 6.0, 'amplitude': 70.0},  # Overlaps with first
                {'x': 150, 'y': 150, 'sigma': 5.0, 'amplitude': 90.0},
                {'x': 155, 'y': 155, 'sigma': 4.0, 'amplitude': 60.0}  # Overlaps with third
            ]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 6: With noise
        print("  📸 Test 6: Blobs with noise")
        image, metadata = self.create_test_image(
            image_id="test_06_with_noise",
            blobs=[
                {'x': 80, 'y': 80, 'sigma': 4.0, 'amplitude': 120.0},
                {'x': 180, 'y': 180, 'sigma': 6.0, 'amplitude': 100.0}
            ],
            noise_level=5.0
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 7: Different amplitudes
        print("  📸 Test 7: Different amplitudes")
        image, metadata = self.create_test_image(
            image_id="test_07_different_amplitudes",
            blobs=[
                {'x': 64, 'y': 128, 'sigma': 4.0, 'amplitude': 30.0},  # Weak
                {'x': 128, 'y': 128, 'sigma': 4.0, 'amplitude': 80.0},  # Medium
                {'x': 192, 'y': 128, 'sigma': 4.0, 'amplitude': 150.0}  # Strong
            ]
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Test Case 8: Empty image (negative control)
        print("  📸 Test 8: Empty image (should find no blobs)")
        image, metadata = self.create_test_image(
            image_id="test_08_empty_image",
            blobs=[],
            noise_level=2.0,
            background_level=5.0
        )
        self.save_test_image(image, metadata)
        test_cases.append(metadata)

        # Save master test metadata
        master_metadata = {
            'test_suite_version': '1.0',
            'total_test_cases': len(test_cases),
            'generated_timestamp': datetime.datetime.now().isoformat(),
            'test_cases': test_cases
        }

        with open(self.output_dir / "test_suite_metadata.json", 'w') as f:
            json.dump(master_metadata, f, indent=2)

        print(f"✅ Generated {len(test_cases)} test cases in: {self.output_dir}")
        print(f"📁 Test images saved to: {self.output_dir / 'test_images'}")

    def save_test_image(self, image: np.ndarray, metadata: Dict) -> None:
        """Save test image and metadata"""
        image_id = metadata['image_id']

        # Save image as TIFF (Igor Pro can read this)
        from PIL import Image

        # Convert to 16-bit for better precision
        image_16bit = (image * 65535 / np.max(image)).astype(np.uint16)
        pil_image = Image.fromarray(image_16bit)
        pil_image.save(self.output_dir / "test_images" / f"{image_id}.tiff")

        # Also save as numpy array for Python
        np.save(self.output_dir / "test_images" / f"{image_id}.npy", image)

        # Save metadata
        with open(self.output_dir / "test_images" / f"{image_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class ValidationRunner:
    """Runs validation comparisons between Igor and Python results"""

    def __init__(self, test_dir: str = "validation_tests"):
        self.test_dir = Path(test_dir)
        self.igor_dir = self.test_dir / "igor_results"
        self.python_dir = self.test_dir / "python_results"
        self.reports_dir = self.test_dir / "comparison_reports"

    def run_python_implementation(self) -> None:
        """Run Python implementation on all test images"""
        if not PYTHON_IMPLEMENTATION_AVAILABLE:
            print("❌ Python implementation not available. Cannot run validation.")
            return

        print("🐍 Running Python implementation on test images...")

        test_images_dir = self.test_dir / "test_images"

        # Standard parameters for validation (matching Igor defaults)
        validation_params = np.array([
            1,  # scaleStart (minimum size in pixels)
            120,  # layers (maximum size in pixels)
            1.5,  # scaleFactor
            -2,  # detHResponseThresh (-2 for interactive, but we'll handle this)
            1,  # particleType (1 for positive blobs)
            1,  # subPixelMult
            0,  # allowOverlap (0 = no overlap)
            -np.inf,  # minH (minimum height)
            np.inf,  # maxH (maximum height)
            -np.inf,  # minA (minimum area)
            np.inf,  # maxA (maximum area)
            -np.inf,  # minV (minimum volume)
            np.inf  # maxV (maximum volume)
        ])

        for image_file in test_images_dir.glob("*.npy"):
            if "_metadata" in image_file.name:
                continue

            print(f"  🔄 Processing {image_file.name}")

            try:
                # Load test image
                image = np.load(image_file)

                # Run Python implementation with standard parameters
                result_folder = HessianBlobs(image, params=validation_params)

                if result_folder:
                    # Copy results to python_results directory
                    import shutil
                    result_name = image_file.stem + "_results"
                    target_dir = self.python_dir / result_name

                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.copytree(result_folder, target_dir)

                    print(f"    ✅ Results saved to {target_dir}")
                else:
                    print(f"    ❌ Python implementation failed for {image_file.name}")

            except Exception as e:
                print(f"    ❌ Error processing {image_file.name}: {e}")

    def compare_results(self, test_image_id: str) -> Dict:
        """Compare Igor and Python results for a single test image"""

        igor_path = self.igor_dir / f"{test_image_id}_results"
        python_path = self.python_dir / f"{test_image_id}_results"

        if not igor_path.exists():
            return {"error": f"Igor results not found: {igor_path}"}
        if not python_path.exists():
            return {"error": f"Python results not found: {python_path}"}

        comparison = {
            "test_image_id": test_image_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "files_compared": [],
            "differences": {},
            "summary": {}
        }

        # Files to compare
        key_files = ["Heights.npy", "Volumes.npy", "Areas.npy", "AvgHeights.npy", "COM.npy"]

        for filename in key_files:
            igor_file = igor_path / filename
            python_file = python_path / filename

            if igor_file.exists() and python_file.exists():
                try:
                    igor_data = np.load(igor_file)
                    python_data = np.load(python_file)

                    comparison["files_compared"].append(filename)

                    # Compare arrays
                    diff_stats = self.compare_arrays(igor_data, python_data, filename)
                    comparison["differences"][filename] = diff_stats

                except Exception as e:
                    comparison["differences"][filename] = {"error": str(e)}
            else:
                missing = []
                if not igor_file.exists():
                    missing.append("igor")
                if not python_file.exists():
                    missing.append("python")
                comparison["differences"][filename] = {"missing": missing}

        # Generate summary statistics
        comparison["summary"] = self.generate_summary(comparison["differences"])

        return comparison

    def compare_arrays(self, igor_array: np.ndarray, python_array: np.ndarray,
                       array_name: str) -> Dict:
        """Compare two numpy arrays and return statistics"""

        stats = {
            "igor_shape": igor_array.shape,
            "python_shape": python_array.shape,
            "igor_count": igor_array.size,
            "python_count": python_array.size,
        }

        if igor_array.shape != python_array.shape:
            stats["shape_match"] = False
            stats["error"] = "Array shapes don't match"
            return stats

        stats["shape_match"] = True

        # Basic statistics
        stats["igor_mean"] = float(np.mean(igor_array))
        stats["python_mean"] = float(np.mean(python_array))
        stats["igor_std"] = float(np.std(igor_array))
        stats["python_std"] = float(np.std(python_array))

        # Difference statistics
        diff = igor_array - python_array
        stats["max_absolute_difference"] = float(np.max(np.abs(diff)))
        stats["mean_absolute_difference"] = float(np.mean(np.abs(diff)))
        stats["rms_difference"] = float(np.sqrt(np.mean(diff ** 2)))

        # Relative differences (avoid division by zero)
        igor_nonzero = igor_array[igor_array != 0]
        python_nonzero = python_array[igor_array != 0]
        if len(igor_nonzero) > 0:
            rel_diff = np.abs((igor_nonzero - python_nonzero) / igor_nonzero)
            stats["max_relative_difference"] = float(np.max(rel_diff))
            stats["mean_relative_difference"] = float(np.mean(rel_diff))

        # Correlation
        if len(igor_array.flatten()) > 1:
            correlation = np.corrcoef(igor_array.flatten(), python_array.flatten())[0, 1]
            stats["correlation"] = float(correlation) if not np.isnan(correlation) else 0.0

        # Agreement assessment
        stats["arrays_identical"] = np.array_equal(igor_array, python_array)
        stats["arrays_close"] = np.allclose(igor_array, python_array, rtol=1e-10, atol=1e-10)
        stats["arrays_close_relaxed"] = np.allclose(igor_array, python_array, rtol=1e-6, atol=1e-6)

        return stats

    def generate_summary(self, differences: Dict) -> Dict:
        """Generate overall summary of comparison"""

        summary = {
            "total_files_compared": len(differences),
            "files_with_errors": 0,
            "files_identical": 0,
            "files_close": 0,
            "files_different": 0,
            "overall_agreement": "unknown"
        }

        for filename, diff_stats in differences.items():
            if "error" in diff_stats:
                summary["files_with_errors"] += 1
            elif diff_stats.get("arrays_identical", False):
                summary["files_identical"] += 1
            elif diff_stats.get("arrays_close_relaxed", False):
                summary["files_close"] += 1
            else:
                summary["files_different"] += 1

        # Overall assessment
        total_valid = summary["total_files_compared"] - summary["files_with_errors"]
        if total_valid == 0:
            summary["overall_agreement"] = "no_valid_comparisons"
        elif summary["files_identical"] == total_valid:
            summary["overall_agreement"] = "perfect"
        elif summary["files_identical"] + summary["files_close"] == total_valid:
            summary["overall_agreement"] = "excellent"
        elif summary["files_different"] <= total_valid // 2:
            summary["overall_agreement"] = "good"
        else:
            summary["overall_agreement"] = "poor"

        return summary

    def run_full_validation(self) -> None:
        """Run complete validation on all test cases"""
        print("🔍 Running full validation comparison...")

        # Load test metadata
        metadata_file = self.test_dir / "test_suite_metadata.json"
        if not metadata_file.exists():
            print("❌ Test suite metadata not found. Run test generation first.")
            return

        with open(metadata_file, 'r') as f:
            test_metadata = json.load(f)

        all_comparisons = []

        for test_case in test_metadata["test_cases"]:
            test_id = test_case["image_id"]
            print(f"  🔄 Comparing {test_id}")

            comparison = self.compare_results(test_id)
            all_comparisons.append(comparison)

            # Save individual comparison report
            report_file = self.reports_dir / f"{test_id}_comparison.json"
            with open(report_file, 'w') as f:
                json.dump(comparison, f, indent=2)

        # Generate master validation report
        master_report = {
            "validation_timestamp": datetime.datetime.now().isoformat(),
            "test_suite_info": test_metadata,
            "individual_comparisons": all_comparisons,
            "overall_summary": self.generate_overall_summary(all_comparisons)
        }

        master_file = self.reports_dir / "master_validation_report.json"
        with open(master_file, 'w') as f:
            json.dump(master_report, f, indent=2)

        print(f"✅ Validation complete! Master report: {master_file}")
        self.print_validation_summary(master_report["overall_summary"])

    def generate_overall_summary(self, comparisons: List[Dict]) -> Dict:
        """Generate summary across all test cases"""

        overall = {
            "total_test_cases": len(comparisons),
            "successful_comparisons": 0,
            "failed_comparisons": 0,
            "perfect_matches": 0,
            "close_matches": 0,
            "poor_matches": 0,
            "agreement_by_file": {}
        }

        for comparison in comparisons:
            if "error" in comparison:
                overall["failed_comparisons"] += 1
                continue

            overall["successful_comparisons"] += 1

            summary = comparison.get("summary", {})
            agreement = summary.get("overall_agreement", "unknown")

            if agreement == "perfect":
                overall["perfect_matches"] += 1
            elif agreement in ["excellent", "good"]:
                overall["close_matches"] += 1
            else:
                overall["poor_matches"] += 1

        return overall

    def print_validation_summary(self, summary: Dict) -> None:
        """Print a nice summary of validation results"""
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total test cases: {summary['total_test_cases']}")
        print(f"Successful comparisons: {summary['successful_comparisons']}")
        print(f"Failed comparisons: {summary['failed_comparisons']}")
        print(f"Perfect matches: {summary['perfect_matches']}")
        print(f"Close matches: {summary['close_matches']}")
        print(f"Poor matches: {summary['poor_matches']}")

        if summary['successful_comparisons'] > 0:
            success_rate = summary['perfect_matches'] + summary['close_matches']
            success_percentage = (success_rate / summary['successful_comparisons']) * 100
            print(f"\n🎉 Overall success rate: {success_percentage:.1f}%")

        print("=" * 60)


def main():
    """Main function - handles command line interface"""

    print("🔬 HESSIAN BLOB VALIDATION SYSTEM")
    print("=" * 50)

    # Create generator and validator instances
    generator = BlobTestGenerator()
    validator = ValidationRunner()

    while True:
        print("\n📋 MENU:")
        print("1. Generate test images")
        print("2. Run Python implementation on test images")
        print("3. Run full validation (compare Igor vs Python)")
        print("4. View validation reports")
        print("5. Exit")

        choice = input("\n👉 Enter your choice (1-5): ").strip()

        if choice == "1":
            print("\n🔧 Generating test images...")
            generator.generate_test_suite()
            print("\n📢 NEXT STEPS FOR YOU:")
            print("   1. Go to validation_tests/test_images/")
            print("   2. Open Igor Pro")
            print("   3. Load each .tiff file")
            print("   4. Run HessianBlobs() on each image")
            print("   5. Copy the results to validation_tests/igor_results/")
            print("   6. Name each result folder as: [test_name]_results")
            print("   (e.g., test_01_single_blob_results)")

        elif choice == "2":
            print("\n🐍 Running Python implementation...")
            validator.run_python_implementation()

        elif choice == "3":
            print("\n🔍 Running validation comparison...")
            validator.run_full_validation()

        elif choice == "4":
            reports_dir = Path("validation_tests/comparison_reports")
            if reports_dir.exists():
                print(f"\n📊 Reports available in: {reports_dir}")
                for report_file in reports_dir.glob("*.json"):
                    print(f"   📄 {report_file.name}")
            else:
                print("\n❌ No reports found. Run validation first.")

        elif choice == "5":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()