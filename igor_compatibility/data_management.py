# ========================================================================
# igor_compatibility/data_management.py
# ========================================================================

"""
Data Management

Igor Pro-compatible data folder management and file I/O operations.
Maintains the exact Igor Pro workflow and folder structure.
"""

import os
import numpy as np
import json
import datetime
from typing import Dict, Optional
try:
    import igor.binarywave as bw
except ImportError:
    bw = None
from PIL import Image
from core.error_handling import handle_error, safe_print, HessianBlobError

class DataManager:
    """Manages file I/O and data folder organization"""

    @staticmethod
    def create_igor_folder_structure(base_path: str, folder_type: str = "particles") -> str:
        """Create Igor Pro-compatible folder structure"""
        try:
            os.makedirs(base_path, exist_ok=True)

            if folder_type == "particles":
                # Create standard particle analysis structure
                subdirs = []
            elif folder_type == "series":
                # Series analysis doesn't need subdirs initially
                subdirs = []

            for subdir in subdirs:
                os.makedirs(os.path.join(base_path, subdir), exist_ok=True)

            safe_print(f"Created Igor Pro folder structure: {base_path}")
            return base_path

        except Exception as e:
            raise HessianBlobError(f"Failed to create folder structure: {e}")

    @staticmethod
    def save_wave_data(data: np.ndarray, filepath: str, wave_info: Dict = None) -> bool:
        """Save wave data in Igor Pro-compatible format"""
        try:
            # Save as .npy for Python compatibility
            np.save(filepath, data)

            # Save metadata if provided
            if wave_info:
                metadata_file = filepath.replace('.npy', '_info.json')
                with open(metadata_file, 'w') as f:
                    json.dump(wave_info, f, indent=2, default=str)

            return True

        except Exception as e:
            handle_error("save_wave_data", e, f"file: {filepath}")
            return False

    @staticmethod
    def load_image_file(filepath: str) -> Optional[np.ndarray]:
        """Load image from various formats"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            file_ext = os.path.splitext(filepath)[1].lower()

            if file_ext == '.ibw' and bw is not None:
                wave = bw.load(filepath)
                image_data = wave['wave']['wData']
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            elif file_ext == '.npy':
                image_data = np.load(filepath)
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            elif file_ext in ['.tiff', '.tif', '.png', '.jpg', '.jpeg']:
                img = Image.open(filepath)
                image_data = np.array(img)
                if not image_data.flags.writeable:
                    image_data = image_data.copy()
                return image_data
            else:
                raise HessianBlobError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            handle_error("load_image_file", e, f"file: {filepath}")
            return None

    @staticmethod
    def create_particle_info(particle_data: Dict, particle_id: int) -> Dict:
        """Create particle information dictionary matching Igor format"""
        info = {
            'Parent': particle_data.get('parent', 'image'),
            'Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Height': float(particle_data.get('height', 0)),
            'Avg Height': float(particle_data.get('avg_height', 0)),
            'Volume': float(particle_data.get('volume', 0)),
            'Area': float(particle_data.get('area', 0)),
            'Perimeter': float(particle_data.get('perimeter', 0)),
            'Scale': float(particle_data.get('scale', 0)),
            'xCOM': float(particle_data.get('com', [0, 0])[0]),
            'yCOM': float(particle_data.get('com', [0, 0])[1]),
            'pSeed': int(particle_data.get('p_seed', 0)),
            'qSeed': int(particle_data.get('q_seed', 0)),
            'rSeed': int(particle_data.get('r_seed', 0)),
            'subPixelXCenter': float(particle_data.get('p_seed', 0)),
            'subPixelYCenter': float(particle_data.get('q_seed', 0))
        }
        return info