import numpy as np

if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'object'):
    np.object = object

import struct
import os

# If you have igor2 installed via pip
try:
    from igor2 import binarywave

    HAS_IGOR2 = True
except ImportError:
    HAS_IGOR2 = False
    # We'll implement a basic IBW reader below


def load_ibw_file(filepath):
    """
    Load an IBW (Igor Binary Wave) file and return the image data and scaling info.

    Parameters:
    filepath (str): Path to the IBW file

    Returns:
    tuple: (image_data, wave_info) where wave_info contains scaling information
    """

    if HAS_IGOR2:
        # Use igor2 library if available
        try:
            # Load the IBW file
            data = binarywave.load(filepath)

            # Extract wave data
            wave = data['wave']['wData']

            # Extract wave header information
            wave_header = data['wave']['wave_header']

            # Get scaling information
            sfA = wave_header.get('sfA', [0] * 4)  # offsets
            sfB = wave_header.get('sfB', [1] * 4)  # deltas (scaling)

            # Build wave info dictionary
            wave_info = {
                'dimensions': list(wave.shape),
                'shape': wave.shape,
                'data_type': str(wave.dtype),
                'note': '',
                'x_start': sfA[0] if len(sfA) > 0 else 0,
                'x_delta': sfB[0] if len(sfB) > 0 else 1,
                'y_start': sfA[1] if len(sfA) > 1 else 0,
                'y_delta': sfB[1] if len(sfB) > 1 else 1,
            }

            # Get wave note if available
            if 'wave' in data and 'note' in data['wave']:
                note = data['wave']['note']
                if isinstance(note, bytes):
                    wave_info['note'] = note.decode('utf-8', errors='ignore')
                else:
                    wave_info['note'] = str(note)

            # Handle different dimensional data
            if wave.ndim == 1:
                raise ValueError(f"IBW file contains 1D data, not an image: {filepath}")
            elif wave.ndim == 2:
                # 2D data is what we expect for images
                return wave, wave_info
            elif wave.ndim == 3:
                # For 3D data (like multi-layer images), take the first layer
                # This matches Igor Pro behavior where im[p][q][0] is used
                print(f"Warning: 3D data detected in {filepath}, using first layer")
                return wave[:, :, 0], wave_info
            else:
                raise ValueError(f"Unsupported data dimensions in IBW file: {wave.ndim}D")

        except Exception as e:
            raise IOError(f"Failed to load IBW file {filepath}: {str(e)}")

    else:
        # Fallback: Basic IBW reader implementation
        return _load_ibw_basic(filepath)


def _load_ibw_basic(filepath):
    """
    Basic IBW file reader (simplified, may not support all IBW versions)
    """
    with open(filepath, 'rb') as f:
        # Read magic number and version
        magic = f.read(4)
        if magic not in [b'IGOR', b'ROGI']:
            raise ValueError(f"Not a valid IBW file: {filepath}")

        version = struct.unpack('<h', f.read(2))[0]

        if version == 5:
            # IBW version 5 (most common for modern Igor)
            return _read_ibw_v5(f, filepath)
        else:
            raise ValueError(f"Unsupported IBW version {version}. Please install igor2 package: pip install igor")


def _read_ibw_v5(f, filepath):
    """Read IBW version 5 file"""
    # This is a simplified reader - for full support, use igor2 package

    # Skip to wave header (after version info)
    f.seek(8)

    # Read creation date and mod date
    f.read(8)  # Skip dates

    # Read data type
    data_type = struct.unpack('<h', f.read(2))[0]

    # Skip some fields
    f.seek(48)

    # Read dimensions
    dims = struct.unpack('<4i', f.read(16))

    # Read scaling factors
    sfA = struct.unpack('<4d', f.read(32))  # offsets
    sfB = struct.unpack('<4d', f.read(32))  # deltas

    # Skip to data offset
    f.seek(384)

    # Determine numpy dtype from Igor data type
    dtype_map = {
        2: np.float32,
        4: np.float64,
        8: np.int8,
        16: np.int16,
        32: np.int32,
    }

    if data_type not in dtype_map:
        raise ValueError(f"Unsupported data type: {data_type}")

    dtype = dtype_map[data_type]

    # Calculate data size
    n_dims = sum(1 for d in dims if d > 0)
    shape = [d for d in dims if d > 0][::-1]  # Igor uses column-major order

    if n_dims < 2:
        raise ValueError(f"IBW file contains {n_dims}D data, not an image")

    # Read data
    n_points = np.prod(shape)
    data = np.frombuffer(f.read(n_points * dtype().itemsize), dtype=dtype)

    # Reshape data
    if n_dims == 2:
        wave = data.reshape(shape[1], shape[0]).T
    elif n_dims == 3:
        wave = data.reshape(shape[2], shape[1], shape[0]).transpose(2, 1, 0)
        print(f"Warning: 3D data detected in {filepath}, using first layer")
        wave = wave[:, :, 0]
    else:
        raise ValueError(f"Unsupported dimensions: {n_dims}D")

    # Build wave info
    wave_info = {
        'dimensions': list(wave.shape),
        'shape': wave.shape,
        'data_type': str(dtype),
        'note': '',
        'x_start': sfA[0],
        'x_delta': sfB[0],
        'y_start': sfA[1],
        'y_delta': sfB[1],
    }

    return wave, wave_info


def save_ibw_file(data, filepath, wave_name=None, x_start=0, x_delta=1, y_start=0, y_delta=1, note=''):
    """
    Save data as an IBW file.

    For now, this is not implemented. Use numpy save or other formats.
    """
    raise NotImplementedError("Saving IBW files is not yet implemented. Use numpy.save() or export to other formats.")


# Alternative: If you can't get igor2 to work, you can use igor.py package
try:
    import igor.binarywave as igor_bw


    def load_ibw_file_igor_py(filepath):
        """Alternative loader using igor.py package"""
        data = igor_bw.load(filepath)

        wave = data['wave']['wData']
        wave_header = data['wave']['wave_header']

        # Extract scaling
        x_scale = wave_header.get('xScale', 1.0)
        y_scale = wave_header.get('yScale', 1.0)
        x_offset = wave_header.get('xOffset', 0.0)
        y_offset = wave_header.get('yOffset', 0.0)

        wave_info = {
            'dimensions': list(wave.shape),
            'shape': wave.shape,
            'data_type': str(wave.dtype),
            'note': data['wave'].get('note', '').decode('utf-8', errors='ignore') if isinstance(
                data['wave'].get('note', ''), bytes) else '',
            'x_start': x_offset,
            'x_delta': x_scale,
            'y_start': y_offset,
            'y_delta': y_scale,
        }

        if wave.ndim == 3:
            wave = wave[:, :, 0]

        return wave, wave_info

except ImportError:
    pass