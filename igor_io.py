import numpy as np
from igor2 import binarywave


def load_ibw_file(filepath):
    """
    Load an IBW (Igor Binary Wave) file and return the image data and scaling info.

    Parameters:
    filepath (str): Path to the IBW file

    Returns:
    tuple: (image_data, wave_info) where wave_info contains scaling information
    """
    try:
        # Load the IBW file
        data = binarywave.load(filepath)
        wave = data['wave']['wData']

        # Extract wave header information
        wave_header = data['wave']['wave_header']

        # Build wave info dictionary
        wave_info = {
            'dimensions': wave_header.get('nDim', [len(s) for s in wave.shape]),
            'shape': wave.shape,
            'data_type': wave.dtype,
            'note': data['wave'].get('note', b'').decode('utf-8', errors='ignore') if isinstance(
                data['wave'].get('note', b''), bytes) else str(data['wave'].get('note', '')),
            # Get scaling information - Igor uses sfA for offset and sfB for delta
            'x_start': wave_header.get('sfA', [0, 0])[0],
            'x_delta': wave_header.get('sfB', [1, 1])[0],
            'y_start': wave_header.get('sfA', [0, 0])[1] if len(wave_header.get('sfA', [0, 0])) > 1 else 0,
            'y_delta': wave_header.get('sfB', [1, 1])[1] if len(wave_header.get('sfB', [1, 1])) > 1 else 1,
        }

        # Handle different dimensional data
        if wave.ndim == 1:
            # For 1D data, we can't use it as an image
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


def save_ibw_file(data, filepath, wave_name=None, x_start=0, x_delta=1, y_start=0, y_delta=1, note=''):
    """
    Save data as an IBW file.

    Parameters:
    data (np.ndarray): The data to save
    filepath (str): Path where to save the IBW file
    wave_name (str): Name of the wave (optional)
    x_start, x_delta, y_start, y_delta: Scaling parameters
    note (str): Note to attach to the wave
    """
    # This would require implementing IBW writing functionality
    # For now, raise NotImplementedError
    raise NotImplementedError("Saving IBW files is not yet implemented")