"""
Igor Pro Compatibility Layer
Recreates Igor Pro native functions to maintain 1-1 code compatibility
Complete implementation with all necessary functions for blob detection
Fixed version with missing functions added
"""

import numpy as np
from scipy import ndimage, fft
from scipy.optimize import curve_fit
import warnings
import tkinter as tk

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex


class Wave:
    """
    Mimics Igor Pro wave structure with complete functionality
    """

    def __init__(self, data=None, name="", note=""):
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data, dtype=np.float64)
        self.name = str(name)
        self.note = str(note)

        # Scaling information for each dimension
        self.scaling = {
            'x': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            'y': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            'z': {'offset': 0.0, 'delta': 1.0, 'units': ''},
            't': {'offset': 0.0, 'delta': 1.0, 'units': ''}
        }

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Wave '{self.name}': shape={self.data.shape}, dtype={self.data.dtype}"

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def SetScale(self, axis, offset, delta, units=""):
        """Set scaling for wave dimension - matches Igor Pro SetScale"""
        axis_key = axis.lower()
        if axis_key in self.scaling:
            self.scaling[axis_key] = {
                'offset': float(offset),
                'delta': float(delta),
                'units': str(units)
            }

    def GetScale(self, axis):
        """Get scaling information for a dimension"""
        axis_key = axis.lower()
        if axis_key in self.scaling:
            return self.scaling[axis_key]
        return {'offset': 0.0, 'delta': 1.0, 'units': ''}

    def copy(self):
        """Create a deep copy of the wave"""
        new_wave = Wave(self.data.copy(), self.name + "_copy", self.note)
        new_wave.scaling = self.scaling.copy()
        return new_wave

    def save(self, filename):
        """Save wave data to file"""
        np.save(filename, self.data)


# Wave dimension and scaling functions
def DimSize(wave, dimension):
    """Igor DimSize function equivalent"""
    if dimension < len(wave.data.shape):
        return wave.data.shape[dimension]
    return 0


def DimOffset(wave, dimension):
    """Igor DimOffset function equivalent"""
    axis_map = {0: 'x', 1: 'y', 2: 'z', 3: 't'}
    if dimension in axis_map:
        return wave.GetScale(axis_map[dimension])['offset']
    return 0.0


def DimDelta(wave, dimension):
    """Igor DimDelta function equivalent"""
    axis_map = {0: 'x', 1: 'y', 2: 'z', 3: 't'}
    if dimension in axis_map:
        return wave.GetScale(axis_map[dimension])['delta']
    return 1.0


def SetScale(wave, axis, offset, delta, units=""):
    """Igor SetScale function equivalent"""
    wave.SetScale(axis, offset, delta, units)


# Wave creation and manipulation functions
def Make(wave_name, n_points, x_start=0, x_delta=1):
    """Igor Make function equivalent"""
    data = np.zeros(n_points)
    wave = Wave(data, wave_name)
    wave.SetScale('x', x_start, x_delta)
    return wave


def Duplicate(source_wave, dest_name=None):
    """Igor Duplicate function equivalent"""
    if dest_name is None:
        dest_name = source_wave.name + "_copy"

    new_wave = Wave(source_wave.data.copy(), dest_name, source_wave.note)
    # Copy all scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


def Redimension(wave, *dimensions):
    """Igor Redimension function equivalent"""
    if len(dimensions) == 1 and hasattr(dimensions[0], '__iter__'):
        dimensions = dimensions[0]
    wave.data = np.reshape(wave.data, dimensions)


def KillWaves(*wave_names):
    """Igor KillWaves function equivalent (placeholder)"""
    # In Igor, this would remove waves from memory
    # In Python, we'll just pass since garbage collection handles this
    pass


def Concatenate(dest_wave, *source_waves):
    """Igor Concatenate function equivalent"""
    arrays = [dest_wave.data] + [w.data for w in source_waves]
    dest_wave.data = np.concatenate(arrays)


# Statistical functions
def WaveMax(wave):
    """Igor WaveMax function equivalent"""
    return np.nanmax(wave.data)


def WaveMin(wave):
    """Igor WaveMin function equivalent"""
    return np.nanmin(wave.data)


def WaveStats(wave):
    """Igor WaveStats function equivalent - returns dict of statistics"""
    data = wave.data.flatten()
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return {
            'numpoints': 0,
            'numinfs': 0,
            'numnans': len(data),
            'avg': np.nan,
            'sum': np.nan,
            'sdev': np.nan,
            'sem': np.nan,
            'rms': np.nan,
            'min': np.nan,
            'max': np.nan
        }

    return {
        'numpoints': len(valid_data),
        'numinfs': np.sum(np.isinf(data)),
        'numnans': np.sum(np.isnan(data)),
        'avg': np.mean(valid_data),
        'sum': np.sum(valid_data),
        'sdev': np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0,
        'sem': np.std(valid_data, ddof=1) / np.sqrt(len(valid_data)) if len(valid_data) > 1 else 0,
        'rms': np.sqrt(np.mean(valid_data ** 2)),
        'min': np.min(valid_data),
        'max': np.max(valid_data)
    }


# Mathematical functions
def Sqrt(x):
    """Igor sqrt function equivalent"""
    return np.sqrt(x)


def Exp(x):
    """Igor exp function equivalent"""
    return np.exp(x)


def Log(x):
    """Igor log function equivalent (natural log)"""
    return np.log(x)


def Ln(x):
    """Igor ln function equivalent (natural log)"""
    return np.log(x)


def Log10(x):
    """Igor log10 function equivalent"""
    return np.log10(x)


def Sin(x):
    """Igor sin function equivalent"""
    return np.sin(x)


def Cos(x):
    """Igor cos function equivalent"""
    return np.cos(x)


def Tan(x):
    """Igor tan function equivalent"""
    return np.tan(x)


def ASin(x):
    """Igor asin function equivalent"""
    return np.arcsin(x)


def ACos(x):
    """Igor acos function equivalent"""
    return np.arccos(x)


def ATan(x):
    """Igor atan function equivalent"""
    return np.arctan(x)


def ATan2(y, x):
    """Igor atan2 function equivalent"""
    return np.arctan2(y, x)


def Abs(x):
    """Igor abs function equivalent"""
    return np.abs(x)


def Sign(x):
    """Igor sign function equivalent"""
    return np.sign(x)


def Max(*args):
    """Igor max function equivalent"""
    if len(args) == 1:
        return np.nanmax(args[0])
    else:
        return np.maximum.reduce([np.array(arg) for arg in args])


def Min(*args):
    """Igor min function equivalent"""
    if len(args) == 1:
        return np.nanmin(args[0])
    else:
        return np.minimum.reduce([np.array(arg) for arg in args])


def Ceil(x):
    """Igor ceil function equivalent"""
    return np.ceil(x)


def Floor(x):
    """Igor floor function equivalent"""
    return np.floor(x)


def Round(x):
    """Igor round function equivalent"""
    return np.round(x)


def Trunc(x):
    """Igor trunc function equivalent"""
    return np.trunc(x)


def Mod(x, y):
    """Igor mod function equivalent"""
    return np.mod(x, y)


def SelectNumber(condition, false_value, true_value):
    """Igor SelectNumber function equivalent"""
    return np.where(condition, true_value, false_value)


# Interpolation functions
def Interp(x_wave, y_wave, x_val):
    """Igor Interp function equivalent"""
    return np.interp(x_val, x_wave.data, y_wave.data)


def Interp2D(wave, x, y):
    """Igor Interp2D function equivalent"""
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate arrays based on wave scaling
    if len(wave.data.shape) >= 2:
        x_coords = np.arange(wave.data.shape[1]) * DimDelta(wave, 0) + DimOffset(wave, 0)
        y_coords = np.arange(wave.data.shape[0]) * DimDelta(wave, 1) + DimOffset(wave, 1)

        # Note: RegularGridInterpolator expects (y, x) order
        interpolator = RegularGridInterpolator((y_coords, x_coords), wave.data,
                                               bounds_error=False, fill_value=0)

        return interpolator([y, x])[0]
    else:
        return 0


def BilinearInterpolate(wave, x, y, z=0):
    """Bilinear interpolation function matching Igor Pro implementation"""
    # Convert coordinates to indices
    p_mid = (x - DimOffset(wave, 0)) / DimDelta(wave, 0)
    p0 = max(0, int(np.floor(p_mid)))
    p1 = min(wave.data.shape[1] - 1, int(np.ceil(p_mid)))

    q_mid = (y - DimOffset(wave, 1)) / DimDelta(wave, 1)
    q0 = max(0, int(np.floor(q_mid)))
    q1 = min(wave.data.shape[0] - 1, int(np.ceil(q_mid)))

    # Handle edge cases
    if p0 == p1:
        p_frac = 0
    else:
        p_frac = p_mid - p0

    if q0 == q1:
        q_frac = 0
    else:
        q_frac = q_mid - q0

    if len(wave.data.shape) == 2:
        # 2D interpolation
        v00 = wave.data[q0, p0]
        v01 = wave.data[q1, p0]
        v10 = wave.data[q0, p1]
        v11 = wave.data[q1, p1]

        # Bilinear interpolation
        v0 = v00 * (1 - p_frac) + v10 * p_frac
        v1 = v01 * (1 - p_frac) + v11 * p_frac
        result = v0 * (1 - q_frac) + v1 * q_frac

        return result

    elif len(wave.data.shape) == 3:
        # 3D interpolation (trilinear)
        r_mid = (z - DimOffset(wave, 2)) / DimDelta(wave, 2)
        r0 = max(0, int(np.floor(r_mid)))
        r1 = min(wave.data.shape[2] - 1, int(np.ceil(r_mid)))

        if r0 == r1:
            r_frac = 0
        else:
            r_frac = r_mid - r0

        # Get 8 corner values
        v000 = wave.data[q0, p0, r0]
        v001 = wave.data[q0, p0, r1]
        v010 = wave.data[q0, p1, r0]
        v011 = wave.data[q0, p1, r1]
        v100 = wave.data[q1, p0, r0]
        v101 = wave.data[q1, p0, r1]
        v110 = wave.data[q1, p1, r0]
        v111 = wave.data[q1, p1, r1]

        # Trilinear interpolation
        v00 = v000 * (1 - r_frac) + v001 * r_frac
        v01 = v010 * (1 - r_frac) + v011 * r_frac
        v10 = v100 * (1 - r_frac) + v101 * r_frac
        v11 = v110 * (1 - r_frac) + v111 * r_frac

        v0 = v00 * (1 - p_frac) + v01 * p_frac
        v1 = v10 * (1 - p_frac) + v11 * p_frac
        result = v0 * (1 - q_frac) + v1 * q_frac

        return result

    return 0


# Image processing functions
def ImageTransform(transform_type, wave, *args):
    """Igor ImageTransform function equivalent"""
    if transform_type.lower() == 'fliprows':
        wave.data = np.flipud(wave.data)
    elif transform_type.lower() == 'flipcols':
        wave.data = np.fliplr(wave.data)
    elif transform_type.lower() == 'transpose':
        wave.data = wave.data.T
    elif transform_type.lower() == 'rotatecw':
        wave.data = np.rot90(wave.data, -1)
    elif transform_type.lower() == 'rotateccw':
        wave.data = np.rot90(wave.data, 1)


def ImageStats(wave):
    """Igor ImageStats function equivalent"""
    return WaveStats(wave)


def ImageFilter(filter_type, wave, *args):
    """Igor ImageFilter function equivalent"""
    if filter_type.lower() == 'gauss':
        if len(args) > 0:
            sigma = args[0]
        else:
            sigma = 1.0
        wave.data = ndimage.gaussian_filter(wave.data, sigma)
    elif filter_type.lower() == 'median':
        if len(args) > 0:
            size = args[0]
        else:
            size = 3
        wave.data = ndimage.median_filter(wave.data, size)
    elif filter_type.lower() == 'min':
        if len(args) > 0:
            size = args[0]
        else:
            size = 3
        wave.data = ndimage.minimum_filter(wave.data, size)
    elif filter_type.lower() == 'max':
        if len(args) > 0:
            size = args[0]
        else:
            size = 3
        wave.data = ndimage.maximum_filter(wave.data, size)


def Smooth(wave, smoothing_factor):
    """Igor Smooth function equivalent"""
    # Apply Gaussian smoothing
    sigma = smoothing_factor / 2.355  # Convert FWHM to sigma
    wave.data = ndimage.gaussian_filter(wave.data, sigma)


def Differentiate(wave, axis=0):
    """Igor Differentiate function equivalent"""
    wave.data = np.gradient(wave.data, axis=axis)


def Integrate(wave, axis=0):
    """Igor Integrate function equivalent"""
    wave.data = np.cumsum(wave.data, axis=axis) * DimDelta(wave, axis)


# FFT functions
def FFT(wave, dest_wave=None):
    """Igor FFT function equivalent"""
    if dest_wave is None:
        dest_wave = Wave(name=wave.name + "_FFT")

    fft_data = np.fft.fft(wave.data)
    dest_wave.data = fft_data

    # Set up frequency scaling
    n = len(wave.data)
    dt = DimDelta(wave, 0)
    freq_delta = 1.0 / (n * dt)
    dest_wave.SetScale('x', 0, freq_delta)

    return dest_wave


def IFFT(wave, dest_wave=None):
    """Igor IFFT function equivalent"""
    if dest_wave is None:
        dest_wave = Wave(name=wave.name + "_IFFT")

    ifft_data = np.fft.ifft(wave.data)
    dest_wave.data = ifft_data

    return dest_wave


# String functions
def StringFromList(item_index, list_string, separator=";"):
    """Igor StringFromList function equivalent"""
    items = list_string.split(separator)
    if 0 <= item_index < len(items):
        return items[item_index]
    return ""


def ItemsInList(list_string, separator=";"):
    """Igor ItemsInList function equivalent"""
    if not list_string:
        return 0
    return len(list_string.split(separator))


def AddListItem(item, list_string, separator=";"):
    """Igor AddListItem function equivalent"""
    if list_string:
        return list_string + separator + item
    return item


def RemoveListItem(item_index, list_string, separator=";"):
    """Igor RemoveListItem function equivalent"""
    items = list_string.split(separator)
    if 0 <= item_index < len(items):
        items.pop(item_index)
    return separator.join(items)


# Utility functions
def NumType(value):
    """Igor NumType function equivalent"""
    if np.isnan(value):
        return 2  # NaN
    elif np.isinf(value):
        return 1  # Inf
    else:
        return 0  # Normal number


def Exists(name):
    """Igor Exists function equivalent"""
    # This would check if a wave/variable exists in Igor
    # For our purposes, we'll implement a simple version
    return name is not None and name != ""


def CleanupName(name, replacement_char="_"):
    """Igor CleanupName function equivalent"""
    import re
    # Replace invalid characters with underscore
    clean_name = re.sub(r'[^\w]', replacement_char, name)
    # Ensure it doesn't start with a number
    if clean_name and clean_name[0].isdigit():
        clean_name = replacement_char + clean_name
    return clean_name


def UniqueName(base_name, name_type=1, index=0):
    """Igor UniqueName function equivalent"""
    # For simplicity, just append a counter
    counter = index
    candidate = base_name
    # In a real implementation, you'd check against existing names
    while counter > 0:  # Simple implementation
        candidate = f"{base_name}_{counter}"
        counter += 1
        if counter > 1000:  # Prevent infinite loop
            break
    return candidate


# Variable functions
def Variable(name, value=0):
    """Igor Variable declaration equivalent"""
    # In Igor, this creates a global variable
    # For Python, we'll just return the value
    return value


def NVAR(name):
    """Igor NVAR (numeric variable reference) equivalent"""
    # This would reference a global numeric variable in Igor
    # For Python, we'll implement a simple registry
    if not hasattr(NVAR, 'registry'):
        NVAR.registry = {}
    return NVAR.registry.get(name, 0)


def SVAR(name):
    """Igor SVAR (string variable reference) equivalent"""
    # This would reference a global string variable in Igor
    # For Python, we'll implement a simple registry
    if not hasattr(SVAR, 'registry'):
        SVAR.registry = {}
    return SVAR.registry.get(name, "")


# Display and UI functions (placeholders)
def DoAlert(alert_type, message):
    """Igor DoAlert function equivalent"""
    import tkinter.messagebox as msgbox
    if alert_type == 0:  # Note
        msgbox.showinfo("Note", message)
        return 1
    elif alert_type == 1:  # Caution
        result = msgbox.askyesno("Caution", message)
        return 1 if result else 2
    elif alert_type == 2:  # Stop
        msgbox.showerror("Error", message)
        return 1
    return 1


def GetDataFolder(flag):
    """Igor GetDataFolder function equivalent"""
    if flag == 1:
        return "root:"
    return "root"


def SetDataFolder(path):
    """Igor SetDataFolder function equivalent"""
    # Placeholder - in Igor this changes the current data folder
    pass


def NewDataFolder(name):
    """Igor NewDataFolder function equivalent"""
    # Placeholder - in Igor this creates a new data folder
    pass


def KillDataFolder(path):
    """Igor KillDataFolder function equivalent"""
    # Placeholder - in Igor this removes a data folder
    pass


# Print and output functions
def Print(*args):
    """Igor Print function equivalent"""
    print(*args)


def Printf(format_string, *args):
    """Igor Printf function equivalent"""
    print(format_string % args)


def Sprintf(format_string, *args):
    """Igor Sprintf function equivalent"""
    return format_string % args


# Error handling
class IgorError(Exception):
    """Custom exception for Igor Pro compatibility"""
    pass


def Abort(message=""):
    """Igor Abort function equivalent"""
    if message:
        raise IgorError(message)
    else:
        raise IgorError("Operation aborted")


# Initialize any required global state
def InitializeIgorCompatibility():
    """Initialize Igor Pro compatibility layer"""
    # Set up any global variables or state needed
    pass


# Call initialization
InitializeIgorCompatibility()