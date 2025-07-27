"""
Igor Pro Compatibility Layer
Recreates Igor Pro native functions to maintain 1-1 code compatibility
Complete implementation with all necessary functions for blob detection
FIXED: Added missing Duplicate function and other required functions
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
        """Create a copy of this wave"""
        new_wave = Wave(self.data.copy(), self.name + "_copy", self.note)
        for axis in self.scaling:
            new_wave.scaling[axis] = self.scaling[axis].copy()
        return new_wave


# FIXED: Added missing Duplicate function
def Duplicate(source_wave, new_name):
    """
    Create a duplicate of a wave - matches Igor Pro Duplicate function
    This was the missing function causing errors in main_functions.py

    Parameters:
    source_wave : Wave - Source wave to duplicate
    new_name : str - Name for the new wave

    Returns:
    Wave - New wave that is a copy of the source
    """
    new_data = source_wave.data.copy()
    new_wave = Wave(new_data, new_name, source_wave.note)

    # Copy scaling information
    for axis in ['x', 'y', 'z', 't']:
        scale_info = source_wave.GetScale(axis)
        new_wave.SetScale(axis, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return new_wave


# Igor Pro function equivalents for wave scaling
def DimOffset(wave, dimension):
    """Igor DimOffset function equivalent"""
    if dimension == 0:
        return wave.GetScale('x')['offset']
    elif dimension == 1:
        return wave.GetScale('y')['offset']
    elif dimension == 2:
        return wave.GetScale('z')['offset']
    elif dimension == 3:
        return wave.GetScale('t')['offset']
    else:
        return 0.0


def DimDelta(wave, dimension):
    """Igor DimDelta function equivalent"""
    if dimension == 0:
        return wave.GetScale('x')['delta']
    elif dimension == 1:
        return wave.GetScale('y')['delta']
    elif dimension == 2:
        return wave.GetScale('z')['delta']
    elif dimension == 3:
        return wave.GetScale('t')['delta']
    else:
        return 1.0


def DimSize(wave, dimension):
    """Igor DimSize function equivalent"""
    if wave.data.ndim > dimension:
        return wave.data.shape[dimension]
    else:
        return 0


def DimUnits(wave, dimension):
    """Igor DimUnits function equivalent"""
    if dimension == 0:
        return wave.GetScale('x')['units']
    elif dimension == 1:
        return wave.GetScale('y')['units']
    elif dimension == 2:
        return wave.GetScale('z')['units']
    elif dimension == 3:
        return wave.GetScale('t')['units']
    else:
        return ""


def WaveMin(wave):
    """Igor WaveMin function equivalent"""
    return np.nanmin(wave.data)


def WaveMax(wave):
    """Igor WaveMax function equivalent"""
    return np.nanmax(wave.data)


def NumPnts(wave):
    """Igor numpnts function equivalent"""
    return wave.data.size


def NumDimensions(wave):
    """Igor NumDimensions function equivalent"""
    return wave.data.ndim


# Mathematical functions
def Sqrt(x):
    """Igor sqrt function equivalent"""
    return np.sqrt(x)


def Exp(x):
    """Igor exp function equivalent"""
    return np.exp(x)


def Log(x):
    """Igor ln function equivalent (natural log)"""
    return np.log(x)


def Log10(x):
    """Igor log function equivalent (base 10)"""
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


# Wave creation functions
def Make(name, dimensions, wave_type=None, value=0):
    """Igor Make function equivalent"""
    if isinstance(dimensions, (int, float)):
        # 1D wave
        data = np.full(int(dimensions), value, dtype=np.float64)
    elif isinstance(dimensions, (list, tuple)):
        # Multi-dimensional wave
        data = np.full(dimensions, value, dtype=np.float64)
    else:
        raise ValueError("Invalid dimensions for Make function")

    return Wave(data, name)


def SetScale(wave, axis, start, delta, units=""):
    """Igor SetScale function equivalent"""
    wave.SetScale(axis.lower(), start, delta, units)


def IndexToScale(wave, index, dimension):
    """Igor IndexToScale function equivalent"""
    offset = DimOffset(wave, dimension)
    delta = DimDelta(wave, dimension)
    return offset + index * delta


def ScaleToIndex(wave, value, dimension):
    """Igor ScaleToIndex function equivalent"""
    offset = DimOffset(wave, dimension)
    delta = DimDelta(wave, dimension)
    if delta == 0:
        return 0
    return int((value - offset) / delta)


def Concatenate(waves, output_name, axis=2):
    """Igor Concatenate function equivalent"""
    if isinstance(waves, list):
        data_list = [w.data for w in waves]
    else:
        # Assume waves is a list of wave names or similar
        raise NotImplementedError("String-based concatenation not implemented")

    # Concatenate along specified axis
    concatenated_data = np.stack(data_list, axis=axis)

    # Create output wave
    output_wave = Wave(concatenated_data, output_name)

    # Copy scaling from first wave
    if waves:
        first_wave = waves[0]
        for dim_name in ['x', 'y', 'z', 't']:
            scale_info = first_wave.GetScale(dim_name)
            output_wave.SetScale(dim_name, scale_info['offset'], scale_info['delta'], scale_info['units'])

    return output_wave


def Redimension(wave, *new_dimensions):
    """Igor Redimension function equivalent"""
    if len(new_dimensions) == 1:
        # 1D redimension
        new_size = new_dimensions[0]
        if new_size > wave.data.size:
            # Extend with zeros
            padding = new_size - wave.data.size
            wave.data = np.pad(wave.data.flatten(), (0, padding), 'constant')
        else:
            # Truncate
            wave.data = wave.data.flatten()[:new_size]
    else:
        # Multi-dimensional redimension
        try:
            wave.data = np.resize(wave.data, new_dimensions)
        except Exception as e:
            print(f"Redimension error: {e}")


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

        # Note: RegularGridInterpolator expects (y, x) order for 2D data
        interpolator = RegularGridInterpolator((y_coords, x_coords), wave.data,
                                               method='linear', bounds_error=False, fill_value=0)

        return interpolator([y, x])[0]
    else:
        raise ValueError("Wave must be at least 2D for Interp2D")


# Display functions (simplified - these would typically create Igor windows)
def Display(*waves, **kwargs):
    """Igor Display function equivalent (simplified)"""
    print(f"Would display waves: {[w.name for w in waves]}")
    # In a full implementation, this would create a plot window


def AppendToGraph(*waves, **kwargs):
    """Igor AppendToGraph function equivalent (simplified)"""
    print(f"Would append to graph: {[w.name for w in waves]}")


def ModifyGraph(**kwargs):
    """Igor ModifyGraph function equivalent (simplified)"""
    print(f"Would modify graph with: {kwargs}")


# String functions
def StringFromList(position, str_list, list_sep=";"):
    """Igor StringFromList function equivalent"""
    items = str_list.split(list_sep)
    if 0 <= position < len(items):
        return items[position]
    else:
        return ""


def StringMatch(string, pattern):
    """Igor StringMatch function equivalent (simplified)"""
    import fnmatch
    return fnmatch.fnmatch(string, pattern)


def StrLen(string):
    """Igor strlen function equivalent"""
    return len(str(string))


def LowerStr(string):
    """Igor LowerStr function equivalent"""
    return str(string).lower()


def UpperStr(string):
    """Igor UpperStr function equivalent"""
    return str(string).upper()


def Num2Str(number):
    """Igor Num2Str function equivalent"""
    return str(number)


def Str2Num(string):
    """Igor Str2Num function equivalent"""
    try:
        if '.' in str(string) or 'e' in str(string).lower():
            return float(string)
        else:
            return int(string)
    except (ValueError, TypeError):
        return np.nan


def NameOfWave(wave):
    """Igor NameOfWave function equivalent"""
    return wave.name


def Note(wave, note_text=None):
    """Igor Note function equivalent"""
    if note_text is None:
        return wave.note
    else:
        wave.note = str(note_text)
        return wave.note


# Igor Pro constants
NT_FP32 = 2
NT_FP64 = 4
NT_I8 = 8
NT_I16 = 16
NT_I32 = 32
NT_CMPLX = 1


def Testing(string_input, number_input):
    """Testing function for igor_compatibility module"""
    print(f"Igor compatibility testing: {string_input}, {number_input}")
    return len(string_input) + number_input