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
        """Create a copy of this wave"""
        new_wave = Wave(self.data.copy(), self.name + "_copy", self.note)
        for axis in self.scaling:
            new_wave.scaling[axis] = self.scaling[axis].copy()
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

    # Get fractional parts
    fp = p_mid - p0
    fq = q_mid - q0

    # Handle edge cases
    if p0 == p1:
        fp = 0
    if q0 == q1:
        fq = 0

    # Extract corner values
    if wave.data.ndim == 2:
        v00 = wave.data[q0, p0]
        v01 = wave.data[q1, p0]
        v10 = wave.data[q0, p1]
        v11 = wave.data[q1, p1]
    elif wave.data.ndim == 3:
        layer = max(0, min(wave.data.shape[2] - 1, int(z)))
        v00 = wave.data[q0, p0, layer]
        v01 = wave.data[q1, p0, layer]
        v10 = wave.data[q0, p1, layer]
        v11 = wave.data[q1, p1, layer]
    else:
        return 0

    # Bilinear interpolation
    v0 = v00 * (1 - fp) + v10 * fp
    v1 = v01 * (1 - fp) + v11 * fp
    value = v0 * (1 - fq) + v1 * fq

    return value


# Statistical functions
def StatsQuantile(wave, quantile):
    """Igor StatsQuantile function equivalent"""
    return np.nanquantile(wave.data, quantile)


def StatsMedian(wave):
    """Igor StatsMedian function equivalent"""
    return np.nanmedian(wave.data)


def Variance(wave):
    """Igor Variance function equivalent"""
    return np.nanvar(wave.data)


def StdDev(wave):
    """Igor StdDev function equivalent"""
    return np.nanstd(wave.data)


# Complex number functions
def Real(z):
    """Igor real function equivalent"""
    return np.real(z)


def Imag(z):
    """Igor imag function equivalent"""
    return np.imag(z)


def Cmplx(real, imag):
    """Igor cmplx function equivalent"""
    return complex(real, imag)


def Conj(z):
    """Igor conj function equivalent"""
    return np.conj(z)


def Mag(z):
    """Igor mag function equivalent (magnitude of complex)"""
    return np.abs(z)


def Phase(z):
    """Igor phase function equivalent"""
    return np.angle(z)


# Utility functions
def NaN():
    """Igor NaN constant equivalent"""
    return np.nan


def Inf():
    """Igor Inf constant equivalent"""
    return np.inf


def Pi():
    """Igor Pi constant equivalent"""
    return np.pi


def E():
    """Igor e constant equivalent"""
    return np.e


def NumType(wave):
    """Igor NumType function equivalent"""
    dtype = wave.data.dtype
    if dtype == np.float32:
        return 2  # NT_FP32
    elif dtype == np.float64:
        return 4  # NT_FP64
    elif dtype == np.int8:
        return 8  # NT_I8
    elif dtype == np.int16:
        return 16  # NT_I16
    elif dtype == np.int32:
        return 32  # NT_I32
    elif dtype == np.complex64:
        return 1  # NT_CMPLX
    elif dtype == np.complex128:
        return 5  # NT_CMPLX
    else:
        return 4  # Default to NT_FP64


# Curve fitting functions
def CurveFit(fit_func, wave_x, wave_y, coef_wave, **kwargs):
    """Igor CurveFit function equivalent (simplified)"""
    try:
        # Use scipy.optimize.curve_fit
        popt, pcov = curve_fit(fit_func, wave_x.data, wave_y.data,
                               p0=coef_wave.data if len(coef_wave.data) > 0 else None)

        # Update coefficient wave
        coef_wave.data = popt

        # Return chi-square statistic (simplified)
        residuals = wave_y.data - fit_func(wave_x.data, *popt)
        chi_square = np.sum(residuals ** 2)

        return chi_square

    except Exception as e:
        print(f"CurveFit error: {e}")
        return np.inf


# Wave manipulation functions
def DeletePoints(start_point, num_points, wave):
    """Igor DeletePoints function equivalent"""
    if wave.data.ndim == 1:
        wave.data = np.delete(wave.data, slice(start_point, start_point + num_points))
    else:
        print("DeletePoints only implemented for 1D waves")


def InsertPoints(point, num_points, wave):
    """Igor InsertPoints function equivalent"""
    if wave.data.ndim == 1:
        insert_data = np.zeros(num_points)
        wave.data = np.insert(wave.data, point, insert_data)
    else:
        print("InsertPoints only implemented for 1D waves")


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