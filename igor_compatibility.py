"""
Igor Pro Compatibility Layer
Recreates Igor Pro native functions to maintain 1-1 code compatibility
Complete implementation with all necessary functions for blob detection
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
        try:
            np.save(filename, self.data)
            return True
        except Exception as e:
            print(f"Error saving wave: {e}")
            return False

    def load(self, filename):
        """Load wave data from file"""
        try:
            self.data = np.load(filename)
            return True
        except Exception as e:
            print(f"Error loading wave: {e}")
            return False


# Dimension access functions
def DimSize(wave, dim):
    """Igor DimSize function equivalent"""
    if hasattr(wave, 'data'):
        if dim < len(wave.data.shape):
            return wave.data.shape[dim]
    return 0


def DimOffset(wave, dim):
    """Igor DimOffset function equivalent"""
    if hasattr(wave, 'scaling'):
        if dim == 0:
            return wave.scaling['x']['offset']
        elif dim == 1:
            return wave.scaling['y']['offset']
        elif dim == 2:
            return wave.scaling['z']['offset']
        elif dim == 3:
            return wave.scaling['t']['offset']
    return 0.0


def DimDelta(wave, dim):
    """Igor DimDelta function equivalent"""
    if hasattr(wave, 'scaling'):
        if dim == 0:
            return wave.scaling['x']['delta']
        elif dim == 1:
            return wave.scaling['y']['delta']
        elif dim == 2:
            return wave.scaling['z']['delta']
        elif dim == 3:
            return wave.scaling['t']['delta']
    return 1.0


def DimUnits(wave, dim):
    """Igor DimUnits function equivalent"""
    if hasattr(wave, 'scaling'):
        if dim == 0:
            return wave.scaling['x']['units']
        elif dim == 1:
            return wave.scaling['y']['units']
        elif dim == 2:
            return wave.scaling['z']['units']
        elif dim == 3:
            return wave.scaling['t']['units']
    return ""


# Wave creation and manipulation functions
def Make(shape, name="", dtype=np.float64, value=0):
    """Igor Make function equivalent - creates a new wave"""
    if isinstance(shape, int):
        data = np.full(shape, value, dtype=dtype)
    else:
        data = np.full(shape, value, dtype=dtype)
    return Wave(data, name)


def Duplicate(source_wave, dest_name="", range_spec=None):
    """Igor Duplicate function equivalent"""
    if range_spec is None:
        new_data = source_wave.data.copy()
    else:
        # Handle range specifications
        new_data = source_wave.data[range_spec].copy()

    new_wave = Wave(new_data, dest_name if dest_name else f"{source_wave.name}_dup")
    new_wave.scaling = source_wave.scaling.copy()
    new_wave.note = source_wave.note
    return new_wave


def Redimension(wave, new_shape):
    """Igor Redimension function equivalent"""
    try:
        wave.data = wave.data.reshape(new_shape)
        return True
    except ValueError as e:
        print(f"Error redimensioning wave: {e}")
        return False


def MatrixOp(operation, *args, **kwargs):
    """Igor MatrixOp equivalent for basic operations"""
    if operation == "transpose":
        return Wave(np.transpose(args[0].data))
    elif operation == "inverse":
        return Wave(np.linalg.inv(args[0].data))
    elif operation == "dot":
        return Wave(np.dot(args[0].data, args[1].data))
    elif operation == "add":
        return Wave(args[0].data + args[1].data)
    elif operation == "multiply":
        return Wave(args[0].data * args[1].data)
    elif operation == "subtract":
        return Wave(args[0].data - args[1].data)
    else:
        raise ValueError(f"Unknown MatrixOp operation: {operation}")


def Concatenate(wave_list, dest_wave, axis=0):
    """Igor Concatenate function equivalent"""
    data_list = [w.data for w in wave_list]
    concatenated = np.concatenate(data_list, axis=axis)
    dest_wave.data = concatenated


def DeletePoints(start, num_points, wave):
    """Igor DeletePoints function equivalent"""
    if len(wave.data.shape) == 1:
        wave.data = np.delete(wave.data, slice(start, start + num_points))
    else:
        # For multi-dimensional, delete along first axis
        wave.data = np.delete(wave.data, slice(start, start + num_points), axis=0)


def InsertPoints(point, num_points, wave, value=0):
    """Igor InsertPoints function equivalent"""
    if len(wave.data.shape) == 1:
        to_insert = np.full(num_points, value, dtype=wave.data.dtype)
        wave.data = np.insert(wave.data, point, to_insert)
    else:
        # For multi-dimensional, insert along first axis
        shape = list(wave.data.shape)
        shape[0] = num_points
        to_insert = np.full(shape, value, dtype=wave.data.dtype)
        wave.data = np.insert(wave.data, point, to_insert, axis=0)


# Mathematical functions
def Multithread(wave, expression):
    """Igor Multithread equivalent - applies expression to wave"""
    # This is a simplified version - would need full expression parser for complete functionality
    if expression == "NaN":
        wave.data[:] = np.nan
    elif expression == "0":
        wave.data[:] = 0
    elif expression == "-1":
        wave.data[:] = -1
    elif "=" in expression:
        # Simple assignment
        value = float(expression.split("=")[1].strip())
        wave.data[:] = value


def WaveStats(wave):
    """Igor WaveStats function equivalent - returns dictionary of statistics"""
    data = wave.data.flatten()

    # Remove NaN values for statistics
    valid_data = data[~np.isnan(data)]

    stats = {
        'V_npnts': len(valid_data),
        'V_numNaNs': np.sum(np.isnan(data)),
        'V_avg': np.mean(valid_data) if len(valid_data) > 0 else np.nan,
        'V_sum': np.sum(valid_data) if len(valid_data) > 0 else np.nan,
        'V_sdev': np.std(valid_data, ddof=1) if len(valid_data) > 1 else np.nan,
        'V_sem': np.std(valid_data, ddof=1) / np.sqrt(len(valid_data)) if len(valid_data) > 1 else np.nan,
        'V_rms': np.sqrt(np.mean(valid_data ** 2)) if len(valid_data) > 0 else np.nan,
        'V_adev': np.mean(np.abs(valid_data - np.mean(valid_data))) if len(valid_data) > 0 else np.nan,
        'V_skew': compute_skewness(valid_data) if len(valid_data) > 2 else np.nan,
        'V_kurt': compute_kurtosis(valid_data) if len(valid_data) > 3 else np.nan,
        'V_min': np.min(valid_data) if len(valid_data) > 0 else np.nan,
        'V_max': np.max(valid_data) if len(valid_data) > 0 else np.nan,
        'V_minloc': np.argmin(data) if len(data) > 0 else -1,
        'V_maxloc': np.argmax(data) if len(data) > 0 else -1,
    }

    return stats


def compute_skewness(data):
    """Compute skewness of data"""
    if len(data) < 3:
        return np.nan

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        return np.nan

    n = len(data)
    skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
    return skew


def compute_kurtosis(data):
    """Compute kurtosis of data"""
    if len(data) < 4:
        return np.nan

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0:
        return np.nan

    n = len(data)
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4) - (
                3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return kurt


def WaveMax(wave):
    """Igor WaveMax function equivalent"""
    return np.nanmax(wave.data)


def WaveMin(wave):
    """Igor WaveMin function equivalent"""
    return np.nanmin(wave.data)


def Sum(wave, start=None, end=None):
    """Igor Sum function equivalent"""
    if start is None and end is None:
        return np.nansum(wave.data)
    else:
        return np.nansum(wave.data[start:end])


def Mean(wave, start=None, end=None):
    """Igor Mean function equivalent"""
    if start is None and end is None:
        return np.nanmean(wave.data)
    else:
        return np.nanmean(wave.data[start:end])


def Variance(wave, start=None, end=None):
    """Igor Variance function equivalent"""
    if start is None and end is None:
        return np.nanvar(wave.data, ddof=1)
    else:
        return np.nanvar(wave.data[start:end], ddof=1)


def StdDev(wave, start=None, end=None):
    """Igor StdDev function equivalent"""
    if start is None and end is None:
        return np.nanstd(wave.data, ddof=1)
    else:
        return np.nanstd(wave.data[start:end], ddof=1)


# Complex number functions
def Cmplx(real, imag):
    """Igor Cmplx function equivalent"""
    return complex(real, imag)


def Real(z):
    """Igor Real function equivalent"""
    return np.real(z)


def Imag(z):
    """Igor Imag function equivalent"""
    return np.imag(z)


def Conj(z):
    """Igor Conj function equivalent"""
    return np.conj(z)


def Cabs(z):
    """Igor cabs function equivalent"""
    return np.abs(z)


def Phase(z):
    """Igor phase function equivalent"""
    return np.angle(z)


# Trigonometric functions
def Sin(x):
    """Igor sin function"""
    return np.sin(x)


def Cos(x):
    """Igor cos function"""
    return np.cos(x)


def Tan(x):
    """Igor tan function"""
    return np.tan(x)


def ASin(x):
    """Igor asin function"""
    return np.arcsin(x)


def ACos(x):
    """Igor acos function"""
    return np.arccos(x)


def ATan(x):
    """Igor atan function"""
    return np.arctan(x)


def ATan2(y, x):
    """Igor atan2 function"""
    return np.arctan2(y, x)


# Exponential and logarithmic functions
def Exp(x):
    """Igor exp function equivalent"""
    return np.exp(x)


def Log(x):
    """Igor log function equivalent (natural log)"""
    return np.log(x)


def Log10(x):
    """Igor log10 function equivalent"""
    return np.log10(x)


def Ln(x):
    """Igor ln function equivalent"""
    return np.log(x)


def Sqrt(x):
    """Igor sqrt function equivalent"""
    return np.sqrt(x)


def Pow(x, y):
    """Igor ^ operator equivalent"""
    return np.power(x, y)


# Utility functions
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
        v0 = v00 + (v10 - v00) * p_frac
        v1 = v01 + (v11 - v01) * p_frac

        return v0 + (v1 - v0) * q_frac
    else:
        # 3D interpolation
        z_idx = int(z)
        if z_idx < 0 or z_idx >= wave.data.shape[2]:
            return 0

        v00 = wave.data[q0, p0, z_idx]
        v01 = wave.data[q1, p0, z_idx]
        v10 = wave.data[q0, p1, z_idx]
        v11 = wave.data[q1, p1, z_idx]

        # Bilinear interpolation
        v0 = v00 + (v10 - v00) * p_frac
        v1 = v01 + (v11 - v01) * p_frac

        return v0 + (v1 - v0) * q_frac


# Curve fitting functions
def CurveFit(func_name, coef_wave, data_wave, mask_wave=None, **kwargs):
    """Igor CurveFit function equivalent"""
    try:
        x = np.arange(len(data_wave.data))
        y = data_wave.data.copy()

        if mask_wave is not None:
            mask = mask_wave.data.astype(bool)
            x = x[mask]
            y = y[mask]

        if func_name == "line":
            def line_func(x, a, b):
                return a + b * x

            popt, _ = curve_fit(line_func, x, y)
            coef_wave.data[:len(popt)] = popt

        elif func_name.startswith("Poly"):
            order = int(func_name.replace("Poly", "")) - 1
            popt = np.polyfit(x, y, order)
            coef_wave.data[:len(popt)] = popt[::-1]  # Igor order is opposite

        elif func_name == "exp":
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c

            popt, _ = curve_fit(exp_func, x, y, maxfev=5000)
            coef_wave.data[:len(popt)] = popt

        elif func_name == "gauss":
            def gauss_func(x, a, b, c, d):
                return a * np.exp(-((x - b) / c) ** 2) + d

            # Initial guess
            p0 = [np.max(y) - np.min(y), x[np.argmax(y)], (x[-1] - x[0]) / 4, np.min(y)]
            popt, _ = curve_fit(gauss_func, x, y, p0=p0, maxfev=5000)
            coef_wave.data[:len(popt)] = popt

        return True

    except Exception as e:
        print(f"CurveFit error: {e}")
        coef_wave.data.fill(0)
        return False


def Poly(coef_wave, x):
    """Igor Poly function equivalent"""
    coeffs = coef_wave.data[::-1]  # Reverse for numpy polyval
    return np.polyval(coeffs, x)


# Histogram and statistics functions
def Histogram(source_wave, dest_wave, bins=None, bin_width=None, range_min=None, range_max=None):
    """Igor Histogram function equivalent"""
    data = source_wave.data.flatten()
    data = data[~np.isnan(data)]  # Remove NaN values

    if len(data) == 0:
        dest_wave.data = np.array([])
        return

    if range_min is None:
        range_min = np.min(data)
    if range_max is None:
        range_max = np.max(data)

    if bins is not None:
        hist, bin_edges = np.histogram(data, bins=bins, range=(range_min, range_max))
    elif bin_width is not None:
        bins = int((range_max - range_min) / bin_width)
        hist, bin_edges = np.histogram(data, bins=bins, range=(range_min, range_max))
    else:
        hist, bin_edges = np.histogram(data, range=(range_min, range_max))

    dest_wave.data = hist.astype(np.float64)

    # Set scaling for histogram
    dest_wave.SetScale('x', bin_edges[0], bin_edges[1] - bin_edges[0])


def Sort(sort_wave, *other_waves):
    """Igor Sort function equivalent"""
    sort_indices = np.argsort(sort_wave.data.flatten())

    # Sort the primary wave
    sort_wave.data = sort_wave.data.flatten()[sort_indices]

    # Sort other waves using the same indices
    for wave in other_waves:
        wave.data = wave.data.flatten()[sort_indices]


# String functions for completeness
def StrLen(string):
    """Igor strlen function equivalent"""
    return len(str(string))


def NumToStr(number):
    """Igor num2str function equivalent"""
    return str(number)


def StrToNum(string):
    """Igor str2num function equivalent"""
    try:
        return float(string)
    except (ValueError, TypeError):
        return np.nan


# Testing function
def TestIgorCompatibility():
    """Test the Igor compatibility functions"""
    print("Testing Igor compatibility layer...")

    # Test Wave creation
    test_wave = Wave(np.random.rand(10, 10), "TestWave")
    test_wave.SetScale('x', 0, 0.1)
    test_wave.SetScale('y', 0, 0.1)

    print(f"✓ Wave creation: {test_wave}")
    print(f"✓ Wave scaling: x={DimOffset(test_wave, 0)}, delta={DimDelta(test_wave, 0)}")

    # Test statistics
    stats = WaveStats(test_wave)
    print(f"✓ Wave statistics: mean={stats['V_avg']:.4f}, std={stats['V_sdev']:.4f}")

    # Test mathematical functions
    result = Sqrt(4.0)
    print(f"✓ Mathematical functions: sqrt(4) = {result}")

    # Test duplicate
    dup_wave = Duplicate(test_wave, "DuplicatedWave")
    print(f"✓ Wave duplication: {dup_wave.name}")

    print("Igor compatibility test completed successfully!")
    return True


if __name__ == "__main__":
    TestIgorCompatibility()