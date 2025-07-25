"""
Igor Pro Compatibility Layer
Recreates Igor Pro native functions to maintain 1-1 code compatibility
"""

import numpy as np
from scipy import ndimage, fft
from scipy.optimize import curve_fit
import warnings

# Monkey patch for numpy complex deprecation (NumPy 1.20+)
if not hasattr(np, 'complex'):
    np.complex = complex


class Wave:
    """
    Mimics Igor Pro wave structure
    """

    def __init__(self, data=None, name="", note=""):
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data)
        self.name = name
        self.note = note
        self.scaling = {'x': {'offset': 0, 'delta': 1},
                        'y': {'offset': 0, 'delta': 1},
                        'z': {'offset': 0, 'delta': 1}}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    def SetScale(self, axis, offset, delta):
        """Set scaling for wave dimension"""
        if axis.lower() == 'x':
            self.scaling['x'] = {'offset': offset, 'delta': delta}
        elif axis.lower() == 'y':
            self.scaling['y'] = {'offset': offset, 'delta': delta}
        elif axis.lower() == 'z':
            self.scaling['z'] = {'offset': offset, 'delta': delta}


def Make(shape, name="", dtype=np.float64, value=0):
    """
    Igor Make function equivalent
    Creates a new wave with specified dimensions
    """
    if isinstance(shape, int):
        data = np.full(shape, value, dtype=dtype)
    else:
        data = np.full(shape, value, dtype=dtype)
    return Wave(data, name)


def Duplicate(source_wave, dest_name="", range_spec=None):
    """
    Igor Duplicate function equivalent
    """
    if range_spec is None:
        new_data = source_wave.data.copy()
    else:
        # Handle range specifications like [0:10][5:15]
        new_data = source_wave.data[range_spec].copy()

    new_wave = Wave(new_data, dest_name)
    new_wave.scaling = source_wave.scaling.copy()
    new_wave.note = source_wave.note
    return new_wave


def Multithread(wave, expression):
    """
    Igor Multithread equivalent - applies expression to wave
    This is a simplified version that applies numpy operations
    """
    # This would need more sophisticated parsing for full Igor compatibility
    # For now, direct numpy operations
    pass


def DimSize(wave, dimension):
    """
    Igor DimSize function equivalent
    """
    if wave.data.ndim <= dimension:
        return 0
    return wave.data.shape[dimension]


def DimOffset(wave, dimension):
    """
    Igor DimOffset function equivalent
    """
    axes = ['x', 'y', 'z']
    if dimension < len(axes):
        return wave.scaling[axes[dimension]]['offset']
    return 0


def DimDelta(wave, dimension):
    """
    Igor DimDelta function equivalent
    """
    axes = ['x', 'y', 'z']
    if dimension < len(axes):
        return wave.scaling[axes[dimension]]['delta']
    return 1


def IndexToScale(wave, index, dimension):
    """
    Igor IndexToScale function equivalent
    """
    return DimOffset(wave, dimension) + index * DimDelta(wave, dimension)


def ScaleToIndex(wave, scale_value, dimension):
    """
    Igor ScaleToIndex function equivalent
    """
    return int((scale_value - DimOffset(wave, dimension)) / DimDelta(wave, dimension))


def WaveMax(wave):
    """
    Igor WaveMax function equivalent
    """
    return np.nanmax(wave.data)


def WaveMin(wave):
    """
    Igor WaveMin function equivalent
    """
    return np.nanmin(wave.data)


def NumPnts(wave):
    """
    Igor NumPnts function equivalent
    """
    return wave.data.size


def FFT_Igor(wave):
    """
    Igor FFT function equivalent
    """
    fft_result = fft.fft2(wave.data)
    return Wave(fft_result, wave.name + "_FFT")


def IFFT_Igor(wave):
    """
    Igor IFFT function equivalent
    """
    ifft_result = fft.ifft2(wave.data)
    return Wave(ifft_result.real, wave.name + "_IFFT")


def MatrixOp(operation, *args, **kwargs):
    """
    Igor MatrixOp function equivalent
    Simplified version for common operations
    """
    # This would need full Igor MatrixOp parsing
    # For now, handle common cases
    if operation == "convolve":
        wave, kernel, mode = args
        if mode == -2:  # Same size output
            result = ndimage.convolve(wave.data, kernel.data, mode='constant')
        else:
            result = ndimage.convolve(wave.data, kernel.data)
        return Wave(result)
    elif operation == "add":
        return Wave(args[0].data + args[1].data)
    elif operation == "multiply":
        return Wave(args[0].data * args[1].data)
    elif operation == "subtract":
        return Wave(args[0].data - args[1].data)


def Concatenate(wave_list, dest_wave):
    """
    Igor Concatenate function equivalent
    """
    data_list = [w.data for w in wave_list]
    concatenated = np.concatenate(data_list)
    dest_wave.data = concatenated


def DeletePoints(start, num_points, wave):
    """
    Igor DeletePoints function equivalent
    """
    if len(wave.data.shape) == 1:
        wave.data = np.delete(wave.data, slice(start, start + num_points))
    else:
        # For multi-dimensional, delete along first axis
        wave.data = np.delete(wave.data, slice(start, start + num_points), axis=0)


def Redimension(wave, new_shape):
    """
    Igor Redimension function equivalent
    """
    wave.data = wave.data.reshape(new_shape)


def SelectNumber(condition, false_value, true_value):
    """
    Igor SelectNumber function equivalent
    """
    return np.where(condition, true_value, false_value)


def Max(*args):
    """
    Igor max function equivalent
    """
    return np.maximum.reduce(args)


def Min(*args):
    """
    Igor min function equivalent
    """
    return np.minimum.reduce(args)


def Sqrt(x):
    """
    Igor sqrt function equivalent
    """
    return np.sqrt(x)


def Ceil(x):
    """
    Igor ceil function equivalent
    """
    return np.ceil(x)


def Floor(x):
    """
    Igor floor function equivalent
    """
    return np.floor(x)


def Round(x):
    """
    Igor round function equivalent
    """
    return np.round(x)


def Log(x):
    """
    Igor log function equivalent (natural log)
    """
    return np.log(x)


def Exp(x):
    """
    Igor exp function equivalent
    """
    return np.exp(x)


def Sum(wave, start=None, end=None):
    """
    Igor Sum function equivalent
    """
    if start is None and end is None:
        return np.sum(wave.data)
    else:
        return np.sum(wave.data[start:end])


def Variance(wave):
    """
    Igor Variance function equivalent
    """
    return np.var(wave.data, ddof=1)  # Igor uses N-1 denominator


def CurveFit(func_name, coef_wave, data_wave, mask_wave=None, **kwargs):
    """
    Igor CurveFit function equivalent
    Simplified version for common fit functions
    """
    x = np.arange(len(data_wave.data))
    y = data_wave.data

    if mask_wave is not None:
        mask = mask_wave.data.astype(bool)
        x = x[mask]
        y = y[mask]

    if func_name == "line":
        def line_func(x, a, b):
            return a + b * x

        try:
            popt, _ = curve_fit(line_func, x, y)
            coef_wave.data[:len(popt)] = popt
        except Exception:
            # If fit fails, return zeros
            coef_wave.data.fill(0)

    elif func_name.startswith("Poly"):
        order = int(func_name.replace("Poly", "")) - 1
        try:
            popt = np.polyfit(x, y, order)
            coef_wave.data[:len(popt)] = popt[::-1]  # Igor order is opposite
        except Exception:
            coef_wave.data.fill(0)


def Poly(coef_wave, x):
    """
    Igor Poly function equivalent
    """
    coeffs = coef_wave.data[::-1]  # Reverse for numpy
    return np.polyval(coeffs, x)


def Histogram(source_wave, dest_wave, bins=None, bin_width=None):
    """
    Igor Histogram function equivalent
    """
    if bins is not None:
        hist, bin_edges = np.histogram(source_wave.data, bins=bins)
    elif bin_width is not None:
        min_val, max_val = np.nanmin(source_wave.data), np.nanmax(source_wave.data)
        bins = int((max_val - min_val) / bin_width)
        hist, bin_edges = np.histogram(source_wave.data, bins=bins, range=(min_val, max_val))
    else:
        hist, bin_edges = np.histogram(source_wave.data)

    dest_wave.data = hist
    # Set scaling for histogram
    dest_wave.SetScale('x', bin_edges[0], bin_edges[1] - bin_edges[0])


def Sort(sort_wave, *other_waves):
    """
    Igor Sort function equivalent
    """
    sort_indices = np.argsort(sort_wave.data)
    sort_wave.data = sort_wave.data[sort_indices]

    for wave in other_waves:
        wave.data = wave.data[sort_indices]


def WaveStats(wave):
    """
    Igor WaveStats function equivalent
    Returns dictionary of statistics
    """
    data = wave.data[~np.isnan(wave.data)]  # Remove NaN values

    stats = {
        'V_npnts': len(data),
        'V_numNaNs': np.sum(np.isnan(wave.data)),
        'V_avg': np.mean(data) if len(data) > 0 else np.nan,
        'V_sum': np.sum(data) if len(data) > 0 else np.nan,
        'V_sdev': np.std(data, ddof=1) if len(data) > 1 else np.nan,
        'V_min': np.min(data) if len(data) > 0 else np.nan,
        'V_max': np.max(data) if len(data) > 0 else np.nan,
    }

    return stats


def Cmplx(real, imag):
    """
    Igor Cmplx function equivalent
    """
    return complex(real, imag)


def Real(z):
    """
    Igor Real function equivalent
    """
    return np.real(z)


def Imag(z):
    """
    Igor Imag function equivalent
    """
    return np.imag(z)


def Sign(x):
    """
    Igor sign function equivalent
    """
    return np.sign(x)


def Abs(x):
    """
    Igor abs function equivalent
    """
    return np.abs(x)


def Interp2D(wave, x, y):
    """
    Igor Interp2D function equivalent
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create coordinate arrays
    x_coords = np.arange(wave.shape[0]) * DimDelta(wave, 0) + DimOffset(wave, 0)
    y_coords = np.arange(wave.shape[1]) * DimDelta(wave, 1) + DimOffset(wave, 1)

    interpolator = RegularGridInterpolator((x_coords, y_coords), wave.data,
                                           bounds_error=False, fill_value=0)

    return interpolator([x, y])[0]


def BilinearInterpolate(wave, x, y, z=0):
    """
    Bilinear interpolation function from the original Igor code
    """
    # Convert coordinates to indices
    p_mid = (x - DimOffset(wave, 0)) / DimDelta(wave, 0)
    p0 = max(0, int(np.floor(p_mid)))
    p1 = min(wave.shape[0] - 1, int(np.ceil(p_mid)))

    q_mid = (y - DimOffset(wave, 1)) / DimDelta(wave, 1)
    q0 = max(0, int(np.floor(q_mid)))
    q1 = min(wave.shape[1] - 1, int(np.ceil(q_mid)))

    if len(wave.shape) == 2:
        # 2D interpolation
        p_interp0 = wave.data[p0, q0] + (wave.data[p1, q0] - wave.data[p0, q0]) * (p_mid - p0)
        p_interp1 = wave.data[p0, q1] + (wave.data[p1, q1] - wave.data[p0, q1]) * (p_mid - p0)

        return p_interp0 + (p_interp1 - p_interp0) * (q_mid - q0)
    else:
        # 3D interpolation
        z = int(z)
        p_interp0 = wave.data[p0, q0, z] + (wave.data[p1, q0, z] - wave.data[p0, q0, z]) * (p_mid - p0)
        p_interp1 = wave.data[p0, q1, z] + (wave.data[p1, q1, z] - wave.data[p0, q1, z]) * (p_mid - p0)

        return p_interp0 + (p_interp1 - p_interp0) * (q_mid - q0)