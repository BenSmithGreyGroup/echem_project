import pandas as pd
import numpy as np

from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter


def dqdq_plotting(capacity, voltage, window_size=201, polyorder=5, smoothing=1e-6):
    """
    Function to find the dQdV from a given capacity and voltage values.
    :param capacity: Array-like capacity values
    :param voltage: Array-like voltage values - should be in increasing order
    :param window_size: Window size for initial smoothing using scipy.signal.savgol_filter
    :param polyorder: polynomail order for smoothing using scipy.signal.savgol_filter
    :param smoothing: the smoothing value for fitting the spline to
    see scipy.interpolate.splrep for details - is the s value for this function
    :return: voltage, dqdv_rough, dqdv_smooth, capacity
    """
    # Literal spline - functionalising  voltage as a function of capacity - perfectly fit to every data point,
    # linearly interpolating between each point
    f_lit = splrep(capacity, voltage, s=0.0, k=1)

    lin_cap = np.linspace(min(capacity), max(capacity), num=1000)
    lin_volt = splev(lin_cap, f_lit)

    # Smoothing the set length linear voltage
    smooth_v = savgol_filter(lin_volt, window_size, polyorder, mode='interp')

    # Making sure voltage runs from low to high so splines work - no dV < 0  - after smoothing
    smooth_df = pd.DataFrame({'Voltage': smooth_v, 'Capacity': lin_cap})
    smooth_df.sort_values('Voltage', inplace=True)
    smooth_df['dV'] = smooth_df['Voltage'].diff()
    smooth_df.drop(smooth_df[smooth_df['dV'] < 0].index, inplace=True)

    # Final smooth voltage and capacity values - will be the same length, possiblt less than 1000
    smooth_v = smooth_df['Voltage']
    x_cap = smooth_df['Capacity']

    if sum(np.diff(smooth_v)) < 0:
        f = splrep(np.flip(smooth_v, axis=0), np.flip(x_cap, axis=0), s=smoothing, k=3)
    else:
        f = splrep(smooth_v, x_cap, s=smoothing, k=3)

    # x = voltage, y = dQ/dV
    x = np.linspace(voltage.min(), voltage.max(), 1000)
    y = abs(splev(x, f, der=1))

    spline_cap = abs(splev(x, f))

    smooth_dqdv = savgol_filter(y, 201, 5)
    return x, y, smooth_dqdv, spline_cap
