import analytic_wfm as aw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def dqdv_extraction(capacity, voltage, smoothing=(0.0, 1e-5, 0.00005, 0.0001,
                                                  0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)):

    from scipy.interpolate import splrep, splev

    def autocorr(x, t=1):
        return np.corrcoef(np.array([x[0:len(x) - t], x[t:len(x)]]))[0][1]

    # Try different smoothing factors until the smoothing measure is fulfilled

    for s in smoothing:
        # Fit the spline
        f = splrep(voltage, capacity, s=s, k=5)
        # x = voltage, y = dQ/dV
        x = np.linspace(voltage.min(), voltage.max(), 1000)
        y = abs(splev(x, f, der=1))
        smoothness = autocorr(y)
        if smoothness > 0.999:
            # Find the peaks

            main_peaks = aw.peakdetect(y, x, lookahead=lookahead, delta=delta)
            maxima, maxima_height = zip(*main_peaks[0])
            minima, minima_height = zip(*main_peaks[1])
            # If there are more than 5 peaks then continue to smooth - just as an extra catch all

            while len(maxima) > 2:
                s += 0.00001
                f = splrep(voltage, capacity, s=s, k=5)
                y = abs(splev(x, f, der=1))
                main_peaks = aw.peakdetect(y, x, lookahead=lookahead, delta=delta)
                maxima, maxima_height = zip(*main_peaks[0])
                minima, minima_height = zip(*main_peaks[1])

            return x, y




def iterative_peak_finding(voltage, dqdv):
    """
    Function for finding the dqdv peaks for a sample, assumes that the best guess for the next cycle is the parameters
    from the previous cycle

    :param voltage: array of voltage (x) values
    :param dqdv: array of dqdv (y) values
    :return:

    """
    sns.set(font_scale=1.5)
    sns.set_style("white")
    sns.set_style('ticks')

    from lmfit import models
    from scipy.signal import savgol_filter

    return init_peak_center, initi_peak_height, peak_center, peak_height, peak_width