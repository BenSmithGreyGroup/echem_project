import analytic_wfm as aw
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def dqdv_peak_extraction(capacity, voltage, sample, cycle, plot=True,
                    smoothing=(0.0, 1e-6, 0.00001, 0.00005, 0.0001),
                    lookahead=100, delta=0.0,
                    smoothing_cond=0.99, auto_correlation_window=5):
    """

    :param capacity: Raw capacity data in array-like form
    :param voltage: Raw voltage data in array-like form
    :param sample: sample id
    :param cycle: cycle number
    :param plot: if you want to produce a plot
    :param smoothing: the smoothing values to try - in list-like format
    :param lookahead: the number of data points separating peaks (minimum) - will be 1000 data points in the voltage
    range
    :param delta: the minimum height above the adjacent minima for the peak
    :param smoothing_cond: the autocorrelation smoothing value to accept -
    defines how smooth the differentiated function is
    :param auto_correlation_window: the autocorrelation window length
    :return:
    """

    def plot_dqdv_from_extraction(raw_cap, raw_volt, linear_capacity, smoothed_voltage, spline, linear_voltage,
                                  dqdv, final_smooth_dqdv, peak_pos, peak_height, smoothing_factor,
                                  label=sample, cycle_num=cycle):
        sns.set_style('white')
        sns.set_style('ticks')

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(raw_cap, raw_volt, 'bd', label='raw data')
        ax[0].set_ylabel('Voltage / V')
        ax[0].set_xlabel('Capacity / mAh')
        ax[0].plot(linear_capacity, smoothed_voltage, label='initial smoothing', color='red')
        ax[0].plot(abs(splev(linear_voltage, spline)), linear_voltage, label='spline fitted', color='green')
        ax[0].legend()

        ax[1].plot(linear_voltage, dqdv, color='blue', label='Spline output')
        ax[1].plot(linear_voltage, final_smooth_dqdv, color='red', label='Smoothed')

        if peak_pos is not None:
            ax[1].scatter(peak_pos, peak_height, color='orange', zorder=5)
        else:
            print(f'{sample}, cycle {(cycle_num+1)/2} - No peaks found')

        ax[1].set_xlabel('Voltage / V')
        ax[1].set_ylabel('dQ/dV / mAh/V')
        ax[1].set_title(f'{label} cycle {(cycle_num+1)/2}, smoothing = {smoothing_factor}')
        ax[1].legend()
        plt.show()
        plt.close()
        return

    def autocorr(x, t=1):
        return np.corrcoef(np.array([x[0:len(x) - t], x[t:len(x)]]))[0][1]

    from scipy.interpolate import splrep, splev
    from scipy.signal import savgol_filter

    # Literal spline - functionalising  voltage as a function of capacity - perfectly fit to every data point,
    # linearly interpolating between each point
    f_lit = splrep(capacity, voltage, s=0.0, k=1)

    lin_cap = np.linspace(min(capacity), max(capacity), num=1000)
    lin_volt = splev(lin_cap, f_lit)

    # Smoothing the set length linear voltage
    smooth_v = savgol_filter(lin_volt, 201, 5, mode='interp')

    # Making sure voltage runs from low to high so splines work - no dV < 0  - after smoothing
    smooth_df = pd.DataFrame({'Voltage': smooth_v, 'Capacity': lin_cap})
    smooth_df.sort_values('Voltage', inplace=True)
    smooth_df['dV'] = smooth_df['Voltage'].diff()
    smooth_df.drop(smooth_df[smooth_df['dV'] < 0].index, inplace=True)

    # Final smooth voltage and capacity values - will be the same length, possiblt less than 1000
    smooth_v = smooth_df['Voltage']
    x_cap = smooth_df['Capacity']

    # Try different smoothing factors until the smoothing measure is fulfilled
    for s in smoothing:
        # Fit the spline
        if sum(np.diff(smooth_v)) < 0:
            f = splrep(np.flip(smooth_v, axis=0), np.flip(x_cap, axis=0), s=s, k=3)
        else:
            f = splrep(smooth_v, x_cap, s=s, k=3)

        # x = voltage, y = dQ/dV
        x = np.linspace(voltage.min(), voltage.max(), 1000)
        y = abs(splev(x, f, der=1))
        smoothness = autocorr(y, t=auto_correlation_window)

        # If smoothing condition is satisfied then plot and return results
        if smoothness > smoothing_cond:
            break

    smoothed_dqdv = savgol_filter(y, 301, 5)
    main_peaks = aw.peakdetect(smoothed_dqdv, x, lookahead=lookahead, delta=delta)
    print(main_peaks)

    if len(main_peaks[0]) > 0:
        maxima, maxima_height = zip(*main_peaks[0])
    else:
        maxima = None
        maxima_height = None

    if plot:
        plot_dqdv_from_extraction(capacity, voltage, x_cap, smooth_v, f, x, y, smoothed_dqdv, maxima,
                              maxima_height, s, sample, cycle)

    return x, smoothed_dqdv, s, main_peaks, f
