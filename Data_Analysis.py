from scipy.signal import savgol_filter, find_peaks, peak_widths

import Data_Visualization


def find_spring_peaks(time_series, window_length, polyorder, prominence_threshold, distance, show_plot=False, save_plot=False):
    # Smooth the signal using Savitzky-Golay filter
    smoothed_signal = savgol_filter(time_series.values, window_length, polyorder)

    # Find peaks in the smoothed signal
    peaks, _ = find_peaks(smoothed_signal, prominence=prominence_threshold, distance=distance)

    # Calculate peak width using the smoothed signal
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed_signal, peaks, rel_height=0.5)

    # Create a DataFrame for peak information
    peak_data = pd.DataFrame({'Datetime': time_series.index[peaks], 'Peak Value': smoothed_signal[peaks], 'Peak Width': widths})

    if show_plot:
        Data_Visualization.show_interactive_peak_plot(time_series, smoothed_signal, peaks)
    if save_plot:
        pass  # write a function that creates a static matplotlib plot that is saved as a pdf

    # Return the peak width values
    return widths, peak_data

def newFun():
    pass  # test