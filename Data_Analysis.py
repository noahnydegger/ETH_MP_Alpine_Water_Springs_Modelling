# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import csv  # read and write csv files
import os  # interaction with operating system
from scipy.signal import savgol_filter, find_peaks, peak_widths

import Data_Visualization
import Helper


def find_spring_peaks(name, discharge_df, path_to_plot_folder, window_length, polyorder, prominence_threshold, distance, show_plot=False, save_plot=False):
    # select the spring data
    discharge_df = discharge_df['discharge(L/min)']
    # Smooth the signal using Savitzky-Golay filter
    smoothed_signal = savgol_filter(discharge_df.values, window_length, polyorder)

    # Find peaks in the smoothed signal
    peaks, _ = find_peaks(smoothed_signal, prominence=prominence_threshold, distance=distance)

    # Calculate peak width using the smoothed signal
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed_signal, peaks, rel_height=0.5)

    # Create a DataFrame for peak information
    peak_data = pd.DataFrame({'Datetime': discharge_df.index[peaks], 'Peak Value(L/min)': smoothed_signal[peaks], 'Peak Width(h)': widths/10})

    # Convert peak width to date range
    peak_data['Start Time'] = peak_data['Datetime'] - pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')
    peak_data['End Time'] = peak_data['Datetime'] + pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')

    if show_plot:
        Data_Visualization.show_interactive_peak_plot(name, discharge_df, smoothed_signal, peaks)
    if save_plot:
        save_path = os.path.join(path_to_plot_folder, 'spring_plots', 'peak_detection')
        Helper.create_directory(save_path)
        Data_Visualization.save_static_peak_plot(name, discharge_df, smoothed_signal, peaks, save_path)

    # Return the peak width values
    return peak_data
