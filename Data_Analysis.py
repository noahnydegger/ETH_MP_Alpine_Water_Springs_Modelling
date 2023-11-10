# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import csv  # read and write csv files
import os  # interaction with operating system
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt, peak_widths, correlate

import Data_Visualization
import Helper


def cross_correlation_time_series_multiple(spring_name, meteo_names, resampled_spring_dfs, resampled_meteo_dfs, path_to_plot_folder, data_type, resolution='H', range_of_days=10):
    if data_type == 'precipitation':
        col_names = ('discharge(L/min)', 'rre150h0')
    elif data_type == 'temperature':
        col_names = ('temperature(C)', 'temperature(C)')
    else:
        print("No such data type. Choose: 'precipitation' or 'temperature'")
        return "-"
        # Handle other cases

    # select spring dataframe
    spring_df = resampled_spring_dfs[spring_name][resolution][col_names[0]]

    # Ensure that the spring_df series has a date-time index
    if not isinstance(spring_df.index, pd.DatetimeIndex):
        raise ValueError("The spring series must have a date-time index.")

    # Fill rows containing nan from the spring_df: otherwise correlate does not work
    if data_type == 'precipitation':
        spring_df = spring_df.ffill().bfill()
    else:  # for temperature
        spring_df = spring_df.rolling(window=7*24).mean().ffill().bfill()

    # Initialize dicts and lists to store time lags and Pearson correlations for each meteo_df
    corr_dfs = {}

    for i, meteo_name in enumerate(meteo_names):
        # select meteo dataframe
        meteo_df = resampled_meteo_dfs[meteo_name][resolution][col_names[1]]
        # Ensure that the meteo_df series has a date-time index
        if not isinstance(meteo_df.index, pd.DatetimeIndex):
            raise ValueError("The meteo_df series must have a date-time index.")

        # Fill rows containing nan from the meteo_df: otherwise correlate does not work
        if data_type == 'precipitation':
            meteo_df.fillna(0, inplace=True)
        else:  # for temperature
            meteo_df = meteo_df.rolling(window=7*24).mean().ffill().bfill()

        # Align the spring and meteo_df series by the date-time index
        common_index = spring_df.index.intersection(meteo_df.index)
        spring_df_aligned = spring_df.loc[common_index]
        meteo_df_aligned = meteo_df.loc[common_index]

        # Calculate the standard deviations
        std1 = np.std(spring_df_aligned)
        std2 = np.std(meteo_df_aligned)

        # Calculate the cross-correlation using scipy.correlate
        cross_corr = correlate(spring_df_aligned.values, meteo_df_aligned.values, mode='full', method='fft')

        # Calculate the Pearson correlation coefficient
        pearson_corr = cross_corr / (len(spring_df_aligned) * std1 * std2)

        # Extract the datetime index from the time series
        index = spring_df_aligned.index

        # Calculate the time lags based on the index
        time_lags_neg = index[0] - index
        time_lags_neg = time_lags_neg[::-1]
        time_lags_pos = index[1:] - index[0]
        time_lags = time_lags_neg.union(time_lags_pos)
        time_lags_in_days = time_lags.total_seconds() / (60 * 60 * 24)  # Convert to days

        # Store time lags and Pearson correlations as a dataframe in the dict
        corr_df = pd.DataFrame(index=time_lags)
        corr_df['time_lag(days)'] = time_lags_in_days
        corr_df['Pearson_corr'] = pearson_corr
        corr_df['cross_corr'] = cross_corr

        corr_dfs[meteo_name] = corr_df

    # Plot the cross-correlation for all meteo_df
    save_path = os.path.join(path_to_plot_folder, 'spring_plots', f'spring_{data_type}_correlation')
    Helper.create_directory(save_path)
    Data_Visualization.plot_cross_correlation_spring_meteo_multiple(spring_name, meteo_names, corr_dfs, range_of_days, save_path)

    return corr_dfs


def find_spring_peaks(spring_name, discharge_df, path_to_plot_folder, window_length, polyorder, prominence_threshold, distance, show_plot=False, save_plot=False):
    # select the spring data
    discharge_df = discharge_df['discharge(L/min)']
    # Smooth the signal using Savitzky-Golay filter
    smoothed_signal = savgol_filter(discharge_df.values, window_length, polyorder)
    smoothed_signal = discharge_df.rolling(window=12*60//10).mean().ffill().bfill()

    # calculate the prominence threshold
    prominence_threshold = max(discharge_df.mean()//10, prominence_threshold)

    # Find peaks in the smoothed signal
    peaks, _ = find_peaks(smoothed_signal, prominence=prominence_threshold, distance=distance)

    # Calculate peak width using the smoothed signal
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed_signal, peaks, rel_height=0.3)

    # Create a DataFrame for peak information
    peak_data = pd.DataFrame({'Datetime': discharge_df.index[peaks], 'Peak Value(L/min)': smoothed_signal[peaks], 'Peak Width(h)': widths/10})

    # Convert peak width to date range
    peak_data['Start Time'] = peak_data['Datetime'] - pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')
    peak_data['End Time'] = peak_data['Datetime'] + pd.to_timedelta(peak_data['Peak Width(h)'], unit='hours')

    if show_plot:
        Data_Visualization.show_interactive_peak_plot(spring_name, discharge_df, smoothed_signal, peaks)
    if save_plot:
        save_path = os.path.join(path_to_plot_folder, 'spring_plots', 'peak_detection')
        Helper.create_directory(save_path)
        Data_Visualization.save_static_peak_plot(spring_name, discharge_df, smoothed_signal, peaks, save_path)

    # Return the peak width values
    return peak_data


def spring_peaks_statistics(resampled_spring_data_dfs, spring_peaks_dfs):
    column_list = ['data_duration(days)', 'peak_count(-)', 'mean_width(hours)', 'std_width(hours)', 'min_width(hours)', '25%_width(hours)', '50%_width(hours)', '75%_width(hours)', 'max_width(hours)']
    index_list = list(spring_peaks_dfs.keys())
    peak_statistics = pd.DataFrame(columns=column_list, index=spring_peaks_dfs.keys())
    stats = []
    for spring_name, peak_df in spring_peaks_dfs.items():
        spring_df = resampled_spring_data_dfs[spring_name]['D']['discharge(L/min)']
        stats.append(spring_df.dropna().shape[0])  # data duration = number of days not counting nan
        if not peak_df.empty:
            stats.append(peak_df['Peak Value(L/min)'].count())  # peak count
            stats.append(peak_df['Peak Width(h)'].mean())  # mean width
            stats.append(peak_df['Peak Width(h)'].std())  # mean width
            stats.append(peak_df['Peak Width(h)'].min())  # min width
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.25))  # 1st quartile
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.5))  # median width
            stats.append(peak_df['Peak Width(h)'].quantile(q=0.75))  # d3rd quartile
            stats.append(peak_df['Peak Width(h)'].max())  # min width
        else:
            stats.extend([0] * (len(column_list) - 1))

        peak_statistics.loc[spring_name] = stats
        stats = []

    return peak_statistics
