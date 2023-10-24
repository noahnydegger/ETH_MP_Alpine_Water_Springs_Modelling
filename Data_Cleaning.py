import pandas as pd  # data processing
import numpy as np  # data processing
import os  # interaction with operating system
import Helper


def keep_only_valid_data(df):
    return df[df.valid]


def interpolate_below_threshold(df, column_name='discharge(L/min)', threshold=5, method='linear'):
    df_copy = df.copy()  # Create a copy of the DataFrame
    mask = df_copy[column_name] < threshold
    df_copy.loc[mask, column_name] = np.NaN  # Replace values below the threshold with NaN
    df_copy[column_name] = df_copy[column_name].interpolate(method=method)
    return df_copy


def add_timezone_to_dataframe(df, timezone_str):
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(timezone_str)
    else:
        df.index = pd.to_datetime(df.index).tz_localize(timezone_str)


def resample_and_save_spring_data(df, resolutions, save_path, spring_name):
    # create the folder for the spring if it does not exist yet
    spring_folder = os.path.join(save_path, spring_name)
    Helper.create_directory(spring_folder)
    # save the original valid data
    filename = f"{spring_name}_{'10min'}.csv"
    file_path = os.path.join(spring_folder, filename)
    df.to_csv(file_path)

    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)

    resampled_dfs = {'10min': df}  # Dictionary to store the resampled dataframes

    for resolution in resolutions:
        # Resample the data based on the specified resolution
        resampled_df = df.resample(resolution).mean()

        resampled_dfs[resolution] = resampled_df  # Store the resampled dataframe in the dictionary

        # Save the resampled dataframe as a CSV file
        filename = f"{spring_name}_{resolution}.csv"
        file_path = os.path.join(spring_folder, filename)
        resampled_df.to_csv(file_path)

    return resampled_dfs


def resample_and_save_precip_data(df_list, resolutions, save_path, prefix):
    resampled_dfs = {}  # Dictionary to store the resampled dataframes

    # create the folder if it does not exist yet
    Helper.create_directory(save_path)
    # save the original 10 min data
    if 'rre150z0' in df_list[0].columns:
        filename = f"{prefix}_{'10min'}.csv"
        file_path = os.path.join(save_path, filename)
        df_10min = df_list[0].loc[:, ['rre150z0']]
        df_10min.to_csv(file_path)
        resampled_dfs['10min'] = df_10min

    # save the original hourly data
    filename = f"{prefix}_{'H'}.csv"
    file_path = os.path.join(save_path, filename)
    df_H = df_list[1].loc[:, ['rre150h0']]
    df_H.to_csv(file_path)
    resampled_dfs['H'] = df_H

    # Convert the index to datetime if it's not already
    df_H.index = pd.to_datetime(df_H.index)

    for resolution in resolutions:
        # Resample the data based on the specified resolution
        resampled_df = df_H.resample(resolution).sum()

        resampled_dfs[resolution] = resampled_df  # Store the resampled dataframe in the dictionary

        # Save the resampled dataframe as a CSV file
        filename = f"{prefix}_{resolution}.csv"
        file_path = os.path.join(save_path, filename)
        resampled_df.to_csv(file_path)

    return resampled_dfs
