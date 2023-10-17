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


def resample_and_save_dataframes(df, resolutions, path, prefix):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)

    resampled_dfs = {'raw': df}  # Dictionary to store the resampled dataframes

    for resolution in resolutions:
        # Resample the data based on the specified resolution
        resampled_df = df.resample(resolution).mean()

        resampled_dfs[resolution] = resampled_df  # Store the resampled dataframe in the dictionary

        # Save the resampled dataframe as a CSV file
        Helper.create_directory(path)
        filename = f"{prefix}_{resolution}.csv"
        file_path = os.path.join(path, filename)
        resampled_df.to_csv(file_path)

    return resampled_dfs
