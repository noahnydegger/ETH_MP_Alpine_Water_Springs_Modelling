import pandas as pd  # data processing
import numpy as np  # data processing
from dateutil.parser import parse
from io import StringIO
import os  # interaction with operating system


def import_spring_data(data_directory):
    spring_names = []
    spring_description = []
    spring_data_paths = []
    spring_data_dfs = {}

    files = os.listdir(data_directory)
    files.sort()
    for filename in files:
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.csv'):  # check only csv files
            spring_names.append(filename.replace('_discharge.csv', ''))
            spring_description.append(spring_description_from_filename(filename))
            spring_data_paths.append(filepath)
            spring_data_dfs[spring_names[-1].split('.')[-1]] = import_data_from_csv_file(filepath)
    return spring_names, spring_description, spring_data_paths, spring_data_dfs


def import_data_from_csv_file(filepath):
    df = pd.read_csv(filepath)  # read csv
    df['datetime'] = pd.to_datetime(df['datetime'])  # convert column to datetime format
    df.set_index('datetime', inplace=True)  # set date as index
    return df


def spring_description_from_filename(filename):
    filename_split = filename.split('.')
    spring_name_split = filename_split[2].split('_')
    spring_name_split.remove('discharge')
    spring_description = '{} spring at {}'.format(' '.join([item for item in spring_name_split]), filename_split[0])
    return spring_description


def find_file_with_pattern(folder_path, pattern='data.txt'):

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if pattern in filename:
                file_path = os.path.join(root, filename)

    return file_path


def read_multi_dataset_txt_file(file_path, delimiter, dataset_separator=''):
    """
    Read a .txt file containing multiple datasets into a list of Pandas DataFrames.

    Args:
        file_path (str): The path to the .txt file.
        delimiter (str): The delimiter used to separate columns within each dataset.
        dataset_separator (str): The string that separates datasets within the file.

    Returns:
        dataframes (list of pd.DataFrame): List of DataFrames, each containing one dataset.
    """
    dataframes = []
    current_dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == dataset_separator:
                if current_dataset:
                    dataframes.append(create_dataframe_from_data_list(current_dataset, delimiter))
                current_dataset = []
            else:
                current_dataset.append(line)

        # If the last dataset is not followed by a separator, add it to the list
        if current_dataset:
            dataframes.append()

    return dataframes


def create_dataframe_from_data_list(data_list, delimiter=';'):
    # Split the header to obtain column names
    header = data_list[0].split(delimiter)

    # Create a dictionary to store data for each column
    data_dict = {col: [] for col in header}
    # Infer datetime format based on the number of characters in the 'time' column
    time_format = '%Y%m%d%H' if len(data_list[1].split(delimiter)[1]) == 10 else '%Y%m%d%H%M'

    # Iterate through the data lines and split values accordingly
    for line in data_list[1:]:
        values = line.split(delimiter)
        data_dict[header[0]].append(values[0])  # add station spring_name
        data_dict['time'].append(values[1])
        for col, value in zip(header[2:], values[2:]):  # Start from the 3rd column
            if value == '-':
                data_dict[col].append(np.nan)
            else:
                data_dict[col].append(value)

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    # Convert specific columns to appropriate data types if needed
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'], format=time_format)
    # set time colum as index
    df.set_index('datetime', inplace=True)
    for col in header[2:]:  # Assuming columns from the 3rd onwards are numeric values
        df[col] = df[col].astype(float)  # Convert to float

    return df
