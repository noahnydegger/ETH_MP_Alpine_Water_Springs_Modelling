import pandas as pd  # data processing
import numpy as np  # data processing
import os  # interaction with operating system


def import_spring_data(data_directory):
    spring_names = []
    spring_description = []
    spring_data_paths = []
    spring_data_dfs = []

    files = os.listdir(data_directory)
    files.sort()
    for filename in files:
        filepath = os.path.join(data_directory, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.csv'):  # check only csv files
            spring_names.append(filename.replace('_discharge.csv', ''))
            spring_description.append(spring_description_from_filename(filename))
            spring_data_paths.append(filepath)
            spring_data_dfs.append(import_data_from_csv_file(filepath))
    return spring_names, spring_description, spring_data_paths, spring_data_dfs


def import_data_from_csv_file(filepath):
    df = pd.read_csv(filepath)  # read csv
    df['datetime'] = pd.to_datetime(df['datetime'])  # convert column to datetime format
    df.set_index('datetime', inplace=True)  # set date as index
    return df


def import_data_from_url():
    pass


def spring_description_from_filename(filename):
    filename_split = filename.split('.')
    spring_name_split = filename_split[2].split('_')
    spring_name_split.remove('discharge')
    spring_description = '{} spring at {}'.format(' '.join([item for item in spring_name_split]), filename_split[0])
    return spring_description
