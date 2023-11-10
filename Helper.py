import pandas as pd  # data processing
import os  # interaction with operating system


def csv_data_to_dataframe(filepath, columns_dict={}, decimalFormat='.', encodingFormat='utf-8', dateFormat=None):
    df = pd.read_csv(filepath, usecols=list(columns_dict.keys()), sep=";", decimal=decimalFormat, encoding=encodingFormat)  # read csv
    df.rename(columns=columns_dict, inplace=True)  # rename columns
    df['Date'] = pd.to_datetime(df['Date'], format=dateFormat)  # convert column to datetime format
    df.set_index('Date', inplace=True)  # set date as index
    return df


def excel_data_to_dataframe(filepath, sheetname):
    excelfile = pd.ExcelFile(filepath)
    df = pd.read_excel(excelfile, sheetname)
    #wb_df.(columns=columns_dict, inplace=True)  # rename columns
    return df


def find_file_by_name(filename, startFolder, filetype):
    # search for the filepath of a single file
    if filetype.lower() not in filename.lower():
        filename = '{}.{}'.format(filename, filetype.lower())  # add filetype

    filefound = False
    for root, dirs, files in os.walk(startFolder):  # Walking top-down from the startFolder looking for the file
        if filename.lower() in [file.lower() for file in files]:
            filefound = True
            path_to_file = os.path.join(root, filename)

    if filefound:
        print('{} has been found in directory {}'.format(filename, os.path.dirname(path_to_file)))
    else:
        path_to_file = ''
        print('{} not found within directory {}'.format(filename, startFolder))
    return path_to_file


def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
