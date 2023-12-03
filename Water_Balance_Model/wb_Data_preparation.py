from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle


def precipitation_correlation(path_to_data_folder):
    save_path = path_to_data_folder / 'meteo_data' / 'model_input_data' / 'precipitation'
    if not save_path.exists():
        save_path.mkdir(parents=True)


def create_model_input_discharge(spring_name, path_to_data_folder):
    path_to_spring_pkl = path_to_data_folder / 'spring_data' / 'resampled_data' / 'resampled_spring_data_dfs.pkl'
    with open(path_to_spring_pkl, 'rb') as file:
        resampled_spring_data_dfs = pickle.load(file)

    wb_discharge = resampled_spring_data_dfs[spring_name]['H']
    wb_discharge.drop(columns=['temperature(C)'], inplace=True)

    wb_discharge['valid'] = True  # set to False later, where spring or meteo data is missing

    # exclude (set valid to False) data gaps and invalid measurements from the calibration period
    wb_discharge['valid'] = np.where(wb_discharge['discharge(L/min)'].isna(), False, wb_discharge['valid'])
    wb_discharge['valid'] = np.where(wb_discharge['discharge(L/min)'] < 5.0, False, wb_discharge['valid'])
    wb_discharge['valid'] = np.where(wb_discharge['discharge(L/min)'] > 2000.0, False, wb_discharge['valid'])

    # limit values to 2000
    #wb_discharge['discharge(L/min)'] = np.minimum(wb_discharge['discharge(L/min)'], 2000)

    wb_discharge['discharge(L/min)'].interpolate(inplace=True)
    #wb_discharge['discharge(L/min)'] = wb_discharge['discharge(L/min)'].ffill().bfill()
    #wb_discharge['discharge(L/min)'].bfill(inplace=True)

    # export as pickle and csv
    file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_discharge.pkl'
    file_path.parent.mkdir(parents=True, exist_ok=True)
    wb_discharge.to_pickle(file_path)
    wb_discharge.to_csv(file_path.with_suffix('.csv'))

    return wb_discharge


def create_model_input_precipitation(spring_name, meteo_names, path_to_data_folder):
    path_to_precip_pkl = path_to_data_folder / 'meteo_data' / 'resampled_precip_data' / 'resampled_precip_data_dfs.pkl'
    with open(path_to_precip_pkl, 'rb') as file:
        resampled_precip_data_dfs = pickle.load(file)

    main_station = meteo_names[0]
    wb_precip = resampled_precip_data_dfs[main_station]['H']
    wb_precip['valid'] = True  # set to False later, where spring or meteo data is missing

    wb_precip.rename(columns={'rre150h0': main_station}, inplace=True)
    wb_precip['precipitation(mm)'] = wb_precip[main_station]
    wb_precip['valid'] = np.where(wb_precip['precipitation(mm)'].isna(), False, wb_precip['valid'])
    for meteo_name in meteo_names[1:]:
        wb_precip = wb_precip.merge(resampled_precip_data_dfs[meteo_name]['H'], how='left',
                                                          left_index=True,  right_index=True)
        wb_precip.rename(columns={'rre150h0': meteo_name}, inplace=True)
        wb_precip['precipitation(mm)'] = wb_precip['precipitation(mm)'].fillna(wb_precip[meteo_name])

    wb_precip['precipitation(mm)'].fillna(0, inplace=True)

    # remove column with raw data
    wb_precip.drop(columns=meteo_names, inplace=True)

    # export as pickle and csv
    file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_precipitation.pkl'
    file_path.parent.mkdir(parents=True, exist_ok=True)
    wb_precip.to_pickle(file_path)
    wb_precip.to_csv(file_path.with_suffix('.csv'))

    return wb_precip


def create_model_input_temperature(spring_name, meteo_names, elevation_difference, path_to_data_folder):
    path_to_temp_pkl = path_to_data_folder / 'meteo_data' / 'resampled_temp_data' / 'resampled_temp_data_dfs.pkl'
    with open(path_to_temp_pkl, 'rb') as file:
        resampled_temp_data_dfs = pickle.load(file)

    main_station = meteo_names[0]
    wb_temp = resampled_temp_data_dfs[main_station]['H']
    wb_temp['valid'] = True  # set to False later, where spring or meteo data is missing

    wb_temp.rename(columns={'temperature(C)': main_station}, inplace=True)

    wb_temp['final_temp'] = wb_temp[main_station]

    wb_temp['valid'] = np.where(wb_temp['final_temp'].isna(), False, wb_temp['valid'])

    for meteo_name in meteo_names[1:]:
        wb_temp = wb_temp.merge(resampled_temp_data_dfs[meteo_name]['H'], how='left',
                                    left_index=True, right_index=True)
        wb_temp.rename(columns={'temperature(C)': meteo_name}, inplace=True)
        wb_temp['final_temp'] = wb_temp['final_temp'].fillna(wb_temp[meteo_name])# - (0.5 * elevation_difference / 100))
        #wb_temp[f'lapsed{meteo_name}'] = wb_temp[meteo_name] - (0.5 * elevation_difference / 100)

    # fill nan values
    wb_temp['final_temp'] = fill_nan_with_monthly_average(wb_temp['final_temp'])

    wb_temp.rename(columns={'final_temp': 'temperature(C)'}, inplace=True)

    '''
    average_difference = (wb_temp[main_station] - wb_temp[meteo_name]).mean()
    print("\nAverage Difference:", average_difference)
    print("\nLapse rate", - (0.5 * elevation_difference / 100))
    # Create a scatter plot
    plt.scatter(wb_temp[main_station], wb_temp[meteo_name], facecolors='none', edgecolors='red', marker='o', s=100)
    plt.scatter(wb_temp[main_station], wb_temp[f'lapsed{meteo_name}'], facecolors='none', edgecolors='green', marker='o', s=100)
    plt.plot(wb_temp[main_station], wb_temp[main_station], color='blue', linestyle='--')

    # Add labels and title
    plt.xlabel(main_station)
    plt.ylabel(meteo_name)
    plt.title('Scatter Plot of Precipitation Data')

    # Show the plot
    plt.show()
    '''

    # remove column with raw data
    wb_temp.drop(columns=meteo_names, inplace=True)

    # export as pickle and csv
    file_path = path_to_data_folder / 'water_balance_model' / spring_name / f'wb_{spring_name}_temperature.pkl'
    file_path.parent.mkdir(parents=True, exist_ok=True)
    wb_temp.to_pickle(file_path)
    wb_temp.to_csv(file_path.with_suffix('.csv'))

    return wb_temp


# Function to fill NaN values with the monthly average
def fill_nan_with_monthly_average(series):
    return series.fillna(series.groupby([series.index.month, series.index.hour]).transform('mean'))


def correlation_function(spring_name, path_to_data_folder):
    path_to_precip_pkl = path_to_data_folder / 'meteo_data' / 'resampled_precip_data' / 'resampled_precip_data_dfs.pkl'
    with open(path_to_precip_pkl, 'rb') as file:
        resampled_precip_data_dfs = pickle.load(file)
    station1 = resampled_precip_data_dfs['Freienbach']['H'].fillna(0)
    station2 = resampled_precip_data_dfs['Oberriet_Moos']['H'].fillna(0)

    common_index = station1.index.intersection(station2.index)
    station1_aligned = station1.loc[common_index]
    station2_aligned = station2.loc[common_index]

    # Create a scatter plot
    plt.scatter(station1_aligned, station2_aligned, color='blue', )

    # Add labels and title
    plt.xlabel('Precipitation Series 1')
    plt.ylabel('Precipitation Series 2')
    plt.title('Scatter Plot of Precipitation Data')

    # Show the plot
    plt.show()

    cross_corr = signal.correlate(station1_aligned.values, station2_aligned.values, mode='full', method='fft')

    # Calculate the standard deviations
    std1 = np.std(station1_aligned)
    std2 = np.std(station2_aligned)
    # Calculate the Pearson correlation coefficient
    pearson_corr = cross_corr / (len(station1_aligned) * std1 * std2)

    # Extract the datetime index from the time series
    index = station1_aligned.index

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




def verify_model_input_data(spring_name, path_to_data_folder):
    for var in ['precipitation', 'temperature']:
        folder = precipitation_folder = path_to_data_folder / 'meteo_data' / 'model_input_data' / var
        if not folder.exists():
            folder.mkdir(parents=True)


def import_data_from_csv_file(filepath):
    df = pd.read_csv(filepath)  # read csv
    df['datetime'] = pd.to_datetime(df['datetime'])
    # convert column to datetime format
    df.set_index('datetime', inplace=True)  # set date as index
    return df
