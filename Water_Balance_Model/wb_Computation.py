import pandas as pd  # data processing
import numpy as np  # data processing


def compute_average_temperature():
    pass


def compute_rain_and_snow(df, melt_rate, melt_temp):
    df['rain(mm)'] = df['precipitation(mm)']

    '''
    # compute snow fall
    wb_df['rain(mm)'] = np.where(wb_df['min_temperature(C)'] > melt_temp, wb_df['precipitation(mm)'], 0)
    wb_df['snow_fall(mm)'] = np.where(wb_df['max_temperature(C)'] < melt_temp, wb_df['precipitation(mm)'], 0)

    # Compute Snow melt
    if T_b[i] > Tsm:
        M[i] = min(melt_rate * (T_b[i] - Tsm), SC[i - 1])

    # Compute Snow cover (SC)
    SC[i] = SC[i - 1] - M[i] + S[i]
    pass
    S(Tmax_b<=0)=P(Tmax_b<=0);
    R(Tmin_b>0)=P(Tmin_b>0);
    R(Tmin_b<=0&Tmax_b>0)=(Tmax_b(Tmin_b<=0&Tmax_b>0).*P(Tmin_b<=0&Tmax_b>0))./(Tmax_b(Tmin_b<=0&Tmax_b>0)-Tmin_b(Tmin_b<=0&Tmax_b>0));
    S(Tmin_b<=0&Tmax_b>0)=P(Tmin_b<=0&Tmax_b>0)-R(Tmin_b<=0&Tmax_b>0);
    '''


def compute_daily_PET_by_Hamon():
    # Compute ET with routine by Hamon
    '''
    if T_b[i] > 0:
        delta = 0.4093 * np.sin((2 * np.pi / 365) * Jday[i] - 1.405)
        omega_s = np.arccos(-np.tan(2 * np.pi * Lat / 360) * np.tan(delta))
        Nt = 24 * omega_s / np.pi
        es = PETa * np.exp(PETb * T_b[i] / (T_b[i] + PETc))
        PET[i] = (2.1 * (Nt ** 2) * es) / (T_b[i] + 273.3)
        ET[i] = (Ss[i - 1] / max_saturation) * PET[i]
        '''


def compute_water_balance_iterative(wb_df, fixed_parameters, variable_parameters):
    use_sigmoid = True
    # Parameters for calibration
    area, max_saturation, rg, melt_rate, fr, melt_temp, lapse_rate = variable_parameters

    # set initial values
    wb_df.loc[wb_df.index[0], ['storage_soil(mm)']] = [10]

    #compute_rain_and_snow(wb_df, melt_rate, melt_temp)
    #compute_daily_PET_by_Hamon()

    y = wb_df.index.tolist()[0]
    # iterate over df row by row
    for i in wb_df.index.tolist():
        # Compute potential soil zone water content without accounting for percolation
        wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][y] + wb_df['rain_fall(mm)'][i] - wb_df['aET(mm)'][i]

        if use_sigmoid:
            sigma = soil_storage_sigmoid(wb_df.loc[i, 'storage_soil(mm)'], max_saturation)
            # Compute surface runoff
            wb_df.loc[i, 'runoff(mm)'] = (wb_df['rain_fall(mm)'][i] + wb_df['snow_melt(mm)'][i]) * fr * sigma
            wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][i] - wb_df['runoff(mm)'][i]

            # Compute percolation to groundwater reservoir
            sigma = soil_storage_sigmoid(wb_df.loc[i, 'storage_soil(mm)'], max_saturation)
            wb_df.loc[i, 'percolation_gw(mm)'] = max((wb_df['storage_soil(mm)'][i] - max_saturation), 0)
            # adjust soil storage
            wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][i] - wb_df['percolation_gw(mm)'][i]
        else:
            # Compute surface runoff
            if wb_df['storage_soil(mm)'][i] > max_saturation:
                wb_df.loc[i, 'runoff(mm)'] = (wb_df['rain_fall(mm)'][i] + wb_df['snow_melt(mm)'][i]) * fr
                wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][i] - wb_df['runoff(mm)'][i]

            # Compute percolation to groundwater reservoir
            if wb_df['storage_soil(mm)'][i] > max_saturation:
                wb_df.loc[i, 'percolation_gw(mm)'] = wb_df['storage_soil(mm)'][i] - max_saturation
                wb_df.loc[i, 'storage_soil(mm)'] = max_saturation


        # only for one tank model
        #wb_df.loc[i, 'discharge_sim(mm)'] = (1 / rg) * wb_df['storage_soil(mm)'][i] + wb_df['percolation_gw(mm)'][i]
        #wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][i] - wb_df['discharge_sim(mm)'][i]


        # Groundwater reservoir water balance
        wb_df.loc[i, 'discharge_sim(mm)'] = (1 / rg) * wb_df['storage_gw(mm)'][y]
        wb_df.loc[i, 'storage_gw(mm)'] = max(wb_df['storage_gw(mm)'][y] + wb_df['percolation_gw(mm)'][i] - wb_df['discharge_sim(mm)'][i], 0)

        y = i  # current day is new yesterday

    # transform units
    wb_df['discharge_sim(L/min)'] = wb_df['discharge_sim(mm)'] * area / (60 * 24)
    gof_values = compute_gof_values(wb_df)

    return wb_df, gof_values


def soil_storage_sigmoid(soil_storage, max_saturation, beta=1):
    return 1 / (1 + np.exp(-beta * (soil_storage - max_saturation)))


def compute_water_balance_direct(wb_df, fixed_parameters, variable_parameters):
    # Parameters for calibration
    area, max_saturation, rg, melt_rate, fr, melt_temp, lapse_rate = variable_parameters

    # set initial values
    wb_df.loc[wb_df.index[0], ['storage_soil(mm)']] = [10]

    #compute_rain_and_snow(wb_df, melt_rate, melt_temp)
    #compute_daily_PET_by_Hamon()

    #input = wb_df['rain_fall(mm)'] + wb_df['snow_melt(mm)']

    # Compute potential soil zone water content without accounting for percolation
    wb_df['storage_soil(mm)'] = wb_df['storage_soil(mm)'].shift(-1) + input * (1 - fr)

    # Compute surface runoff
    wb_df['runoff(mm)'] = input * fr * soil_storage_sigmoid(wb_df['storage_soil(mm)'], max_saturation)


    # Compute percolation to groundwater reservoir
    if wb_df['storage_soil(mm)'][i] > max_saturation:
        wb_df.loc[i, 'percolation_gw(mm)'] = wb_df['storage_soil(mm)'][i] - max_saturation
        wb_df.loc[i, 'storage_soil(mm)'] = max_saturation


    # only for one tank model
    #wb_df.loc[i, 'discharge_sim(mm)'] = (1 / rg) * wb_df['storage_soil(mm)'][i] + wb_df['percolation_gw(mm)'][i]
    #wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][i] - wb_df['discharge_sim(mm)'][i]


    # Groundwater reservoir water balance
    wb_df.loc['discharge_sim(mm)'] = (1 / rg) * wb_df['storage_gw(mm)'].shift(-1)
    wb_df.loc['storage_gw(mm)'] = max(wb_df['storage_gw(mm)'].shift(-1) + wb_df['percolation_gw(mm)'] - wb_df['discharge_sim(mm)'], 0)

    # transform units
    wb_df['discharge_sim(L/min)'] = wb_df['discharge_sim(mm)'] * area / (60 * 24)
    gof_values = compute_gof_values(wb_df)

    return wb_df, gof_values


def compute_gof_values(wb_df):
    # Store dataframe columns in separate variables
    q_meas = wb_df.loc[wb_df['valid'], 'discharge_meas(L/min)']
    q_sim = wb_df.loc[wb_df['valid'], 'discharge_sim(L/min)']

    # Calculate Goodness-of-Fit values
    gof_values = {
        'NSE': 1 - np.sum((q_meas - q_sim) ** 2) / np.sum(
            (q_meas - np.nanmean(q_meas)) ** 2),
        'KGE': 1 - np.sqrt(
            (np.corrcoef(q_meas, q_sim)[0, 1] - 1) ** 2 +
            ((np.std(q_sim) / np.std(q_meas) - 1) ** 2) +
            ((np.mean(q_sim) / np.mean(q_meas) - 1) ** 2)),
        'Bias': 1 / len(q_meas) * np.sum(q_sim - q_meas),
        'PBias': 100 * (np.mean(q_meas) - np.mean(q_sim)) / np.mean(q_meas),
        'MAE': 1 / len(q_meas) * np.sum(np.abs(q_meas - q_sim)),
        'RMSE': np.sqrt(1 / len(q_meas) * np.sum((q_sim - q_meas) ** 2)),
        'MAD': np.max(q_sim - q_meas),
        'MPD': np.max(q_sim) - np.max(q_meas)
    }

    return gof_values


def callback(xk, convergence=None):
    # xk is the current parameter vector
    # convergence is the current convergence value (optional)

    # Custom logging or storage of intermediate results
    print("Current parameters:", xk)


def optimize_over_full_timeseries(variable_parameters, fixed_parameters, wb_df):
    # Calculate model predictions
    wb_df, gof_values = compute_water_balance(wb_df, fixed_parameters, variable_parameters)
    return gof_values['RMSE']