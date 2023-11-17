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


def compute_water_balance(wb_df, fixed_parameters, variable_parameters):
    # params, Lat, Tsm, PETa, PETb, PETc, Qobs, R, S, T_b, Jday, SC, M, Ss, PET, ET, Qsurf, PER_gw, Qgw, Sg, Qsim
    # Parameters for calibration
    area, max_saturation, rg, melt_rate, fr, melt_temp, lapse_rate = tuple(variable_parameters.values())

    # set initial values
    wb_df.loc[wb_df.index[0], ['storage_soil(mm)']] = [10]

    #compute_rain_and_snow(wb_df, melt_rate, melt_temp)
    #compute_daily_PET_by_Hamon()

    y = wb_df.index.tolist()[0]
    # iterate over df row by row
    for i in wb_df.index.tolist():
        #i = wb_df.index[j]
        # Compute Snow melt
        if wb_df['mean_temperature(C)'][i] > melt_temp:
            wb_df.loc[i, 'snow_melt(mm)'] = np.minimum(melt_rate * (wb_df['mean_temperature(C)'][i] - melt_temp),
                                                       wb_df['snow_cover(mm)'][y])

        # Compute Snow cover
        wb_df.loc[i, 'snow_cover(mm)'] = (wb_df['snow_cover(mm)'][y] - wb_df['snow_melt(mm)'][i] +
                                          wb_df['snow_fall(mm)'][i])

        # Compute potential soil zone water content without accounting for percolation
        wb_df.loc[i, 'storage_soil(mm)'] = wb_df['storage_soil(mm)'][y] + wb_df['rain_fall(mm)'][i] - wb_df['aET(mm)'][i]

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
        #Qgw[Sg < 0] = 0
        y = i  # current day is new yesterday

    gof_values = compute_gof_values(wb_df, variable_parameters)

    return wb_df, gof_values


def compute_gof_values(wb_df, variable_parameters):
    # Store dataframe columns in separate variables
    discharge_meas = wb_df['discharge_meas(mm)'] * variable_parameters['area'] / (60 * 24)
    discharge_sim = wb_df['discharge_sim(mm)'] * variable_parameters['area'] / (60 * 24)

    # Calculate Goodness-of-Fit values
    gof_values = {
        'NSE': 1 - np.sum((discharge_meas - discharge_sim) ** 2) / np.sum(
            (discharge_meas - np.nanmean(discharge_meas)) ** 2),
        'KGE': 1 - np.sqrt(
            (np.corrcoef(discharge_meas, discharge_sim)[0, 1] - 1) ** 2 +
            ((np.std(discharge_sim) / np.std(discharge_meas) - 1) ** 2) +
            ((np.mean(discharge_sim) / np.mean(discharge_meas) - 1) ** 2)),
        'Bias': 1 / len(discharge_meas) * np.sum(discharge_sim - discharge_meas),
        'PBias': 100 * (np.mean(discharge_meas) - np.mean(discharge_sim)) / np.mean(discharge_meas),
        'MAE': 1 / len(discharge_meas) * np.sum(np.abs(discharge_meas - discharge_sim)),
        'RMSE': np.sqrt(1 / len(discharge_meas) * np.sum((discharge_sim - discharge_meas) ** 2)),
        'MAD': np.max(discharge_sim - discharge_meas),
        'MPD': np.max(discharge_sim) - np.max(discharge_meas)
    }

    return gof_values
