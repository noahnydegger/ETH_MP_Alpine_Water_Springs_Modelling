import pandas as pd  # data processing
import numpy as np  # data processing
from scipy import optimize
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime


def compute_water_balance_iterative(wb_df, fixed_parameters, variable_parameters, resolution):
    use_sigmoid = True
    # fixed parameters_for_df
    melt_temp = fixed_parameters['melting_temperature']
    # Parameters for calibration
    area, max_saturation, rg, fr, melt_rate = variable_parameters
    if resolution == 'H':
        rg = rg / 24
        melt_rate = melt_rate / 24

    # set initial values
    wb_df.loc[wb_df.index[0], ['storage_soil(mm)']] = [max_saturation / 2]

    y = wb_df.index.tolist()[0]
    # iterate over df row by row
    for i in wb_df.index.tolist():
        # Compute Snow melt
        if wb_df['temperature(C)'][i] >= melt_temp:
            wb_df.loc[i, 'snow_melt(mm)'] = np.minimum(melt_rate * (wb_df['temperature(C)'][i] - melt_temp),
                                                       wb_df['snow_cover(mm)'][y])

        # Compute Snow cover
        wb_df.loc[i, 'snow_cover(mm)'] = (wb_df['snow_cover(mm)'][y] - wb_df['snow_melt(mm)'][i] +
                                          wb_df['snow_fall(mm)'][i])

        # compute actual evapotranspiration
        wb_df.loc[i, 'aET(mm)'] = (wb_df['storage_soil(mm)'][y] / max_saturation) * wb_df['pET(mm)'][i]

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
        'RMSE': np.sqrt(1 / len(q_meas) * np.sum((q_sim - q_meas) ** 2)),
        'Bias': 1 / len(q_meas) * np.sum(q_sim - q_meas),
        'PBias': 100 * (np.mean(q_meas) - np.mean(q_sim)) / np.mean(q_meas),
        'MeanAbsErr': 1 / len(q_meas) * np.sum(np.abs(q_meas - q_sim)),
        'MaxAbsDiff': np.max(q_sim - q_meas),
        'MaxPeakDiff': np.max(q_sim) - np.max(q_meas)
    }

    return gof_values


def callback_minimize(intermediate_result):
    intermediate_params.append(intermediate_result.x)
    intermediate_gof.append(intermediate_result.fun)
    #print(f"x = {x}, Objective function value = {fun}")


def callback_evolution(xk, convergence):
    intermediate_params.append(xk)


def callback_basin(x, f, accepted):
    intermediate_params.append(x)


def optimize_over_full_timeseries(variable_parameters, fixed_parameters, wb_df, resolution, optimize_metric, tscv):
    # Calculate model predictions
    wb_df, gof_values = compute_water_balance_iterative(wb_df, fixed_parameters, variable_parameters, resolution)

    if optimize_metric == 'NSE':
        gof = -gof_values['NSE']
    elif optimize_metric == 'RMSE':
        gof = gof_values['RMSE']
    return gof


def optimize_over_calibration_period(variable_parameters, fixed_parameters, wb_df, resolution, optimize_metric, tscv):
    # Calculate model predictions
    wb_df, gof_values = compute_water_balance_iterative(wb_df, fixed_parameters, variable_parameters, resolution)

    if optimize_metric == 'NSE':
        gof = -gof_values['NSE']
    elif optimize_metric == 'RMSE':
        gof = gof_values['RMSE']
    return gof


def optimize_with_timeseries_split(variable_parameters, fixed_parameters, wb_df, resolution, optimize_metric, tscv):
    total_fit_value = 0.0

    for train_index, test_index in tscv.split(wb_df):
        #train_data, test_data = wb_df.iloc[train_index].copy(), wb_df.iloc[test_index]
        train_data = wb_df.iloc[train_index].copy()
        train_data, gof_values = compute_water_balance_iterative(train_data, fixed_parameters, variable_parameters, resolution)
        if optimize_metric == 'NSE' or optimize_metric == 'KGE':
            gof = -gof_values[optimize_metric]
        else:
            gof = gof_values[optimize_metric]

        total_fit_value += gof

    return total_fit_value


def optimize_wb_model(spring_name, variable_parameters, fixed_parameters, wb_df, resolution,
                      optimize_method='evolution', optimize_metric='NSE', max_iter=1):
    tscv = TimeSeriesSplit(n_splits=2)  # Adjust the number of splits as needed

    initial_guess = tuple(variable_parameters.values())

    # Define parameter bounds for the variable parameters_for_df
    if spring_name == 'Ulrika':
        bounds = [(initial_guess[0] - 0.5 * initial_guess[0], initial_guess[0] + 0.5 * initial_guess[0]),
                  (5, 120),  # soil storage capacity
                  (8, 12),  # residence time, qualitativ fixed
                  (0.01, 0.1),  # runoff fraction
                  (0.5, 6)]  # melt rate different papers, depends on elevation and season
    elif spring_name == 'Paliu_Fravi':
        bounds = [(initial_guess[0] - 0.5 * initial_guess[0], initial_guess[0] + 0.5 * initial_guess[0]),
                  (5, 120),  # soil storage capacity
                  (12, 27),  # residence time, qualitativ fixed
                  (0.01, 0.1),  # runoff fraction
                  (0.5, 6)]  # melt rate different papers, depends on elevation and season


    # Initialize lists to store intermediate results
    global intermediate_params
    global intermediate_gof
    intermediate_params = []
    intermediate_gof = []

    # for testing
    '''
    start_date = pd.to_datetime('2021-11-01').tz_localize('UTC')
    end_date = pd.to_datetime('2021-11-15').tz_localize('UTC')
    wb_df = wb_df.loc[(wb_df.index >= start_date) & (wb_df.index <= end_date)].copy()
    bounds = [(initial_guess[0] - 0.1 * initial_guess[0], initial_guess[0] + 0.1 * initial_guess[0]),
              (21, 23),  # soil storage capacity
              (9, 11),  # residence time, qualitativ fixed
              (0.08, 0.1),  # runoff fraction
              (2.5, 3.5)]  # melt rate different papers, depends on elevation and season
              '''

    # select subset for calibration
    wb_df = wb_df[wb_df['calibration_period']].copy()
    # Run the global optimization
    if optimize_method == 'Evolution':
        result = optimize.differential_evolution(
            optimize_over_calibration_period, #optimize_over_full_timeseries,
            bounds=bounds,
            args=(fixed_parameters, wb_df, resolution, optimize_metric, tscv),
            callback=callback_evolution,
            disp=True,
            polish=False,
            maxiter=max_iter,  # Specify the maximum number of iterations (maxiter) here
        )
    elif optimize_method == 'Basin':
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "args": (fixed_parameters, wb_df, resolution, optimize_metric, tscv, )}
        result = optimize.basinhopping(
            optimize_over_calibration_period, #optimize_over_full_timeseries,
            x0=initial_guess,
            minimizer_kwargs=minimizer_kwargs,
            disp=True,
            niter=max_iter,
            stepsize=0.1,
            callback=callback_basin,
        )
    elif optimize_method == 'Minimize':  # only local optimization
        result = optimize.minimize(
            optimize_over_full_timeseries,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            args=(fixed_parameters, wb_df),
            callback=callback_minimize,
            options={'maxiter': max_iter})


    # create a DataFrame for the intermediate results
    param_names = list(variable_parameters.keys())
    intermediate_results_df = pd.DataFrame()

    # Add lists to the DataFrame
    intermediate_results_df[param_names] = intermediate_params
    for xk in intermediate_params:
        wb_df, gof_values = compute_water_balance_iterative(wb_df[wb_df['calibration_period']].copy(),
                                                            fixed_parameters, xk, resolution)
        intermediate_gof.append(list(gof_values.values()))
    intermediate_results_df[list(gof_values.keys())] = intermediate_gof

    return result, intermediate_results_df
