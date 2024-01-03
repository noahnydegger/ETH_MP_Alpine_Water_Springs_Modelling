# import all required packages; add more if required
import pandas as pd
import numpy as np  # data processing
import pickle  # copy figures
import matplotlib.pyplot as plt  # create plots
import plotly.graph_objects as go  # create interactive plots
from plotly.subplots import make_subplots
import os  # interaction with operating system
from datetime import datetime


def first_impression_plot(spring_name, wb_df):
    max_value = max(wb_df['discharge_meas(L/min)'].max(), wb_df['discharge_sim(L/min)'].max())
    # Create a Matplotlib figure and axis
    fig, ax_flow = plt.subplots(figsize=(10, 6))

    valid = (wb_df['valid'])

    # Plot the raw signal in blue
    ax_flow.plot(wb_df.index, wb_df['discharge_meas(L/min)'], linewidth=1, color='blue', label='discharge measured')

    #ax_flow.plot(wb_df.index[valid], wb_df.loc[valid, 'discharge_meas(L/min)'], linewidth=1, color='blue', label='discharge measured valid')
    #ax_flow.plot(wb_df.index[~valid], wb_df.loc[~valid, 'discharge_meas(L/min)'], linewidth=1, color='lightblue', label='discharge measured invalid')

    # Plot the smoothed signal in orange
    ax_flow.plot(wb_df.index, wb_df['discharge_sim(L/min)'], linewidth=1, color='orange', label='discharge simulated')
    #ax_flow.plot(wb_df.index[valid], wb_df.loc[valid, 'discharge_sim(L/min)'], linewidth=1, color='orange',label='discharge measured')
    #ax_flow.plot(wb_df.index[~valid], wb_df.loc[~valid, 'discharge_sim(L/min)'], linewidth=1, color='lightcoral',label='discharge measured')

    ax_flow.fill_between(wb_df.index, 0, max_value, where=valid, color='lightblue', alpha=0.5,
                    label='Calibration period')

    #ax_flow.plot(wb_df.index, wb_df['storage_soil(mm)'], linewidth=1, color='lightgreen', label='soil storage')
    #ax_flow.plot(wb_df.index, wb_df['storage_gw(mm)'], linewidth=1, color='darkgreen', label='gw storage')
    #ax_flow.plot(wb_df.index, wb_df['aET(mm)'], linewidth=1, color='red', label='aET')

    ax_prec = ax_flow.twinx()
    ax_prec.bar(wb_df.index, wb_df['rain_fall(mm)'] + wb_df['snow_melt(mm)'], color='darkblue', alpha=0.5,
                width=0.8,
                align='edge', label=f'rain and snowmelt sum [mm]')

    ax_prec.invert_yaxis()

    # Customize the layout of the plot
    ax_flow.tick_params(axis='x', rotation=45)
    ax_flow.set_ylabel('Discharge [L/min]')
    ax_prec.set_ylabel(f'rain and snowmelt sum [mm]', color='darkblue')
    ax_flow.set_title(f'{spring_name}')
    ax_flow.legend()

    #ax_flow.set_xlim([pd.to_datetime('2021-10-01'), pd.to_datetime('2022-07-20')])
    #ax_flow.set_ylim([-5, 300])
    plt.show()

    # Save the plot as a PDF
    current_time_str = datetime.now().strftime("%m_%d_%H")  # Convert the datetime object to a string
    #save_path = os.path.join('Plots', 'Water_Balance_Model', current_time_str)
    #Helper.create_directory(save_path)
    #fig.savefig(os.path.join(save_path, f'{spring_name}_model_.pdf'), bbox_inches='tight')


def overview_plot(spring_name, wb_df, gof_values_cal, gof_values_val):
    max_value = max(wb_df['discharge_meas(L/min)'].max(), wb_df['discharge_sim(L/min)'].max())
    # Create a Matplotlib figure and axis
    fig, ax_flow = plt.subplots(figsize=(10, 6))

    # Plot the measured signal in blue
    ax_flow.plot(wb_df.index, wb_df['discharge_meas(L/min)'], linewidth=1, color='blue', label='discharge measured')

    # Plot the simulated signal in orange
    ax_flow.plot(wb_df.index, wb_df['discharge_sim(L/min)'], linewidth=1, color='orange', label='discharge simulated')

    # show the valid period excluding data gaps
    valid = (wb_df['valid'])
    ax_flow.fill_between(wb_df.index, 0, max_value, where=valid, color='lightblue', alpha=0.5,
                    label='Calibration period')

    #ax_flow.plot(wb_df.index, wb_df['storage_soil(mm)'], linewidth=1, color='lightgreen', label='soil storage')
    #ax_flow.plot(wb_df.index, wb_df['storage_gw(mm)'], linewidth=1, color='darkgreen', label='gw storage')
    #ax_flow.plot(wb_df.index, wb_df['aET(mm)'], linewidth=1, color='red', label='aET')

    ax_prec = ax_flow.twinx()
    ax_prec.bar(wb_df.index, wb_df['rain_fall(mm)'] + wb_df['snow_melt(mm)'], color='darkblue', alpha=0.5,
                width=0.8,
                align='edge', label=f'rain and snowmelt sum')

    ax_prec.invert_yaxis()

    # Customize the layout of the plot
    ax_flow.tick_params(axis='x', rotation=45)
    ax_flow.set_ylabel('Discharge [L/min]')
    ax_prec.set_ylabel(f'rain and snowmelt sum [mm]', color='darkblue')
    ax_flow.set_title(f'{spring_name}')
    ax_flow.legend()

    gov_text = (f"Training NSE: {np.round(gof_values_cal['NSE'], 2)}, "
                f"Training RMSE: {np.round(gof_values_cal['RMSE'], 2)}\n"
                f"Testing NSE: {np.round(gof_values_val['NSE'], 2)}, "
                f"Testing RMSE: {np.round(gof_values_val['RMSE'], 2)}")
    plt.gcf().text(0.12, 0.9, gov_text, fontsize=10)

    return fig


def cal_val_period_plot(spring_name, wb_df, gof_values_train, gof_values_test, gof_values_val, gof_values_all):
    #wb_df = wb_df[wb_df['validation_period']]

    max_value = max(wb_df['discharge_meas(L/min)'].max(), wb_df['discharge_sim(L/min)'].max())
    # Create a Matplotlib figure and axis
    fig, ax_flow = plt.subplots(figsize=(10, 4))  # 10, 5

    train_period = wb_df['calibration_period']
    test_period = wb_df['validation_period']
    val_period = wb_df['final_validation_period']
    not_used = ~(wb_df['calibration_period'] | wb_df['validation_period'] | wb_df['final_validation_period'])

    # show the valid period excluding data gaps
    valid = (wb_df['valid'])
    ax_flow.fill_between(wb_df.index, 0, max_value, where=valid, color='lightblue', alpha=0.3,
                         label='data availability')

    # add precipitation
    ax_prec = ax_flow.twinx()
    ax_prec.bar(wb_df.index, wb_df['rain_fall(mm)'] + wb_df['snow_melt(mm)'], color='darkblue', alpha=0.3,
                width=0.8,
                align='edge', label=f'rain and snow melt')

    ax_prec.invert_yaxis()

    # Plot the measured signal
    #ax_flow.plot(wb_df.index[not_used], wb_df.loc[not_used, 'discharge_meas(L/min)'], linewidth=1, color='black',
    #             label='not used')
    ax_flow.plot(wb_df.index[train_period], wb_df.loc[train_period, 'discharge_meas(L/min)'], linewidth=1, color='blue', label='Training data')
    ax_flow.plot(wb_df.index[test_period], wb_df.loc[test_period, 'discharge_meas(L/min)'], linewidth=1, color='orange',
                 label='Testing data')
    ax_flow.plot(wb_df.index[val_period], wb_df.loc[val_period, 'discharge_meas(L/min)'], linewidth=1,
                 color='green', label='Validation')

    # Plot the simulated signal in red
    ax_flow.plot(wb_df.index, wb_df['discharge_sim(L/min)'], linewidth=1, color='red', label='Model')

    ax_prec.set_ylabel(f'Rain and Snow melt [mm/d]', color='darkblue')
    ax_prec.tick_params(axis='y', labelcolor='darkblue')

    # Customize the layout of the plot
    ax_flow.grid(axis='y', linestyle='-', alpha=0.7)
    #ax_flow.tick_params(axis='x', rotation=45)
    ax_flow.set_ylabel('Discharge [L/min]')
    #ax_flow.set_title(f'{spring_name}')
    ax_flow.legend(loc='upper left', fontsize=10)

    # Set xlim using the min and max date values
    #ax_flow.set_xlim(wb_df.index.min(), pd.Timestamp('2024-01-05', tz='UTC'))
    #ax_flow.set_xlim(pd.Timestamp('2020-12-15', tz='UTC'), pd.Timestamp('2024-01-05', tz='UTC'))
    #2020-12-15

    periods = ['Training', 'Testing', 'Validation', 'Overall']
    gov_textl = (f"{periods[0]:<10} NSE: {np.round(gof_values_train['NSE'], 3):>5}, "
                f"Bias: {np.round(gof_values_train['Bias'], 1):>5} L/min\n"
                f"{periods[1]:<10} NSE: {np.round(gof_values_test['NSE'], 3):>5}, "
                f"Bias: {np.round(gof_values_test['Bias'], 1):>5} L/min")

    gov_textr = (f"{periods[2]:<10} NSE:{np.round(gof_values_val['NSE'], 3):>5}, "
                f"Bias: {np.round(gof_values_val['Bias'], 1):>5} L/min\n"
                f"{periods[3]:<10} NSE:{np.round(gof_values_all['NSE'], 3):>5}, "
                f"Bias: {np.round(gof_values_all['Bias'], 1):>5} L/min")

    '''
    ax_flow.annotate(gov_textl, xy=(0.00, 1.02), xycoords="axes fraction", ha='left', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.1), fontproperties={'family': 'DejaVu Sans Mono'})
    ax_flow.annotate(gov_textr, xy=(1.0, 1.02), xycoords="axes fraction", ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle="round", alpha=0.1), fontproperties={'family': 'DejaVu Sans Mono'})
    '''
    plt.subplots_adjust(bottom=0.2)

    return fig

# Add legend with training and testing metrics

def snow_plot(spring_name, wb_df, gof_values_all):
    #wb_df = wb_df[wb_df['validation_period']]

    max_value = wb_df['snow_cover(mm)'].max()

    # Create a Matplotlib figure and axis
    fig, ax_snow = plt.subplots(figsize=(15, 2))  # 10, 5

    train_period = wb_df['calibration_period']
    test_period = wb_df['validation_period']
    val_period = wb_df['final_validation_period']
    not_used = ~(wb_df['calibration_period'] | wb_df['validation_period'] | wb_df['final_validation_period'])

    # show the valid period excluding data gaps
    valid = (wb_df['valid'])
    #ax_snow.fill_between(wb_df.index, 0, max_value, where=valid, color='lightblue', alpha=0.3,
    #                     label='data availability')

    ax_snow.bar(wb_df.index, wb_df['snow_cover(mm)'], color='green', alpha=1,
                width=0.8,
                align='edge', label=f'Snow cover')

    # add rain and snowmelt
    ax_prec = ax_snow.twinx()
    ax_prec.bar(wb_df.index, wb_df['rain_fall(mm)'], color='darkblue', alpha=1,
                width=0.8,
                align='edge', label=f'Rain')

    ax_prec.bar(wb_df.index, wb_df['snow_melt(mm)'], color='lightgreen', alpha=1,
                width=0.8,
                align='edge', label=f'Snow melt')

    ax_prec.invert_yaxis()

    ax_prec.set_ylabel(f'Rain and Snow melt [mm/d]')
    #ax_prec.tick_params(axis='y', labelcolor='darkblue')

    # Customize the layout of the plot
    ax_snow.grid(axis='y', linestyle='-', alpha=0.7)
    #ax_snow.tick_params(axis='x', rotation=45)
    ax_snow.set_ylabel('Snow cover [mm]')
    ax_snow.set_title(f'{spring_name}')
    ax_snow.legend(loc='upper left', fontsize=10)
    ax_prec.legend(loc='upper right', fontsize=10)

    # Set xlim using the min and max date values
    #ax_snow.set_xlim(wb_df.index.min(), pd.Timestamp('2024-01-05', tz='UTC'))
    ax_snow.set_xlim(pd.Timestamp('2020-12-15', tz='UTC'), pd.Timestamp('2024-01-05', tz='UTC'))
    #2020-12-15

    plt.subplots_adjust(bottom=0.2)

    return fig
