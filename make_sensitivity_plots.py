from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import Data
from Water_Balance_Model import wb_Visualization

spring_name = 'Paliu_Fravi'
selected_time = {
    'Ulrika': '20231227_1847',  # 20231209_1534 area & smax & meltrate, 20231210_0206 area & smax & fr, 20231227_1847 smax & rg
    'Paliu_Fravi': '20231227_1921'  # 20231209_1542 area & smax & meltrate, 20231210_0316 area & smax & fr, 20231227_1921 smax & rg
}
param_unit = {
    'area': ['Area', '(m$^2$)'],
    'storage_capacity': ['Storage capacity', '(mm)'],
    'residence_time': ['Residence time', '(days)'],
    'runoff_fraction': ['Runoff fraction', '(-)'],
    'melting_rate': ['Melt rate', '(mm/°C/d)']
}

path_to_sensitivity_df = (Data.datafolder / 'water_balance_model' / f'Results_{spring_name}' / 'Sensitivity_Results' /
                          selected_time[spring_name] /
                          f'{selected_time[spring_name]}_sensitivity_results_{spring_name}.pkl')

sensitivity_df = pd.read_pickle(path_to_sensitivity_df)

param_x = 'residence_time'
param_y = 'storage_capacity'
gof = 'NSE'
nr_levels = 10
#filtered_df = sensitivity_df[(sensitivity_df['area'] > 181820.0) & (sensitivity_df['area'] < 181821.0)]
#filtered_df = sensitivity_df[(sensitivity_df['storage_capacity'] == 13.0)]
filtered_df = sensitivity_df
filtered_df = filtered_df[(filtered_df['storage_capacity'] > 9.0)]  # bei Paliu Fravi

pivot_df = filtered_df.pivot(index=param_y, columns=param_x, values=gof)

# Get the parameter values and z values from the pivot table
x_values = pivot_df.columns
y_values = pivot_df.index
z_values = pivot_df.values

# Create a figure and axis
fs = 18
fig, ax = plt.subplots(figsize=(8, 6))

# Create a filled contour plot with a continuous color map
contour_filled = plt.contourf(x_values, y_values, z_values, levels=nr_levels, cmap='YlOrRd')

# Overlay contour lines on top of the filled contour plot
contour_lines = plt.contour(x_values, y_values, z_values, levels=contour_filled.levels, colors='k', linestyles='solid')

# Add labels to contour lines indicating GOF values
ax.clabel(contour_lines, inline=True, inline_spacing=25, fontsize=fs, fmt='%1.3f')#, manual=[(2, i) for i in range(2, nr_levels+1, 2)])

# Add a continuous colorbar to the plot
cbar = plt.colorbar(contour_filled, ax=ax)
cbar.set_label(gof, fontsize=fs)  # Add your title here
cbar.ax.tick_params(labelsize=fs-2)

# Add labels and a title
ax.set_xlabel(f"{param_unit[param_x][0]} {param_unit[param_x][1]}", fontsize=fs)
ax.set_ylabel(f"{param_unit[param_y][0]} {param_unit[param_y][1]}", fontsize=fs)
#ax.set_title(f'{spring_name}', fontsize=fs)
# Adjust font size for x-axis and y-axis ticks
ax.tick_params(axis='x', labelsize=fs-2)
ax.tick_params(axis='y', labelsize=fs-2)
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.2)

fig_path = Data.datafolder / 'water_balance_model' / 'Plots' / spring_name / f'contour_{spring_name}_{param_x}_{param_y}_{gof}.pdf'
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, bbox_inches='tight')
plt.show()
