from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import Data
from Water_Balance_Model import wb_Visualization

path_to_reference_ET_data = Data.datafolder / 'water_balance_model' / 'ET_Comparison'
ref_ET_df = pd.read_pickle(path_to_reference_ET_data / 'ET_data_Chur.pkl')
Hamon_ET_df = pd.read_pickle(path_to_reference_ET_data / 'wb_Chur_input_D.pkl')

ET_df = ref_ET_df.merge(Hamon_ET_df[Hamon_ET_df['valid_meteo']]['pET(mm)'],
                            how='right', left_index=True, right_index=True)

x_value = 'ets150d0'
x_value = 'erefaod0'
y_value = 'pET(mm)'
fs = 14

# Perform linear regression using numpy.polyfit
coefficients = np.polyfit(ET_df[x_value], ET_df[y_value], 1)
slope, intercept = coefficients

# Create a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ET_df[x_value], ET_df[y_value], facecolors='none', edgecolors='red', marker='o', s=50)

# Add the dashed line
maxET = max(max(ET_df[x_value]), max(ET_df[y_value]))
x_dash = [0, maxET]
y_dash = [0, maxET]
ax.plot(x_dash, y_dash, linewidth=2, linestyle='--', color='black', label='Identity Line')

regression_line = np.polyval(coefficients, ET_df[x_value])
#ax.plot(ET_df[x_value], regression_line, color='blue', label=f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')

# Add labels and title
ax.set_xlabel('Reference pET (mm/d)', fontsize=fs)
ax.set_ylabel('Hamon pET (mm/d)', fontsize=fs)
ax.set_title('Meteo station Chur', fontsize=fs)
ax.legend(loc='upper left', fontsize=fs)

# Set tick label font size
ax.tick_params(axis='both', labelsize=fs)

# Show the plot
fig_path = path_to_reference_ET_data / f'pET_Chur_{x_value}_{y_value}.pdf'
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, bbox_inches='tight')
plt.show()

