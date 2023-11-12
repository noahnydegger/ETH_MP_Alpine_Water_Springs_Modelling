# import the required packages. Add more if necessary
import pandas as pd # data processing
import numpy as np # data processing
import matplotlib.pyplot as plt # create plots
import plotly.graph_objects as go # create interactive plots
import os # interaction with operating system
import numpy as np
import pandas as pd
import Data_Import
import Data_Cleaning
import Data_Analysis
import Data_Visualization
import Helper
import WatBalMod
import AR_Model as ar
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np
import sammhelper as sh
import plotly.express as px

fontSize = 25 # Size of numbers and text in plots
lineWidth = 1.5 # linewidth in plots

# adapts the plot settings for all future plots

plt.rcParams['font.size'] = fontSize
plt.rcParams['lines.linewidth'] = lineWidth

def plot_frq_vs_amp_static(df_ulrika,path_to_save_ar_figures):
    frq, amp = sh.fft(df_ulrika)
    plt.figure()
    plt.grid()
    plt.plot(frq, amp)
    plt.xlabel('Frequency [d$^{-1}$]')
    plt.ylabel('Discharge [L/min]')
    plt.savefig(os.path.join(path_to_save_ar_figures, "frq_vs_amp.pdf"))
    plt.show()
def plot_frq_vs_amp(df_ulrika,path_to_save_ar_figures):
    frq, amp = sh.fft(df_ulrika)
    # Plotting an interactive plot using plotly express
    # Create a dataframe for plotting

    # Create a DataFrame for plotting
    plot_data = {'frq': frq, 'amp': amp}
    df = pd.DataFrame(plot_data)

    # Plotting an interactive scatter plot using plotly express
    fig = px.line(df, x='frq', y='amp', title='Interactive Plot: Frequency vs. Amplitude')
    fig.update_layout(xaxis_title='Frequency', yaxis_title='Amplitude')

    # Save the interactive plot as an HTML file
    fig.write_html(os.path.join(path_to_save_ar_figures, "amp_vs_frq.pdf"))
    # Show the plot
    fig.show()

    return frq, amp

def acf(ulrika_d, lags,path_to_save_ar_figures):
    # Plot the autocorrelation function with confidence bounds
    plot_acf(ulrika_d['discharge(L/min)'], lags=lags, alpha=0.05)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    #plt.title('Autocorrelation Function with Confidence Bounds')
    plt.savefig(os.path.join(path_to_save_ar_figures, "autocorr_function"))
    plt.show()

