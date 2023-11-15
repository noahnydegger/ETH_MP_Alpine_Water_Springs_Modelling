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

def model_cycle(ulrika_d,plot, time, d_avg,d_1,d_2,d_3, frq_1,frq_2,frq_3,path_to_save_ar_figures):
    # Make the model
    # read from the fft Plot the values
    #d_avg = 373.63
    #d_155 = 128.27
    #d_55 = 97.196
    #d_28 = 72.51

    # Parameters: Phase Shifts to obtain with curve fit:
    phi_y = 0  # [-]
    phi_m = 0  # [-]
    phi_w = 0  # [-]

    # Model
    def model(t, param):
        phi_y, phi_m, phi_w = param
        d_mod = (d_avg
                 + d_1 * np.cos(2 * np.pi * t * (frq_1) + phi_y)
                 + d_2 * np.cos(2 * np.pi * t * (frq_2) + phi_m)
                 + d_3 * np.cos(2 * np.pi * t * (frq_3) + phi_m))
        return d_mod

    # Curve Fit
    phi_y, phi_m, phi_w = sh.curve_fit(model, ulrika_d['discharge(L/min)'], time, param=[phi_y, phi_m, phi_w])

    # Modelled Data
    d_mod = model(time, param=[phi_y, phi_m, phi_w])

    if plot:
        plt.figure('curve fit')
        plt.grid()
        plt.plot(time, ulrika_d['discharge(L/min)'], label="measured")
        plt.plot(time, d_mod, label="modelled")
        plt.xlabel('Time [d]')
        plt.ylabel('Discharge [L/min]')
        plt.legend()
        plt.savefig(os.path.join(path_to_save_ar_figures, 'cycle_model.pdf'))
        plt.show()

    return d_mod

def model_ar_cycle(discharge,d_mod,time,d_avg,d_1,d_2,d_3,plot, path_to_save_ar_figures,frq_1,frq_2, frq_3):
    # Autocorrelation Term
    res = discharge - d_mod
    res_1 = sh.delay(time, res, (time[1] - time[0]))

    def regression(res_1, param):
        kappa, alpha = param
        res = kappa + alpha * res_1
        return res

    kappa = 0
    alpha = 1
    v = res
    d = res[1:]
    kappa, alpha = sh.curve_fit(regression, res[1:], res_1[1:], param=[kappa, alpha])

    autoregr = regression(res_1[1:], param=[kappa, alpha])


    # Noise Term (Std. of Residuals)
    h = np.array(res[1:]) - autoregr
    h_1 = sh.delay(time, h, time[1] - time[0])
    sigh = np.std(h, ddof=1)
    time_ext = np.arange(0, 2000, 1)

    # AR(1) Term
    AR = np.zeros(len(time))
    AR[0] = np.random.normal(0, sigh, 1)
    for i in range(1, len(time)):
        AR[i] = AR[i - 1] * alpha + np.random.normal(0, sigh, 1)

    # Full Model
    b = 0.001
    a = d_avg
    # Parameters: Phase Shifts to obtain with curve fit:
    phi_y = 0  # [-]
    phi_m = 0  # [-]
    phi_w = 0  # [-]

    def model(t, param):
        phi_y, phi_m, phi_w, a, b = param
        d_mod_AR = ((a + b * t
                     + d_1 * np.cos(2 * np.pi * t * (frq_1) + phi_y)
                     + d_2 * np.cos(2 * np.pi * t * (frq_2) + phi_m)
                     + d_3 * np.cos(2 * np.pi * t * (frq_3) + phi_m))
                    + AR)
        return d_mod_AR

    # Curve Fit
    phi_y, phi_m, phi_w, a, b = sh.curve_fit(model, discharge, time, param=[phi_y, phi_m, phi_w, a, b])
    # Modelled Data
    d_mod_AR = model(time, param=[phi_y, phi_m, phi_w, a, b])

    if plot:

        #Plot autoregression
        plt.plot(res_1[1:], res[1:], label="residuals", marker=".", linestyle="")
        plt.plot(res_1[1:], autoregr, label="autoregression")
        # Plot Modelled Data
        plt.figure('new model',figsize=(15, 10))
        plt.grid()
        plt.plot(time, discharge, label="measured")
        plt.plot(time, d_mod_AR, label="modelled")
        plt.xlabel('Time [d]',fontsize =16)
        plt.ylabel('Discharge [L/min]', fontsize = 20)
        plt.legend()

        plt.savefig(os.path.join(path_to_save_ar_figures,  'ar_model.pdf'))

        # Plot residuals with autoregression line
        plt.figure('res')
        plt.grid()
        plt.plot(res_1[1:], res[1:], label="residuals", marker=".", linestyle="")
        plt.plot(res_1[1:], autoregr, label="autoregression")
        plt.xlabel('res$_{i-1}$ [L/min]')
        plt.ylabel('res$_i$ [L/min]')
        plt.legend()

        plt.savefig(os.path.join(path_to_save_ar_figures, 'res_with_corr.pdf'))
        # Plot residuals without autocorrelation
        plt.figure('h')
        plt.grid()
        plt.plot(h_1[1:], h[1:], label="residuals", marker=".", linestyle="")
        plt.xlabel('h$_{i-1}$ [L/min]')
        plt.ylabel('h$_{i}$ [L/min]')
        plt.legend()

        plt.savefig(os.path.join(path_to_save_ar_figures, "res_minus_corr.pdf"))

    return d_mod_AR,res,res_1,h,h_1

def model_ar(discharge,time,d_avg, path_to_save_ar_figures):
    # Autocorrelation Term
    res = discharge
    res_1 = sh.delay(time, res, (time[1] - time[0]))

    def regression(res_1, param):
        kappa, alpha = param
        res = kappa + alpha * res_1
        return res

    kappa = 0
    alpha = 1
    v = res
    d = res[1:]
    kappa, alpha = sh.curve_fit(regression, res[1:], res_1[1:], param=[kappa, alpha])

    autoregr = regression(res_1[1:], param=[kappa, alpha])

    # Noise Term (Std. of Residuals)
    h = np.array(res[1:]) - autoregr
    h_1 = sh.delay(time, h, time[1] - time[0])
    sigh = np.std(h, ddof=1)
    time_ext = np.arange(0, 2000, 1)

    # AR(1) Term
    AR = np.zeros(len(time))
    AR[0] = np.random.normal(0, sigh, 1)
    for i in range(1, len(time)):
        AR[i] = AR[i - 1] * alpha + np.random.normal(0, sigh, 1)


    # Modelled Data
    d_mod_AR = d_avg + AR
    # Plot autoregression
    plt.plot(res_1[1:], res[1:], label="residuals", marker=".", linestyle="")
    plt.plot(res_1[1:], autoregr, label="autoregression")
    # Plot Modelled Data
    plt.figure('new model', figsize=(15, 10))
    plt.grid()
    plt.plot(time, discharge, label="measured")
    plt.plot(time, d_mod_AR, label="modelled")
    plt.xlabel('Time [d]', fontsize=16)
    plt.ylabel('Discharge [L/min]', fontsize=20)
    plt.legend()

    plt.savefig(os.path.join(path_to_save_ar_figures, 'ar_only_model.pdf'))
    return d_mod_AR

