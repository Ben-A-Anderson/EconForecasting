# Import and Configure Standard Libraries
import os, sys
from matplotlib.dates import date2num
import pandas as pd# imports pandas for dataframe manipulation
import numpy as np# imports numpy
import matplotlib as mpl# for data visualization
mpl.rcParams['figure.figsize'] = (25,8)# sets the plot size to 12x8
from matplotlib import pyplot as plt# for shorter lines with plotting
from statsmodels.tsa.stattools import adfuller  # Statistical test for stationary data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from fitter import Fitter
import warnings # to hide warning messages
warnings.filterwarnings('ignore')

#Configure and Import Custom functions
# Define location for custom functions
module_path = os.path.abspath(os.path.join('../Functions'))
# Verify it's accessible for loading
if (module_path not in sys.path) & (os.path.isdir(module_path)):
    sys.path.append(module_path)
    print('Added', module_path, 'to system paths')
elif (module_path in sys.path) & (os.path.isdir(module_path)):
    print(module_path, 'ready to be used for import')
else:
    print(module_path, 'is not a valid path')
# Import Multi_Plot
try: from multi_plot import * # Allows for plotting of multiple columns in a data frame
except: print('failed to load multi_plot')
default: print('Loaded multiplot')
# Import StationaryTools
try: from StationaryTools import *
except: print('Failed to load StationaryTools')
default: print('Loaded StationaryTools')
# Import RegressionTools
try: from RegressionTools import *
except: print('Failed to load RegressionTools')
default: print('Loaded RegressionTools')

def GBM(dfinp, fit_col, start_date, end_date, test_end, odatafolder = None, ofilename = None, makestat=True, thres=0.05, nsteps=10, nsims=1000, GBMsigma = 1, plotting=True, fitting=False, verbose=True):
    """
    GBM performs Geometric Brownian Motion (GBM) prediction of a single data column
        OUTPUT:
        GBM = Pandas dataFrame of all simulations
        df_forecast = Pandas DataFrame of original data plus all average and STD forecasted data

        INPUT:
        dfinp = Pandas Data Frame with datetimeindex
        fit_col = String, of the column within df to be predicted
        start_date = Date to start Training from
        end_date = Date to stop Training on
        test_end = Date to stop Testing on
        odatafolder = Folder location to store the output of the prediciton
        ofilename = Filename to save the data as
        
        makestat = Control whether to allow automated testing and stationary processing using ADFuller thres (default=True)
        thres = ADFuller Threshold for stationary testing (default=0.05)
        nsteps = Number of steps into the future to predict (default=10)
        nsims = Number of GBM prediction loops to execute (default=1000)
        GBMsigma = modifier for STD to artificially inflate the error (default = 1)
        plotting = controls if the function outputs plots (default=True)
        fitting = controls the use of Fitter to provide the distribution of the data (default=False)
        verbose = controls if the function prints information as it runs (default=True)

        Notes:
        * Cannot use data that has been interpolated. It will skew the statistics used to calculate Mu and Sigma
    """
    ####################
    # Data Preparation #
    ####################

    # Extract only single column of values to be used in the simulation
    df_orig = pd.DataFrame(dfinp[fit_col]).dropna()#.rename(columns={fit_col:'data'})
    df_orig = df_orig[start_date : test_end]
    # Extract training data for posterity
    df_train = df_orig[(df_orig.index >= start_date) & (df_orig.index <= end_date)]
    # Extract testing data
    df_test = df_orig[(df_orig.index >= end_date) & (df_orig.index <= test_end)]
    # Extract Training Data
    df = df_orig[(df_orig.index >= start_date) & (df_orig.index <= end_date)]

    print(df.index[-1])
    print(df_test.index[0])
    #############################
    # Exploratory Data Analysis #
    #############################
    # Print the Description and Info about the single data colomn
    if verbose: print('Training Data Description\n',df.describe(),'\nTesting Data Description\n',df_test.describe())

    # Perform naive regression of data and return plots if desired
    reg, regmu, freg, fmu = LinRegress(df, df_test)

    if makestat:
        df, stat_test = MakeStationaryLPC(df, thres = thres)

    if plotting:
        plt.figure(1)
        plt.hist(df, bins=20)
        if makestat:
            plt.xlabel('Ln(%Change)')
        else:
            plt.xlabel('Price')
        plt.ylabel('Count')
        plt.show()

    # Perform automated histogram fitting to determine df shape
    if fitting:
        from fitter import Fitter
        nbins = int(np.ceil(len(df)/20))
        print(nbins)
        dfhist = pd.DataFrame(np.histogram(df, bins = 20)).T.rename(columns={0:'count',1:'edge'}).set_index('edge').dropna()
        #print(dfhist)
        f = Fitter(dfhist)
        f.fit()
        f.summary()

    dT = df.index[-1]-df.index[-2]
    horizon = pd.date_range(start=df.index[-2]+dT, freq=dT, periods=nsteps)
    if verbose: 
        print('Last date interval of',dT, 'will be used for intervals into the future. ', len(horizon) , 'points after', df.index[-1])
        print('Forecasting will be performed for each step size above from',horizon[0], 'to', horizon[-1])

    # Determine mu (mean) and sigma (std) for historical data
    mu = np.mean(df)[0]
    sigma = 2*np.std(df)[0]

    # Assemble DRIFT using mu and sigma above
    # t = integer range of steps into the future. Must start at 1 so shifted to nsteps+1
    projection_steps = np.arange(1, int(nsteps) + 1)
    #print(projection_steps)
    drift  = (mu - 0.5 * sigma**2) * projection_steps

    # Generate Rand array using normal distribution between 0 and 1
    # rows = nsteps, columns = nsims
    rands = pd.DataFrame(np.random.normal(mu,sigma, size=(nsims, nsteps)))
    #print('Rand\n', rands.head(2))

    # Generate random path by adding second step(row) to first and 3rd to second and so on to nsteps
    sum_rands = rands.cumsum(axis=1)
    #print('Sum Rand\n', sum_rands.head(2))

    # Finalize SHOCK Term by multiplying calthor by sigma
    shock = sigma * sum_rands
    #print('Shock\n',shock.head(2))

    # Test if final value in df is 0, if it is make it just larger than zero, if it isn't then us it as the starting point
    if df.iloc[-1][0] == 0:
        So = mu
    else:
        So = df.iloc[-1][0]
        

    # Generate exponential term for GBM model
    eterm = pd.DataFrame(drift + shock)

    # Generate GBM path into the future
    GBM = So * np.exp(eterm)
    GBM = pd.DataFrame(GBM.T)
    GBM.index = horizon

    GBM_Stats = pd.DataFrame(GBM.mean(axis=1)).rename(columns={0:"mean"})
    GBM_Stats['std'] = GBM.std(axis=1)

    # Generate a probability density function for the random walk
    PDF = np.log(So) + eterm
    PDF = pd.DataFrame(PDF.T)
    PDF.index = horizon

    if verbose:
        print('   mu=', mu)
        print('sigma=', sigma)
        print('   S0=', So)
        print('\nDrift:\n', drift[::5])
        #print('\nShock:\n',shock.iloc[::100,::10])
        #print('eterm:\n', eterm.iloc[::100,::10])
        print('\nGBM:\n',GBM.shape,'\n',GBM.iloc[::10,::150])
        #print('\nPDF:\n',PDF.iloc[::10,::100])

    if plotting:
        plt.figure(2)
        nplots = 50
        multi_plot(GBM.iloc[:,::int(nsims/nplots)],"Geometric Brownian Motion Predictions")
        #multi_plot(GBM.iloc[:,::int(nsims/nplots)],"Probability Density Function")
        plt.rcParams['figure.figsize'] = (25,8)
        plt.errorbar(GBM_Stats.index, 'mean', yerr='std', data=GBM_Stats)
        plt.title('Geometric Brownian Motion Errorbar Plot')
        plt.xlabel('Date')
        if makestat:
            plt.ylabel('log(%Change)')
        else:
            plt.ylabe('Price')
        plt.show()

    # Convert GBM predictions back into pricing units if LPC was perfomred to make stationary
    if makestat:
        df_corr = unLPC(df_train, GBM)
    else:
        df_corr = GBM
    
    if verbose: df_corr.head(5)
    if plotting: multi_plot(df_corr.iloc[:,::int(nsims/nplots)],"Geometric Brownian Motion Prediction")

    df_corr_stats = pd.DataFrame(df_corr.mean(axis=1)).rename(columns={0:fit_col})
    df_corr_stats['std'] = df_corr.std(axis=1)*GBMsigma
    
    if plotting:
        plt.figure(3)
        plt.rcParams['figure.figsize'] = (25,8)
        plt.errorbar(df_corr_stats.index, fit_col, yerr='std', data=df_corr_stats)
        plt.xlim(df_corr_stats.index[0]-dT, df_corr_stats.index[-1]+dT)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Pricing Forecast Errorbar plot')
        plt.show()

        plt.figure(4)
        plt.rcParams['figure.figsize'] = (25,15)
        plt.plot(df_train.index, df_train, 'b', label= 'Training Data')
        plt.plot(df_test.index, df_test, '--k', label = 'Testing Data')
        plt.errorbar(df_corr_stats.index, fit_col, yerr='std', data=df_corr_stats, label='Geometric Brownian Motion Forecast')
        plt.plot(freg, '--r', label = 'Linear regression')
        plt.plot(fmu, '--g', label = 'Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Full Analysis Plot')
        plt.xlim(df_train.index[0]-dT, df_corr_stats.index[-1]+dT)
        plt.ylim(
            np.amin([int(df_train.min()), int(df_test.min()), int(df_corr_stats[fit_col].min())])-50, 
            np.amax([int(df_train.max()), int(df_test.max()), int(df_corr_stats[fit_col].max())])+50
            )
        plt.legend()
        plt.show()
    
    df_forecast = df_orig.append(df_corr_stats)

    if not(odatafolder == None and ofilename == None):
        print('Exporting data to:', odatafolder, ofilename)
        GBM.to_csv(odatafolder&ofilename)
        df_forecast.to_csv(odatafolder&ofilename)

    return GBM, df_forecast