# Load deafult packages
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import sys, os
import numpy as np 

# Define location for custom functions
module_path = os.path.abspath(os.path.join('./'))
try: os.path.isdir(module_path) and sys.path.append(module_path)
except: print('Custom function directory failed for loading')
# Import Custom Functions
try: from multi_plot import * # Allows for plotting of multiple columns in a data frame
except: print('failed to load multi_plot')
default: print('Loaded multiplot')
try: from StationaryTools import *
except: print('Failed to load StationaryTools')
default: print('Loaded StationaryTools')

def MakeStationaryDiff(df, thres=0.05, diffrng=1, difflim=3, verbose=True, plotting=True):
    """
    MakeStationaryDiff(df, nonstat, difflim, verbose)
    MakeStationaryDiff takes a DataFrame (df) and uses diff to make it stationary. The loop differencing will run up to it's limit (difflim) and will perform the differencing over
         a specified interval (diffrng).

    df      = Pandas dataframe with any number of non-stationary columns
    thres   = The ADFuller threshold for stationary p-values (default = 0.05)
    diffrng = The length of differencing to be applied (default = 1, i.e. previous day)
    difflim = The maximum number of differencing loops to perform on difficult columns
    verbose = Control informational printing during function execution (default = True)
    plotting = Control is plots are displayed during runs (default = True)
    """
    ##############################################
    # Determine which columns are not stationary #
    ##############################################
    # Create temporary dataframes
    stat_test = pd.DataFrame()
    diff_df = pd.DataFrame()
    fstat_test = pd.DataFrame()
    statdf = pd.DataFrame()

    # Find non-stationary columns
    for ncol in range(len(df.columns)):
        # Populate stat_test with each column name and it's p-value
        stat_test = pd.concat([stat_test, pd.DataFrame([df.columns[ncol], adfuller(df.iloc[:,ncol])[1]])],axis = 1)
    stat_test = stat_test.T.rename(columns={0:"Feature",1:"p-value"})

    if len(stat_test) == 0:
        print('All data is stationary')
    else:
        if verbose: print('Beginning stationary correction via differencing')
        # Extract only those Feature names who's values are above threshold
        nonstat = pd.DataFrame(stat_test[stat_test['p-value'] > thres])['Feature']

        # Create dataframe of only stationary data
        statdf = df.drop(columns = nonstat.to_list(), axis = 1)

        if verbose: print('There are', len(nonstat), 'non-stationary features')
        if verbose: print('Diff\tp-value')
        lcnt = 1
        while (lcnt <= difflim) and (len(nonstat)!=0): 
            # Only loop as many times as difflim allows and test that there are still columns to be differenced
            #if verbose: print('...running iteration:', str(diffrng * lcnt))
            for col in nonstat: 
                # For each listing in nonstat take the difference at the multiple of diffrng and loop counter
                # Perform the differencing at this iteration
                tdiff = df[col].diff(diffrng * lcnt).iloc[1:]
                adfpval = adfuller(tdiff)[1]
                if  adfpval < thres:
                    # If the ADFUller test is < threshold add this column into diff_df
                    diff_df[col+"_Delta"+str(lcnt)] = tdiff
                    # Also Remove it from nonstat
                    nonstat = nonstat[nonstat != col]
                    # Populate stat_test with each column name and it's p-value
                    fstat_test = pd.concat([fstat_test, pd.DataFrame([col, adfpval])],axis = 1)
                    if verbose: print(str(lcnt),'\t',str(adfpval),'\t',col)

                #stat_test_diff = pd.concat([stat_test_diff, pd.DataFrame([col, adfuller(tdiff)[1]])],axis = 1)
            #stat_test = stat_test.T.rename(columns={0:"Feature",1:"p-value"})
            if verbose: print('...running iteration:', str(lcnt), 'complete\n',str(len(nonstat)),'values remain to be corrected')
            lcnt += 1

        # Merge stationary and original data
        statdf = pd.concat([statdf, diff_df], axis=1)
        # Backfill missing values as differencing removes leading values equal to order of differencing
        statdf = statdf.interpolate(method='bfill')
        # Rearrange fstat_test so that it will print nicely for the user
        fstat_test = fstat_test.T.rename(columns={0:"Feature",1:"p-value"})
        # Plot the entire dataframe after stationary corrections finish
        if plotting: multi_plot(statdf, "Data after differencing is complete")
    
    return statdf, fstat_test

def MakeStationaryLPC(df, AllCols=False, calcstat=True, thres=0.05, diffrng=1, difflim=3, verbose=True, plotting=False):
    """
    MakeStationaryLPC(df, nonstat, difflim, verbose)
    MakeStationaryLPC takes a DataFrame (df) and uses Log Percent Change to make it stationary. The loop differencing will run up to it's limit (difflim) and will perform the differencing over
         a specified interval (diffrng).

    df      = Pandas dataframe with any number of non-stationary columns
    thres   = The ADFuller threshold for stationary p-values (default = 0.05)
    diffrng = The length of differencing to be applied (default = 1, i.e. previous day)
    difflim = The maximum number of differencing loops to perform on difficult columns
    verbose = Control informational printing during function execution (default = True)
    plotting = Control is plots are displayed during runs (default = True)
    """
    ##############################################
    # Determine which columns are not stationary #
    ##############################################
    # Create temporary dataframes
    stat_test = pd.DataFrame()
    diff_df = pd.DataFrame()
    fstat_test = pd.DataFrame()
    statdf = pd.DataFrame()

    # Must Calculate statistics if not performing stationary work on all columns
    if not AllCols: calcstat = True

    # Calculate stationarity columns
    if calcstat:
        if verbose: print('Calculationg AD Fuller values for all columns')
        for ncol in range(len(df.columns)):
            # Populate stat_test with each column name and it's p-value
            stat_test = pd.concat([stat_test, pd.DataFrame([df.columns[ncol], adfuller(df.iloc[:,ncol])[1]])],axis = 1)
        stat_test = stat_test.T.rename(columns={0:"Feature",1:"p-value"})
    else:
        stat_test = pd.DataFrame()

    if AllCols:
        if verbose: print('Performing Percent Change Calculation on all columns')
        # Make percent change of all columns
        statdf = df.pct_change()
        # remove the first row as it's all NaN and replace any other NaN with 0 (pct_change NaN -> NaN = NaN)
        statdf = statdf.iloc[1:,:].fillna(0)
        # Replace all -inf and inf values with NaN (pct_change(num -> NaN) = inf or -inf)
        statdf = statdf.replace([np.inf, -np.inf], np.nan).fillna(0)
        

    else:
        if verbose: print('Testing and making non-stationary columns stationary with LPC\n!VERIFY THIS OUTPUT WHEN USING IT FOR THE FIRST TIME!')
        if len(stat_test) == 0:
            print('All data is stationary')
        else:
            if verbose: print('Beginning stationary correction via differencing')
            # Extract only those Feature names who's values are above threshold
            nonstat = pd.DataFrame(stat_test[stat_test['p-value'] > thres])['Feature']

            # Create dataframe of only stationary data
            statdf = df.drop(columns = nonstat.to_list(), axis = 1)

            if verbose: print('There are', len(nonstat), 'non-stationary features')
            if verbose: print('Diff\tp-value')
            lcnt = 1
            while (lcnt <= difflim) and (len(nonstat)!=0): 
                # Only loop as many times as difflim allows and test that there are still columns to be differenced
                #if verbose: print('...running iteration:', str(diffrng * lcnt))
                for col in nonstat: 
                    # For each listing in nonstat take the difference at the multiple of diffrng and loop counter
                    # Perform the differencing at this iteration
                    tdiff = df[col].pct_change(diffrng * lcnt).iloc[1:]
                    adfpval = adfuller(tdiff)[1]
                    if  adfpval < thres:
                        # If the ADFUller test is < threshold add this column into diff_df
                        diff_df[col+"_Delta"+str(lcnt)] = tdiff
                        # Also Remove it from nonstat
                        nonstat = nonstat[nonstat != col]
                        # Populate stat_test with each column name and it's p-value
                        fstat_test = pd.concat([fstat_test, pd.DataFrame([col, adfpval])],axis = 1)
                        if verbose: print(str(lcnt),'\t',str(adfpval),'\t',col)

                    #stat_test_diff = pd.concat([stat_test_diff, pd.DataFrame([col, adfuller(tdiff)[1]])],axis = 1)
                #stat_test = stat_test.T.rename(columns={0:"Feature",1:"p-value"})
                if verbose: print('...running iteration:', str(lcnt), 'complete\n',str(len(nonstat)),'values remain to be corrected')
                lcnt += 1

            # Merge stationary and original data
            statdf = pd.concat([statdf, diff_df], axis=1)
            # Backfill missing values as differencing removes leading values equal to order of differencing
            statdf = statdf.interpolate(method='bfill')
            # Rearrange fstat_test so that it will print nicely for the user
            stat_test = fstat_test.T.rename(columns={0:"Feature",1:"p-value"})
        
    # Plot the entire dataframe after stationary corrections finish
    if plotting: multi_plot(statdf, "Data after differencing is complete")
    
    return statdf, fstat_test, df

def unLPC(df, LPC):
    """
    unLPC performs the reversal of a (natural)Log Percent Change calculation. LPC is usually performed as df.pct_change()
    df = The original DataFrame (true units not LPC) The last row value will be used as the S0 value
    LPC = Log Percent Change data frame. Will be converted
    """
    S0 = float(df.iloc[-1])
    fixed = pd.DataFrame().reindex_like(LPC)
    rcnt = 1
    while rcnt < len(fixed):
        #print(fixed.iloc[[rcnt-1]].shape)
        #print('row',str(rcnt),'\n',fixed.iloc[rcnt,:])
        fixed.iloc[[rcnt]] = LPC.iloc[[rcnt-1]]  * S0 #fixed.iloc[[rcnt-1]] 
        rcnt += 1
    return fixed

def unPCall(df, PC):
    """
    unPCall performs the reverse of percent change for all columns.
    OUTPUT:
        corrected = df of values corrected to original values
    INPUT:
        df = original data frame
        PC = dataframe of percent changed values
    """
    # Build the output df the same shape as the PC df, then remove the row index to allow for math to apply correctly
    corrected = pd.DataFrame().reindex_like(df)#.reset_index(drop=True)
    # Set the first row the last known value as the basis for un-percent change operation
    corrected.iloc[0] = df.iloc[0]
    # Fill the remainder of the corrected df with the %changes
    corrected.iloc[1:,:] = PC.add(1)
    
    # Set the first value in the corrected df as it differs from the loop logic
    #corrected.iloc[[0]] = df.iloc[[0]].reset_index(drop=True) * PC.iloc[[0]].reset_index(drop=True)

    # Print shapes for validation
    print('Output will be len',len(corrected))
    print('df is len', len(df))
    print('PC is len', len(PC))
    # Initialize loop counter
    rcnt = 1
    # Loop through all rows >0 performing calculation of previous row * PC for current day
    for rcnt in range(1, corrected.index.size):
        # Print step info every 10 steps
        if (rcnt % 10) == 0: print(rcnt, end='\r')
        # Current row values = Last row values * %Change from previous day to today as stored in current row
        corrected.iloc[rcnt] = corrected.iloc[rcnt-1] * corrected.iloc[rcnt]
    
    print('Finished All of', len(corrected))
    
    # Reassign index to corrected values using index of PC
    #corrected = corrected.set_index(PC.index)

    # Return corrected df
    return corrected