import pandas as pd
from pathlib import Path
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
from datetime import datetime, timedelta, date
import shelve
from time import perf_counter
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
import requests
import json
from skopt import BayesSearchCV
from sklearn_genetic import GASearchCV
#from genetic_selection import 

#####################
# MODEL PREPARATION #
#####################

# Create Lagged columns
def lagger(df, target=None, lag_delta=1, lag_limit=30, verbose=True, plotting=True):
	"""
	lagger performs lagging on all columns other than target with a provided delta and a limit. The resulting dataframe will have shape:
	N + (N-1)*(lag_limit / lag_delta) where N=# of columns in df. The default case will provide an output df of shape N+(N-1)*30

	OUTPUT:
		lag_df = dataframe of original df plus each lagged column

	INPUT:
		df = Pandas dataframe to be lagged
		target = column name that will not be lagged (default=None)
		lag_delta = provides the lagging length, will be used as the shift value (default=1)
		lag_limit = provides the lagging limit for how many iterations of the delta to perform (default=30)
		verbose = controls printing during run (default=True)
	"""
	if verbose: print('Original df shape:', df.shape)
	# List all columns that will be lagged
	lag_cols = [col for col in df.columns if col != target]
	
	# Initialize the lag_df as a full copy of the input df
	lag_df = df.copy(deep=True)

	# Loop through each lag column
	for col in lag_cols:
		# Initialize the step counter
		step = lag_delta
		
		# Do the lagging in a loop 
		while step <= lag_delta * lag_limit:
			# Perform the lagging and save it as a new column with name based on the current step
			lag_df[col+"_lag_"+str(step)] = lag_df[col].shift(step)
			# Increment the step based on the current step and the lag_delta
			step = step + lag_delta
		
	if verbose: print('Lagged df shape (with NaN):', lag_df.shape)
	lag_df = lag_df.dropna(axis=0)

	if verbose: 
		print('Lagged df shape (without NaN):', lag_df.shape)
		print('There are', df.isna().sum().sum(), 'NaN values remaining')
	
	
	return lag_df, df

# Create test train windows
def get_windows(y, cv):
    """Generate windows used in Time Series Forecasting"""
    train_windows = []
    test_windows = []
    for i, (train, test) in enumerate(cv.split(y)):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows

# Create expanding window cv iterables
def ts_cv_exwindow(df, iw, sl, forecast_horizon, verbose=True):
    """
    ts_cv_builder does all assembly work to create an iterable time series CV object for use in RFE or HP Optimizations
    
    df = dataframe to slice into cv chunks
    iw = initial window for first training set
    sl = step length, how many points into the future are added in each cv step
    forecast_horizon = how many points are tested at each cv training interval

    verbose = print during run? (default=True)
    """
    # Create CV object using expanding window
    cv = ExpandingWindowSplitter(initial_window=iw, fh=forecast_horizon, step_length=sl)
    # Generate number of splits
    n_splits = cv.get_n_splits(df)
    
    if verbose: 
        print('Initial training window ends on: ', df.iloc[[iw]].index[0])
        print(f"Number of Folds = {n_splits}")

    # Extract window data from df
    train_windows, test_windows = get_windows(df, cv)
    
    # Create iterable windows that can be fed into the CV function of the optimizaation script
    cv_iterables = zip(train_windows, test_windows)

    return cv_iterables

# Create sliding window cv iterables
def ts_cv_slwindow(df, iw, sl, forecast_horizon, verbose=True):
    """
    ts_cv_builder does all assembly work to create an iterable time series CV object for use in RFE or HP Optimizations
    
    df = dataframe to slice into cv chunks
    iw = initial window for first training set
    sl = step length, how many points into the future are added in each cv step
    forecast_horizon = how many points are tested at each cv training interval

    verbose = print during run? (default=True)
    """
    # Create CV object using expanding window
    cv = SlidingWindowSplitter(initial_window=iw, fh=forecast_horizon, step_length=sl)
    # Generate number of splits
    n_splits = cv.get_n_splits(df)
    
    if verbose: 
        print('Initial training window ends on: ', df.iloc[[iw]].index[0])
        print(f"Number of Folds = {n_splits}")

    # Extract window data from df
    train_windows, test_windows = get_windows(df, cv)
    
    # Create iterable windows that can be fed into the CV function of the optimizaation script
    cv_iterables = zip(train_windows, test_windows)

    return cv_iterables

#################
# RANDOM FOREST #
#################

# RandomForest Optimization with RFE, RandomSearchCV and Lagging (unused, RFE is ineffective)
def RF_RFE_RandomSearchCV(df, iw, sl, fh, min_feat, random_grid, pred, targ, search_iters=10, max_iter=10, init_params=None, verbose=True, debugging=False):
    """
    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    min_feat = Minimum Features to select in RFE
    random_grid = dictionary of grid options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    max_iter = maximum number of times the RFE+RandomSearch loop can execute (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbose':0,
            'n_jobs':-1
        }
    # Define initial lcnt for multi-index storing of parameters during runs
    lcnt = 0
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()
    rfe_col_log = pd.DataFrame()   
    rfe_cv_log = pd.DataFrame()
    #df_rfe = df.copy(deep=True)                             
    # Define predictor and target columns
    predictors = pred
    target = targ
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, 13*3, 13, 13, verbose=True)
    # Create dataframe of Target
    df_target = df[targ]
    # Limit df to only include predictor data
    df = df[pred]
    #####################################################
    # Loop through RFE and HyperParameter optimizations #
    #####################################################
    while lcnt < max_iter:
        print('\nLoop', str(lcnt))
        ##################################
        # Recurrsive Feature Elimination #
        ##################################
        # Build cross validataion iterables
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Using generated rf_params (basic to start and then updated after each SearchCV run)
        alg = RandomForestRegressor(**rf_params)
        rfe_alg = RFECV(
                    alg, 
                    step=0.1, 
                    min_features_to_select=min_feat, 
                    cv=cv_iterables, 
                    scoring='neg_mean_absolute_error', 
                    verbose=0, 
                    n_jobs=-1, 
                    importance_getter='auto'
                    )
        # Get time, in seconds, before optimization starts (tic)
        tic = perf_counter()
        # Run CV optimization
        rfe_out = rfe_alg.fit(df[predictors], df_target)
        # Extract Selected features and information about them (rank and if included in output)
        df_features = pd.DataFrame(columns = ['feature', 'selected', 'ranking'])
        if debugging: print('df_features is shape',df_features.shape, '\ndf is shape',df.shape,'df.shape[1] is',df.shape[1],'\nLoop Range is',str(range(df.shape[1])))
        for i in range(df.shape[1]):
            if debugging: print(i, end='\r')
            row = {'feature': df.columns[i], 'selected': rfe_out.support_[i], 'ranking': rfe_out.ranking_[i]}
            df_features = df_features.append(row, ignore_index=True)
        if debugging: print('')
        df_rfe = df[df.columns[rfe_out.get_support(1)]]
        #df_rfe = pd.concat([df_rfe, df_target], axis=1)
        # Get time, in seconds, after optimization finishes (toc)
        toc= perf_counter()
        # Calculate run_time in seconds
        run_time = toc-tic
        if verbose: print('RFE runtime was',str(timedelta(seconds=run_time)),'.  df_rfe shape: ', df_rfe.shape)
        # Add current run to rfe_col_log 
        new_rfe = pd.concat([df_features.set_index('feature')], keys=[lcnt], names=['Iteration', 'Data'], axis=1)
        rfe_col_log = pd.concat([rfe_col_log, new_rfe], axis =1)
        # Add current run to rfe_cv_log
        new_cv = pd.concat([pd.DataFrame.from_dict(rfe_out.cv_results_)], keys=[lcnt], names=['iteration', 'data'], axis=1)
        rfe_cv_log = pd.concat([rfe_cv_log, new_cv], axis=1)
        # Add current run rfe feature importance to log
        new_rfe_fi = pd.DataFrame({'feature':df_rfe.columns, lcnt:rfe_out.estimator_.feature_importances_}).set_index('feature')
        rfe_fi_log = pd.concat([rfe_fi_log, new_rfe_fi], axis=1)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df_rfe, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        # Use no bias when beginnng the search so do not include rf_params
        alg = RandomForestRegressor(verbose=0, n_jobs=-1) #**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = RandomizedSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_distributions = random_grid, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring='neg_mean_absolute_error' # closest to 0 is best fit
            )
        # Get time, in seconds, before optimization starts (tic)
        tic = perf_counter()
        # Build list of predictors that are in df_rfe still
        #predictors = [x for x in df_rfe.columns if x not in [target]+[IDcol]]
        # Run CV optimization
        fit_out = search_alg.fit(df_rfe, df_target)
        # Get time, in seconds, after optimization finishes (toc)
        toc= perf_counter()
        # Calculate run_time in seconds
        run_time = toc-tic
        if verbose: print('Optimization runtime was',str(timedelta(seconds=run_time)))
        # Set rf_params to output for next loop
        rf_params = fit_out.best_params_
        # Log current iteration rf_params
        best_param_df = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').rename(columns={0:lcnt})
        rf_params_log = pd.concat([rf_params_log, best_param_df], axis =1)
        # Increment lcnt for next iteration
        lcnt = lcnt + 1
    

    ####
    # Need to incorporate threshold validation for RFE and RandomSearch to drop out when stead state is reached
    ####
    
    return df_rfe, search_alg, rfe_alg, rfe_col_log, rf_params_log

# RandomForest Optimization with RandomSearchCV and Lagging
def Lag_RF_RandomSearchCV(df, iw, sl, fh, random_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=10, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform RandomSearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    random_grid = dictionary of grid options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbose':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLag', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = RandomForestRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = RandomizedSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_distributions = random_grid, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method # closest to 0 is best fit
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Random Grid hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# RandomForest Optimization with Bayesian SearchCV and Lagging
def Lag_RF_BayesSearchCV(df, iw, sl, fh, search_params, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_root_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform BayesOptCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    search_params = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbose':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLag', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = RandomForestRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = BayesSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            search_spaces = search_params, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            #n_points= 32, # How many points to optimize simultaneously
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=0, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method, # closest to 0 is best fit
            optimizer_kwargs = {'base_estimator': 'RF'} # Use Random Forest Surrogate function
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()



    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Bayesian hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# RandomForest Optimization with Genetic SearchCV and Lagging
def Lag_RF_GeneticSearchCV(df, iw, sl, fh, param_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform GASearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    param_grid = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbose':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = RandomForestRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = GASearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_grid = param_grid, # Grid or distribution of hyperparameters to sample
            population_size=20,
            generations=search_iters, 
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=True, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            criteria='min',
            scoring=score_method, # closest to 0 is best fit
            keep_top_k = 1
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Evolutionary hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log


###########
# XGBoost #
###########

def Lag_XGB_RandomSearchCV(df, iw, sl, fh, random_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=10, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Lag_XGB_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    Iteratively lag df from 1 to 'lags' and perform RandomSearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
        df = dataframe to be processed
        iw = initial window for CV
        sl = step length for CV
        fh = forecast horizion for CV
        random_grid = dictionary of grid options to search during hyperparameter optimizataion
        pred = list of predictors to use in first loop
        targ = target column to be predicted
        search_iters = number of iterations to search within random_grid (default=10)
        init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
        verbose = controls standard print during run (default=True)
        debugging = controls debugging print during run (default=False)
    
    OUTPUT:
        df = Last RFE reduced dataframe
        search_alg = last output of RandomSearchCV optimization Model
        rfe_alg = last output of RFECV model
        rfe_col_log = Log of all RFE removed columns in each iteration
        rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbosity':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = xgb.XGBRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = RandomizedSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_distributions = random_grid, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method # closest to 0 is best fit
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Random Grid hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# XGBoost Optimization with Bayesian SearchCV and Lagging
def Lag_XGB_BayesSearchCV(df, iw, sl, fh, search_params, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform BayesOptCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    search_params = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbosity':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLap', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = xgb.XGBRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = BayesSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            search_spaces = search_params, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            #n_points= 32, # How many points to optimize simultaneously
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=0, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method, # closest to 0 is best fit
            #optimizer_kwargs = {'base_estimator': 'RF'} # Use Random Forest Surrogate function
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Bayesian hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# XGBoost Optimization with Genetic SearchCV and Lagging
def Lag_XGB_GeneticSearchCV(df, iw, sl, fh, param_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform GASearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    param_grid = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbosity':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = xgb.XGBRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = GASearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_grid = param_grid, # Grid or distribution of hyperparameters to sample
            population_size=50,
            generations=35, 
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method, # closest to 0 is best fit
            keep_top_k = 4
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Genetic hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

####################################
# Multi-layer Perceptron regressor #
####################################

# MLPR Optimization with Bayesian SearchCV and Lagging
def Lag_MLPR_RandomSearchCV(df, iw, sl, fh, random_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=10, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Lag_XGB_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    Iteratively lag df from 1 to 'lags' and perform RandomSearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
        df = dataframe to be processed
        iw = initial window for CV
        sl = step length for CV
        fh = forecast horizion for CV
        random_grid = dictionary of grid options to search during hyperparameter optimizataion
        pred = list of predictors to use in first loop
        targ = target column to be predicted
        search_iters = number of iterations to search within random_grid (default=10)
        init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
        verbose = controls standard print during run (default=True)
        debugging = controls debugging print during run (default=False)
    
    OUTPUT:
        df = Last RFE reduced dataframe
        search_alg = last output of RandomSearchCV optimization Model
        rfe_alg = last output of RFECV model
        rfe_col_log = Log of all RFE removed columns in each iteration
        rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'shuffle':False
            , 'early_stopping':False
            }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = MLPRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = RandomizedSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_distributions = random_grid, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method # closest to 0 is best fit
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Random Grid hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# MLPR Optimization with Bayesian SearchCV and Lagging
def Lag_MLPR_BayesSearchCV(df, iw, sl, fh, search_params, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform BayesOptCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    search_params = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'shuffle':False
            , 'early_stopping':False
            }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = MLPRegressor(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = BayesSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            search_spaces = search_params, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            #n_points= 32, # How many points to optimize simultaneously
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method # closest to 0 is best fit
            #optimizer_kwargs = {'base_estimator': 'RF'} # Use Random Forest Surrogate function
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Bayesian hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

# MLPR Optimization with Genetic SearchCV and Lagging
def Lag_MLPR_GeneticSearchCV(df, iw, sl, fh, param_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=50, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Iteratively lag df from 1 to 'lags' and perform GASearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
    RF_RFE_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    df = dataframe to be processed
    iw = initial window for CV
    sl = step length for CV
    fh = forecast horizion for CV
    param_grid = dictionary of space options to search during hyperparameter optimizataion
    pred = list of predictors to use in first loop
    targ = target column to be predicted
    search_iters = number of iterations to search within random_grid (default=10)
    init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
    verbose = controls standard print during run (default=True)
    debuggin = controls debugging print during run (default=False)
    
    OUTPUT:
    df_rfe = Last RFE reduced dataframe
    search_alg = last output of RandomSearchCV optimization Model
    rfe_alg = last output of RFECV model
    rfe_col_log = Log of all RFE removed columns in each iteration
    rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {
            'verbose':0,
            'n_jobs':-1
        }
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        alg = MLPRegressorr(**rf_params)
        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = GASearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_grid = param_grid, # Grid or distribution of hyperparameters to sample
            population_size=50,
            generations=35, 
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method, # closest to 0 is best fit
            keep_top_k = 4
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Genetic hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

##################################
# Long Short Term Memory (Keras) #
##################################

# MLPR Optimization with Bayesian SearchCV and Lagging
def Lag_kLSTM_RandomSearchCV(df, iw, sl, fh, random_grid, pred, targ, fit_shelf, payload_url, lags=13, search_iters=10, init_params=None, score_method='neg_mean_absolute_error', verbose=True, debugging=False):
    """
    Lag_XGB_RandomSearchCV performs RFE and RandomSearch with CV to optimize RandomForestRegression()
    Iteratively lag df from 1 to 'lags' and perform RandomSearchCV to optimize each over the given interval
    Does NOT perform any prediction work, that will be done seperately this only optimizes the hyperparameters for the given lag with CV

    INPUT:
        df = dataframe to be processed
        iw = initial window for CV
        sl = step length for CV
        fh = forecast horizion for CV
        random_grid = dictionary of grid options to search during hyperparameter optimizataion
        pred = list of predictors to use in first loop
        targ = target column to be predicted
        search_iters = number of iterations to search within random_grid (default=10)
        init_params = initial RandomForest parameters used when peforming run 0 RFE (default=None)
        verbose = controls standard print during run (default=True)
        debugging = controls debugging print during run (default=False)
    
    OUTPUT:
        df = Last RFE reduced dataframe
        search_alg = last output of RandomSearchCV optimization Model
        rfe_alg = last output of RFECV model
        rfe_col_log = Log of all RFE removed columns in each iteration
        rf_params_log = log of all selected hyperparameter sets in each iteration
    """
    # Configure initial RandomForestRegression Parameters
    if init_params:
        rf_params = init_params
    else:
        rf_params = {0: 'LSTM(units=50, return_sequences=True, input_shape=('& df.shape[1] &', 1))'}
        rf_params[1] = 'Dropout(0.2)'
        rf_params[2] = 'LSTM(units=50,return_sequences=True)'
        rf_params[3] = 'Dropout(0.2)'
        rf_params[4] = 'LSTM(units=50,return_sequences=True)'
        rf_params[5] = 'Dropout(0.2)'
        rf_params[6] = 'LSTM(units=50)'
        rf_params[7] = 'Dropout(0.2)'
        rf_params[8] = 'Dense(units=1)'
        rf_params[9] = " optimizer='adam',loss=" & score_method
    
    # Open shelf to store fits
    s = shelve.open(fit_shelf, flag='c', writeback=True)
    if verbose: print(fit_shelf+' opened ')
    # Pre-Define data frames
    rf_params_log = pd.DataFrame()                    
    # Create and print cv data once
    cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=True)
    # Initialize fit parameter log
    df_fit_params = pd.DataFrame()
    # Initizlize cv_result log
    df_cv_results = pd.DataFrame()
    # Initialize run_time_log
    run_time_log = pd.DataFrame()
    # Store original df for recall later
    df_orig = df.copy(deep=True)
    # Set run_date in case optimizations spans multiple days
    run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date is: {run_date}')

    #############################################
    # Loop through HyperParameter optimizations #
    #############################################
    tic_overall = perf_counter()
    for lcnt in range(1, lags+1, 1):
        print('\nLoop', str(lcnt))
        # Get time, in seconds, before iteration starts (tic)
        tic = perf_counter()
        ##############################
        # Perform Lagging Operations #
        ##############################
        # Revert df to original
        df = df_orig.copy(deep=True)
        # Shift prediction values forward lcnt steps
        df[pred] = df[pred].shift(lcnt)
        # Drop NaN values due to shift
        df = df.dropna(axis=0)
        if verbose: print('Shape of lag', lcnt, 'iteration df is', df.shape)
        ################################
        # Hyperparameter Optimizataion #
        ################################
        # create cv_iterable for this run
        cv_iterables = ts_cv_exwindow(df, iw, sl, fh, verbose=False)
        # Define algorithm for modeling
        mbcnt = 0 # define (m)odel (b)uild counter
        alg = Sequential()
        while mbcnt < max(rf_params): # add each parameter set incrementally
            alg.add(rf_params[mbcnt])
        alg.compile(rf_params[max(rf_params)]) # use final rf_params set as compile param

        # Define a Random Search of fitting function that will sample the hyperparameter space and perform CV
        search_alg = RandomizedSearchCV(  # Original values were: cv=3 , n_iter=100
            estimator = alg, # Defined above, RandomForestRegression or XGBoost
            param_distributions = random_grid, # Grid or distribution of hyperparameters to sample
            n_iter = search_iters, # How many iterations of param_distribution will be sampled
            cv = cv_iterables, # Iterable ist of CV space created by ExpandingWindowSplitter for TS
            verbose=1, # High = more printing, 0=No Print, 10=all/max?
            #random_state=42, # Forced random number state to allow comparison between runs
            n_jobs = -1, # How many cores, -1 = All available
            return_train_score = True, # Computationally expensive, but returns training scores to over/under fit comparison of CV space
            refit = True, # will allow the use of the best estimator with .predict() after run
            scoring=score_method # closest to 0 is best fit
            )
        # Run CV optimization
        fit_out = search_alg.fit(df[pred], df[targ])
        # Pickle and store current model on shelf
        s[run_date+"_lag_"+str(lcnt)] = fit_out
        # Log current iteration best_params
        new_fit_param = pd.DataFrame.from_dict(fit_out.best_params_, orient='index').T
        new_fit_param['run_date'] = run_date
        new_fit_param['lag'] = lcnt
        df_fit_params = pd.concat([df_fit_params, new_fit_param], axis=0)
        # Log current iteration cv_results_
        new_cv_result = pd.DataFrame.from_dict(fit_out.cv_results_)
        new_cv_result['run_date'] = run_date
        new_cv_result['lag'] = lcnt
        new_cv_result = new_cv_result.reset_index().rename(columns={'index':'CV_Fold'})
        df_cv_results = pd.concat([df_cv_results, new_cv_result], axis=0, ignore_index=False)
        # Log current fit runtime from tic-toc
        if verbose: print('Optimization runtime was',str(timedelta(seconds=perf_counter()-tic)))
        new_run_time = pd.DataFrame([perf_counter()-tic]).rename(columns={0:'run_time'})
        new_run_time['run_date'] = run_date
        new_run_time['lag'] = lcnt
        run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=False)
    # Log total function run_time as lag=0 value
    total_run_time = perf_counter() - tic_overall
    new_run_time['run_time'] = total_run_time
    new_run_time['run_date'] = run_date
    new_run_time['lag'] = 0
    run_time_log = pd.concat([run_time_log, new_run_time], axis=0, ignore_index=True)
    # Close (and sync) the shelf
    s.close()
    # Send a Teams message using the Benzene Forecasting Team Incoming Webhook with output info from the model
    payload = {
        "text": "Random Grid hyperparameter search with lagging complete. <br><br> \
            Run time:<br>&nbsp;&nbsp;" + str(timedelta(seconds=total_run_time)) + "<br><br> \
            Total Lags:<br>" + str(lcnt) # +  "<br><br> \
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(payload_url, headers=headers, data=json.dumps(payload))
    #print(response.text.encode('utf8'))
    print('Total Runtime:', str(timedelta(seconds=total_run_time)),' H:M:S.sss')
    return df_fit_params, df_cv_results, run_time_log

###############
# PREDICTIONS #
###############
# These function are Algorithm agnostic and run whatever is contained in the Shelf

# Prediction using any model optimized above
def Lag_Predict(df, pred, targ, models, shelf, pred_out, lpred_out, param_out, fake_model_date=None, fake_run_date=None, verbose=True, debugging=False):
    """
    Lag_RF_Predict uses optimized model parameters from "2 Benzene Model Optimization" to Train on new data and predict each lag step included in the model

    INPUTS:
        df = dataframe of data to train and utilize for prediction
        pred = list of predictor column names
        targ = target column name
        models = List of models held in the shelf to be used for each lagged step
        shelf = shelve opject containing pickles of models
        pred_out = file location to store prediction data
        lpred_out = file location to store long prediction data
        verbose = control printing during runs (default=True)
    OUTPUTS:
        preds = wide dataframe of prediction values
        lpreds = long dataframe of prediction values
        preds_hist = entry history of preds (as stored in parquet file)
        lpreds_hist = entry history of lpreds (as stored in parquet file)
    """
    # Initialize dataframe
    tpred = pd.DataFrame()
    preds = pd.DataFrame(); # preds['run_date']=""; preds['lag']=""
    lpreds = pd.DataFrame()
    fit_params = pd.DataFrame()

    # Get Algorithm name
    model_name = str(shelf[models[0]].best_estimator_).split('(')[0]
    if debugging: print(model_name)
    if model_name in ['RandomForestRegressor','XGBoost']:
        feat_import_toggle = True
    else:
        feat_import_toggle = False
        

    # Set run_date for use later
    if fake_run_date:
        run_date=fake_run_date
    else:
        run_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    
    # Capture date for when the routine was run
    if fake_model_date:
        model_date = fake_model_date
    else:
        model_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
        print(f'Model Date: {model_date}')

    # Rebuild df so that it includes all future lags
    df_orig = pd.concat([df, pd.DataFrame(pd.date_range(df.index[-1], periods=len(models)+1, freq='w'))[1:].set_index(0)], axis=0)
    
    # Loop for each model to perform prediction
    for lcnt in range(0, len(models)):
        lag = lcnt +1
        if debugging: print('Lag', lag)
        # Revert df to original data for next iteration
        df = df_orig.copy(deep=True)
        # Shift Forward by lag
        df[pred] = df[pred].shift(lag)
        # Extract prediciton data set
        df_pred = df[pred][-len(models):].dropna(axis=0)
        if debugging: print('Prediction shape:', df_pred.shape)
        # Drop training rows with any NaN (includes prediction data as target = NaN)
        df = df.dropna(axis=0)
        # Load alg from Shelf by loop count (lcnt)
        #alg = shelf[list(shelf)[lcnt]].best_estimator_
        alg = shelf[models[lcnt]].best_estimator_
        if debugging: print('Current Alg: ', alg)
        # Fit current lag data using current lag optimized model
        alg.fit(df[pred], df[targ])
        # Predict using fit
        new_pred = alg.predict(df_pred)
        if debugging: print('\t',new_pred)
        new_pred = pd.DataFrame(new_pred).set_index(df_pred.index).T
        new_pred.columns = new_pred.columns.map(lambda t: t.strftime('%Y%m%d'))
        tpred = pd.concat([tpred, new_pred], axis=0, ignore_index=True)
        # Store Feature importance for this model if type supports it
        if feat_import_toggle:
            feat_import = pd.DataFrame(alg.feature_importances_).rename(columns={0:str(lag)})
            feat_name = pd.DataFrame(alg.feature_names_in_).rename(columns={0:"Feature"})
            tfit_params = pd.concat([feat_name, feat_import], axis=1).set_index('Feature')
            fit_params = pd.concat([fit_params, tfit_params], axis=1, ignore_index=False)
    # Add rundate identifier to fit_params
    if feat_import_toggle: 
        fit_params['run_date'] = run_date
        fit_params['model_date'] = model_date
        fit_params = fit_params.reset_index()
    # Convert tpreds to lpreds
    for ccnt in range(0, tpred.shape[1]):
        lfut = pd.DataFrame(tpred.iloc[ccnt])
        lfut['lag'] = lfut.columns[0]+1
        lfut['run_date'] = run_date
        lfut= lfut.dropna(axis=0).reset_index().rename(columns={ccnt:'value', 'index':'pred_date'})
        lpreds = pd.concat([lpreds, lfut], axis=0, ignore_index=True)
    lpreds['model_date'] = model_date
    # Add identifier columns to tpred before merging with preds
    tpred['run_date'] = run_date
    tpred['model_date'] = model_date
    tpred = tpred.reset_index().rename(columns={'index':'lag'})
    tpred['lag'] = tpred['lag']+1
    preds = pd.concat([preds, tpred], axis=0, ignore_index=True)
    # Export wide prediction data
    if Path(pred_out).is_file():
        preds_hist = pd.read_parquet(pred_out)
        preds_hist = pd.concat([preds_hist, preds], axis=0, ignore_index=True)
        preds_hist.to_parquet(path=pred_out, engine='pyarrow', compression=None, index=True)
    else:
        preds_hist = preds # No current history so set it to current df values
        preds_hist.to_parquet(path=pred_out, engine='pyarrow', compression=None, index=True)
    # Export long prediction data
    if Path(lpred_out).is_file():
        lpreds_hist = pd.read_parquet(lpred_out)
        lpreds_hist = pd.concat([lpreds_hist, lpreds], axis=0, ignore_index=True)
        lpreds_hist.to_parquet(path=lpred_out, engine='pyarrow', compression=None, index=True)
    else:
        lpreds_hist = lpreds # No current history so set it to current df values
        lpreds_hist.to_parquet(path=lpred_out, engine='pyarrow', compression=None, index=True)
    # Exporting fit parameters
    if feat_import_toggle:
        if Path(param_out).is_file():
            param_hist = pd.read_parquet(param_out)
            param_hist = pd.concat([param_hist, fit_params], axis=0, ignore_index=True)
            param_hist.to_parquet(path=param_out, engine='pyarrow', compression=None, index=True)
        else:
            param_hist = fit_params # No current history so set it to current df values
            param_hist.to_parquet(path=param_out, engine='pyarrow', compression=None, index=True)
    
    if feat_import_toggle:
        return preds, lpreds, preds_hist, lpreds_hist, fit_params, param_hist
    else:
        return preds, lpreds, preds_hist, lpreds_hist

# Prediction of history to present using any model optimized above
def Lag_History_Predict(df, iw_int, predictors, target, model_list, s, pred_loc, lpred_loc, param_loc, verbose=True):
    """
    Lag_RF_History_Predict simulates future predictions as weeks are added to the dataset and the next set of weeks is predicted
    INPUTS
        df = dataframe with all future dates to be walked through, will be trimmed to match df_train
        iw_int = integer index to end training on the first loop, will walk forward to end of dataset from this date
        predictors = list of columns to act as predictors
        target = column name to be predicted
        model_list = list of modesl for each lag to be used
        s = Shelf object that contains model_list
        pred_loc = where to save the wide prediction data
        lpred_loc = where to save the long prediction data
    OUTPUTS
        preds = wide dataframe of prediction values
        lpreds = long dataframe of prediction values
        preds_hist = entry history of preds (as stored in parquet file)
        lpreds_hist = entry history of lpreds (as stored in parquet file)
    """
    df_preds = pd.DataFrame()
    df_predsl = pd.DataFrame()
    df_predsh = pd.DataFrame()
    df_predshl = pd.DataFrame()

    # Get Algorithm name
    model_name = str(s[model_list[0]].best_estimator_).split('(')[0]
    if verbose: print(model_name)
    if model_name in ['RandomForestRegressor','XGBoost']:
        feat_import_toggle = True
    else:
        feat_import_toggle = False

    # Force model date to be at initiation of the function so that it is identical for all runs
    sim_model_date = datetime.now().strftime('%Y/%m/%d %H:%M %Z')
    print(f'Model Date: {sim_model_date}')
    
    # Loop through all integers between iw_int+1 and df.shape[0]
    if feat_import_toggle:
        for tcnt in range(iw_int+1, df.shape[0]): # iw_int+1 because slice is < and not <= so it misses final value @ iw_int
            sim_run_date = pd.Timestamp(df.iloc[tcnt-1].name).strftime('%Y/%m/%d %H:%M %Z')
            if verbose: print('Simulating run on ', sim_run_date , 'to predict weeks after',str(df.iloc[tcnt,:].name))
            df_preds, df_predsl, df_predsh, df_predshl, fit_params, param_hist = Lag_Predict(df.iloc[:tcnt,:], predictors, target, model_list, s, pred_loc, lpred_loc, param_loc, fake_model_date=sim_model_date, fake_run_date=sim_run_date, verbose=verbose)
        return df_preds, df_predsl, df_predsh, df_predshl, fit_params, param_hist

    else:
        for tcnt in range(iw_int+1, df.shape[0]): # iw_int+1 because slice is < and not <= so it misses final value @ iw_int
            sim_run_date = pd.Timestamp(df.iloc[tcnt-1].name).strftime('%Y/%m/%d %H:%M %Z')
            if verbose: print('Simulating run on ', sim_run_date , 'to predict weeks after',str(df.iloc[tcnt,:].name))
            df_preds, df_predsl, df_predsh, df_predshl = Lag_Predict(df.iloc[:tcnt,:], predictors, target, model_list, s, pred_loc, lpred_loc, param_loc, fake_model_date=sim_model_date, fake_run_date=sim_run_date, verbose=verbose)
        return df_preds, df_predsl, df_predsh, df_predshl


##############
# OUTPUTTING #
##############
def export_optimization(fit_params, cv_res_log, runtimelog, param_file='./params.parquet', cv_file='./cv.parquet', runlog_file='./run_log.parquet', verbose=True):
    """
    export_optimization stores optimization output files in defined locations
    [fit_params, cv_res_log, runtimelog] are returned from each optimization run
    [param_file, cv_file, runlog_file] are the storage location for each, respectively
    """

    # Export fit_params
    if verbose: print('Exporting fit parameters')
    if Path(param_file).is_file():
        param_hist = pd.read_csv(param_file)
        print('\t', param_file + ' dataset loaded with shape', param_hist.shape, 'and', param_hist.isna().sum().sum(), 'NaN values')
    else:
        param_hist = pd.DataFrame()
        print('\tStorage file does not exist. Beginning with empty DataFrame.')
    param_hist = pd.concat([param_hist, fit_params], axis=0)
    param_hist.to_csv(param_file, index=False)
    
    # Export cv_res_log
    if verbose: print('Exporting cv log')
    if Path(cv_file).is_file():
        cv_hist = pd.read_csv(cv_file)
        print('\t', cv_file + ' dataset loaded with shape', cv_hist.shape, 'and', cv_hist.isna().sum().sum(), 'NaN values')
    else:
        cv_hist = pd.DataFrame()
        print('\tStorage file does not exist. Beginning with empty DataFrame.')
    cv_hist = pd.concat([cv_hist, cv_res_log], axis=0)
    cv_hist.to_csv(cv_file, index=False)

    # Export runtimelog
    if verbose: print('Exporting runtime log')
    if Path(runlog_file).is_file():
        run_hist = pd.read_csv(runlog_file)
        print('\t', runlog_file + ' dataset loaded with shape', run_hist.shape, 'and', run_hist.isna().sum().sum(), 'NaN values')
    else:
        run_hist = pd.DataFrame()
        if verbose: print('\tStorage file does not exist. Beginning with empty DataFrame.')
    run_hist = pd.concat([run_hist, runtimelog], axis=0)
    run_hist.to_csv(runlog_file, index=False)

    if verbose: print('All exports complete')

