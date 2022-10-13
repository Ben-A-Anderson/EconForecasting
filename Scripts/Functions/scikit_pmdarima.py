import pandas as pd
import numpy as np
from scipy.sparse import data
from sklearn import metrics
from prophet import Prophet
import os

def pmdarima_scikit(alg, df, dtrain, dtest, dpred, predictors, target, datafolder ,filename, verbose=True):
    """
    pmdarima_scikit(alg, dtrain, dtest, dpred, predictors, target, IDcol, filename)
    * Performs a scikit ML fit of a given training dataset (dtrain), validates the fit versus a given test dataset (dtest)
    * Facebook Prophet is used to forecast the predictors into the future so that the target can be forecast using an 
    ensamble of the the provided algorithm (alg) and the previously Prophet forecast predictors.

    - Scikit regressions are unable to forecast unless each predictor is known, but can be very accurate as they are multivariate
    - FB Prophet predictions are generally accurate, but are only univariate so they cannot relate additional features as part of the model
    
    alg = Scikit Algorithm used for prediciton of the target
    dtrain = training dataset
    dtest = testing dataset
    dpred = Pandas dataframe of dates into the future to be predicted by both Prophet and alg
    predictors = list of column names to be used for prediction
    target = column name to be determined into the future
    IDcol = reference column for predictiction (date)
    filename = where to store output files
    """
    #####################
    ## SciKit Modeling ##
    #####################
    # Fit the algorithm on the training data
    if verbose: print("Training fit", end='\r')
    alg.fit(dtrain[predictors], dtrain[target])
    if verbose: print("Training fit...Complete")
        
    # Predict training set using the fit:
    if verbose: print("Training prediction", end='\r')
    dtrain_predictions = alg.predict(dtrain[predictors])
    coefs = pd.DataFrame(alg.feature_importances_, predictors).reset_index().rename(columns={'index':'Feature', 0:'Importance'})
    if verbose: print('Training prediction...Complete')

    # Print train fit model report:
    #print("\nTraining Model Report")
    if verbose: print(" > RMSE (Train original vs Train predict): %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))

    # Predict on testing data:
    if verbose: print('Testing prediction', end='\r')
    dtest_predictions = alg.predict(dtest[predictors])
    if verbose: print('Testing prediction...Complete')
    # Print model report:
    #print("\nTesting Model Report")
    if verbose: print(" > RMSE (Test original vs Test predict): %.4g" % np.sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions)))

    ###################################
    ## Prophet Predictor Forecasting ##
    ###################################
    # Initialize forecasted dfs which will include original and Prophet forecasted info
    fdf = df.copy(deep = True) # Average Prophet forecasted values
    fdfu = df.copy(deep = True) # Low range of Prophet forecasted values
    fdfl = df.copy(deep = True) # High range of Prophet forecasted values

    # Initialize temporary Prophet only forecasted values
    tfdf = pd.DataFrame()
    tfdfu = pd.DataFrame()
    tfdfl = pd.DataFrame()

    # Execture model and forecast for each predictor column
    if verbose: print("Prophet is forecasting predictors")
    fcnt = 1
    for x in predictors:
        if verbose: print("...Predicting indicator",str(fcnt),"of",str(len(predictors)), end='\r')
        # Initialize Prophet Model for the next run
        m = Prophet(yearly_seasonality=False, daily_seasonality=False) # Use m to call the Prophet forecaster, can include additional controls in Prophet here
        #print("\t",x)
        m.fit(df[x].reset_index().rename_axis(None,axis=1).rename(columns ={'date':'ds',x:'y'}))
        forecast = m.predict(dpred).rename(columns={'ds':'date'}).set_index('date')
        tfdf = pd.concat([tfdf, forecast['yhat']], axis=1).rename(columns={'yhat': x})
        tfdfu = pd.concat([tfdfu, forecast['yhat_upper']], axis=1).rename(columns={'yhat_upper': x})
        tfdfl = pd.concat([tfdfl, forecast['yhat_lower']], axis=1).rename(columns={'yhat_lower': x})
        fcnt += 1
    if verbose: print();print("Prophet forecasting of predictors...complete")

    ########################
    ## Telling the Future ##
    ########################
    if verbose: print("Foretelling the future with", str(alg), "based on Prophet forecasts", end='\r')
    future = pd.concat([dpred, pd.DataFrame(alg.predict(tfdf[predictors]))], axis=1).rename(columns={0:target, 'ds':'date'}).set_index('date')
    futureu = pd.concat([dpred, pd.DataFrame(alg.predict(tfdfu[predictors]))], axis=1).rename(columns={0:target, 'ds':'date'}).set_index('date')
    futurel = pd.concat([dpred, pd.DataFrame(alg.predict(tfdfl[predictors]))], axis=1).rename(columns={0:target, 'ds':'date'}).set_index('date')
    if verbose: print("Foretelling the future with", str(alg), "based on Prophet forecasts...Complete")

    ######################
    ## Merging Datasets ##
    ######################
    # Merge Predictors with Target
    tfdf = pd.concat([tfdf, future], axis=1, ignore_index=False)
    tfdfu = pd.concat([tfdfu, futureu], axis=1, ignore_index=False)
    tfdfl = pd.concat([tfdfl, futurel], axis=1, ignore_index=False)
    # Merge past and future
    fdf = pd.concat([df, tfdf], axis=0, ignore_index=False)
    fdfu = pd.concat([df, tfdfu], axis=0, ignore_index=False)
    fdfl = pd.concat([df, tfdfl], axis=0, ignore_index=False)

    #################
    ## Export Data ##
    #################
    #try: 
    #    os.path.isdir(datafolder)
    #except:
    #    os.mkdir(datafolder)
    
    if os.path.isdir(datafolder):
        fdf.to_parquet(path=datafolder+filename+'.parquet', engine='pyarrow', compression=None, index=True)
        fdfu.to_parquet(path=datafolder+filename+'_upper.parquet', engine='pyarrow', compression=None, index=True)
        fdfl.to_parquet(path=datafolder+filename+'_lower.parquet', engine='pyarrow', compression=None, index=True)
        coefs.to_parquet(path=datafolder+filename+'_coefs.parquet', engine='pyarrow', compression=None, index=True)
        if verbose: print("Data saved in:", datafolder)
    else:
        os.mkdir(datafolder)
        print("Export Folder Created")
        fdf.to_parquet(path=datafolder+filename+'.parquet', engine='pyarrow', compression=None, index=True)
        fdfu.to_parquet(path=datafolder+filename+'_upper.parquet', engine='pyarrow', compression=None, index=True)
        fdfl.to_parquet(path=datafolder+filename+'_lower.parquet', engine='pyarrow', compression=None, index=True)
        coefs.to_parquet(path=datafolder+filename+'_coefs.parquet', engine='pyarrow', compression=None, index=True)
        if verbose: print("Data saved in:", datafolder)

    #######################
    ## Return Dataframes ##
    #######################
    return fdf, fdfu, fdfl, coefs