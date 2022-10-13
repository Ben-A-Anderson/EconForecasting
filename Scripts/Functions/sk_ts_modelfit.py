import pandas as pd
import numpy as np
import calendar
from datetime import datetime
from datetime import timedelta
import datetime as dt
from sklearn import metrics
from matplotlib import pyplot as plt

##################################################
# modelfit
##################################################
def modelfit(alg, dtrain, dtest, test_known, predictors, target, verbose=True, plotting=True):
    """
    modelfit is a simple implementation for any SciKit Learn ML model
    OUTPUT:
        yhat = Predicted values for the test predictor set
        alg = full algorithm that can be further called to predict values

    INPUT:
        alg = a fully defined SKLearn algorithm
        dtrain = dataframe used for traiing
        dtest = values to be used for prediciton
        predictors = list of predictor that will be trained/tested across
        target = column name that is being predicted using predictors
        
    """
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Print model report:
    if verbose:
        print("\nTraining Report")
        print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    if verbose:
        print('\nTesting Report')
        print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtest[target].values, test_known)))

    # Plot training values vs predicted values for comparison
    if plotting:
        plt.plot(dtrain[target], 'b', label='Training Known')
        plt.plot(dtrain_predictions, '--b', label='Training Prediction')
        plt.plot(test_known, 'k', label='Test Known')
        plt.plot(dtest[target], '--k', label='Test Prediction')
        plt.legend()
        plt.show()

    #Export submission file:
    #IDcol.append(target)
    #yhat = pd.DataFrame({ x: dtest[x] for x in IDcol})
    #yhat.to_csv(filename, index=False)
    return dtest, alg

##################################################
# sk_ts_modelfit
##################################################
def sk_ts_modelfit(alg, dtrain, dtest, dpred, predictors, target, IDcol, filename):
    """
    modelfit allows for a single function to execute any sklearn algorithm.

    alg = algorithm name as defined by scikit
    dtrain = training dataframe
    dtest  = testing dataframe
    dpred = prediction dataframe
    predictors = list of all features to be used as predictors
    target = column name that is to be predicted
    IDcol = reference column(s) that will predict the target
    filename = where to store the prediction output
    """
    
    # Fit the algorithm on the training data
    alg.fit(dtrain[predictors], dtrain[target])
        
    # Predict training set using the fit:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Print train fit model report:
    #print("\nTraining Model Report")
    
    
    # Predict on testing data:
    dtest_predictions = alg.predict(dtest[predictors])
    
    # Print model report:
    #print("\nTesting Model Report")
    

    # Predict the unknown
    dpred_predictions = alg.predict(dpred[predictors])

    # Export prediction file:
    dpred.append(target)
    prediction = pd.DataFrame({ x: dpred[x] for x in IDcol})
    prediction.to_csv(filename, index=False)

##################################################
# add_month 
##################################################
def add_month(df, target, IDcol, forecast_length, forecast_period = 'Day'):
    """
    Add weeks/months to dataframe
    df = dataframe to add to
    forecast_length = # of periods to add
    forecast_period = Weeks or Months
    """
    end_point = len(df)
    df1 = pd.DataFrame(index=range(forecast_length), columns=range(2))
    df1.columns = [target, IDcol]
    df = df.append(df1)
    df = df.reset_index(drop=True)
    x = df.at[end_point - 1, IDcol]
    x = pd.to_datetime(x, format='%Y-%m-%d')
    if forecast_period == 'Week':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], IDcol] = x + timedelta(days=7 + 7 * i)
            df.at[df.index[end_point + i], target] = 0
    elif forecast_period == 'Month':
        days_in_month=calendar.monthrange(x.year, x.month)[1]
        for i in range(forecast_length):
            df.at[df.index[end_point + i], IDcol] = x + timedelta(days=days_in_month + days_in_month * i)
            df.at[df.index[end_point + i], target] = 0
    elif forecast_period == 'Day':
        for i in range(forecast_length):
            df.at[df.index[end_point + i], IDcol] = x + timedelta(days=1 + i)
            df.at[df.index[end_point + i], target] = 0
    #df['date'] = pd.to_datetime(df[IDcol], format='%Y-%m-%d')
    #df['month'] = df['date'].dt.month
    #df = df.drop(['date'], axis=1)
    return df

##################################################
# create_lag
##################################################
def create_lag(df3, n=30):
    """
    Creates lagging features
    df3 = input dataframe
    n = number of days to create lag for
    """
    dataframe = pd.DataFrame()
    for i in range(n, 0, -1):
        for cols in df3.columns:
            dataframe['t-' + str(i)] = df3[cols].shift(i)
    df4 = pd.concat([df3, dataframe], axis=1)
    df4.dropna(inplace=True)
    return df4

##################################################
# sk_forecast
##################################################
def sk_forecast(alg, df, dtrain, dtest, dpred, num_periods, period_type, predictors, target, IDcol, filename):
    """
    modelfit allows for a single function to execute any sklearn algorithm.

    alg = algorithm name as defined by scikit
    df = complete dataframe to be predicted
    dtrain = training dataframe
    dtest  = testing dataframe
    dpred = prediction dataframe
    nmonths = number of months to forecast
    predictors = list of all features to be used as predictors
    target = column name that is to be predicted
    IDcol = reference column(s) that will predict the target
    filename = where to store the prediction output
    """
    
    # Fit the algorithm on the training data
    print("Training fit")
    alg.fit(dtrain[predictors], dtrain[target])
    print("...Complete")
        
    # Predict training set using the fit:
    print("\nTraining prediction")
    dtrain_predictions = alg.predict(dtrain[predictors])
    print('...Complete')

    # Print train fit model report:
    #print("\nTraining Model Report")
    print(" > RMSE (Train original vs Train predict): %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    
    # Predict on testing data:
    print('\nTesting prediction')
    dtest_predictions = alg.predict(dtest[predictors])
    print('...Complete')
    # Print model report:
    #print("\nTesting Model Report")
    print(" > RMSE (Test original vs Test predict): %.4g" % np.sqrt(metrics.mean_squared_error(dtest[target].values, dtest_predictions)),"\n")

    # Predict the future 
    print('\nFuture prediction')
    #df3 = df.reset_index().loc[:, [target, IDcol]]
    #df3 = add_month(df3, num_periods, period_type)
    finaldf = pd.concat([dtrain, dtest, dpred], ignore_index=True)
    #finaldf = pd.concat([df, pdates]).reset_index()
    end_point = len(finaldf)
    print('endpoint=',end_point)

    y = end_point-1
    print('y= ', y)

    #finaldf.tail(10)
    inputfile = finaldf.loc[y:end_point, :]
    print(len(inputfile.columns))

    inputfile_x = inputfile.loc[:, inputfile.columns != target]
    inputfile_x = inputfile_x.loc[:, inputfile_x.columns != IDcol]
    print(len(inputfile_x.columns))

    #finaldf = add_month(df3, num_periods, period_type)
    #finaldf = create_lag(df3)
    #finaldf = finaldf.reset_index(drop=True)
    yhat = pd.DataFrame()
    end_point = len(finaldf)
    df_end = len(df)
    for i in range(len(dpred), 0, -1):
        y = end_point - i
        inputfile = finaldf.loc[y:end_point, :]
        inputfile_x = inputfile.loc[:, inputfile.columns != target]
        inputfile_x = inputfile_x.loc[:, inputfile_x.columns != IDcol]
        pred_set = inputfile_x.head(1)
        pred = alg.predict(pred_set)
        df.at[df.index[df_end - i], target] = pred[0]
        finaldf = create_lag(df)
        finaldf = finaldf.reset_index(drop=True)
        yhat.append(pred)
    yhat = np.array(yhat)

    # Export prediction file:
    #prediction = pd.concat([dpred, pd.DataFrame(dpred_predictions)], axis=1)
    #prediction.to_csv(filename, index=False)
    #print('Outputs of',str(alg), '\n  saved to',filename)
    return yhat