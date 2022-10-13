import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

def LinRegress(df, df_fut = None, plotting = False, verbose = True):
    """
    LinearRegression performs a simple linear regression using sklearn. It outputs the fit array as well as the statistics of the fit
    df = dataframe with index for row names
    """
    # Create an instance of a linear regression model and fit it to the data with the fit() function:
    x = pd.DataFrame(np.arange(0, len(df)))
    model = LinearRegression().fit(x, df) 

    # The following section will get results by interpreting the created instance: 

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_sq = model.score(x, df)
    

    if verbose:
        # print Coefficient of determination
        print('Coefficient of Determination:', r_sq)

        # Print the Intercept:
        print('intercept(b):', model.intercept_)

        # Print the Slope:
        print('slope(m):', model.coef_) 

    # Predict a Response and print it:
    y_pred = model.predict(x)
    predict = pd.DataFrame(y_pred).set_index(df.index)

    # Make an average of the df and extrapolate
    mu = pd.DataFrame(np.ones_like(df) * np.average(df)).set_index(df.index)
    
    if df_fut.empty:
        #print('No future data')
        if plotting:
            plt.plot(df, 'b', label='Training')
            plt.plot(predict, '--r', label='Linear Regression')
            plt.plot(mu, '--g', label='Average')
        return predict, mu
    else:
        #print('processing future data')
        dffull = df.append(df_fut)
        x2 = pd.DataFrame(np.arange(0, len(dffull)))
        y2_pred = model.predict(x2)
        fpredict = pd.DataFrame(y2_pred).set_index(dffull.index)
        fmu = pd.DataFrame(np.ones_like(fpredict) * np.average(df)).set_index(fpredict.index)
        if plotting:
            plt.plot(df, 'b', label = 'Training')
            plt.plot(df_fut, 'k', label = 'Testing')
            plt.plot(fpredict, '--r', label = 'Linear Regresiion')
            plt.plot(fmu, '--g', label = 'Average')
        return predict, mu, fpredict, fmu