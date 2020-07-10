import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np

import pandas_profiling as PP

import quandl
quandl.read_key()

import yfinance as yf

import matplotlib.pyplot as plt

import plotly

# Set directories
#rep_dir = 'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\rep_dir\'
#raw_dir = 'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\'

# Assign Stock Tickers to pull data from
stock_tickers = ['DJI','NDAQ','INX','AAPL','AXP','BA','CAT','CSCO','CVX','DD','DIS','GE','GS','HD','IBM','INTC','JNJ','JPM',\
                 'KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','UNH','UTX','V','VZ','WMT','XOM']

# Use Pandas Data Reader to load all tickers in a single go
yf.pdr_override()

# download dataframe of stock values
#data = pdr.get_data_yahoo(stock_tickers, start="2017-01-01", end="2017-04-30")
df = pdr.get_data_yahoo(stock_tickers, period="max") # , group_by='ticker'
df.shape
print('\nShape of original df {}'.format(df.shape))
df.to_csv(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\stocks1.csv')

# Make a complete copy of the stock df and combine multi-index headers for pandas profiling
df2 = df.copy(deep=True)
df2.columns = ['_'.join(col[::-1]).strip() for col in df2.columns.values]
df2.columns = df2.columns.str.replace(' ', '_')
print('Shape of copied df {}'.format(df2.shape))
df2.to_csv(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\stocks2.csv')

stocks_profile = PP.ProfileReport(df2, title='Pandas Profiling Report', explorative=True)
#stocks_profile.to_file(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\rep_data\Stocks_Profile.html")

# Europe Brent Crude
eubrent = quandl.get("FRED/DCOILBRENTEU")
brent_profile = PP.ProfileReport(eubrent, title='Pandas Profiling Report', explorative=True)
brent_profile.to_file(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\profile_reports\Brent_Report.html')
eubrent.to_csv(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\brent.csv')

# All Crude Pricing
crude = quandl.get("BP/CRUDE_OIL_PRICES")
crude_profile = PP.ProfileReport(crude, title='Pandas Profiling Report', explorative=True)
crude_profile.to_file(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\profile_reports\Crude_Report.html')
crude.to_csv(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\crude.csv')

# Federal Reserve Data
fedres = quandl.get('FRED/NROUST')
fedres_profile = PP.ProfileReport(fedres, title='Pandas Profiling Report', explorative=True)
fedres_profile.to_file(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\profile_reports\FedRes_Profile.html')
fedres.to_csv(r'c:\Users\benan\Google Drive\DataScience\EconForecasting\VSCode\raw_data\fedres.csv')
