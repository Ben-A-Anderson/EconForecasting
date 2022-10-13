import pyodbc
import pandas as pd
from fredapi import Fred
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.cElementTree as et

def SQLImport(ServerURL, DatabaseNAME, username, query, driver = 'ODBC Driver 17 for SQL Server', Authentication='ActiveDirectoryInteractive'):
    """
    SQLImport performs routine SQL import using ActiveDirectoryInteractive method

    OUTPUTS:
    df = pandas dataframe of resulting data from query

    INPUTS:
    ServerURL = URL for SQL Server
    DatabaseNAME = Name of Database on server
    username = first.last@lyondellbasell.com to authenticate to server with
    query = multiple line text field of query that will return df
    """
    
    if driver in pyodbc.drivers():
        # Connect to PRD FinanceDa Database
        try:
            ServerConnection = pyodbc.connect('DRIVER='+driver+
                                    ';SERVER='+ServerURL+
                                    ';PORT=1433;DATABASE='+DatabaseNAME+
                                    ';UID='+username+
                                    ';AUTHENTICATION='+Authentication
                                    )
            print("FinanceDA PRD\t\tEstablished ")
            df = pd.read_sql(query, ServerConnection)
            print("Query Extraction\tComplete")
            print('Extracted data has size:', df.shape)
            return df

        except:
            print("Connection to:",ServerURL," - ",DatabaseNAME, "Failed")
    else:
        print("Necessary Driver (ODBC Driver 17 for SQL Server), not installed, cannot connect to database ")

def platts_cleaning(df):
    """
    platts_cleaning performs all necessary post processing for Platts data extracted form SQL which contain only date, location, and value columns
    OUTPUT:
        df_long = long format dataframe, useful for appending
        df_wide = wide format dataframe, useful for performing calculations or in ML
    INPUT:
        df = dataframe of Platts data
    """
    print('Original shape:\t', df.shape)
    # Convert the Platts data to long format so that each value is in it's own row
    df_long = pd.melt(df, id_vars=['date','LOCATION'], value_vars=['HIGHLOW2','AVERAGE','CLOSE','HIGH','LOW'])

    # Combine the Location and variable columns into a single description feature column
    df_long['description'] = df_long[['LOCATION','variable']].agg('-'.join, axis=1)
    # Drop the original columns
    df_long = df_long.drop(columns=['LOCATION','variable'])
    print('Long shape:\t', df_long.shape)

    # Pivot the data and keep only the first if multiple values exist at an intersection (safer than mean or sum to keep first)
    df_wide = pd.pivot_table(df_long, index='date', columns='description', values='value', aggfunc='first')
    print('Pivot shape:\t', df_wide.shape)

    return df_long, df_wide

def fedex_cleaning(df_long):
    """
    fedex performs all necessary post processing for Federal Exchange Rate data extracted form SQL which contain only date, EXCHANGERATE, and UNIT columns
    OUTPUT:
        df_long = long format dataframe, useful for appending
        df_wide = wide format dataframe, useful for performing calculations or in ML
    INPUT:
        df = dataframe of Federal Exchange Rate data
    """
    print('Original shape:\t', df_long.shape)
    df_long = df_long.rename(columns={'UNIT':'description','EXCHANGERATE':'value'})
    print(df_long.columns)
    # Pivot the data and keep only the first if multiple values exist at an intersection (safer than mean or sum to keep first)
    df_wide = pd.pivot_table(df_long, index='date', columns='description', values='value', aggfunc = 'first')
    print('Pivot shape:\t', df_wide.shape)
    return df_long, df_wide

def multi_fred(param_list, fred_api_key): # , start_date, end_date
    """
    multi-fred performs data import from the FRED database using a dictionary to query and rename data
    https://fred.stlouisfed.org/categories
    https://fred.stlouisfed.org/docs/api/api_key.html

    OUTPUTS: 
        df = DataFrame of all imported data
    INPUTS:
        param_list = dictionary with {'FRED Value':'Human readable name'}
        fred_api_key = API key obtained from link above
    """
    # Initialize the fred package with the provided api key
    fred = Fred(api_key = fred_api_key)
    # Initialize df for concatenation
    df = pd.DataFrame()
    # Loop through each item in param_list 
    for item in param_list:
        # Import current item as tdf
        tdf = pd.DataFrame(fred.get_series(item)).rename(columns={0:param_list[item]})
        # Merge newest item with existing items
        df = pd.concat([df, tdf], axis=1)
    return df


def icis_excel_import(file_loc, sheet, keep_cols='B:Z', header=12, footer=11):
    # Import file_location and skip defined hearder and footer rows
    df_wide = pd.read_excel(file_loc, usecols=keep_cols, sheet_name=sheet, skiprows=header, skipfooter=footer)
    df_wide = df_wide.rename(columns={'Date':'date'})
    # Melt data 
    df_long = df_wide.melt(id_vars='date').rename(columns={'variable':'description'})

    return df_wide, df_long

def icis_api_series(series,series_name, uname, passwd, constraints="", options="", structure_export=False, verbose=False):
    """" 
    icis_api_series utilizes the ICIS RESTful API to query commodity data and return it into a dataframe.
    The data structure of the API returned XML file is faily complexe and can be returned for additional use if needed (off by default)

    INPUT
        series = full URL for the series to be returned
        uname = ICIS configured username to authenticate
        passwd = ICIS configured password. This is passed in plaintext to the API
        constraints = additional constraints to be passed into the API. Should be multi-line text surrounted with tripe quotes (single or double)
        options = additonal options to pass into the API request. max-results is hardcoded to 99999 to ensure all data is returned.

    OUTPUT
        df = data frame with columns [date, low, high]
        df = data frame with columns [date, description, value]
        struct = (optional) fully parsed xml.etree.cElementTree (et) structure of the API return

    EXAMPLE
    df = icis_api_series("http://iddn.icis.com/series/petchem/6002007", 
                
                'your_user_name', 
                'you_password', 
                constraints='''
                    <compare field="c:series-order" op="ge" value="2016-01-01"/>
                    <compare field="c:series-order" op="le" value="2016-05-01"/>
                    '''
            )

    """
    #Define API URL to handle authentication and request handoff
    API_URL = 'https://api.icis.com/v1/search'

    # Assemble scope using standard and user defined components
    scope = """
    <request xmlns="http://iddn.icis.com/ns/search">
        <scope>
            <series>""" + series + """</series>
        </scope>
        <constraints>
        """ + constraints + """
        </constraints>
        <options>
            <max-results>99999</max-results>
            """ + options + """</options>
    </request>"""

    if verbose: print(scope)

    # Execute API post request
    try:
        response = requests.post(API_URL, # + icis_url_2, 
            auth=HTTPBasicAuth(uname, passwd)
            ,headers = {"Content-Type": "application/xml"}
            ,data = scope
        )
        if verbose: print(response)
    except requests.exceptions.RequestException as e:  
        raise SystemExit(e)

    # extract content from the reponse using its own encoding
    resp_data = response.content.decode(response.encoding)
    
    # Generate a tree from the decoded XML contents above
    root = et.fromstring(resp_data)

    ## Parse File
    # Iteratively extract date, low, and high data from the XML tree above

    # Pre define the temporary list which will house the extracted data
    temp_list = []

    for child in root: # Step through each entry (data point) in the tree
        if child.tag == "{http://www.w3.org/2005/Atom}entry":
            for schild in child: # Within each entry access the content (data)
                if schild.tag == '{http://www.w3.org/2005/Atom}content':
                    for tchild in schild: # Within the current Entry Conents append the date, low, and high values to the previous entryes
                        temp_list.append([tchild[8].text, tchild[9].text, tchild[10].text])
    
    # Convert the list into a DataFrame with the correct column names and assign date as the index
    df = pd.DataFrame(temp_list).rename(columns={0:'date',1:'low',2:'high'})#.set_index('date')

    df_l = pd.melt(df, id_vars='date')
    df_l['name']=series_name
    df_l['description'] = df_l['name'] +"_" + df_l['variable']
    df_l = df_l.drop(columns={'variable','name'}, axis=1)

    if structure_export:
        return df, df_l, root
    else:
        return df, df_l

def icis_api_series_dict(series_dict, uname, passwd, constraints="", options="", structure_export=False, verbose=False):
    """" 
    icis_api_series_dict leverage the icis_api_series funtion and wraps a dictionary around it to permit multiple imports with a dictionary.

    INPUT
        series_dict = Dictionary of {Series_URL:Series_Name}
        uname = ICIS configured username to authenticate
        passwd = ICIS configured password. This is passed in plaintext to the API
        constraints = additional constraints to be passed into the API. Should be multi-line text surrounted with tripe quotes (single or double)
        options = additonal options to pass into the API request. max-results is hardcoded to 99999 to ensure all data is returned.

    OUTPUT
        df = data frame with columns [date, low, high]
        df = data frame with columns [date, description, value]
        struct = (optional) fully parsed xml.etree.cElementTree (et) structure of the API return
    """
    df_l = pd.DataFrame()
    for item in series_dict:
        tdf, tdf_l = icis_api_series(item, 
                    series_dict[item],
                    'ben.anderson@lyondellbasell.com', 
                    'MyICISPassword1234', 
                    constraints,
                    options,
                    structure_export,
                    verbose=False
                )
        df = pd.concat([df_l, tdf_l], axis=0)

        return df