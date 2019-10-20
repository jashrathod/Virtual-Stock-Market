# ---------------------------------------------------
# IMPORTING NECESSARY LIBRARIES
# ---------------------------------------------------

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from sklearn import linear_model

# ---------------------------------------------------


# ---------------------------------------------------
# SELECTING DURATION OF DATA
# ---------------------------------------------------

end = dt.date.today()
start = end - dt.timedelta(days=1*100)

# ---------------------------------------------------


# ---------------------------------------------------
# GETTING THE LIST OF STOCKS
# ---------------------------------------------------

stocks_list = pd.read_csv('stocks_list.csv')
# print(stocks_list)

# ---------------------------------------------------


# ---------------------------------------------------
# RETRIEVAL AND SAVING OF DATA
# ---------------------------------------------------

for index, row in stocks_list.iterrows():
    abbr = row['abbreviation']
    df = web.DataReader(abbr, 'yahoo', start, end)
    df = df.drop(['High', 'Low', 'Open', 'Close', 'Volume'], 1)
    df = df.reset_index(drop=True)
    df.to_csv('data/stock_data_' + abbr + '.csv')

# ---------------------------------------------------


# ---------------------------------------------------
# PREPARATION FOR FEATURE ENGINEERING
# ---------------------------------------------------

stocks_abbr = list(stocks_list['abbreviation'])
df_results = pd.DataFrame()

# ---------------------------------------------------


# ---------------------------------------------------
# FEATURE ENGINEERING, TRAINING, AND PREDICTIONS
# ---------------------------------------------------

for z in stocks_abbr:

    df = pd.read_csv('data/stock_data_' + z + '.csv')

    # ---------------------------------------------------
    # GENERATING NEW FEATURES
    # ---------------------------------------------------

    for j in range (1, 8):
        df1 = df['Adj Close'].tolist()
        n_1 = []
        for i in range(len(df1)):
            if i >= j:
                n_1.append(df1[i-j])
            else:
                n_1.append(np.mean(df1[:10]))
        df['before_' + str(j) + '_Adj_Close'] = pd.DataFrame(n_1)
    
    # ---------------------------------------------------


    # ---------------------------------------------------
    # PREPARATION OF DATA FOR NEXT DAY'S PREDICTIONS
    # ---------------------------------------------------

    df = df.append({'before_1_Adj_Close' : df.loc[df.shape[0]-1, 'Adj Close'] , 
                    'before_2_Adj_Close' : df.loc[df.shape[0]-2, 'Adj Close'] ,
                    'before_3_Adj_Close' : df.loc[df.shape[0]-3, 'Adj Close'] ,
                    'before_4_Adj_Close' : df.loc[df.shape[0]-4, 'Adj Close'] ,
                    'before_5_Adj_Close' : df.loc[df.shape[0]-5, 'Adj Close'] ,
                    'before_6_Adj_Close' : df.loc[df.shape[0]-6, 'Adj Close'] ,
                    'before_7_Adj_Close' : df.loc[df.shape[0]-7, 'Adj Close'] } , ignore_index=True)
    
    df = df.drop('Unnamed: 0', 1)

    # ---------------------------------------------------


    # ---------------------------------------------------
    # SPLITTING INTO TRAINING AND TESTING DATA
    # ---------------------------------------------------

    X = df.drop('Adj Close', axis=1) 
    y = df['Adj Close']

    X_train = X[:-1]
    X_test = X[-1:]
    y_train = y[:-1]
    y_test = y[-1:]

    # ---------------------------------------------------


    # ---------------------------------------------------
    # LINEAR REGRESSION TRAINING 
    # ---------------------------------------------------

    reg = linear_model.LinearRegression() 
    reg.fit(X_train, y_train)

    # ---------------------------------------------------


    # ---------------------------------------------------
    # MAKING PREDICTIONS AND TABULATING RESULTS
    # ---------------------------------------------------

    y_pred = reg.predict(X_test)
    val = round(float(y_pred), 3)
    df_results = df_results.append({'Stock': z, 'Prediction': val} , ignore_index=True)
    
    # ---------------------------------------------------

# ---------------------------------------------------


# ---------------------------------------------------
# SAVING RESULTS FOR ALL COMPANIES
# ---------------------------------------------------

df_results = df_results.join(stocks_list['company'])
df_results.to_csv('stocks_results.csv')

# ---------------------------------------------------