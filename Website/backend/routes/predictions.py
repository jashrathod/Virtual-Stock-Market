# IMPORTING NECESSARY LIBRARIES

import datetime as dt
import sys
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from sklearn import linear_model


# FUNCTION TO BE CALLED FROM JAVASCRIPT CODE

def pred_function(stock_name):

    # SELECTING DURATION OF DATA
    
    end = dt.date.today()
    start = end - dt.timedelta(days=1*100)

         
    # OBTAINING DATA
     
    df = web.DataReader(stock_name, 'yahoo', start, end)
    df = df.drop(['High', 'Low', 'Open', 'Close', 'Volume'], 1)
    df = df.reset_index(drop=True)


    # GENERATING NEW FEATURES
     
    for j in range (1, 8):
        df1 = df['Adj Close'].tolist()
        n_1 = []
        for i in range(len(df1)):
            if i >= j:
                n_1.append(df1[i-j])
            else:
                n_1.append(np.mean(df1[:10]))
        df['before_' + str(j) + '_Adj_Close'] = pd.DataFrame(n_1)
        
    
    # PREPARATION OF DATA FOR NEXT DAY'S PREDICTIONS
     
    df = df.append({'before_1_Adj_Close' : df.loc[df.shape[0]-1, 'Adj Close'] , 
                    'before_2_Adj_Close' : df.loc[df.shape[0]-2, 'Adj Close'] ,
                    'before_3_Adj_Close' : df.loc[df.shape[0]-3, 'Adj Close'] ,
                    'before_4_Adj_Close' : df.loc[df.shape[0]-4, 'Adj Close'] ,
                    'before_5_Adj_Close' : df.loc[df.shape[0]-5, 'Adj Close'] ,
                    'before_6_Adj_Close' : df.loc[df.shape[0]-6, 'Adj Close'] ,
                    'before_7_Adj_Close' : df.loc[df.shape[0]-7, 'Adj Close'] } , ignore_index=True)

     
    # SPLITTING INTO TRAINING AND TESTING DATA
     
    X = df.drop('Adj Close', axis=1) 
    y = df['Adj Close']

    X_train = X[:-1]
    X_test = X[-1:]
    y_train = y[:-1]
    y_test = y[-1:]

     
    # LINEAR REGRESSION TRAINING 
     
    reg = linear_model.LinearRegression() 
    reg.fit(X_train, y_train)

     
    # MAKING PREDICTIONS

    y_pred = reg.predict(X_test)
    val = round(float(y_pred), 3)

    
    # SAVING RESULTS FOR ALL COMPANIES

    return val 

ans = pred_function(sys.argv[1])    
print(ans)
# sys.stdout.flush()    

# if __name__ == 'main':
#     return pred_function(sys.argv[1])