# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:25 2019

@author: liaot
"""
import pandas as pd
import numpy as np
import json


def trading_representative(component, window_size, alpha, t):

    ## find the intersection of the S&P500 stocks during the past year
    for k in range(1):
        stocks = list(set(comp_df['Symbols'][k]).intersection(set(comp_df['Symbols'][k+1])))
        
    # use local data
    # Method: use dictionary to store all the adj close information and then transform it into a dataframe
    adj_close_dict = {}
    # open local files and extract the corresponding adj close prices
    for stock in stocks:
        filename = './Code-master(new)/data1/' + stock + '.csv'
        file = pd.read_csv(filename)
        file.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)          
        # extract the adj close prices
        close = file['Adj Close']
        close.index = file['Date']
        adj_close_dict[stock] = close

    adj_close = pd.DataFrame.from_dict(data = adj_close_dict)
    adj_close = adj_close.dropna(axis = 0, how = 'all')
    adj_close = adj_close.dropna(axis = 1, how = 'any')
    # compute the price relative
    pr = adj_close / adj_close.shift(1)    
    pr = pr.fillna(value = 1)
    
    # slice the adj close and pr dataframes according to the start_date and end_date
    end_date = str(comp_df['Date'][t])[:10]
    start_date = str(comp_df['Date'][t-1])[:10]
    sliced_close = adj_close.loc[start_date:end_date, :]
    [T, nstock] = sliced_close.shape
    sliced_pr = pr.loc[start_date:end_date, :]
#    [T1, nstock1] = sliced_pr.shape
    
    # test
#    data_col = sliced_data.columns
#    close_col = sliced_close.columns
#    for s in close_col:
#        if s not in data_col:
#            print(s)
    
    # compute the three trading representative series
    # initialize the ema series
    ema_array = np.ones(shape=(T, nstock))
    # update the ema series for each intra-day
    for i in range(1,T):
        ema_array[i] = (1 - alpha) * ema_array[i-1] / sliced_pr.iloc[i,] + alpha 
    ema = pd.DataFrame(data = ema_array, columns = sliced_pr.columns, index = sliced_pr.index)


    # update the pp & sma series for each intra-day
    pp = sliced_close.rolling(5).max() / sliced_close
    sma = sliced_pr.rolling(5).sum() / sliced_pr  
    sma = sma / win_size

    # put the three trading representatives dataframe into a larger dataframe
    #tr_dict = {'ema': ema, 'sma': sma, 'pp': pp}
    #tr = pd.DataFrame.from_dict(data = tr_dict)
    return ema, sma, pp

if __name__ == '__main__':
    # global variables
    win_size = 5    # time window
    alpha = 0.5     # parameter in computing EMA

    with open('./sp500-historical-components.json','r') as f:
        total_sp500 = json.loads(f.read())   # total_sp500 is a list

    # clean the data
    comp_df = pd.DataFrame(data = total_sp500)
    comp_df = comp_df.sort_values(by = 'Date', ascending = True)  # sort the dataframe by date
    comp_df = comp_df.reset_index(drop = True)
    comp_df['Date'] = pd.to_datetime(comp_df['Date'], format = '%Y/%m/%d')
    total_mon = len(comp_df['Date'])
    

    # iterate through time
    for t in range(1,total_mon):
        [EMA, SMA, PP] = trading_representative(comp_df, win_size, alpha, t)
        # test the correlation between the trading representatives
        stks = EMA.columns
        e_s = []
        s_p = []
        e_p = []
        for s in stks:
            ema_sma = EMA[s].corr(SMA[s])
            sma_pp = SMA[s].corr(PP[s])
            ema_pp = EMA[s].corr(PP[s])
            e_s.append(ema_sma)
            s_p.append(sma_pp)
            e_p.append(ema_pp)
        corr = pd.DataFrame(data = [e_s, s_p, e_p], columns = stks, index = ['EMA_SMA','SMA_PP','EMA_PP'])
    
    