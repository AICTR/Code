'''
Created by Yanbin Liu on September 17 2019
Email: ybliu17@bu.edu
This file contains multiple useful tools that can be used in various context
'''

import yfinance as yf
from datetime import datetime


def DownloadData(ticker, start_date, end_date, storage_path=None):
		'''
		download price data to local
		source: yahoo! finance
		This module uses the api from yfinance
		installing this package by "pip install yfinance"
		:param ticker:  list of str tickers
		:param start_date: str '2019-01-01'
		:param end_date: str '2019-01-01'
		:param storage_path: str, storage folder, eg. '/home/'. Set None if not store.
		:return: the download data
		'''
		t = ''
		for i in ticker:
			t = t+i+' '
		data = yf.download(t, start_date, end_date)
		data.index = [datetime.strftime(i, '%Y-%m-%d') for i in data.index]
		data = data.T.swaplevel(0,1).T
		if  storage_path:
			if len(ticker)==1:
				data.to_csv(storage_path + ticker[0] + '.csv')
			else:
				for name in ticker:
					data[name].to_csv(storage_path+name+'.csv')
		return data



if __name__ == "__main__":
	DownloadData(['XLB','XLP'], '2010-01-01', '2019-08-31', 'data1/')