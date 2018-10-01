#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:16:24 2018

@author: shashi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import MinMaxScaler

#read stock data 04-04-2013 to 04-04-2018
stock_data = pd.read_csv('./complete_data_set_v1/KOTAKBANK.NS.csv')
#read updated stock data 05-04-2018 to 11-05-2018
live_data = pd.read_csv('./updated/KOTAKBANK.NS.csv')

stock_price_after_30_days = pd.DataFrame()
stock_price_after_30_days['Close'] = stock_data['Close']
#stock_price_after_30_days['Open'] = stock_data['Open']

#live res for final testing on live_data of 05-04-2018-11-05-2018
live_res = pd.DataFrame()
live_res['Close'] = live_data['Close']
#live_res['Open'] = live_data['Open']

def remove_nans_Result(data,stk_data):
    index = pd.isnull(data).any(1).nonzero()[0]
    data = data.drop(data.index[index])
    stk_data = stk_data.drop(stk_data.index[index])
    
    return stk_data,data

stock_data ,stock_price_after_30_days = remove_nans_Result(stock_price_after_30_days,stock_data)
#append live data to stock data to extract features
stock_data = stock_data.append(live_data)


def extract_features(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    open_ = data['Open']
    
    data['ADX'] = ta.ADX(high, low, close, timeperiod=19)
    data['CCI'] = ta.CCI(high, low, close, timeperiod=19)  
    data['CMO'] = ta.CMO(close, timeperiod=14)
    data['MACD'], X, Y = ta.MACD(close, fastperiod=10, slowperiod=30, signalperiod=9)
    data['MFI'] = ta.MFI(high, low, close, volume, timeperiod=19)
    data['MOM'] = ta.MOM(close, timeperiod=9)
    data['ROCR'] = ta.ROCR(close, timeperiod=12) 
    data['RSI'] = ta.RSI(close, timeperiod=19)  
    data['STOCHSLOWK'], data['STOCHSLOWD'] = ta.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    data['TRIX'] = ta.TRIX(close, timeperiod=30)
    data['WILLR'] = ta.WILLR(high, low, close, timeperiod=14)
    data['OBV'] = ta.OBV(close, volume)
    data['TSF'] = ta.TSF(close, timeperiod=14)
    data['NATR'] = ta.NATR(high, low, close)#, timeperiod=14)
    data['ULTOSC'] = ta.ULTOSC(high, low, close)
    data['AROONOSC'] = ta.AROONOSC(high, low, timeperiod=14)
    data['BOP'] = ta.BOP(open_, high, low, close)
    data['LINEARREG'] = ta.LINEARREG(close)
    data['AP0'] = ta.APO(close, fastperiod=9, slowperiod=23, matype=1)
    data['TEMA'] = ta.TRIMA(close, 29)
    
    return data

def Fillna_with_mean(data):
    data['High'] = data[['High']].fillna(value=data['High'].mean())
    data['Low'] = data[['Low']].fillna(value=data['Low'].mean())
    data['Open'] = data[['Open']].fillna(value=data['Open'].mean())
    data['Close'] = data[['Close']].fillna(value=data['Close'].mean())
    data['Volume'] = data[['Volume']].fillna(value=data['Volume'].mean())
    data['Adj Close'] = data[['Adj Close']].fillna(value=data['Adj Close'].mean())
    data['ADX'] = data[['ADX']].fillna(value=data['ADX'].mean())
    data['CCI'] = data[['CCI']].fillna(value=data['CCI'].mean())
    data['CMO'] = data[['CMO']].fillna(value=data['CMO'].mean())
    data['MACD'] = data[['MACD']].fillna(value=data['MACD'].mean())
    data['MFI'] = data[['MFI']].fillna(value=data['MFI'].mean())
    data['MOM'] = data[['MOM']].fillna(value=data['MOM'].mean())
    data['ROCR'] = data[['ROCR']].fillna(value=data['ROCR'].mean())
    data['RSI'] = data[['RSI']].fillna(value=data['RSI'].mean())
    data['STOCHSLOWK'] = data[['STOCHSLOWK']].fillna(value=data['STOCHSLOWK'].mean())
    data['STOCHSLOWD'] = data[['STOCHSLOWD']].fillna(value=data['STOCHSLOWD'].mean())
    data['TRIX'] = data[['TRIX']].fillna(value=data['TRIX'].mean())
    data['WILLR'] = data[['WILLR']].fillna(value=data['WILLR'].mean())
    data['OBV'] = data[['OBV']].fillna(value=data['OBV'].mean())
    data['TSF'] = data[['TSF']].fillna(value=data['TSF'].mean())
    data['NATR'] = data[['NATR']].fillna(value=data['NATR'].mean())
    data['TRIX'] = data[['TRIX']].fillna(value=data['TRIX'].mean())
    data['WILLR'] = data[['WILLR']].fillna(value=data['WILLR'].mean())
    data['OBV'] = data[['OBV']].fillna(value=data['OBV'].mean())
    data['TSF'] = data[['TSF']].fillna(value=data['TSF'].mean())
    data['NATR'] = data[['NATR']].fillna(value=data['NATR'].mean())
    data['ULTOSC'] = data[['ULTOSC']].fillna(value=data['ULTOSC'].mean())
    data['AROONOSC'] = data[['AROONOSC']].fillna(value=data['AROONOSC'].mean())
    data['BOP'] = data[['BOP']].fillna(value=data['BOP'].mean())
    data['LINEARREG'] = data[['LINEARREG']].fillna(value=data['LINEARREG'].mean())
    data['AP0'] = data[['AP0']].fillna(value=data['AP0'].mean())
    data['TEMA'] = data[['TEMA']].fillna(value=data['TEMA'].mean())
    
    return data

stock_data = extract_features(stock_data)

stock_data = Fillna_with_mean(stock_data)
live_data = stock_data.iloc[-26:,:]

#getting rid of live_data
stock_data = stock_data.iloc[:-26,:]

#feature scaling 
scale_stock = MinMaxScaler()

stock_data = stock_data.drop(['Date'],axis=1)
live_data = live_data.drop(['Date'],axis=1)
stock_data = scale_stock.fit_transform(stock_data)
live_data = scale_stock.transform(live_data)

stock_data_ = []
stock_price_after_30_days_ = []
for i in range(30,1230):
    stock_data_.append(stock_data[i-30:i])
    stock_price_after_30_days_.append(stock_price_after_30_days.iloc[i])

stock_data, stock_price_after_30_days = np.array(stock_data_),np.array(stock_price_after_30_days_)
   
stock_data = stock_data.reshape(stock_data.shape[0],stock_data.shape[1],27)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(stock_data,stock_price_after_30_days)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers

def build_model():
    model = Sequential()
    model.add(LSTM(64,activation='relu',input_shape=(30,27)))
    #model.add(LSTM(16,activation='tanh',return_sequences=True,return_state=True))
    #model.add(Dense(16,kernel_initializer='uniform',activation='sigmoid',input_dim=27))
    model.add(Dropout(0.2))
    model.add(Dense(64,kernel_initializer='uniform',activation ='relu'))
    
    model.add(Dense(units=1,activation='relu'))
    #sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'])
    return model

Model = build_model()

Model.fit(X_train,y_train,batch_size=32,epochs=300)
prediction = Model.predict(X_test[:60])
#Y = result_scale.inverse_transform(y_train)
#prediction = result_scale.inverse_transform(prediction)
plt.plot(y_test[:60],color='green')
plt.plot(prediction,color='red')

np.concatenate([live_data],live_data[21:25])

prediction = Model.predict(live_data)
#prediction = result_scale.inverse_transform(prediction)
plt.plot(live_res.values,color='green')
plt.ylabel("KOTAK BANK")
plt.plot(prediction,color='red')
