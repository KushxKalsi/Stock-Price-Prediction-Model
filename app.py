import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model("StockPricePredictionModel.keras")

st.header('Stock Market Predictior')

stock = st.text_input('Enter Stock Symnbol','GOOG')
start=st.text_input('Enter Start Date','2012-01-01')
end=st.text_input('Enter End Date','2024-08-01')

data= yf.download(stock,start,end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data. Close [0: int(len(data) *0.80)])
data_test = pd.DataFrame(data. Close [int (len(data) *0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

st.subheader('Price vs Mooving Array 50')
ma_50_days=data.Close.rolling(50).mean()
fig1=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r',label="Moving Array 50 Days")
plt.plot(data.Close,'g',label="Price")
plt.show()
st.pyplot(fig1)

st.subheader('Price vs Mooving Array 50 vs Mooving Array 100')
ma_100_days=data.Close.rolling(100).mean()
fig2=plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r',label="Moving Array 50 Days")
plt.plot(ma_100_days,'b', label="Moving Array 100 Days")
plt.plot(data.Close,'g')
plt.legend()
plt.show()
st.pyplot(fig2)

x=[]
y=[]


for i in range(100,data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])

x,y=np.array(x), np.array(y)

predict= model.predict(x)

scale=1/scaler.scale_

predict=predict*scale
y=y*scale

st.subheader('Orignal Price vs Pridiced Price')
fig4=plt.figure(figsize=(8,6))
plt.plot(predict,'r',label= 'Predicted Price')
plt.plot(y,'g',label='Orignal Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
