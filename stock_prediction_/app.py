import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from joblib import load


start_date = '2001-01-01'
end_date = '2023-01-01'
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker','AAPL')


df = yf.download(user_input, start=start_date, end=end_date)

st.subheader('Data From 2001 to 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

data_training=pd.DataFrame(df['Close'])[0:int(len(df)*0.7)]
data_testing=pd.DataFrame(df['Close'])[int(len(df)*0.7):]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

data_training_array=scaler.fit_transform(data_training)

x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)


model=load('model_keras_tensorflow.joblib')

past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing])
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test=np.array(x_test)
y_test=np.array(y_test)

y_pred=model.predict(x_test)

scale=scaler.scale_

a=y_test*(1/scale[0])
b=y_pred*(1/scale[0])


st.subheader('Prediction vs Orginal')

fig2=plt.figure(figsize=(12,6))
plt.plot(a,label='Original Price')
plt.plot(b,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
