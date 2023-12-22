#!/usr/bin/env python
# coding: utf-8

# In[223]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as data


# In[224]:


import yfinance as yf

# Specify the stock symbol and date range
stock_symbol = 'AAPL'
start_date = '2001-01-01'
end_date = '2023-01-01'

# Fetch the data
df = yf.download(stock_symbol, start=start_date, end=end_date)






# In[225]:


df.head()
df.shape


# In[226]:


df=df.reset_index()


# In[227]:


df.head()


# In[228]:


df=df.drop(['Date','Adj Close'],axis=1)


# In[229]:


plt.plot(df.Close)


# In[230]:


ma100=df.Close.rolling(100).mean()


# In[231]:


ma100


# In[232]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')


# In[233]:


ma200=df.Close.rolling(200).mean()


# In[234]:


plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.show()


# In[235]:


data_training=pd.DataFrame(df['Close'])[0:int(len(df)*0.7)]
data_testing=pd.DataFrame(df['Close'])[int(len(df)*0.7):]


# In[236]:


data_training.shape


# In[237]:


data_testing.head()


# In[238]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[239]:


data_training_array=scaler.fit_transform(data_training)
data_training_array


# In[277]:


x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    


# In[278]:


x_train,y_train=np.array(x_train),np.array(y_train)


# In[279]:


y_train


# In[280]:


from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout


# In[281]:


model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))




# In[282]:


model.summary()


# In[283]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)


# In[285]:


import joblib as jb


# In[286]:


jb.dump(model,'model_keras_tensorflow.joblib')


# In[ ]:





# In[287]:


#model.save('keras-model.h5')


# In[288]:


data_testing.head()


# In[289]:


past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing])


# In[290]:


final_df.head()


# In[291]:


input_data=scaler.fit_transform(final_df)


# In[292]:


input_data


# In[293]:


x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[294]:


x_test=np.array(x_test)
y_test=np.array(y_test)


# In[295]:


y_test


# In[296]:


y_pred=model.predict(x_test)


# In[298]:


y_pred.max()


# In[300]:


scale=scaler.scale_
scale


# In[301]:


a=y_test*(1/scale)
b=y_pred*(1/scale)


# In[303]:


b


# In[306]:


plt.figure(figsize=(12,6))
plt.plot(a,label='Original Price')
plt.plot(b,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()



# In[264]:





# In[265]:





# In[266]:


plt.figure(figsize=(12,6))
plt.plot(y_test)
plt.plot(y_pred,'r')


# In[267]:


x_test.shape


# In[268]:


model.predict(x_test)


# In[269]:


x_test


# In[270]:


y_test


# In[274]:


len(y_pred)


# In[273]:





# In[ ]:




