import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("GOOG.csv")

L = len(df)
print(L)

High = np.array([df.loc[:,'High']])
Low = np.array([df.loc[:,'Low']])
Close = np.array([df.loc[:,'Close']])


plt.figure(1)
H, = plt.plot(High[0, :])
L, = plt.plot(Low[0, :])
C, = plt.plot(Close[0, :])


plt.legend([H,L,C], ["High", "Low", "Close"])
plt.show(block=False);

X = np.concatenate([High, Low],axis=0)
X = np.transpose(X)

Y = Close
# print(Y)
Y = np.transpose(Y)
# print(Y)

scaler = MinMaxScaler()
scaler.fit(X);
X = scaler.transform(X)


scaler1 = MinMaxScaler()
scaler1.fit(Y)

Y = scaler1.transform(Y)

X = np.reshape(X, (X.shape[0], 1,  X.shape[1]))

model = Sequential()
model.add(LSTM(100, activation='tanh', input_shape=(1,2), recurrent_activation='hard_sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.mae])
model.fit(X,Y,epochs=15,batch_size=1,verbose=2)

Predict = model.predict(X,verbose=1)
print(Predict)

plt.figure(2)
plt.scatter(Y, Predict)
plt.show(block=False)

plt.figure(3)
Test, = plt.plot(Y)
Predict, = plt.plot(Predict)
plt.legend([Predict, Test], ["Predicted data", "Real"])
plt.show()
