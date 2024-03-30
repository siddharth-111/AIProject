#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:17:41 2024

@author: sn22wex
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

df = pd.read_csv("NY.csv")

TAVG = np.array([df.iloc[:, 5]])
TMAX = np.array(df.iloc[:, 4])
TMIN = np.array(df.iloc[:, 3])

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(TMAX, TMIN, TAVG, marker='o')
ax.set_xlabel('TMIN')
ax.set_ylabel('TMAX')
ax.set_zlabel('TAVG')
plt.show()



