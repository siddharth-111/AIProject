#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:52:41 2024

@author: sn22wex
"""

import pandas as pd
import seaborn as sn

df = pd.read_csv("heights.csv")

# print(df.height.describe)

sn.histplot(df.height, kde=False)

mean = df.height.mean()
std_deviation = df.height.std()

# print(mean, std_deviation)

mean_low = mean - 3 * std_deviation;
mean_high = mean + 3 * std_deviation;

res = df[(df.height < mean_low) | (df.height > mean_high)]
# print(res)

res = df[(df.height > mean_low) & (df.height < mean_high)]
print(res.shape)
# res_high = df[df.height > mean_high]
# print(res_high)

df['zscore'] = (df.height - df.height.mean()) / df.height.std()
# print(df[df.zscore > 3])

res2 = df[(df.zscore < 3) & (df.zscore > -3)] 
print(res2.)
