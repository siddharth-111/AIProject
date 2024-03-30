#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:56:57 2024

@author: sn22wex
"""

import pandas as pd
import seaborn as sn

df = pd.read_csv("bhp.csv", names=["price_per_sqft"], skiprows=[0])

# print(df.shape)
percentile_99 = df.price_per_sqft.quantile(0.999)
# print(percentile_99)

percentile_001 = df.price_per_sqft.quantile(0.001)
# print(percentile_001)

# print(df.shape)

df = df[(df.price_per_sqft > percentile_001) & (df.price_per_sqft < percentile_99)]

# print(df.shape)

mean = df.price_per_sqft.mean();
std = df.price_per_sqft.std();


mean_low = mean - 4 * std;
mean_high = mean + 4 * std;

std_df = df[(df.price_per_sqft > mean_low) & (df.price_per_sqft < mean_high)]
print(std_df.shape)


sn.histplot(std_df.price_per_sqft, kde=True)

df['zscore'] = (df.price_per_sqft - mean) / std;
# print(df.shape)

zscore_df = df[(df.zscore < 4) & (df.zscore > -4)]
print(zscore_df.shape)

sn.histplot(zscore_df.price_per_sqft, kde=True)