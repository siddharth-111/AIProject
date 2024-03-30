#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:12:29 2024

@author: sn22wex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("smartphones.csv")

percentage_count = df['brand_name'].value_counts()
# print(percentage_count.index)


# plt.figure(1)
# percentage_count.plot(kind='bar')
# plt.ylabel("count")


# average_prices = df.groupby('brand_name')['5G_or_not']
# average_prices = average_prices[average_prices < 200000]
# average_prices.plot(kind='bar')

is5g = df['5G_or_not'].value_counts()


# plt.figure(figsize=(6, 4))
# is5g.plot(kind='bar')
# plt.xlabel('Value')
# plt.ylabel('Count')
# plt.title('Distribution of 0s and 1s')
# plt.xticks(ticks=[0, 1], labels=['no', 'yes'], rotation=0)  # Ensures the x-ticks are only 0 and 1
# plt.show(block=False)


brands = df['processor_brand'].value_counts()

plt.figure(figsize=(6, 4))
brands.plot(kind='bar')
plt.xlabel('Brands')
plt.ylabel('Count')
plt.title('Distribution by phone brands')
plt.show(block=False)

chip_with_fast_processing = df[(df['processor_brand'] == 'bionic') & (df['processor_speed'] > 3)]