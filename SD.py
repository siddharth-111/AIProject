#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 12:01:42 2024

@author: sn22wex
"""

import numpy as np
import pandas as pd

df = pd.read_csv('income.csv', names=["name", "income"], skiprows=[0])

# print(df.describe())
# print(df.income.quantile(0.45))
percentile_99 = df.income.quantile(0.99)
# print(df[df.income<=percentile_99])

df.loc[3, "income"] = np.NaN


print(df.fillna(df.income.median()))

