#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:53:28 2024

@author: sn22wex
"""

import pandas as pd
import numpy as np


def get_z_score(value, mean, std):
    return (value - mean) / std

def get_mad(s):
    median = np.median(s)
    diff = abs(s - median)
    MAD = np.median(diff)
    return MAD

def get_modified_z_score(x, median, MAD):
    return 0.6745 * (x - median) / MAD

df = pd.read_csv("movie_revenues.csv")


df['revenue_mln'] = df['revenue'].apply(lambda x: x/1000000)
# print(df.revenue_mln.describe())

_, mean, std, *_ = df.revenue_mln.describe()
df['z_score'] = df.revenue_mln.apply(lambda x : get_z_score(x, mean, std))


MAD = get_mad(df.revenue_mln)
median = np.median(df.revenue_mln)

df['modified_z_score'] = df.revenue_mln.apply(lambda x : get_modified_z_score(x, median, MAD))
print(df[df.modified_z_score >= 3.5])
