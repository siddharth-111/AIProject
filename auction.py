#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:24:26 2024

@author: sn22wex
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("iplauction2023.csv")
# print(df.loc[df.duplicated(subset=['name'])])

missing_values_count = df.isnull().sum()

# print(missing_values_count[0:7])

# print(df.describe())


# sns.countplot(x="franchise", hue="status", data=df)
# sns.countplot(x="status", hue="franchise", data=df)

percentage_count = df['status'].value_counts(normalize=True) * 100

# plt.figure(1)
# plt.pie(percentage_count, labels=percentage_count.index, autopct = '%1.1f%%', startangle=90, colors=['gold',  'lightblue', 'lightcoral'])

new_df = df.fillna({'franchise' : 'no franchise'})

# sns.countplot(x="franchise", hue="status", data=df)
# sns.countplot(x="franchise", hue="status", data=new_df)

g = df.groupby('status')

data = df.groupby('status')[['base price (in lacs)','final price (in lacs)']]



final_df = new_df.fillna({
        'base price (in lacs': 0,
        'final price (in lacs': 0})

missing_values_count = final_df.isnull().sum()
# print(missing_values_count[0:7])

# sns.countplot(x="nationality", data=final_df)
# sns.boxplot(x="nationality", y = "final price (in lacs)", data=final_df)

mydata = pd.crosstab(final_df["status"], final_df["player style"], normalize=True)
sns.countplot(x ='player style', data = final_df)
# sns.boxplot(x="player style", y = "final price (in lacs)", data=final_df)
# sns.countplot(x="franchise", data = final_df)

