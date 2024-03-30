#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:46:08 2024

@author: sn22wex
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


df = pd.read_csv("income.csv", index_col=None, names=["income", "count"], skiprows=1)
# print(df)

print(cosine_similarity([[3,1]],[[6,2]]))
print(cosine_distances([[3,1]],[[6,2]]))

doc1 = """
iphone sales contributed to 70% of revenue. iphone demand is increasing by 20% yoy. 
the main competitor phone galaxy recorded 5% less growth compared to iphone"
"""

doc2 = """
The upside pressure on volumes for the iPhone 12 series, historical outperformance 
in the July-September time period heading into launch event, and further catalysts in relation
to outperformance for iPhone 13 volumes relative to lowered investor expectations implies a 
very attractive set up for the shares.
"""

doc3 = """
samsung's flagship product galaxy is able to penetrate more into asian markets compared to
iphone. galaxy is redesigned with new look that appeals young demographics. 60% of samsung revenues
are coming from galaxy phone sales
"""

doc4 = """
Samsung Electronics unveils its Galaxy S21 flagship, with modest spec improvements 
and a significantly lower price point. Galaxy S21 price is lower by ~20% (much like the iPhone 12A), 
which highlights Samsung's focus on boosting shipments and regaining market share.
"""

df = pd.DataFrame([
    {'Iphone': 3, 'galaxy': 1},
    {'Iphone': 2, 'galaxy': 0},
    {'Iphone': 1, 'galaxy': 3},
    {'Iphone': 1, 'galaxy': 2},
    ], 
    index = ["doc1", "doc2", "doc3", "doc4"]
    )
check = df.loc["doc1" : "doc1"]
similarity = cosine_similarity(df.loc["doc1" : "doc1"], df.loc["doc2" : "doc2"])
distance = cosine_distances(df.loc["doc1" : "doc1"], df.loc["doc2" : "doc2"])