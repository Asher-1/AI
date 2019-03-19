#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: mmy_axis.py
@time: 2019/03/18
"""

import pandas as pd
import numpy as np

dates = pd.date_range("20190318", periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=["A", "B", "C", "D"])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
print(df)

print(np.any(pd.isnull(df)))

print(df.drop("C", axis=1))

# print(df.dropna(axis=1))
