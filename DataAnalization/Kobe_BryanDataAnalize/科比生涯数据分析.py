#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: 科比生涯数据分析.py
@time: 2019/04/07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

# import data
filename = "data.csv"
raw = pd.read_csv(filename)
print(raw.shape)
print(raw.head(5))  # raw.head

# 5000 for test
kobe = raw[pd.notnull(raw['shot_made_flag'])]
print(kobe.shape)

# plt.subplot(211) first is raw second Column
alpha = 0.02
plt.figure(figsize=(10, 10))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(kobe.loc_x, kobe.loc_y, color='R', alpha=alpha)
plt.title('loc_x and loc_y')

# lat and lon
plt.subplot(122)
plt.scatter(kobe.lon, kobe.lat, color='B', alpha=alpha)
plt.title('lat and lon')

raw['dist'] = np.sqrt(raw['loc_x'] ** 2 + raw['loc_y'] ** 2)

loc_x_zero = raw['loc_x'] == 0
# print (loc_x_zero)
raw['angle'] = np.array([0] * len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2

raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

print(kobe.action_type.unique())
print(kobe.combined_shot_type.unique())
print(kobe.shot_type.unique())
print(kobe.shot_type.value_counts())

kobe['season'].unique()

raw['season'] = raw['season'].apply(lambda x: int(x.split('-')[1]))
raw['season'].unique()

print(kobe['team_id'].unique())
print(kobe['team_name'].unique())

pd.DataFrame({'matchup': kobe.matchup, 'opponent': kobe.opponent})

plt.figure(figsize=(5, 5))

plt.scatter(raw.dist, raw.shot_distance, color='blue')
plt.title('dist and shot_distance')

gs = kobe.groupby('shot_zone_area')
print(kobe['shot_zone_area'].value_counts())
print(len(gs))

import matplotlib.cm as cm

plt.figure(figsize=(20, 10))


def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = kobe.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


# shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')

drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)

print(raw['combined_shot_type'].value_counts())
pd.get_dummies(raw['combined_shot_type'], prefix='combined_shot_type')[0:2]

categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)

train_kobe = raw[pd.notnull(raw['shot_made_flag'])]
train_kobe = train_kobe.drop('shot_made_flag', 1)
train_label = train_kobe['shot_made_flag']
test_kobe = raw[pd.isnull(raw['shot_made_flag'])]
test_kobe = test_kobe.drop('shot_made_flag', 1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, log_loss
import time
import numpy as np

range_m = np.logspace(0, 2, num=5).astype(int)
range_m

# find the best n_estimators for RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0, 2, num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train_kobe), n_folds=10, shuffle=True):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
print(best_n, min_score)

# find best max_depth for RandomForestClassifier
print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0, 2, num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    for train_k, test_k in KFold(len(train_kobe), n_folds=10, shuffle=True):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))
print(best_m, min_score)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')

model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train_kobe, train_label)
