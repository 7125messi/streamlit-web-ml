# Import libraries | Standard
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)  
import os
import datetime
import warnings
warnings.filterwarnings("ignore") # ignoring annoying warnings
from time import time
from rich.progress import track

# Import libraries | Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import libraries | Sk-learn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

import xgboost as xgb
from lightgbm import LGBMRegressor

# udf function
from util_func import distribution

def read_data(file):
    features = pd.read_csv('../raw_data/'+ file[0])
    train = pd.read_csv('../raw_data/'+ file[1])
    stores = pd.read_csv('../raw_data/'+ file[2])
    test = pd.read_csv('../raw_data/'+ file[3])
    return features,train,stores,test

filename = ["features.csv","train.csv","stores.csv","test.csv"]
stores = read_data(filename)[2]
features = read_data(filename)[0]
train = read_data(filename)[1]
test = read_data(filename)[3]

#################################################################################### 数据预处理
#################################################################################### (1) 缺失值异常值处理
#################################################################################### stores
# 异常门店信息(含索引)
print(stores[stores['Store'].isin([3,5,33,36])].index)

# index [2,4,32,35] type = 'C'
stores.iloc[2,1] = stores.iloc[4,1] = stores.iloc[32,1] = stores.iloc[35,1] = 'C'

#################################################################################### features
# Features Data | Negative values for MarkDowns
features['MarkDown1'] = features['MarkDown1'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown2'] = features['MarkDown2'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown3'] = features['MarkDown3'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown4'] = features['MarkDown4'].apply(lambda x: 0 if x < 0 else x)
features['MarkDown5'] = features['MarkDown5'].apply(lambda x: 0 if x < 0 else x)

# Features Data | NaN values for multiple columns
for i in track(range(len(features))):
    if features.iloc[i]['Date'] == '2013-04-26':
        CPI_new = features.iloc[i]['CPI']
        Unemployment_new = features.iloc[i]['Unemployment']
    if np.isnan(features.iloc[i]['CPI']):
        features.iat[i, 9] = CPI_new
        features.iat[i, 10] = Unemployment_new

# Columns: MarkDown1, MarkDown2, MarkDown3, MarkDown4 & MarkDown5
features['Week'] = 0
for i in track(range(len(features))):
    features.iat[i, 12] = datetime.date(
        int(features.iloc[i]['Date'][0:4]), 
        int(features.iloc[i]['Date'][5:7]), 
        int(features.iloc[i]['Date'][8:10])
    ).isocalendar()[1]

# missing data for 2012 & 2013
features['Year'] = features['Date'].str.slice(start=0, stop=4)
total = features[features['Year'].isin(['2012','2013'])].isnull().sum().sort_values(ascending=False)
percent = (features[features['Year'].isin(['2012','2013'])].isnull().sum()/
           features[features['Year'].isin(['2012','2013'])].isnull().count()
          ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head())

# Iterate through stores
for i in track(range(1, len(features['Store'].unique()))):
    # For 2010, iterate through weeks 5 thru 52
    for j in range(5, 52):
        idx = features.loc[(features.Year == '2010') & (features.Store == i) & (features.Week == j),['Date']].index[0]

        features.iat[idx, 4] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown1']].values[0]
        features.iat[idx, 5] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown2']].values[0]
        features.iat[idx, 6] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown3']].values[0]
        features.iat[idx, 7] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown4']].values[0]
        features.iat[idx, 8] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown5']].values[0]

    # For 2011, iterate through weeks 1 thru 44
    for j in range(1, 44):
        idx = features.loc[(features.Year == '2011') & (features.Store == i) & (features.Week == j),['Date']].index[0]

        features.iat[idx, 4] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown1']].values[0]
        features.iat[idx, 5] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown2']].values[0]
        features.iat[idx, 6] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown3']].values[0]
        features.iat[idx, 7] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown4']].values[0]
        features.iat[idx, 8] = features.loc[(features.Year == '2012') & (features.Store == i) & (features.Week == j),['MarkDown5']].values[0]  
features.drop(columns=['Year'], axis=1, inplace=True)
features.fillna(0, inplace=True)

#################################################################################### train
# Train Data | Negative Values for Weekly Sales
train['Weekly_Sales'] = train['Weekly_Sales'].apply(lambda x: 0 if x < 0 else x)


#################################################################################### (2) 合并数据集
# Merge the following datasets:
# Stores + Features + Train
# Stores + Features + Test
# Remove duplicate columns from each dataset
train = pd.merge(train, stores, how='left', on=['Store'])
train = pd.merge(train, features, how='left', on=['Store','Date'])
test = pd.merge(test, stores, how='left', on=['Store'])
test = pd.merge(test, features, how='left', on=['Store','Date'])
train.drop(columns=['IsHoliday_y'], axis=1, inplace=True)
test.drop(columns=['IsHoliday_y'], axis=1, inplace=True)
train.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
test.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)


#################################################################################### (3) 特种工程
# Column #1: IsHoliday
train['IsHoliday'] = train['IsHoliday'].apply(lambda x: 1 if x==True else 0)
test['IsHoliday'] = test['IsHoliday'].apply(lambda x: 1 if x==True else 0)

# Column #2: Type
train = pd.get_dummies(train, columns=['Type'])
test = pd.get_dummies(test, columns=['Type'])

# Column #3: Week
train['Week'] = test['Week'] = 0
# For each date, retrive the corresponding week number
for i in track(range(len(train))):
    train.iat[i, 15] = datetime.date(
        int(train.iloc[i]['Date'][0:4]), 
        int(train.iloc[i]['Date'][5:7]), 
        int(train.iloc[i]['Date'][8:10])
    ).isocalendar()[1]

# For each date, retrive the corresponding week number
for i in track(range(len(test))):
    test.iat[i, 14] = datetime.date(
        int(test.iloc[i]['Date'][0:4]), 
        int(test.iloc[i]['Date'][5:7]), 
        int(test.iloc[i]['Date'][8:10])
    ).isocalendar()[1]

# Create checkpoint
train.to_csv('clean_data/train_prescaled.csv', index=False)
test.to_csv('clean_data/test_prescaled.csv', index=False)