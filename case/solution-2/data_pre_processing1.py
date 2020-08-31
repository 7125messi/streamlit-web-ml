############################################################### 1 Libraries and Data Loading
import pandas as pd
pd.set_option('display.max_columns', None)  
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore") # ignoring annoying warnings

from pandasql import sqldf
def pysqldf(q):
    return sqldf(q,globals())

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

############################################################### 2 Exploratory Analysis and Data Cleaning
# features merge stores
feat_sto = features.merge(stores, how='inner', on='Store')
print(feat_sto.head())
print(feat_sto.info())

# train and test
print(train.head())
print(test.head())
print(pd.DataFrame({'Type_Train': train.dtypes, 'Type_Test': test.dtypes}))

# date数据类型转换
feat_sto.Date = pd.to_datetime(feat_sto.Date)
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)

# 特征生成  周 和 年
feat_sto['Week'] = feat_sto.Date.dt.week 
feat_sto['Year'] = feat_sto.Date.dt.year

# 生成宽表 feat_sto与train  ————> train_detail
# 生成宽表 feat_sto与test  ————> test_detail
# key by  'Store', 'Dept' and 'IsHoliday'
train_detail = train.merge(
                        feat_sto, 
                        how='inner',
                        on=['Store','Date','IsHoliday']
                        ).sort_values(
                            by=['Store',
                                'Dept',
                                'Date']
                            ).reset_index(drop=True)
test_detail = test.merge(
                        feat_sto, 
                        how='inner',
                        on=['Store','Date','IsHoliday']
                        ).sort_values(
                            by=['Store',
                            'Dept',
                            'Date']).reset_index(drop=True)
# 删除冗余变量
del features, train, stores, test

# Create checkpoint
train_detail.to_csv('clean_data/train_detail.csv', index=False)
test_detail.to_csv('clean_data/test_detail.csv', index=False)
feat_sto.to_csv('clean_data/feat_sto.csv', index=False)