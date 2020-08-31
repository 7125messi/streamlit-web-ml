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
from util_func import reduce_mem_usage
from util_func import eval_train_predict
from util_func import eval_visualize


train_X = pd.read_csv('train_validation_test_scaled_data/train_X_scaled.csv')
val_X = pd.read_csv('train_validation_test_scaled_data/val_X_scaled.csv')
train_y = pd.read_csv('train_validation_test_scaled_data/train_y.csv')
val_y = pd.read_csv('train_validation_test_scaled_data/val_y.csv')
test_scaled = pd.read_csv('train_validation_test_scaled_data/test_X_scaled.csv')

# reduce_mem_usage 自定义数据格式，减少内存消耗
# train_scaled = reduce_mem_usage(train_scaled)
# val_scaled = reduce_mem_usage(val_scaled)

# Convert Dataframe to Series
train_y = train_y.iloc[:,0]
val_y = val_y.iloc[:,0]

# Initialize base models
model_A = LinearRegression()
model_B = ElasticNet(random_state=1)
model_C = RandomForestRegressor(random_state=1)
model_D = GradientBoostingRegressor(random_state=1)
model_E = xgb.XGBRegressor()
model_F = LGBMRegressor(random_state=1)

samples = len(train_y) # 100% of training set
log_constant = 0

# Collect results on the learners
results = {}
for model in track([model_A, model_B, model_C, model_D, model_E, model_F]):
    model_name = model.__class__.__name__
    results[model_name] = {}
    for i, samples in enumerate([samples]):
        results[model_name][i] = eval_train_predict(model, samples, train_X, train_y, val_X, val_y, 'log', log_constant)

eval_visualize(results)


import json
# 将字典转换为JSON格式的字符串，并将转化后的结果写入文件
filename = "res_log/model_res.json"
with open(filename, 'w', encoding = 'utf-8') as fw:
    json.dump(results, fw,ensure_ascii=False)

# 从文件读取JSON格式的字符串，并将其转化为字典
# with open(filename, 'r', encoding='UTF-8') as fr:
#     res = json.load(fr)
#     print("读取JSON文件中的内容：")
#     print(res)