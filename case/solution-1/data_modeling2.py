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
import streamlit as st

# udf function
from util_func import distribution
from util_func import reduce_mem_usage
from util_func import eval_train_predict
from util_func import eval_visualize
from util_func import train_predict
from util_func import compute_weights
from util_func import weighted_mean_absolute_error

def read_data():
    train_X = pd.read_csv('train_validation_test_scaled_data/train_X_scaled.csv')
    val_X = pd.read_csv('train_validation_test_scaled_data/val_X_scaled.csv')
    train_y = pd.read_csv('train_validation_test_scaled_data/train_y.csv')
    val_y = pd.read_csv('train_validation_test_scaled_data/val_y.csv')
    test_scaled = pd.read_csv('train_validation_test_scaled_data/test_X_scaled.csv')
    # Convert Dataframe to Series
    train_y = train_y.iloc[:,0]
    val_y = val_y.iloc[:,0]
    return train_X,val_X,train_y,val_y,test_scaled

train_X,val_X,train_y,val_y,test_scaled = read_data()

#"从WMAE来看,Random Forest和Light GBM表现效果更好"
######################################  "Evaluate Random Forest (Ensemble)"
log_constant = 0
model_rf_base = RandomForestRegressor(
                                    random_state=42, 
                                    n_estimators=150, 
                                    bootstrap=True, 
                                    max_features=None, 
                                    max_depth=None, 
                                    min_samples_leaf=1,
                                    min_samples_split=3,
                                    verbose=1
                                    )
model_rf_base, pred_y_rf_val = train_predict(
                                            model_rf_base, 
                                            train_X, 
                                            train_y, 
                                            val_X, 
                                            val_y, 
                                            'log', 
                                            log_constant, 
                                            verbose=1
                                        )



###################################### "Evaluate Light GBM (Boosting)"
model_lgbm_base = LGBMRegressor(
       boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.3, max_bin=150,
       max_depth=-1, min_child_samples=5, min_child_weight=0.001,
       min_data_in_leaf=3, min_depth=2, min_split_gain=0.0,
       n_estimators=3000, n_jobs=-1, num_leaves=80, objective='regression',
       random_state=42, reg_alpha=0.1, reg_lambda=2, silent=True,
       subsample=1.0, subsample_for_bin=200000, subsample_freq=0,
       verbose=1
)
model_lgbm_base, pred_y_lgbm_val = train_predict(
                                    model_lgbm_base, 
                                    train_X, 
                                    train_y, 
                                    val_X, 
                                    val_y, 
                                    'log', 
                                    log_constant, 
                                    verbose=1
                                    )

print(type(model_rf_base))
print(type(pred_y_rf_val))
print(type(model_lgbm_base))
print(type(pred_y_lgbm_val))

# 训练好的模型
print(model_rf_base)
print(model_lgbm_base)

# 验证集val 预测结果
# print(pred_y_rf_val)
# print(pred_y_lgbm_val)

# 测试集test 预测结果
pred_y_rf_test = model_rf_base.predict(test_scaled)
pred_y_lgbm_test = model_lgbm_base.predict(test_scaled)



###################################### Model Stacking
# 两种模型预测结果融合(val和test)
pred_y_val = ((np.exp(pred_y_rf_val) - 1 - log_constant) * 0.7) + ((np.exp(pred_y_lgbm_val) - 1 - log_constant) * 0.3)
val_y = np.exp(val_y) - 1 - log_constant

# 验证集val 预测结果评估
print("Weighted Mean Absolute Error: ", weighted_mean_absolute_error(
        pred_y_val, 
        val_y, 
        compute_weights(val_X['IsHoliday'])
    )
)

# 针对test测试集最终的预测结果
# pred_y_test = ((np.exp(pred_y_rf_test) - 1 - log_constant) * 0.7) + ((np.exp(pred_y_lgbm_test) - 1 - log_constant) * 0.3)