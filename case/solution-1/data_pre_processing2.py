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

# Restore checkpoint
train = pd.read_csv("clean_data/train_prescaled.csv")
test = pd.read_csv("clean_data/test_prescaled.csv")

# Create Submission dataframe
submission = test[['Store', 'Dept', 'Date']].copy()
submission['Id'] = submission['Store'].map(str) + '_' + submission['Dept'].map(str) + '_' + submission['Date'].map(str)
submission.drop(['Store', 'Dept', 'Date'], axis=1, inplace=True)

train['Year'] = train['Date'].str.slice(start=0, stop=4)
test['Year'] = test['Date'].str.slice(start=0, stop=4)
# Drop non-numeric columns
train.drop(columns=['Date'], axis=1, inplace=True)
test.drop(columns=['Date'], axis=1, inplace=True)

# Log Transform Skewed Features
skewed = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
train[skewed] = train[skewed].apply(lambda x: np.log(x + 1))
test[skewed] = test[skewed].apply(lambda x: np.log(x + 1))

MarkDown1_min = abs(min(train['MarkDown1'].min(),test['MarkDown1'].min())) 
MarkDown2_min = abs(min(train['MarkDown2'].min(),test['MarkDown2'].min())) 
MarkDown3_min = abs(min(train['MarkDown3'].min(),test['MarkDown3'].min())) 
MarkDown4_min = abs(min(train['MarkDown4'].min(),test['MarkDown4'].min())) 
MarkDown5_min = abs(min(train['MarkDown5'].min(),test['MarkDown5'].min()))

train['MarkDown1'] = train['MarkDown1'].apply(lambda x: np.log(x + 1 + MarkDown1_min)) 
train['MarkDown2'] = train['MarkDown2'].apply(lambda x: np.log(x + 1 + MarkDown2_min)) 
train['MarkDown3'] = train['MarkDown3'].apply(lambda x: np.log(x + 1 + MarkDown3_min)) 
train['MarkDown4'] = train['MarkDown4'].apply(lambda x: np.log(x + 1 + MarkDown4_min)) 
train['MarkDown5'] = train['MarkDown5'].apply(lambda x: np.log(x + 1 + MarkDown5_min))

test['MarkDown1'] = test['MarkDown1'].apply(lambda x: np.log(x + 1 + MarkDown1_min)) 
test['MarkDown2'] = test['MarkDown2'].apply(lambda x: np.log(x + 1 + MarkDown2_min)) 
test['MarkDown3'] = test['MarkDown3'].apply(lambda x: np.log(x + 1 + MarkDown3_min)) 
test['MarkDown4'] = test['MarkDown4'].apply(lambda x: np.log(x + 1 + MarkDown4_min)) 
test['MarkDown5'] = test['MarkDown5'].apply(lambda x: np.log(x + 1 + MarkDown5_min))

log_constant = 0
train['Weekly_Sales'] = train['Weekly_Sales'].apply(lambda x: np.log(x + 1 + log_constant))
distribution(train, ['Weekly_Sales'])

# Analyze Feature Correlation
colormap = plt.cm.RdBu
corr = train.astype(float).corr()

plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.set(font_scale=0.9)
sns.heatmap(
    round(corr,2),
    linewidths=0.1,
    vmax=1.0, 
    square=True, 
    cmap=colormap, 
    linecolor='white', 
    annot=True
)
plt.show()


corr_cutoff = 0.8
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= corr_cutoff:
            if columns[j]:
                columns[j] = False
selected_columns = train.columns[columns]
highcorr_columns = train.columns.difference(selected_columns)
print("highcorr_columns:{0}".format(highcorr_columns))

# MarkDown4 and Type_A are highly correlated to other existing features and have been dropped.
train.drop(columns=highcorr_columns, axis=1, inplace=True)
test.drop(columns=highcorr_columns, axis=1, inplace=True)

#################################################################################### (4) 分割数据集，归一化
# Split Training dataset into Train & Validation
train_X, val_X, train_y, val_y = train_test_split(
    train.drop('Weekly_Sales', axis = 1), 
    train['Weekly_Sales'], 
    test_size = 0.2, 
    random_state = 0
)
# Show the results of the split
print("Training set has {} samples.".format(train_X.shape[0]))
print("Validation set has {} samples.".format(val_X.shape[0]))
# Train & Validation & Test shape
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test.shape)

# print(train_X.columns)
# Index(['Store', 'Dept', 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price',
#        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'CPI',
#        'Unemployment', 'Week', 'Type_B', 'Type_C', 'Year'],
#       dtype='object')

# Scale Datasets
# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = [
    'Store', 'Dept', 'IsHoliday', 'Size', 'Temperature', 'Fuel_Price', 
    'CPI', 'Unemployment', 'Week', 'Type_B', 'Type_C',
    'MarkDown1','MarkDown2','MarkDown3','MarkDown4'
]
train_scaled = pd.DataFrame(data = train_X)
train_scaled[numerical] = scaler.fit_transform(train_X[numerical]) # Year未标准化
print(train_scaled.head())

val_scaled = pd.DataFrame(data = val_X)
val_scaled[numerical] = scaler.transform(val_X[numerical])
print(val_scaled.head())

test_scaled = pd.DataFrame(data = test)
test_scaled[numerical] = scaler.transform(test[numerical])
print(test_scaled.head())

# Create checkpoint
train_scaled.to_csv('train_validation_test_scaled_data/train_X_scaled.csv', index=False)
val_scaled.to_csv('train_validation_test_scaled_data/val_X_scaled.csv', index=False)
train_y.to_csv('train_validation_test_scaled_data/train_y.csv', index=False, header=['Weekly_Sales'])
val_y.to_csv('train_validation_test_scaled_data/val_y.csv', index=False, header=['Weekly_Sales'])
test_scaled.to_csv('train_validation_test_scaled_data/test_X_scaled.csv', index=False)