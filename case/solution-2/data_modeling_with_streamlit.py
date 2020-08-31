import streamlit as st
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

from util_func import make_discrete_plot
from util_func import make_continuous_plot
from util_func import random_forest

data_explore = """
    <div style="background-color:tomato;">
        <p style="color:white;font-size:25px;">机器学习</p>
    </div>
"""
st.markdown(data_explore,unsafe_allow_html=True)

@st.cache(persist=True,allow_output_mutation=True)
def read_data():
    X_train = pd.read_csv('train_test/X_train.csv')
    Y_train = pd.read_csv('train_test/Y_train.csv')
    X_test = pd.read_csv('train_test/X_test.csv')
    return X_train, Y_train, X_test

X_train, Y_train, X_test = read_data()
st.dataframe(X_train.head())

from PIL import Image
image = Image.open('img/WMAE.png')
st.image(image, caption='Model Functions',use_column_width=True)

############################################################### Training Model
# Tuning 'n_estimators' and 'max_depth'
n_estimators = [56, 58, 60]
max_depth = [25, 27, 30]
df1 = random_forest(n_estimators, max_depth, X_train, Y_train)
st.dataframe(df1)

# Tuning 'max_features'
# max_features = [2, 3, 4, 5, 6, 7]
# df2 = random_forest_II(n_estimators=58, max_depth=27, max_features=max_features, X_train, Y_train)
# st.dataframe(df2)

# Tuning 'min_samples_split' and 'min_samples_leaf'
# min_samples_split = [2, 3, 4]
# min_samples_leaf = [1, 2, 3]
# df3 = random_forest_III(n_estimators=58, max_depth=27, max_features=6, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,X_train, Y_train)
# st.dataframe(df3)

# 训练最优模型并持久化模型
# RF = RandomForestRegressor(n_estimators=58, max_depth=27, max_features=6, min_samples_split=3, min_samples_leaf=1)
# RF = RF.fit(X_train, Y_train)
# from sklearn.externals import joblib
# joblib.dump(RF, 'save/RF.pkl')

# 预测结果
# predict = RF.predict(X_test)