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

data_explore = """
    <div style="background-color:tomato;">
        <p style="color:white;font-size:25px;">变量分析</p>
    </div>
"""
st.markdown(data_explore,unsafe_allow_html=True)

@st.cache(persist=True,allow_output_mutation=True)
def read_data():
    train_detail = pd.read_csv('clean_data/train_detail.csv')
    test_detail = pd.read_csv('clean_data/test_detail.csv')
    return train_detail, test_detail

train_detail, test_detail = read_data()
st.dataframe(train_detail.head())

###############################################################

st.write('''
    * Weekly_Sales x IsHoliday
''')
make_discrete_plot('IsHoliday',train_detail)
st.pyplot()
st.write("该字段对于区分周休假很重要。 我们可以看到，与非假日周相比，周假日的销售活动更多。")

st.write('''
    * Weekly_Sales x Type
''')
make_discrete_plot('Type',train_detail)
st.pyplot()
st.write("我们不知道“类型”是什么，但是我们可以假设以Sales Median表示A> B>C。 因此，让我们将其视为序数变量并替换其值。下图说明了有序变量。")

train_detail.Type = train_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
test_detail.Type = test_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))

from PIL import Image
image = Image.open('img/data.png')
st.image(image, caption='数据分类图',use_column_width=True)

st.write('''
    * Weekly_Sales x Temperature
''')
make_continuous_plot('Temperature',train_detail)
st.pyplot()
st.write("尽管偏度发生了变化，但相关性似乎根本没有变化。 我们可以决定删除它。")
train_detail = train_detail.drop(columns=['Temperature'])
test_detail = test_detail.drop(columns=['Temperature'])

st.write('''
    * Weekly_Sales x CPI
''')
make_continuous_plot('CPI',train_detail)
st.pyplot()
train_detail = train_detail.drop(columns=['CPI'])
test_detail = test_detail.drop(columns=['CPI'])

st.write('''
    * Weekly_Sales x Unemployment
''')
make_continuous_plot('Unemployment',train_detail)
st.pyplot()
train_detail = train_detail.drop(columns=['Unemployment'])
test_detail = test_detail.drop(columns=['Unemployment'])

st.write('''
    * Weekly_Sales x Size
''')
make_continuous_plot('Size',train_detail)
st.pyplot()
st.write("最后，我们将继续使用此变量，因为它与“ WeeklySales”具有适度的相关性。")

# 准备训练集
X_train = train_detail[['Store','Dept','IsHoliday','Size','Week','Type','Year']]
Y_train = train_detail['Weekly_Sales']
X_test = test_detail[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Type', 'Year']]

X_train.to_csv('train_test/X_train.csv', index=False)
Y_train.to_csv('train_test/Y_train.csv', index=False)
X_test.to_csv('train_test/X_test.csv', index=False)