import streamlit as st

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

@st.cache(persist=True,allow_output_mutation=True)
def read_data(file):
    st.write('读取数据，被缓存命中~~~')
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

holidays = ['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08', #Super Bowl
           '2010-09-10', '2011-09-09', '2012-09-07', '2013-02-06',  #Labor Day
           '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29',  #Thanksgiving
           '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']  #Christmas

# st.title("XXX门店销售预测")
activity = ['项目简介','数据集探索']
choice = st.sidebar.selectbox("目录",activity)

if choice == "项目简介":
    project_intro = """
        <div style="background-color:tomato;">
            <p style="color:white;font-size:25px;">项目简介</p>
        </div>
    """
    st.markdown(project_intro,unsafe_allow_html=True)

    st.write('''
        ### 比赛描述
        建模零售数据的一个挑战是需要根据有限的历史做出决策。
        如果圣诞节一年一次，那么有机会看到战略决策如何影响到底线。
        在此招聘竞赛中，为求职者提供位于**不同地区的45家沃尔玛商店的历史销售数据**。
        每个商店都包含许多部门，参与者必须为**每个商店中的每个部门预测销售额**。
        要添加挑战，选定的**假日降价事件**将包含在数据集中。
        众所周知，这些降价会影响销售，但预测哪些部门受到影响以及影响程度具有挑战性。

        ### 比赛评估
        本次比赛的加权平均绝对误差（WMAE）评估：
        * n是行数
        * yi是真实销售额
        * wi是权重，如果该周是假日周，wi=5，否则为1
        提交文件：**Id列是通过将Store，Dept和Date与下划线连接而形成的（例如Store_Dept_2012-11-02）**

        对于测试集中的每一行（**商店+部门+日期三元组**），您应该预测该部门的每周销售额。

        ### 数据描述
        您将获得位于不同地区的45家沃尔玛商店的历史销售数据。
        每个商店都包含许多部门，您的任务是预测每个商店的部门范围内的销售额。
        此外，沃尔玛全年举办多项促销降价活动。
        这些降价活动在突出的假期之前，其中最大的四个是**超级碗，劳动节，感恩节和圣诞节**。
        包括这些**假期的周数在评估中的加权比非假日周高五倍**。
        本次比赛提出的部分挑战是**在没有完整/理想的历史数据的情况下模拟降价对这些假期周的影响**。

        * stores.csv:
        此文件包含有关45个商店的匿名信息，指示商店的类型和大小。

        * train.csv:
        这是历史销售数据，涵盖2010-02-05至2012-11-01。在此文件中，您将找到以下字段：

        > Store - 商店编号

        > Dept - 部门编号
        
        > Date - 一周
        
        > Weekly_Sales - 给定商店中给定部门的销售额(目标值)
        
        > sHoliday - 周是否是一个特殊的假日周

        * test.csv:
        此文件**与train.csv相同，但我们保留了每周销售额**。您必须**预测此文件中每个商店，部门和日期三元组的销售额**。

        * features.csv:
        此文件包含与给定日期的商店，部门和区域活动相关的其他数据。它包含以下字段：

        > Store - 商店编号

        > Date - 一周
        
        > Temperature - 该地区的平均温度
        
        > Fuel_Price - 该地区的燃料成本
        
        > MarkDown1-5 - 与沃尔玛正在运营的促销降价相关的匿名数据。MarkDown数据仅在2011年11月之后提供，并非始终适用于所有商店。任何缺失值都标有NA。
        
        > CPI - 消费者物价指数
        
        > Unemployment - 失业率
        
        > IsHoliday - 周是否是一个特殊的假日周

        为方便起见，数据集中的四个假期在接下来的几周内（并非所有假期都在数据中）：

        > 超级碗：2月12日至10日，11月2日至11日，10月2日至12日，2月8日至2月13
        
        > 日劳动节：10月9日至10日，9月9日至9日，9月9日至9月12日-13
        
        > 感恩节：26-Nov- 10,25 -Nov-11,23-Nov-12,29-Nov-13
        
        > 圣诞节：31-Dec-10,30-Dec-11,28-Dec-12,27-Dec -13
    ''')

if choice == "数据集探索":
    # st.subheader("数据集探索")
    # st.info("zhaoyadong@sfmail.sf-express.com")
    data_explore = """
        <div style="background-color:tomato;">
            <p style="color:white;font-size:25px;">数据集探索</p>
        </div>
    """
    st.markdown(data_explore,unsafe_allow_html=True)

    st.write('你选择了以下数据:`{0}, {1}, {2}, {3}`'.format(filename[0],filename[1],filename[2],filename[3]))
    # feat_sto = read_data(filename)[0].merge(read_data(filename)[2], how='inner', on='Store')
    # if st.checkbox("展示features和stores合并后的数据集feat_sto"):
    #     st.dataframe(feat_sto.head())
    # if st.checkbox("展示feat_sto数据类型"):
    #     st.dataframe(
    #         pd.DataFrame(feat_sto.dtypes, columns=['Type'])
    #     )

    st.write('''
        ### 1 stores数据集
    ''')
    if st.checkbox("stores数据集"):
        st.dataframe(stores.head())
        st.dataframe(
            pd.DataFrame(stores.dtypes, columns=['Type'])
        )

    if st.checkbox("stores数据集缺失值情况"):
        total = stores.isnull().sum().sort_values(ascending=False)
        percent = (stores.isnull().sum()/stores.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        st.dataframe(missing_data.head(20))

    if st.checkbox("stores每种类型的商店面积平均大小"):
        st.dataframe(stores['Size'].groupby(stores['Type']).mean())

    if st.checkbox("Plot[Matplotlib]"):
        fig, ax = plt.subplots(1, 2, figsize = (15,6))
        ax[0].bar(stores['Type'].unique(), stores['Size'].groupby(stores['Type']).count())
        ax[0].set_ylabel('# of Stores')
        ax[0].set_xlabel('Store Type')
        ax[0].yaxis.grid(True, linewidth=0.3)

        ax[1].scatter(stores['Type'], stores['Size'])
        ax[1].scatter(stores['Type'].unique(), stores['Size'].groupby(stores['Type']).mean()) #Store Type Average Store Size Vs 
        ax[1].set_ylabel('Store Size (Total / Average)')
        ax[1].set_xlabel('Store Type')
        ax[1].yaxis.grid(True, linewidth=0.3)
        st.pyplot()

    if st.checkbox("商店size小于40000非C类商店"):
        st.dataframe(
            stores[(stores['Size'] < 40000) & (~stores['Type'].isin(['C']))]
        )

    if st.checkbox("商店size分布图[seaborn]"):
        sns.distplot(stores['Size'])
        st.pyplot()

    st.markdown('''
        #### stores数据集探索结论
        * Column TYPE is a candidate for one-hot encoding.
        * Most stores are of TYPE='A'. Only a few stores are of TYPE='C'.
        * TYPE columns seem to be linked to Store Size. Average store size of TYPE 'A' is ~ 175k, TYPE 'B' is ~ 100k and TYPE 'C' is ~40k
        * Four stores [3, 5, 33 & 36] whose size is < 40k, seem to have been incorrectly tagged as Types A & B
    ''')

    st.write('''
        ### 2 features数据集
    ''')
    if st.checkbox("features数据集"):
        st.dataframe(features.head())
        st.dataframe(
            pd.DataFrame(features.dtypes, columns=['Type'])
        )
        st.dataframe(features.describe())

    if st.checkbox("features数据集缺失值情况"):
        total = features.isnull().sum().sort_values(ascending=False)
        percent = (features.isnull().sum()/features.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        st.dataframe(missing_data.head(20))

    if st.checkbox("所有特征列的空值分布"):
        features_missing = features.isna().sum()/len(features) * 100
        st.dataframe(features_missing)

        plt.figure(figsize=(12,10))
        plt.yticks(np.arange(len(features_missing)),features_missing.index,rotation='horizontal')
        plt.xlabel('fraction of rows with missing data')
        plt.barh(np.arange(len(features_missing)), features_missing)
        st.pyplot()

    if st.checkbox("Plot Year Vs # of Records/Unemployment/CPI"):
        fig, ax = plt.subplots(2, 2, figsize = (12,9))
        # Plot 1: Year Vs # of Records
        ax[0,0].barh(features['Date'].str.slice(start=0, stop=4).unique(), 
                features['Date'].str.slice(start=0, stop=4).value_counts())
        ax[0,0].set_xlabel('# of Records')
        ax[0,0].set_ylabel('Year')
        ax[0,0].yaxis.grid(True, linewidth=0.3)

        # Plot 2: Month Vs # of Records with Missing Values - Unemployment
        ax[1,0].barh(features['Date'].str.slice(start=0, stop=7)[features['Unemployment'].isna()].unique(), 
                features['Date'].str.slice(start=0, stop=7)[features['Unemployment'].isna()].value_counts())
        ax[1,0].set_xlabel('# of Records with Missing Values - Unemployment')
        ax[1,0].set_ylabel('Month')
        ax[1,0].yaxis.grid(True, linewidth=0.3)

        # Plot 3: Month Vs # of Records with Missing Values - CPI
        ax[1,1].barh(features['Date'].str.slice(start=0, stop=7)[features['CPI'].isna()].unique(), 
                features['Date'].str.slice(start=0, stop=7)[features['CPI'].isna()].value_counts())
        ax[1,1].set_xlabel('# of Records with Missing Values - CPI')
        ax[1,1].set_ylabel('Month')
        ax[1,1].yaxis.grid(True, linewidth=0.3)

        st.pyplot()

    if st.checkbox("features假期特征"):
        st.write("有效假期")
        st.dataframe(features['IsHoliday'][features['Date'].isin(holidays)].value_counts())
        st.dataframe(features['Date'][features['IsHoliday'].isin([1])][~features['Date'].isin(holidays)].value_counts())

    if st.checkbox("按照store和date分组"):
        store_date_groupby = features[['CPI','Unemployment']].groupby([features['Store'], features['Date'].str.slice(start=0, stop=7)]).mean()
        st.table(store_date_groupby.head(84))

    if st.checkbox("按照date分组"):
        date_count = features.groupby(features['Date'].str.slice(start=0, stop=7))['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'].count()
        st.table(date_count)

    if st.checkbox("features数据集的数值型特征分布"):
        distribution(features, ['CPI','Unemployment']);st.pyplot()
        distribution(features, ['Temperature','Fuel_Price']);st.pyplot()
        distribution(features, ['MarkDown1','MarkDown2']);st.pyplot()
        distribution(features, ['MarkDown3','MarkDown4']);st.pyplot()
        distribution(features, ['MarkDown5']);st.pyplot()

    st.markdown('''
        #### features数据集探索结论
        * Data requires pre-processing
        * Column(s) ISHOLIDAY has been validated
        * Column(s) UNEMPLOYMENT & CPI have missing values for May, Jun & Jul 2013. 
          For these columns as the values dont change significantly month on month, 
          value from Apr 2013 would be propogated over for each store.
        * Column(s) MARKDOWN* have missing values for 2010 (entire year) and 2011 (until Nov). 
          Additionally, there are missing values for other other dates as well.
        * CPI and UNEMPLOYMENT value are a bit skewed. MARKDOWN* columns are skewed.
    ''')

    st.write('''
        ### 3 train数据集
    ''')
    if st.checkbox("train数据集"):
        st.dataframe(train.head())
        st.dataframe(
            pd.DataFrame(train.dtypes, columns=['Type'])
        )
        st.dataframe(train.describe())

    if st.checkbox("探索date年份范围以及对于年份的数据量"):
        st.dataframe(train['Date'].str.slice(start=0, stop=4).value_counts()) # slice(start,stop) 抽取字段片段，这里抽取年份

    if st.checkbox("train假期特征"):
        st.write("有效假期")
        st.dataframe(train['IsHoliday'][train['Date'].isin(holidays)].value_counts())
        st.dataframe(train['Date'][train['IsHoliday'].isin([1])][~train['Date'].isin(holidays)].value_counts())

    if st.checkbox("train数据集的数值型特征分布"):
        distribution(train, ['Weekly_Sales']);st.pyplot()

    if st.checkbox("商店销售额为负的(相对于目标值)的商店数"):
        st.write(train['Store'][train['Weekly_Sales'] < 0].count())

    train_outliers = pd.merge(train, stores, how='left', on=['Store'])
    if st.checkbox("每种商店类型的平均周销"):
        st.dataframe(train_outliers.groupby(['Type'])['Weekly_Sales'].mean())
    
    train_outliers = train_outliers[train_outliers['Store'].isin([3,5,33,36])]
    if st.checkbox("可能是误分类的商店类型的平均周销"):
        st.dataframe(train_outliers.groupby(['Store','Type'])['Weekly_Sales'].mean())

    if st.checkbox("商店类型的平均周销可视化"):
        fig, ax = plt.subplots(1, 2, figsize = (15,6))
        ax[0].bar(train_outliers['Type'].unique(), train_outliers.groupby(['Type'])['Weekly_Sales'].mean())
        ax[0].set_ylabel('Average Weekly Sales')
        ax[0].set_xlabel('Store Type')
        ax[0].yaxis.grid(True, linewidth=0.3)

        ax[1].bar([3,5,33,36], train_outliers.groupby(['Store','Type'])['Weekly_Sales'].mean())
        ax[1].set_ylabel('Average Weekly Sales')
        ax[1].set_xlabel('Store ID')
        ax[1].yaxis.grid(True, linewidth=0.3)

        st.pyplot()
    
    # Free up memory
    train_outliers = None

    st.markdown('''
        #### train数据集探索结论
        * Column DATE is non-numeric and is a candidate for pre-processing.
        * 1285 records with Weekly Sales < 0
        * Data spans years 2010, 2011 and 2012
        * As suspected above, four stores [3, 5, 33 & 36] seem to have incorrectly classified as Type A & B.
          Average Weekly Sales for these stores is in line with the average for Type C. 
          Hence, these would need to be reclassified as Type C.
    ''') 

    st.write('''
        ### 4 test数据集
    ''')
    if st.checkbox("test数据集"):
        st.dataframe(test.head())
        st.dataframe(
            pd.DataFrame(test.dtypes, columns=['Type'])
        )
        st.dataframe(test.describe())

    if st.checkbox("探索date年份和年份对于的数据量"):
        st.dataframe(test['Date'].str.slice(start=0, stop=4).value_counts()) # slice(start,stop) 抽取字段片段，这里抽取年份

    if st.checkbox("test假期特征"):
        st.write("有效假期")
        st.dataframe(test['IsHoliday'][test['Date'].isin(holidays)].value_counts())
        st.dataframe(test['Date'][test['IsHoliday'].isin([1])][~test['Date'].isin(holidays)].value_counts())

    st.markdown('''
        #### test数据集探索结论
        * Column DATE is non-numeric and is a candidate for pre-processing.
        * Data spans years 2012 and 2013
    ''')