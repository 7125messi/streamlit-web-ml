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

from pandasql import sqldf


data_explore = """
    <div style="background-color:tomato;">
        <p style="color:white;font-size:25px;">数据预处理</p>
    </div>
"""
st.markdown(data_explore,unsafe_allow_html=True)


def pysqldf(q):
    return sqldf(q,globals())

@st.cache(persist=True,allow_output_mutation=True)
def read_data():
    train_detail = pd.read_csv('clean_data/train_detail.csv')
    test_detail = pd.read_csv('clean_data/test_detail.csv')
    feat_sto = pd.read_csv('clean_data/feat_sto.csv')
    return train_detail, test_detail, feat_sto

train_detail, test_detail, feat_sto = read_data()
st.dataframe(train_detail.head())

###############################################################

st.write('''
    * 缺失值:
    * 有些列的空值超过60％,如果与目标WeeklySales的相关性为0，则使用它们不是一个好主意。
    * 此外，它们是匿名字段，可能很难知道它们的含义
''')

# print((train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False))
null_columns = (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False).index
null_data = pd.concat([
    train_detail.isnull().sum(axis = 0),
    (train_detail.isnull().sum(axis = 0)/len(train_detail)).sort_values(ascending=False),
    train_detail.loc[:, train_detail.columns.isin(list(null_columns))].dtypes], axis=1)
null_data = null_data.rename(columns={0: '# null', 
                                      1: '% null', 
                                      2: 'type'}).sort_values(ascending=False, by = '% null')
null_data = null_data[null_data["# null"]!=0]
print(null_columns)
print(null_data)
st.table(null_data)

############################################################### Holidays Analysis
st.write('''
    * 在这里，我们将分析每年假期的工作日。 这与知道每个星期有多少节假日相关，这在“ IsHoliday”字段中标记为“ True”。
    * 如果某年某个星期内的假期前天比另一年多，则很有可能假期前天更多的那一年在同一周内会有更大的销售额。 
    * 因此，模型将不会考虑这一点，并且我们可能需要在最后调整预测值。
    * 要考虑的另一件事是，假日周但节假日前几天很少或没有假日可能比前一周的销售额低。
    * 我们可以使用SQL，将每年每个假期的工作日数。 做一些研究，超级碗，劳动节和感恩节是在同一天。 另一方面，圣诞节始终是12月25日，因此工作日可以更改。
''')

df = pysqldf("""
    SELECT
        T.*,
        case
            when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Super Bowl'
            when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Labor Day'
            when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thanksgiving'
            when ROW_NUMBER() OVER(partition by Year order by week) = 4 then 'Christmas'
        end as Holyday,
        case
            when ROW_NUMBER() OVER(partition by Year order by week) = 1 then 'Sunday'
            when ROW_NUMBER() OVER(partition by Year order by week) = 2 then 'Monday'
            when ROW_NUMBER() OVER(partition by Year order by week) = 3 then 'Thursday'
            when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2010 then 'Saturday'
            when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2011 then 'Sunday'
            when ROW_NUMBER() OVER(partition by Year order by week) = 4 and Year = 2012 then 'Tuesday'
        end as Day
    from(
        SELECT 
            DISTINCT
            Year,
            Week,
            case when Date <= '2012-11-01' then 'Train Data' else 'Test Data' end as Data_type
        FROM feat_sto
        WHERE IsHoliday = True
    ) as T
""")
print(df)
st.table(df)

st.write('''
    * (1)所有假期都在同一周
    * (2)测试数据没有劳动节，所以这个假期不是很重要
    * (3)圣诞节在2010年节假日前有0天，在2011年是1天，在2012年是3天。该模型不会考虑2012年测试数据的更多销售，因此我们将在最后进行调整，并提供公式和解释。
    * 让我们看一下每年的平均每周销售量，找出是否有另一个假期高峰销售被“假期”字段所忽略。
''')
weekly_sales_2010 = train_detail[train_detail.Year==2010]['Weekly_Sales'].groupby(train_detail['Week']).mean()
weekly_sales_2011 = train_detail[train_detail.Year==2011]['Weekly_Sales'].groupby(train_detail['Week']).mean()
weekly_sales_2012 = train_detail[train_detail.Year==2012]['Weekly_Sales'].groupby(train_detail['Week']).mean()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1, 53, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Weekly Sales - Per Year', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Week', fontsize=16)
plt.show()
st.pyplot()


st.write('''
    如我们所见，“ IsHoliday”中没有包含一个重要的假期。 今天是复活节。 它总是在星期日，但可以在不同的星期。

    * 2010年是第13周
    * 2011年，第16周
    * 2012年第14周
    * 最后是2013年第13周的测试集
    
    因此，我们可以在每年的这些周更改为“ True”。
''')


# 找出是否有另一个假期高峰销售被“假期”字段所忽略
train_detail.loc[(train_detail.Year==2010) & (train_detail.Week==13), 'IsHoliday'] = True
train_detail.loc[(train_detail.Year==2011) & (train_detail.Week==16), 'IsHoliday'] = True
train_detail.loc[(train_detail.Year==2012) & (train_detail.Week==14), 'IsHoliday'] = True
test_detail.loc[(test_detail.Year==2013) & (test_detail.Week==13), 'IsHoliday'] = True

# 销售的中位数，未除以年份
weekly_sales_mean = train_detail['Weekly_Sales'].groupby(train_detail['Date']).mean()
weekly_sales_median = train_detail['Weekly_Sales'].groupby(train_detail['Date']).median()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)
sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Weekly Sales - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()
st.pyplot()

st.write("正如观察结果所示，均值和中位数非常不同，这表明某些商店/部门的销售额可能比其他商店/部门更大。")

############################################################### Average Sales per Store and Department
weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Store']).mean()
plt.figure(figsize=(20,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()
st.pyplot()

weekly_sales = train_detail['Weekly_Sales'].groupby(train_detail['Dept']).mean()
plt.figure(figsize=(25,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales - per Dept', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Dept', fontsize=16)
plt.show()
st.pyplot()

st.write("各部门之间也存在销售差异。 另外，某些部门不在列表中，例如数字“ 15”。")

############################################################### Variables Correlation
st.write('''
    Correlation Metrics:using Pearson Correlation
    * 0: no correlation at all
    * 0-0.3: weak correlation
    * 0.3-0.7: moderate correlaton
    * 0.7-1: strong correlation
''')

sns.set(style="white")
corr = train_detail.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title('Correlation Matrix', fontsize=18)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()
st.pyplot()

st.write('''
    * 'MarkDown'1到5与'Weekly_Sales'的相关性不强，并且它们有很多空值，然后我们可以将其删除。
    * 同样，“Fuel_Price”与“Year”密切相关。 必须删除其中之一，否则它们将携带与模型类似的信息。 不能删除“Year”，因为它区分“stores” +“dept”的同一周。
    * 可以分析与“ Weekly_Sales”相关性较弱的其他变量，以查看它们是否有用。
''')
train_detail = train_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
test_detail = test_detail.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
st.table(train_detail.head())

# Create checkpoint
# train_detail.to_csv('clean_data/train_detail.csv', index=False)
# test_detail.to_csv('clean_data/test_detail.csv', index=False)