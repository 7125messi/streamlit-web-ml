import streamlit as st
import pandas as pd
import numpy as np

st.title("数据转换")

# 上传文件并展示在页面
st.write('上传csv文件，进行数据转换 :wave:')
file = st.file_uploader('上传文件', type=['csv'], encoding='auto', key=None)

@st.cache
def get_data(file):
    df = pd.DataFrame()
    if file is not None:
        data = []
        for i, line in enumerate(file.getvalue().split('\n')):
             if i == 0:
                 header = line.split(',')
             else:
                 data.append(line.split(','))
        df=pd.DataFrame(data,columns=header)
    return df
df = get_data(file)
st.write(df)


# 针对特定的数据进行长表转宽表操作
##########y值的行转列
@st.cache
def transform_y(df):
    for param_name in df[["PARAM_NAME"]].drop_duplicates().values:
        data_para = df[df["PARAM_NAME"] == param_name[0]]
        final_data = data_para.pivot_table(index=["GLASS_ID", "GLASS_START_TIME", "EQUIP_ID"], columns=["SITE_NAME"],
                                           values=["PARAM_VALUE"], aggfunc=np.sum)
        ncl = [param_name[0] + '_' + str(x + 1) for x in range(final_data.shape[1])]
        final_data = pd.DataFrame(final_data.values, columns=ncl, index=final_data.index)
        final_data = final_data.reset_index()
        final_data["GLASS_ID"] = final_data["GLASS_ID"].fillna(method='ffill')
        final_data.drop_duplicates(subset=["GLASS_ID"], keep='last', inplace=True)
        return final_data

#####x值的行转列
@st.cache
def transform_x(df_x):
    data_x = df_x.pivot_table(index="GLASS_ID", columns="PARAM_NAME", values="PARAM_STR_VALUE",
                                  aggfunc='last')
    return data_x



# 设置button进行调用转换函数
if st.button("X数据转换"):
    data_x = transform_x(df)
    st.write(data_x)
    data_x.to_csv("data/data_x.csv")
#f len(df)!=0:
    #data_x=transform_x(df)
if st.button("X点击下载"):
    data_x = transform_x(df)
    st.write(data_x)
    st.write('http://localhost:8081/data_x.csv')
if st.button("Y数据转换"):
    data_y = transform_y(df)
    st.write(data_y)
    data_y.to_csv("data/data_y.csv")
# if len(df)!=0:
#     data_y=transform_y(df)
if st.button("Y点击下载"):
    data_y = transform_y(df)
    st.write(data_y)
    st.write('http://localhost:8081/data_y.csv')
# 两个button进行数据转换和下载，但是streamlit仅支持50M以内的数据上传和下载，可用于小数据
# 数据存储在本地项目的文件夹下，为了隐藏代码，可新建一个文件夹用于存储数据，但写入网页并不需要此文件夹的路径
