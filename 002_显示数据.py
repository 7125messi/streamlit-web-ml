import streamlit as st
import numpy as np
import pandas as pd

st.write('显示Dataframe')
# 可以通过图表显示数据，也可以原始形式显示数据
# streamlit.dataframe(data=None, width=None, height=None)
# data (pandas.DataFrame, pandas.Styler, numpy.ndarray, Iterable, dict,) –or None The data to display.
df = pd.DataFrame(
   np.random.randn(10, 6),
   columns=('col %d' % i for i in range(6)))
st.dataframe(df)  # Same as st.write(df)
st.write(df)
st.dataframe(df,200,100)

# 还可以传递Pandas Styler对象以更改渲染的DataFrame的样式
st.dataframe(df.style.highlight_max(axis=0))

st.write('''
    显示静态表
    * 与st.dataframe不同之处在于，该表在这种情况下是静态的：它的全部内容都直接放在页面上。
''')
st.table(df)

st.write('''
    显示json
    * 将对象或字符串显示为打印精美的JSON字符串
    * 所有引用的对象也应可序列化为JSON
    * 如果object是字符串，则假定它包含序列化的JSON
''')

st.json({
    'foo': 'bar',
    'baz': 'boz',
    'stuff': [
        'stuff 1',
        'stuff 2',
        'stuff 3',
        'stuff 5',
    ],
})

st.json(df.to_dict())