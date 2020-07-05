import streamlit as st
import pandas as pd



# st.title来设置应用程序的标题
# 使用2个标题级别： st.header和st.subheader
st.title('机器学习项目Web展示')
st.header('机器学习项目Web展示1')
st.subheader('机器学习项目Web展示2')



# 输入st.text
st.text('Streamlit,你好')

# 输入st.markdown
# streamlit.markdown(body,unsafe_allow_html = False)
st.markdown('Streamlit,**你好**')
st.markdown('''
    Streamlit,**你好**
''')

# streamlit.latex
st.latex(r'''
    a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
    \sum_{k=0}^{n-1} ar^k =
    a \left(\frac{1-r^{n}}{1-r}\right)
''')



# 瑞士军刀--命令st.write，该命令接受多个参数和多种数据类型
# streamlit.write(*args, **kwargs)
# *args（任意）一个或多个对象以打印到应用程序。
st.write('''
    * st.write瑞士军刀命令，该命令接受多个参数和多种数据类型
    * streamlit.write(*args, **kwargs),*args（任意）一个或多个对象以打印到应用程序。
        * **write（string）：打印格式化的Markdown字符串**，带有支持LaTeX表达式和表情符号短代码。有关更多信息，请参阅文档。
        * write（data_frame）：将DataFrame显示为表格。
        * write（error）：专门打印异常。
        * write（func）：显示有关功能的信息。
        * write（module）：显示有关模块的信息。
        * write（dict）：在交互式窗口小部件中显示dict。
        * write（obj）：默认是打印str（obj）。
        * write（mpl_fig）：显示Matplotlib图形。
        * write（altair）：显示一个Altair图表。
        * write（keras）：显示Keras模型。
        * write（graphviz）：显示一个Graphviz图。
        * write（plotly_fig）：显示绘图。
        * write（bokeh_fig）：显示散景图。
        * write（sympy_expr）：使用LaTeX打印SymPy表达式。
''')

# 当输入为字符串时，绘制Markdown格式的文本
st.write('Streamlit,你好,:sunglasses:')
# st.write添加markdown文本
st.write('''
    # 一级标题
    * 1
    * 2
    * 3
    > 12345
    ## 二级标题
    ### 三级标题
''')

# st.write接受其他数据格式
# 例如数字，数据框，样式化的数据框和各种对象
st.write(1234)
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
}))

# st.write传入多个参数来执行以下操作
st.write('1 + 1 = ', 2)
data_frame = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
})
st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')

# st.write也接受图表对象
import pandas as pd
import numpy as np
import altair as alt
df = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c']
)
c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c']
    )
st.write(c)

# 魔术命令是Streamlit中的一项功能，可让您只需很少的按键操作就可以将markdown and data写入应用程序

# 它的工作原理很简单：
# 只要Streamlit在自己的行上看到变量或文字值，它就会自动使用st.write将其写入您的应用程序 

# 如果希望更明确地调用Streamlit命令，
# 可以~/.streamlit/config.toml使用以下设置来关闭Magic：
# [runner]
# magicEnabled = false

# 添加Data
df = pd.DataFrame({'col1': [1,2,3]})
df  # <-- Draw the dataframe
st.write(df)


col1 = [1,2,3]
'col1', col1  # <-- Draw the string 'col1' and then the value of col1
col2 = {'a':1,'b':2,'c':3}
'col2', col2

# 
x = st.slider('x')
st.write(x,'square is',x*x)