import streamlit as st
import numpy as np
import pandas as pd
import datetime

st.write('''
    # 1 将小部件添加到边栏
    * 不仅可以使用小部件向报表添加交互性，还可以使用来将其组织到侧边栏中**st.sidebar.[element_name]**
    * 传递到的每个元素st.sidebar都固定在左侧，从而使用户可以专注于应用程序中的内容。
    * 唯一不受支持的元素是：**st.write应该使用st.sidebar.markdown()，st.echo和st.spinner**。。。。。。
''')
cal_date = st.sidebar.date_input(
    label = "请选择计算日期",
    value = datetime.date(2020,6,30)
)

add_selectbox = st.sidebar.selectbox(
    label = "请选择计算表",
    options = ("pro_attribute", "shop_attribute", "shop_pro_attribute"),
    index = 0
)



st.write('''
    # 2 显示代码
    * streamlit.echo(code_location ='above'):在with块中使用可在应用程序上绘制一些代码，然后执行它
    * code_location（“上方” 或“下方”）–是在执行的代码块的结果之前还是之后显示回显的代码
''')
with st.echo():
    st.write('This code will be printed')

# 假设您拥有以下文件，并且希望通过在Streamlit应用程序中显示该中间部分来使其应用程序更不言自明
st.write('''
    ```python
    import streamlit as st
    def get_user_name():
        return 'John'

    # ------------------------------------------------
    # Want people to see this part of the code...

    def get_punctuation():
        return '!!!'

    greeting = "Hi there, "
    user_name = get_user_name()
    punctuation = get_punctuation()

    st.write(greeting, user_name, punctuation)

    # ...up to here
    # ------------------------------------------------

    foo = 'bar'
    st.write('Done!')
    ```
''')

# 让我们st.echo()在应用程序中使代码的中间部分可见
# 同一文件中可以有多个st.echo（）块。随意使用它！
st.write('''
    ```python
    import streamlit as st

    def get_user_name():
        return 'John'

    with st.echo():
        # Everything inside this block will be both printed to the screen
        # and executed.

        def get_punctuation():
            return '!!!'

        greeting = "Hi there, "
        value = get_user_name()
        punctuation = get_punctuation()

        st.write(greeting, value, punctuation)

    # And now we're back to _not_ printing to the screen
    foo = 'bar'
    st.write('Done!')
    ```
''')




st.write('''
    # 3 显示进度和状态
    Streamlit提供了一些方法，可让您向应用程序中添加动画。这些动画包括进度条，状态消息（如警告）和庆祝气球。
    
    * streamlit.progress(value) 显示进度条。 
        * value (int or float) 
        * 0 <= value <= 100 for int 
        * 0.0 <= value <= 1.0 for float
''')
# 以下是进度条随时间增加
import time
my_bar = st.progress(0)
for percent_complete in range(20):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)


st.write('''
    * streamlit.spinner(text='In progress...')  :在执行代码块时临时显示一条消息
    * streamlit.success(body):显示成功消息
    * streamlit.balloons() :绘制庆祝气球
    * streamlit.error(body) :显示错误信息
    * streamlit.warning(body) :显示警告消息
    * streamlit.info(body):显示参考消息
    * streamlit.exception(body)
''')
with st.spinner('Wait for it...'):
    time.sleep(2)
st.success('Done!')
st.balloons()
st.error('This is an error')
st.warning('This is a warning')
st.info('This is a purely informational message')

e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e)




st.write('''
    # 4 修改数据
    使用Streamlit，您可以修改现有元素（图表，表格，数据框）中的数据。

    * DeltaGenerator.add_rows(data=None, **kwargs)
''')
df1 = pd.DataFrame(
   np.random.randn(5, 6),
   columns=('col %d' % i for i in range(6)))
my_table = st.table(df1)

df2 = pd.DataFrame(
   np.random.randn(5, 6),
   columns=('col %d' % i for i in range(6)))
my_table.add_rows(df2)
# Now the table shown in the Streamlit app contains the data for
# df1 followed by the data for df2.


st.write('''
    可以使用绘图执行相同的操作。例如，如果要向折线图添加更多数据
''')
# Assuming df1 and df2 from the example above still exist...
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)
# Now the chart shown in the Streamlit app contains the data for
# df1 followed by the data for df2.