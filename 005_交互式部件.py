import streamlit as st
import numpy as np
import pandas as pd

st.write('''
    # 交互式部件使用
    借助小部件，Streamlit允许您使用按钮，滑块，文本输入等将交互性直接烘焙到应用程序中

    > 对于大量选项（100+），交互式用户体验可能会降低
    * streamlit.button(label, key=None)
    * streamlit.checkbox(label, value=False, key=None)
    * streamlit.radio(label, options, index=0, format_func=<class 'str'>, key=None)
    * streamlit.selectbox(label, options, index=0, format_func=<class 'str'>, key=None)
    * streamlit.multiselect(label, options, default=None, format_func=<class 'str'>, key=None)
''')

# 显示按钮小部件
# 简短的标签，向用户说明此按钮的用途 label
if st.button(label = 'Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
st.write('---')




# 显示复选框小部件
agree = st.checkbox(
    label = 'I agree', # 简短的标签，向用户说明此复选框的用途
    value = False # 首次渲染时预选该复选框,将在内部强制转换为bool
)
if agree:
    st.write('Great!')
st.write('---')



# 显示单选按钮小部件
genre = st.radio(
    label = "What's your favorite movie genre",
    options = ('Comedy', 'Drama', 'Documentary'), # list ，tuple ，numpy.ndarray 或pandas.Series
    index = 1 # 第一次渲染时默认选项的索引
)
if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write("You didn't select comedy.")
st.write('---')



# 显示选择窗口小部件
option = st.selectbox(
    label = 'How would you like to be contacted?',
    options = ('Email', 'Home phone', 'Mobile phone'), # list ，tuple ，numpy.ndarray 或pandas.Series
    index = 1 # 第一次渲染时默认选项的索引
)
st.write('You selected:', option)
st.write('---')



# 显示多选小部件
options = st.multiselect(
    label = 'What are your favorite colors',
    options = ['Green', 'Yellow', 'Red', 'Blue'], # list ，tuple ，numpy.ndarray 或pandas.Series
    default = ['Yellow', 'Red'] # 多选小部件默认开始为空
)
st.write('You selected:', options)



st.write('''
    ---
    ---
    ---
''')



st.write('''
    * streamlit.slider(label, min_value=None, max_value=None, value=None, step=None, format=None, key=None)
    * streamlit.text_input(label, value='', max_chars=None, key=None, type='default')
    * streamlit.number_input(label, min_value=None, max_value=None, value=<streamlit.DeltaGenerator.NoValue object>, step=None, format=None, key=None)
    * streamlit.text_area(label, value='', height=None, max_chars=None, key=None)
    * streamlit.date_input(label, value=None, min_value=datetime.datetime(1, 1, 1, 0, 0), max_value=None, key=None)
    * streamlit.time_input(label, value=None, key=None)
    * streamlit.file_uploader(label, type=None, encoding='auto', key=None)
    * streamlit.beta_color_picker(label, value=None, key=None)
''')


# 显示滑块小部件
value = st.slider(
    label = 'Select a value', # str or None
    min_value = 0.0,          # （int / float 或None）–最小允许值。如果值为int，则默认为0；否则为0.0。
    max_value = 100.0,        # （int / float 或None）–最大允许值。如果值为int，则默认为100；否则为1.0。
    value = 20.0              # (int/float or a tuple/list of int/float or None,滑块首次呈现时的值。如果在此处传递两个值的元组/列表，则将呈现具有上下限的范围滑块。例如，如果设置为（1，10），则滑块的可选范围为1到10。默认值为min_value。
)
st.write('Value:', value)
values = st.slider(
    label = 'Select a range of values',
    min_value = 0.0, 
    max_value = 100.0, 
    value = (25.0, 75.0),
    step  = 5.0               # step（int/float或None）–滑动间隔。如果值为int，则默认为1；否则为0.01，注意要与min_value和max_value数据类型保持一致
)
st.write('Values:', values)
st.write('---')




# 显示单行文本输入小部件
title = st.text_input(
    label = 'Movie title', # 简短的标签，向用户说明此输入的用途
    value = 'Life of Brian' # 首次呈现时的文本值。这将在内部强制转换为str。
)
st.write('The current movie title is', title)
st.write('---')



# 显示数字输入小部件
number = st.number_input(
    label = 'Insert a number', # 简短的标签，向用户解释此输入的用途
    min_value = 0.0,           # min_value（int或float或None）–最小允许值。如果没有，则没有最小值。
    max_value = 100.0,         # max_value（int或float或None）–最大允许值。如果为None，则没有最大值。
    value = 20.0,              # value（int或float或None）–此小部件首次呈现时的值。默认为min_value，如果min_value为None，则为0.0
    step = 0.5                 # step（int/float或None）–滑动间隔。如果值为int，则默认为1；否则为0.01，注意要与min_value和max_value数据类型保持一致
)
st.write('The current number is ', number)
st.write('---')




# 显示多行文本输入小部件
txt = st.text_area(
    label = 'Text to analyze',  # 简短的标签，向用户说明此输入的用途
    value = '''
        It was the best of times, it was the worst of times, it was
        the age of wisdom, it was the age of foolishness, it was
        the epoch of belief, it was the epoch of incredulity, it
        was the season of Light, it was the season of Darkness, it
        was the spring of hope, it was the winter of despair, (...)
    '''   # 首次呈现时的文本值。这将在内部强制转换为str。
)
st.write('Sentiment:', txt)
st.write('---')





# 显示日期输入小部件
import datetime
d = st.date_input(
    label = "When's your birthday",        # –简短的标签，向用户解释此日期输入的含义
    value = datetime.date(2020, 7, 5),     # （datetime.date or datetime.datetime or list/tuple of datetime.date or datetime.datetime or None) -这个widget当它第一次呈现的值。如果提供的列表/元组具有0到2个日期/日期时间值，则日期选择器将允许用户提供范围。默认为今天作为单日选择器。
    min_value = datetime.date(2018, 6, 1), # （datetime.date 或datetime.datetime）–最小可选日期。默认为datetime.min。
    max_value = datetime.date(2021,12,31)  # （datetime.date 或datetime.datetime）–最大可选日期。默认为今天+ 10y。
)
st.write('Your birthday is:', d)
st.write('---')


# 显示时间输入小部件
import datetime
t = st.time_input(
    label = 'Set an alarm for',
    value = datetime.time(8, 45)   # （datetime.time/datetime.datetime）–此小部件首次呈现时的值。这将在内部强制转换为str。默认为当前时间。
)
st.write('Alarm is set for', t)
st.write('---')




# 显示文件上传器小部件
# 默认情况下，上传的文件限制为200MB。您可以使用server.maxUploadSize配置选项进行配置。
uploaded_file = st.file_uploader(
    label = "Choose a CSV file",    # 向用户说明此文件上传器的用途
    type = "csv"                    # str或str的列表或None–允许的扩展数组。['png'，'jpg']默认情况下，允许所有扩展名。
)
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
st.write('---')




# 显示颜色选择器小部件
color = st.beta_color_picker(
    label = 'Pick A Color',     # 简短的标签，向用户说明此输入的用途
    value = '#00f900'           # （str 或None）–首次渲染时此小部件的十六进制值。如果为无，则默认为黑色。
)
st.write('The current color is', color)