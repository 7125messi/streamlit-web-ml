import streamlit as st
import numpy as np
import pandas as pd

st.write('''
    # 显示图表
    * Streamlit支持几种不同的图表库，最基础的库是Matplotlib
    * 还有诸如Vega Lite（2D图表）和 deck.gl（地图和3D图表）之类的交互式图表库
    * **Streamlit原生的图表类型，例如st.line_chart和st.area_chart**

    ## 显示折线图
    streamlit.line_chart(data=None, width=0, height=0, use_container_width=True)
    * 此命令使用数据自己的列和索引来确定图表的规格。在许多“仅绘制此”场景中更易于使用，而可定制性却较低。
    * use_container_width（bool）–如果为True，则将图表宽度设置为列宽。这优先于width参数。
''')
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)


st.write('''
    ## 显示面积图
    streamlit.area_chart(data=None, width=0, height=0, use_container_width=True)
''')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.area_chart(chart_data)


st.write('''
    ## 显示条形图
    streamlit.area_chart(data=None, width=0, height=0, use_container_width=True)
''')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.bar_chart(chart_data)


st.write('''
    ## 显示matplotlib.pyplot图
    streamlit.pyplot(fig=None, clear_figure=None, **kwargs)
    * Matplotlib支持几种不同类型的“后端”。如果使用Matplotlib和Streamlit遇到错误，请尝试将后端设置为“ TkAgg”
    * `echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc`
''')
import matplotlib.pyplot as plt
import numpy as np
arr = np.random.normal(1, 1, size=100)
plt.hist(arr, bins=20)
st.pyplot()


st.write('''
    ## 使用Altair库显示图表
    streamlit.altair_chart(altair_chart, width=0, use_container_width=False)
''')
import pandas as pd
import numpy as np
import altair as alt
df = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c'])
c = alt.Chart(df).mark_circle().encode(
    x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.altair_chart(c, use_container_width=True)


st.write('''
    ## 使用Plotly库显示图表
    streamlit.plotly_chart(figure_or_data, width=0, height=0, use_container_width=False, sharing='streamlit', **kwargs)
''')
import streamlit as st
import plotly.figure_factory as ff
import numpy as np
# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
# Group data together
hist_data = [x1, x2, x3]
group_labels = ['Group 1', 'Group 2', 'Group 3']
# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
# Plot!
st.plotly_chart(fig, use_container_width=True)



st.write('''
    ## 使用Bokeh交互式散景图
    streamlit.bokeh_chart(figure, use_container_width=False)
''')
import streamlit as st
from bokeh.plotting import figure
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]
p = figure(
    title='simple line example',
    x_axis_label='x',
    y_axis_label='y')
p.line(x, y, legend='Trend', line_width=2)
st.bokeh_chart(p, use_container_width=True)




st.write('''
    ## 显示graphviz_chart图
    streamlit.graphviz_chart(figure_or_dot, width=0, height=0, use_container_width=False)
''')

# import streamlit as st
# import graphviz as graphviz
# Create a graphlib graph object
# graph = graphviz.Digraph()
# graph.edge('run', 'intr')
# graph.edge('intr', 'runbl')
# graph.edge('runbl', 'run')
# graph.edge('run', 'kernel')
# graph.edge('kernel', 'zombie')
# graph.edge('kernel', 'sleep')
# graph.edge('kernel', 'runmem')
# graph.edge('sleep', 'swap')
# graph.edge('swap', 'runswap')
# graph.edge('runswap', 'new')
# graph.edge('runswap', 'runmem')
# graph.edge('new', 'runmem')
# graph.edge('sleep', 'runmem')
# st.graphviz_chart(graph)

import streamlit as st
st.graphviz_chart('''
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
''')



st.write('''
    ## 使用PyDeck位置信息图
    streamlit.pydeck_chart(pydeck_obj=None, use_container_width=False)
''')
import pydeck as pdk
df = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))