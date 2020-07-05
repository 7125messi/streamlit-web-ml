import streamlit as st
import numpy as np
import pandas as pd

st.write('''
    ## 显示图像
    streamlit.image(image, caption=None, width=None, use_column_width=False, clamp=False, channels='RGB', format='JPEG')
    * Display an image or list of images.
    * use_container_width（bool）–如果为True，则将图表宽度设置为列宽。这优先于width参数。
''')
from PIL import Image
image = Image.open('sunrise.png')
st.image(image, 
    caption='Sunrise by the mountains',
    use_column_width=True
)



st.write('''
    ## 显示音频
    streamlit.audio(data, format='audio/wav', start_time=0)
''')
# audio_file = open('myaudio.ogg', 'rb')
# audio_bytes = audio_file.read()
# st.audio(audio_bytes, format='audio/ogg')


st.write('''
    ## 显示视频
    streamlit.video(data, format='video/mp4', start_time=0)
''')
video_file = open('myvideo.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)