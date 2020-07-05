import streamlit as st
import numpy as np
import pandas as pd
import datetime
import time

st.write('''
    # Streamlit缓存技术
    Streamlit提供了一种缓存机制，即使在从Web加载数据，处理大型数据集或执行昂贵的计算时，也可以使您的应用程序保持高性能。这是通过@st.cache装饰器完成的。
    当用@st.cache装饰器标记一个函数时，它告诉Streamlit每当调用该函数时，它都需要检查一些事情：
    * 函数名称
    * 组成函数主体的实际代码
    * 用来调用函数的输入参数
        * streamlit的缓存机制是通过@st.cache的装饰器实现的。
        * 每当程序运行至被cache装饰的函数时，当第一次运行时，会正常执行并将结果存入缓存，
        * 当函数被再次运行，首先会判断函数的输入参数，函数主体内容是否发生变化，如果发生变化，则重新运行函数，否则将跳过函数，直接读取缓存结果。
    * 主要限制是**Streamlit的缓存功能不知道在带注释的函数主体之外发生的更改** 

    ```python
    streamlit.cache(
        func=None, 
        persist=False, 
        allow_output_mutation=False, 
        show_spinner=True, 
        suppress_st_warning=False, 
        hash_funcs=None, 
        max_entries=None, 
        ttl=None
    )
    ```

    ```
    函数装饰器，用于记忆函数执行。
    
    参数：	
    
    func（callable）–缓存功能。Streamlit对函数和相关代码进行哈希处理。
    
    persist（boolean）–是否将缓存保留在磁盘上。
    
    allow_output_mutation（boolean）–
    当返回值未突变时，Streamlit通常会显示警告，因为这可能会导致意想不到的后果。这是通过内部对返回值进行哈希处理来完成的。如果您知道自己在做什么，并且想要覆盖此警告，请将其设置为True。

    show_spinner（boolean）–启用微调器。默认值为True，以在发生高速缓存未命中时显示微调框。
    
    suppress_st_warning（boolean）–禁止有关从缓存的函数内调用Streamlit函数的警告。
    
    hash_funcs（dict 或None）–类型或完全限定名称到哈希函数的映射。这用于覆盖Streamlit的缓存机制中哈希器的行为：当哈希器遇到对象时，它将首先检查其类型是否与此字典中的键匹配，如果匹配，则使用提供的函数来生成哈希。有关如何使用此功能的示例，请参见下文。
    
    max_entries（int 或None）–保留在缓存中的最大条目数，或无限制的缓存的None。（将新条目添加到完整缓存后，最早的缓存条目将被删除。）默认值为“无”。
    
    ttl（float 或None）–将条目保留在缓存中的最大秒数；如果缓存条目不应过期，则为None。默认为无。
    ```
''')


st.write('''
    ## 示例1：基本用法
    首先，让我们看一个示例应用程序，该应用程序具有执行昂贵的，长时间运行的计算的功能。如果不进行缓存，则每次刷新应用程序时都会重新运行此功能，从而导致糟糕的用户体验。
    * 我们添加了suppress_st_warning关键字到@ st.cache装饰。
    * 这是因为上面的缓存函数本身使用了Streamlit命令（在本例中为st.write），并且当Streamlit看到该消息时，它会显示一条警告，指出仅在命中缓存命中时命令才会执行。
    * 通常，当您看到该警告时，是因为代码中存在错误。
    * 但是，在本例中，我们使用st.write命令来演示何时命中缓存，因此Streamlit向我们警告的行为正是我们想要的。
    * 结果，我们传入了suppress_st_warning = True来关闭该警告。
''')
# with st.echo():
#     @st.cache(suppress_st_warning=True)  # 👈 Changed this
#     def expensive_computation(a, b):
#         # 👇 Added this
#         st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
#         time.sleep(2)  # This makes the function take 2s to run
#         return a * b

#     a = 2
#     b = 21
#     res = expensive_computation(a, b)

#     st.write("Result:", res)




st.write('''
    ## 示例2：当函数参数更改时
    在不停止先前的应用服务器的情况下，让我们将其中一个参数更改为我们的缓存函数：

    现在，第一次您重新运行该应用程序时，它是缓存未命中。这可以通过显示“ Cache miss”（缓存未命中）文本以及应用以2秒的时间完成运行来证明。
''')
# with st.echo():
#     @st.cache(suppress_st_warning=True)
#     def expensive_computation(a, b):
#         st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
#         time.sleep(2)  # This makes the function take 2s to run
#         return a * b

#     a = 2 
#     b = 210  # 👈 Changed this
#     res = expensive_computation(a, b)

#     st.write("Result:", res)


st.write('''
    ## 示例3：当函数体更改时
    第一次运行是“高速缓存未命中”，但是当您按R时，每个后续运行都是高速缓存命中。
''')
# with st.echo():
#     @st.cache(suppress_st_warning=True)
#     def expensive_computation(a, b):
#         st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
#         time.sleep(2)  # This makes the function take 2s to run
#         return a * b + 1  # 👈 Added a +1 at the end here

#     a = 2
#     b = 210
#     res = expensive_computation(a, b)

#     st.write("Result:", res)




st.write('''
    ## 示例4：内部函数发生变化时
    让我们使缓存的函数在内部依赖于另一个函数

    您所看到的是通常的：
    * 第一次运行会导致缓存未命中。
    * 随后的每个重新运行都会导致缓存命中。
''')
# with st.echo():
#     def inner_func(a, b):
#         st.write("inner_func(", a, ",", b, ") ran")
#         return a * b


#     @st.cache(suppress_st_warning=True)
#     def expensive_computation(a, b):
#         st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
#         time.sleep(2)  # This makes the function take 2s to run
#         return inner_func(a, b) + 1

#     a = 2
#     b = 210
#     res = expensive_computation(a, b)

#     st.write("Result:", res)

st.write('''
    * 现在让我们尝试修改inner_func()
    * Streamlit始终会遍历您的代码及其依赖项，以验证缓存的值仍然有效。
    * 在开发应用程序时，您可以自由编辑代码，而不必担心缓存。
    * 您对应用程序进行的任何更改，Streamlit都应该做正确的事！
    * Streamlit也足够聪明，只需要遍历属于您应用程序的依赖项，而跳过来自已安装的Python库的任何依赖项。
''')
with st.echo():
    def inner_func(a, b):
        st.write("inner_func(", a, ",", b, ") ran")
        return a ** b  # 👈 Changed the * to ** here

    @st.cache(suppress_st_warning=True)
    def expensive_computation(a, b):
        st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
        time.sleep(2)  # This makes the function take 2s to run
        return inner_func(a, b) + 1

    a = 2
    b = 21
    res = expensive_computation(a, b)

    st.write("Result:", res)