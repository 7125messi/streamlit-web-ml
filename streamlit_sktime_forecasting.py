'''
@Author: ydzhao
@Description: 
@Date: 2020-07-16 11:26:20
@LastEditTime: 2020-07-16 15:56:12
@FilePath: \project\streamlit-web-ml\01_streamlit_sktime_forecasting.py
'''
import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting.forecasting import plot_ys

from sktime.forecasting.naive import NaiveForecaster

st.write('''
    ### 1 数据简介
    > 使用著名的Box-Jenkins航空公司数据集，该数据集显示1949-1960年期间每月国际航班的乘客人数。除了使用原始时间序列（这是乘法时间序列的经典示例）之外，我们还将通过对原始数据执行对数转换来创建加法时间序列，因此我们可以将预测器与两种类型的模型进行比较。
''')
y = load_airline()
st.dataframe(y.head())
fig, ax = plot_ys(y)
ax.set(xlabel="Time", ylabel="Number of airline passengers")
st.pyplot()


st.write('''
    ### 2 定义预测任务
    * 接下来，我们将定义一个预测任务。我们将尝试使用前几年的训练数据来预测最近3年的数据。 该系列中的每个点代表一个月，因此我们应保留最后36个点作为测试数据，并使用36步超前的预测范围来评估预测效果。
    * 我们将使用sMAPE（对称平均绝对百分比误差）来量化我们预测的准确性。 较低的sMAPE意味着较高的精度。
    我们可以按以下方式拆分数据：
''')
y_train, y_test = temporal_train_test_split(y, test_size=36)
plot_ys(y_train, y_test, labels=["y_train", "y_test"])
st.pyplot()
st.write("y_train.shape[0],y_test.shape[0]:",y_train.shape[0],y_test.shape[0])

st.write('''
    当我们想要生成预测时，我们需要指定预测范围并将其传递给我们的预测算法。 我们可以将预测范围指定为相对于训练结束时前面步骤的简单numpy数组：
''')
fh = np.arange(len(y_test)) + 1
st.write(fh)

st.write('''
    因此，在此我们有兴趣预测从第一步到第36步。当然，您可以使用其他预测范围。 例如，要仅预测前面的第二和第五步，可以编写：fh = np.array([2, 5])
''')

st.write('''
    ### 3 预测
    像在scikit-learn中一样，为了进行预测，我们需要首先指定（或构建）模型，然后将其拟合到训练数据中，最后调用predict为给定的预测范围生成预测。
    sktime附带了多种预测算法（或预测器）和用于构建复合模型的工具。 
    所有预测器共享一个公共接口。 对预测员进行一系列单一数据的培训，并对提供的预测范围进行预测。

    (1) 基准模型预测
    * 我们总是预测（在训练系列中）观察到的最后一个值
    * 我们预测在同一季节观察到的最后一个值
''')
y_pred = np.repeat(y_train.iloc[-1], len(fh))
y_pred = pd.Series(y_pred, index=y_train.index[-1] + fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
st.pyplot()

st.write('''
    (2) 使用sktime
''')
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
y_last = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
st.pyplot()
st.write("smape_loss(y_last, y_test):",smape_loss(y_last, y_test))

forecaster = NaiveForecaster(strategy="seasonal_last", sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
st.pyplot()
st.write("smape_loss(y_last, y_test):",smape_loss(y_last, y_test))


st.write('''
    ### 4 Forecasting with sktime
    
    ### 4.1 Reduction: from forecasting to regression
    
    从预测到回归sktime为此方法提供了一个元估算器，即：
    * 模块化并与scikit-learn兼容，因此我们可以轻松地应用任何scikit-learn回归器来解决我们的预测问题;
    * 可调整的，允许我们调整超参数，例如窗口长度或生成预测的策略
    * 自适应的，从某种意义上讲，它可以使scikit-learn的估算器界面适应预测者的界面，并确保我们可以调整和正确评估模型
''')

y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)
st.write("y_train.shape[0],y_test.shape[0]:",y_train.shape[0],y_test.shape[0])


from sktime.forecasting.compose import ReducedRegressionForecaster
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=1)
forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=12, strategy="recursive")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))

st.write('''
    为了更好地理解先前的数据转换，我们可以看看如何将训练数据划分为多个窗口。 
    本质上，sktime使用时间时间序列分割器，类似于scikit-learn中的交叉验证分割器。 
    在这里，我们展示了这对于训练数据的前20个观察结果是如何工作的：
''')
with st.echo():
    from sktime.forecasting.model_selection import SlidingWindowSplitter
    cv = SlidingWindowSplitter(window_length=10, start_with_window=True)
    for input_window, output_window in cv.split(y_train.iloc[:20]):
        print(input_window, output_window)
st.write('''
    [0 1 2 3 4 5 6 7 8 9] [10]
    
    [ 1  2  3  4  5  6  7  8  9 10] [11]
    
    [ 2  3  4  5  6  7  8  9 10 11] [12]
    
    [ 3  4  5  6  7  8  9 10 11 12] [13]
    
    [ 4  5  6  7  8  9 10 11 12 13] [14]
    
    [ 5  6  7  8  9 10 11 12 13 14] [15]
    
    [ 6  7  8  9 10 11 12 13 14 15] [16]
    
    [ 7  8  9 10 11 12 13 14 15 16] [17]
    
    [ 8  9 10 11 12 13 14 15 16 17] [18]
    
    [ 9 10 11 12 13 14 15 16 17 18] [19]
    
    [0 1 2 3 4 5 6 7 8 9] [10]
''')

st.write('''
    ### 4.2 Statistical forecasters
    
    sktime基于statsmodels中的实现，具有多种统计预测算法。 
    
    例如，要将指数平滑与可加趋势成分和可乘季节性一起使用，我们可以编写以下内容。注意，由于这是每月数据，所以季节性周期（sp）或每年的周期数为12。
''')

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
forecaster = ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))

st.write('''
    另一个常见模型是ARIMA模型。 
    在sktime中，我们连接pmdarima，这是一个用于自动选择最佳ARIMA模型的软件包。 
    这是因为搜索了许多可能的模型参数，因此可能需要更长的时间。
''')

from sktime.forecasting.arima import AutoARIMA
forecaster = AutoARIMA(sp=12, suppress_warnings=True)
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))


st.write('''
    ### 4.3 Compositite model building
    sktime提供了用于组合模型构建的模块化API，以进行预测。 
    
    * Ensembling
    像scikit-learn一样，sktime提供了一个元预测器来集成多种预测算法。
    例如，我们可以如下组合指数平滑的不同变体：
''')

from sktime.forecasting.compose import EnsembleForecaster
forecaster = EnsembleForecaster([
    ("ses", ExponentialSmoothing(seasonal="multiplicative", sp=12)),
    ("holt", ExponentialSmoothing(trend="add", damped=False, seasonal="multiplicative", sp=12)),
    ("damped", ExponentialSmoothing(trend="add", damped=True, seasonal="multiplicative", sp=12))
])
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))


st.write('''
    * Tuning
    In the `ReducedRegressionForecaster`, 
    both the `window_length` and `strategy arguments` are hyper-parameters which we may want to optimise.
''')
from sktime.forecasting.model_selection import ForecastingGridSearchCV
forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=15, strategy="recursive")
param_grid = {"window_length": [5, 10, 15]}
cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.5))
gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=param_grid)
gscv.fit(y_train)
y_pred = gscv.predict(fh)

plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))
st.write("gscv.best_params_:",gscv.best_params_)

st.write('''
    * Detrending
    请注意，到目前为止，上述减少方法并未考虑任何季节或趋势，但我们可以轻松地指定首先对数据进行趋势去除的管道。
    sktime提供了一个通用的去趋势器，它是一个使用任何预测器并返回预测器预测值的样本内残差的转换器。 
    例如，要删除时间序列的线性趋势，我们可以写成
''')

from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformers.single_series.detrend import Detrender
# liner detrending
forecaster = PolynomialTrendForecaster(degree=1)
transformer = Detrender(forecaster=forecaster)
yt = transformer.fit_transform(y_train)
# internally, the Detrender uses the in-sample predictions of the PolynomialTrendForecaster
forecaster = PolynomialTrendForecaster(degree=1)
fh_ins = -np.arange(len(y_train)) # in-sample forecasting horizon
y_pred = forecaster.fit(y_train).predict(fh=fh_ins)
plot_ys(y_train, y_pred, yt, labels=["y_train", "Fitted linear trend", "Residuals"])
st.pyplot()


st.write('''
    * Pipelining
    让我们在管道中使用**去趋势剂**和**去季节化**。 
    请注意，在预测中，当我们在拟合之前应用数据变换时，我们需要将逆变换应用于预测值。 
    为此，我们提供以下管道类
''')

from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformers.single_series.detrend import Deseasonalizer
forecaster = TransformedTargetForecaster([
    ("deseasonalise", Deseasonalizer(model="multiplicative", sp=12)),
    ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
    ("forecast", ReducedRegressionForecaster(regressor=regressor, window_length=15, strategy="recursive"))
])
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))


st.write('''
    * Dynamic forecasts
    对于模型评估，我们有时希望使用带有测试数据滑动窗口的时间交叉验证来评估多个预测。 
    为此，sktime中的所有预测器都具有update_predict方法。 
    在这里，我们对测试集进行了重复的单步提前预测。
    
    请注意，预测任务已更改：尽管我们仍进行36个预测，但我们不会预测36个步骤，而是进行36个单步预测
''')
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)
cv = SlidingWindowSplitter(fh=1)
y_pred = forecaster.update_predict(y_test, cv)
plot_ys(y_train, y_test, y_pred)
st.pyplot()
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))



st.write('''
    * Prediction intervals
    到目前为止，我们仅关注点预测。 在许多情况下，我们也对预测间隔感兴趣。 
    sktime的界面支持预测间隔，但我们尚未针对所有算法实现它们。在这里，我们使用Theta预测算法
''')
from sktime.forecasting.theta import ThetaForecaster
forecaster = ThetaForecaster(sp=12)
forecaster.fit(y_train)
alpha = 0.05  # 95% prediction intervals
y_pred, pred_ints = forecaster.predict(fh, return_pred_int=True, alpha=alpha)
st.write("smape_loss(y_test, y_pred):",smape_loss(y_test, y_pred))

fig, ax = plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
ax.fill_between(y_pred.index, pred_ints["lower"], pred_ints["upper"], alpha=0.2, color="green", label=f"{1 - alpha}% prediction intervals")
plt.legend()
st.pyplot()


st.write('''
    ### 5 Summary
    * 如我们所见，为了进行预测，我们需要首先指定（或构建）模型，然后将其拟合到训练数据中，最后调用predict为给定的预测范围生成预测。
    * sktime附带了多种预测算法（或预测器）和用于构建复合模型的工具。 所有预测器共享一个公共接口。 对预测员进行一系列单一数据的培训，并对提供的预测范围进行预测。
    * sktime基于statsmodels中的实现，具有多种统计预测算法。 例如，要将指数平滑与可加趋势成分和可乘季节性一起使用。
''')
