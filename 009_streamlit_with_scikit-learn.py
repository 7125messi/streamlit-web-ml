import streamlit as st 
import numpy as np
import pandas as pd
import pymysql

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Streamlit-ML-Demo')

st.write("""
    * 探索不同数据集以及不同的分类器的效果
""")

dataset_name = st.sidebar.selectbox(
    label = '选择数据集',
    options = ('Iris', 'Breast Cancer', 'Wine'),
    index = 0
)
st.write(f"## {dataset_name} 数据集")

# 获取分类器的名字
classifier_name = st.sidebar.selectbox(
    label = '选择分类器',
    options = ('KNN', 'SVM', 'Random Forest'),
    index = 0
)

# 获得数据集
@st.cache(persist=True)
def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('数据集形状:', X.shape)
st.write('类别数:', len(np.unique(y)))

# 添加UI界面的参数
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider(
            label = 'C', 
            min_value = 0.00, 
            max_value = 10.0,
            step = 0.1
        )
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider(
            label = 'K', 
            min_value = 1, 
            max_value = 15,
            step = 1
        )
        params['K'] = K
    else:
        max_depth = st.sidebar.slider(
            label = 'max_depth', 
            min_value = 2, 
            max_value = 15,
            step = 1
        )
        params['max_depth'] = max_depth

        n_estimators = st.sidebar.slider(
            label = 'n_estimators', 
            min_value = 1, 
            max_value = 100,
            step = 5
        )
        params['n_estimators'] = n_estimators
    return params
params = add_parameter_ui(classifier_name)


# 获得分类器
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], 
            random_state=1234
        )
    return clf
clf = get_classifier(classifier_name, params)

#### CLASSIFICATION ####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
#plt.show()
st.pyplot()
