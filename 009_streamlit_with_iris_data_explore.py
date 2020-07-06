import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.sidebar.title("参考资料")
st.sidebar.info(
    "This app shows available data in [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets) repository.\n\n"
    "It is maintained by [Ali](https://www.linkedin.com/in/aliavnicirik/). \n\n"
    "Check the code at https://github.com/aliavni/awesome-data-explorer"
)


@st.cache
def read_iris_csv() -> pd.DataFrame:
    data = datasets.load_iris()
    df = pd.concat([pd.DataFrame(data.data),pd.DataFrame(data.target)],axis=1)
    df.columns = ['sepal.length','sepal.width','petal.length','petal.width','variety']
    return df


def main():
    st.title("Iris Classifier")
    st.header("Data Exploration")

    source_df = read_iris_csv()
    st.subheader("Source Data")
    if st.checkbox("Show Source Data"):
        st.write(source_df.head(10))

    selected_species_df = select_species(source_df)
    if not selected_species_df.empty:
        show_scatter_plot(selected_species_df)
        show_histogram_plot(selected_species_df)
    else:
        st.info("Please select one of more varieties above for further exploration.")
    show_machine_learning_model(source_df)


def select_species(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    选择某个类别去进行数据探索
    """
    selected_species = st.multiselect(
        "Select iris varieties for further exploration below",
        source_df["variety"].unique(),
    )
    selected_species_df = source_df[(source_df["variety"].isin(selected_species))]
    if selected_species:
        st.write(selected_species_df.head(10))
    return selected_species_df


def show_scatter_plot(selected_species_df: pd.DataFrame):
    """
    根据选择的某个类别的两个特征画出散点图
    """
    st.subheader("Scatter plot")
    feature_x = st.selectbox("Which feature on x?", selected_species_df.columns[0:4])
    feature_y = st.selectbox("Which feature on y?", selected_species_df.columns[0:4])
    fig = px.scatter(selected_species_df, x=feature_x, y=feature_y, color="variety")
    st.plotly_chart(fig)


def show_histogram_plot(selected_species_df: pd.DataFrame):
    """根据选择某个类别和选择的某个特征画出条形图
    """
    st.subheader("Histogram")
    feature = st.selectbox("Which feature?", selected_species_df.columns[0:4])
    fig2 = px.histogram(selected_species_df, x=feature, color="variety", marginal="rug")
    st.plotly_chart(fig2)


def show_machine_learning_model(source_df: pd.DataFrame):
    """
    根据选择的ML模型训练数据集并展示模型评估
    """
    st.header("Machine Learning models")

    features = source_df[
        ["sepal.length", "sepal.width", "petal.length", "petal.width"]
    ].values
    labels = source_df["variety"].values
    
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.7, random_state=1
    )
    
    alg = ["Decision Tree", "Support Vector Machine"]
    classifier = st.selectbox("Which algorithm?", alg)

    if classifier == "Decision Tree":
        model = DecisionTreeClassifier()
    elif classifier == "Support Vector Machine":
        model = SVC()
    else:
        raise NotImplementedError()

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    st.write("Accuracy: ", acc.round(2))

    pred_model = model.predict(x_test)
    cm_model = confusion_matrix(y_test, pred_model)
    st.write("Confusion matrix: ", cm_model)

if __name__ == "__main__":
    main()
