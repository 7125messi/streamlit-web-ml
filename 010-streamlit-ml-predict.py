import pickle
import numpy as np
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_PKL_FILE = "model.pkl"
IRIS_PKL_FILE = "iris.pkl"


def main():
    """This function runs/ orchestrates the Machine Learning App Registry"""
    st.markdown(
        """
        # Machine Learning App Registry

        These are projects from Artificial Intelligence Movement(AIM) Lead by
        [Boadzie Daniel](http://boadzie.surge.sh/)"
        """
    )
    sentimental_analysis_component()
    st.markdown("---")
    sallery_predictor_component()
    st.markdown("---")
    iris_predictor_component()


def sentimental_analysis_component():
    """## Sentimental Analysis Component
    A user can input a text string and output a sentiment score
    """
    st.markdown(
        """
        ## App 1: VADER Sentimental Analysis

        Sentimental Analysis is a branch of Natural Language Processing
        which involves the extraction of sentiments in text. The VADER package makes it easy to do
        Sentimental Analysis
        """
    )
    sentence = st.text_area("Write your sentence")

    if st.button("Submit"):
        result = sentiment_analyzer_scores(sentence)
        st.success(result)


def sallery_predictor_component():
    """## Sallery Predictor Component
    A user can input some of his developer features like years of experience and he will get a
    prediction of his sallery
    """
    st.markdown("## App 2: Salary Predictor For Techies")

    experience = st.number_input("Years of Experience")
    test_score = st.number_input("Aptitude Test score")
    interview_score = st.number_input("Interview Score")

    if st.button("Predict"):
        model = get_pickle(MODEL_PKL_FILE)
        features = [experience, test_score, interview_score]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        st.balloons()
        st.success(f"Your Salary per anum is: Ghc {prediction[0]:.0f}")


def iris_predictor_component():
    """## Iris Flower Predictor Component
    A user can input some of the features of an iris flower and see the predicted iris type
    prediction
    """
    st.markdown("## App 3: Iris Flower Classifier")

    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    if st.button("Report"):
        iris = get_pickle(IRIS_PKL_FILE)
        features = [sepal_length, sepal_width, petal_length, petal_width]
        final_features = [np.array(features)]
        prediction = iris.predict(final_features)
        prediction = str(prediction).replace("']", "").split("-")
        st.balloons()
        st.success(f"The flower belongs to the class {prediction[1]}")


@st.cache
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()  # initialize it


@st.cache
def sentiment_analyzer_scores(sentence) -> str:
    score = get_sentiment_analyzer().polarity_scores(sentence)
    return f"The Sentiment is ==> {score}"


@st.cache
def get_pickle(file: str):
    with open(file,'rb') as open_file:
        pickle.load(open_file)
    # return joblib.load(file)


if __name__ == "__main__":
    main()
