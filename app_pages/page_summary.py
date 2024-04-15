import streamlit as st


def page_summary_body():

    st.write('### Quick Project Summary')

    st.info(
        'Skin lesion classifier is a data science and machine learning project. '
        'The business goal of this project is the differentiation of the '
        'benign skin leasion from a malignant one based on a photography '
        'of the skin lesion. The project is realised with the Streamlit Dashboard '
        'and gives the client a possibility to upload a photo of a skin lesion '
        'in order to predict the type of lesion presented. The dashboard '
        'offers the results of the data analysis, description and the '
        'analysis of the project\'s hypothesis, and details about the '
        'performance of the machine learning model.'
        )

    st.write(
        '* For additional information, please visit and read the '
        '[Project\'s README file]'
        '(https://github.com/ispeakfishlanguage/benign-malignant-classification).')

    st.success(
        'The project has 2 business requirements:\n'
        '* 1 - The client would like to have a study of the dataset'
        'collected \n'
        '* 2 - The client would like to have a ML model developed in order '
        'to be able to identify the nature of the skin lesion presented.'
        )