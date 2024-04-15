import streamlit as st
import matplotlib.pyplot as plt


import streamlit as st


def page_project_hypothesis_body():
    st.write('### Project Hypothesis')

    st.success(
        "1. There's a strong conviction that there could be a way to visually "
        "observe and notice the difference in photographies between a "
        "benign skin lesion and one that is malignant. Being in low resolution, "
        "the filtering of the MRI scans and the comparison between the "
        "average scan of the tumor and the healthy brain scan should show "
        "the visible shade difference.\n"
        "2. The deep learning model with convolutional neural network (CNN) "
        "architecture should be able to accurately classify the unseen data "
        "of the skin lesions photographs into two categories, benign and malignant.\n"
        "Data augmentation techniques will help improve model generalization."
    )
    st.write('---')
    