import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.image import imread


def page_lesions_visualizer_body():
    """Contents of Lesion Comparison"""
    st.write("### Lesion Comparison")
    st.info(
        f"* The client is interested in conducting a study to visually differentiate a benign skin lesion from one that is malignant."
    
    )
    version = 'v1'
    if st.checkbox("Difference betweeen average image and variability image"):
        av_healthy = plt.imread(f"outputs/{version}/avg_var_benign.png")
        av_mildew = plt.imread(f"outputs/{version}/avg_var_malignant.png")

        st.image(av_healthy, caption='Benign Lesion - Average and Variability')
        st.image(av_mildew, caption='Malignant Lesion - Average and Variability')
        st.write("---")
        
    if st.checkbox("Differences between Benign and Malignant Lesions."):
        av_diff = plt.imread(f"outputs/{version}/avg_diff.png")

        st.image(av_diff, caption='Difference between average images')
        
    if st.checkbox("Image Montage"):
        st.write("* To update the montage, please click 'Update Montage'")
        my_data_dir = 'input/Benign_Malignant_DataSet/MainData'
        labels = os.listdir(my_data_dir + '/validation')
        display_label = st.selectbox(
            label="Choose label", options=labels, index=0
            )
        if st.button("Update Montage"):      
            image_montage(dir_path= my_data_dir + '/validation',
                      display_label=display_label,
                      nrows=8, ncols=3, figsize=(10,25))
        st.write("---")


def image_montage(dir_path, display_label, nrows, ncols, figsize=(15,10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

  # subset the class you want to display
    if display_label in labels:

    # check if your montage space > subset size
    # num images in folder
        image_list = os.listdir(dir_path+'/'+ display_label)
        if nrows * ncols < len(image_list):
            img_idx = random.sample(image_list, nrows * ncols)
        else:
            print(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are {len(image_list)} in your subset. "
                f"You requested a montage with {nrows * ncols} spaces")
            return
    

        # create list of axes indices based on nrows and ncols
        list_rows= range(0,nrows)
        list_cols= range(0,ncols)
        plot_idx = list(itertools.product(list_rows,list_cols))


        # create a Figure and display images
        fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
        for x in range(0,nrows*ncols):
            img = imread(dir_path + '/' + display_label + '/' + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
        plt.tight_layout()
    
        st.pyplot(fig=fig)
        # plt.show()


    else:
        print("The label you selected doesn't exist.")
        print(f"The existing options are: {labels}")