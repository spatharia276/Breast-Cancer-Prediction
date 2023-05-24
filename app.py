import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the CSV file into a pandas dataframe
st.set_page_config(page_title="Breast Cancer Detection")
df = pd.read_csv('data.csv')
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})
df.drop(['id','Unnamed: 32'], axis=1,inplace=True)
st.title("Breast Cancer Prediction")
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1598885159329-9377168ac375?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=873&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


# Accept user inputs for the feature columns
radius_mean = st.number_input('Radius Mean')
texture_mean = st.number_input('Texture Mean')
perimeter_mean = st.number_input('Perimeter Mean')
area_mean = st.number_input('Area Mean')
smoothness_mean = st.number_input('Smoothness Mean')
compactness_mean = st.number_input('Compactness Mean')
concavity_mean = st.number_input('Concavity Mean')
concave_points_mean = st.number_input('Concave Points Mean')
symmetry_mean = st.number_input('Symmetry Mean')
fractal_dimension_mean = st.number_input('Fractal Dimension Mean')
radius_se = st.number_input('Radius Standard Error')
texture_se = st.number_input('Texture Standard Error')
perimeter_se = st.number_input('Perimeter Standard Error')
area_se = st.number_input('Area Standard Error')
smoothness_se = st.number_input('Smoothness Standard Error')
compactness_se = st.number_input('Compactness Standard Error')
concavity_se = st.number_input('Concavity Standard Error')
concave_points_se = st.number_input('Concave Points Standard Error')
symmetry_se = st.number_input('Symmetry Standard Error')
fractal_dimension_se = st.number_input('Fractal Dimension Standard Error')
radius_worst = st.number_input('Worst Radius')
texture_worst = st.number_input('Worst Texture')
perimeter_worst = st.number_input('Worst Perimeter')
area_worst = st.number_input('Worst Area')
smoothness_worst = st.number_input('Worst Smoothness')
compactness_worst = st.number_input('Worst Compactness')
concavity_worst = st.number_input('Worst Concavity')
concave_points_worst = st.number_input('Worst Concave Points')
symmetry_worst = st.number_input('Worst Symmetry')
fractal_dimension_worst = st.number_input('Worst Fractal Dimension')


# Use the model to make predictions on the user inputs
model =LogisticRegression(C=0.1)
model.fit(df.drop('diagnosis', axis=1), df['diagnosis'])
prediction = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean,smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])

# Display the prediction to the user
if prediction[0] == 0:
    st.write('Prediction: Benign')
else:
    st.write('Prediction: Malignant')
