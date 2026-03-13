import streamlit as st 
import pandas as pd 
import joblib 
import numpy as np 


# then we load the model 
model = joblib.load("model/wine_quality_model.joblib")
scaler = joblib.load("model/wine_quality_scaler.joblib")


st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

st.title("🍷Wine Quality Prediction")

# then we use the single row prediction 

st.subheader("🔢 Single value Prediction")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity")
    citric_acid = st.number_input("Citirc Acid")
    chlorides = st.number_input("Chlorides")
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
    sulphates = st.number_input("Sulphates")

with col2:
    volatile_acidity = st.number_input("Volatile Acidity")
    residual_sugar = st.number_input("Residual Sugar")
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
    ph = st.number_input("pH")
    alcohol = st.number_input("Alcohol")


if st.button("Predict Wine Quality"):
        
        input_data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        ph,
        sulphates,
        alcohol
    ]])
        scaler_input = scaler.transform(input_data)
        prediction = model.predict(scaler_input)[0]

        st.success(f"Predicted Wine Quality: {prediction}")

        

# then for batch prediction 

st.subheader("Batch prediction")

upload_file = st.file_uploader("Upload .csv file", type=["csv"])

if upload_file is not None:
     
     # then we read the file first 
     df = pd.read_csv(upload_file)

     st.write("Uploaded Data Preview")

     st.dataframe(df.head())

     required_cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "pH",
        "sulphates",
        "alcohol"
    ]
     
     if all(col in df.columns for col in required_cols):
          
          
          X_batch = df[required_cols]
          X_scaled = scaler.transform(X_batch)
          predictions =model.predict(X_scaled)

          df["Predict Quality"] = predictions.round(2)

          st.write("Bacth Prediction Result")
          st.dataframe(df)

          csv = df.to_csv(index=False).encode("utf-8")
          st.download_button(
               label="Download Prediction as CSV",
               data=csv,
               file_name = "Wine_Quality_prediction.csv",
               mime = "text/csv"
          )
     else:
          st.error("Uploaded file is not contains required column")

          
                
