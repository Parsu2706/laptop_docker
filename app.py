import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ---------------- STREAMLIT APP ---------------- #
if __name__ == "__main__":
    st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

    st.title("🍷 Wine Quality Prediction Dashboard")

    # Load dataset
    df = pd.read_csv("winequality-red.csv")

    if df is not None:
        st.success("✅ Dataset Loaded Successfully")
    else:
        st.error("❌ Failed to load dataset")

    # Define features
    cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"]

    X = df[cols]
    y = df["quality"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Show metrics
    y_pred = model.predict(X_test)
    st.subheader("📊 Model Performance on Test Data")
    st.write("R² Score:", round(r2_score(y_test, y_pred), 3))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 3))

    st.subheader("🔧 Try Your Own Wine Features")

    # User input sliders
    user_input = {}
    for col in cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.slider(col, min_val, max_val, mean_val)

    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])

    # Predict
    prediction = model.predict(input_df)[0]
    st.subheader("🍇 Predicted Wine Quality:")
    st.success(f"{prediction:.2f} (out of 10)")
