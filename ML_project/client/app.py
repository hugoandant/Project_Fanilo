import streamlit as st
import requests

# Streamlit UI for user input
st.title("Random forest iris model")

# User input for feature values
sepal_length = st.slider("Sepal Length:", 0.0, 10.0)
sepal_width = st.slider("Sepal Width:", 0.0, 10.0)
petal_length = st.slider("Petal Length:", 0.0, 10.0)
petal_width = st.slider("Petal Width:", 0.0, 10.0)

# Button to make prediction
if st.button("Predict"):
    # Prepare input data
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    # Make HTTP request to FastAPI server
    response = requests.post("http://server:8000/predict", json=data)

    # Display prediction result
    if response.status_code == 200:
        prediction = response.json()["predicted_class"]
        st.success(f"Predicted Class: {prediction}")
    else:
        st.error(f"Error: {response.text}")