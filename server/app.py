from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



@app.post("/predict")
def predict(item: Item):
    # Convert input to numpy array for prediction
    features = np.array([[
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
    ]])

    # Make the prediction
    prediction = model.predict(features)

    # Return the predicted class
    return {"predicted_class": int(prediction[0])}

