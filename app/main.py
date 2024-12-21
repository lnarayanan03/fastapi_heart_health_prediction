from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import dill
app = FastAPI()

model_path = 'best_model_heart_prediction.pkl'

# Load the model using dill
with open('./app/best_model_heart_prediction.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


class Heart(BaseModel):                   
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

    

@app.get("/")
def read_root():
    return {
        "Name": "R Lakshmi Narayanan",
        "Project" : "EAS 501 Python Final Project",
        "Model": "Random Forest"
        }


# @app.get("/items/{item_id}")
# def read_item(
#     item_id: int, 
#     q: Union[str, None] = None,
#     x: Union[str, None] = None
#     ):
#     return {"item_id": item_id, "q": q, "x":x}

@app.post("/predict")
def predict(heart : Heart):
    df = pd.DataFrame([heart.model_dump().values()], columns = heart.model_dump().keys())
    y_hat = reloaded_model.predict(df)
    return {"prediction" : y_hat[0]}