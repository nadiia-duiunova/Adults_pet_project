from fastapi import FastAPI
import pandas as pd
import numpy as np
from common_functions import get_data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from joblib import load

app = FastAPI()

X, y = get_data()

oe = OrdinalEncoder(categories=[[' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',
                                            ' 12th',' HS-grad',' Some-college',' Assoc-voc',' Assoc-acdm', 
                                            ' Bachelors',' Masters',' Prof-school',' Doctorate']], dtype = int)
X['Education'] = oe.fit_transform(X[['Education']])

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.2)

model = load('rand_forest_shorten.joblib')

def predict_income(age: int, hours_per_week: int, education: str, capital_gain: int = 0, capital_loss: int = 0):
    model.fit(X_train, y_train)

    if education ==' Preschool':
        education_int = 0
    elif education == ' 1st-4th':
        education_int = 1
    elif education == ' 5th-6th':
        education_int = 2
    elif education == ' 7st-8th':
        education_int = 3
    elif education == ' 9th':
        education_int = 4
    elif education == ' 10th':
        education_int = 5
    elif education == ' 11th':
        education_int = 6
    elif education == ' 12th':
        education_int = 7
    elif education == ' HS-grad':
        education_int = 8
    elif education == ' Some-college':
        education_int = 9
    elif education == ' Assoc-voc':
        education_int = 10
    elif education == ' Assoc-acdm':
        education_int = 11
    elif education == ' Bachelors':
        education_int = 12
    elif education == ' Masters':
        education_int = 13
    elif education == ' Prof-school':
        education_int = 14
    else:
        education_int = 15

    X_predictable = [[age, hours_per_week, education_int, capital_gain, capital_loss]]
    income_prediction = model.predict(X_predictable)[0]
    prob_high_income = round(model.predict_proba(X_predictable)[0][1], 4)

    return {'result': income_prediction,
            'high_inc_prob': float(prob_high_income)*100}




@app.get("/predict_induvidual_income/")
def predict_income_result(age: int, hours_per_week: int, education: str, capital_gain: int = 0, capital_loss: int = 0):
    return predict_income(age, hours_per_week, education, capital_gain, capital_loss)