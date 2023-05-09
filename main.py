from fastapi import FastAPI
import pandas as pd
import numpy as np
from common_functions import get_clean_data, cluster_categorical
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from joblib import load

app = FastAPI()

data = get_clean_data()
TARGET = 'Income'

data = data[~data['Workclass'].isin(['Never-worked', 'Without-pay'])]

def cluster_education(df):
    df.loc[
        lambda x: x["Education-Num"].between(0, 8, "both"), "Education"
    ] = "Under-grad"

    df.loc[
        lambda x: x["Education-Num"] == 9, "Education"
    ] = "HS-grad"

    df.loc[
        lambda x: x["Education-Num"] == 10, "Education"
    ] = "Some-college"

    df.loc[
        lambda x: x["Education-Num"].between(11, 16, 'both'), "Education"
    ] = "Above-grad"

    scale_mapper = {'Under-grad':0, 'Some-college':1, 'HS-grad':2, 'Above-grad':3}
    df["Education"] = df["Education"].replace(scale_mapper)

cluster_education(data)
data = cluster_categorical(data)
data = data.drop(columns=['Education-Num'])

X = data.drop(columns=[TARGET])
y = data[TARGET]

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)

model = load('rand_forest_full.joblib')

def predict_income(age: int, workclass:str, education: str, marital_status: str, occupation: str, relationship: str, 
                   ethnic_group: str, sex: str, country: str, capital_gain: int = 0, capital_loss: int = 0, hours_per_week: int = 40):
    model.fit(X_train, y_train)

    # transform data from user input to the same clusters as train data
    # education
    # marital status
    # relationship
    # country
    

    X_predictable = pd.DataFrame([[age, workclass, education, marital_status, occupation, relationship, ethnic_group, sex, capital_gain, capital_loss, hours_per_week, country]], columns=X_train.columns)
    income_prediction = model.predict(X_predictable)[0]
    prediction_prob = model.predict_proba(X_predictable)

    if income_prediction == '>50K':
        prediction_prob = prediction_prob[0][1]
    else:
        prediction_prob = prediction_prob[0][0]

    return {'result': income_prediction,
            'prediction_prob': float(prediction_prob)*100}




@app.get("/predict_induvidual_income/")
def predict_income_result(age: int, workclass:str, education: str, marital_status: str, occupation: str, relationship: str, 
                   ethnic_group: str, sex: str, country: str, capital_gain: int = 0, capital_loss: int = 0, hours_per_week: int = 40):
    return predict_income(age, workclass, education, marital_status, occupation, relationship, ethnic_group, sex, country, capital_gain, capital_loss, hours_per_week)