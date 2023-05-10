from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from common_functions import cluster_categorical, cluster_education
from joblib import load

app = FastAPI()

# read precleaned data
data = pd.read_csv("./clean_data.csv")
TARGET = 'Income'

# remove examples that do not earn
data = data[~data['Workclass'].isin(['Never-worked', 'Without-pay'])]

# cluster Education values to 4 categories:
data = cluster_education(data)

# cluster Workclass, Marital status, Relationship and Country values
data = cluster_categorical(data)

# separate dataset to features and target
X = data.drop(columns=[TARGET, 'Education-Num'])
y = data[TARGET]

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.2)

# create list of developed countries
countries = pd.read_csv("./country.csv")
developed_countries = []
for i in range(countries.shape[0]):
    if countries['developed'][i]:
        developed_countries.append(countries['value'][i])

# load pretrained model
model = load('rand_forest_full.joblib')

# define target function
def predict_income(age: int, workclass:str, education: str, marital_status: str, occupation: str, relationship: str, 
                   ethnic_group: str, sex: str, country: str, capital_gain: int = 0, capital_loss: int = 0, hours_per_week: int = 40):
    """_summary_

    Args:
        age: int
            user's age
        workclass: str
            user's workclass
        education: str
            user's education
        marital_status: str 
            user's marital status
        occupation: str 
            user's work occupation
        relationship: str 
            whether or not user belongs to the family
        ethnic_group: str 
            user's ethnic group (as initialy defined in dataset)
        sex: str
            user's sex
        country: str 
            user's country of living and working
        capital_gain: int, optional 
            positive number, set if sum of all operations with capital led to profit. Defaults to 0.
        capital_loss: int, optional 
            positive number, set if sum of all operations with capital led to los. Defaults to 0.
        hours_per_week: int, optional 
            user's working hours per week. Defaults to 40.

    Returns:
        dict 
            Dictionary with 2 values: result, containing the predicted class of income; and probability of this prediction to be true, always > 50% 
    """
    
    model.fit(X_train, y_train)

    # transform data from user input to the same clusters as in train data
    if country in developed_countries:
        country = 'Developed'
    else:
        country = 'Developing'
    
    # create dataframe with user's data
    X_predictable = pd.DataFrame([[age, workclass, education, marital_status, occupation, relationship, ethnic_group, 
                                   sex, capital_gain, capital_loss, hours_per_week, country]], columns=X_train.columns)
    
    # predict income class
    income_prediction = model.predict(X_predictable)[0]

    # get probability of that prediction
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