from fastapi import FastAPI
from joblib import load
import pandas as pd
import numpy as np
import matplotlib

import shap
from common_functions import cluster_categorical, cluster_education

matplotlib.use('agg')
app = FastAPI()

# read precleaned data
data = pd.read_csv("./clean_data.csv")
TARGET = 'Income'

# remove examples that do not earn money
data = data[~data['Workclass'].isin(['Never-worked', 'Without-pay'])]

# cluster Education values to 4 categories:
data = cluster_education(data)

# cluster Workclass, Marital status, Relationship and Country values
data = cluster_categorical(data)

# separate dataset to features and target
X = data.drop(columns=[TARGET, 'Education-Num'])
y = data[TARGET]

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
                   ethnic_group: str, sex: str, country: str, capital_gain: int, capital_loss: int, hours_per_week: int):
    """ Predict user's income based on given parametes

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
            positive number, set if sum of all operations with capital led to profit
        capital_loss: int, optional 
            positive number, set if sum of all operations with capital led to los
        hours_per_week: int, optional 
            user's working hours per week

    Returns:
        dict 
            Dictionary with 2 values: result, containing the predicted class of income; and probability of this prediction to be true, always > 50% 
    """
    
    # model.fit(X_train, y_train)
    model.fit(X, y)

    # transform data from user input to the same clusters as in train data
    if country in developed_countries:
        country = 'Developed'
    else:
        country = 'Developing'
    
    # create dataframe with user's data
    X_predictable = pd.DataFrame([[age, workclass, education, marital_status, occupation, relationship, ethnic_group, 
                                   sex, capital_gain, capital_loss, hours_per_week, country]], columns=X.columns)
    
    # predict income class
    income_prediction = model.predict(X_predictable)[0]
    prediction_prob = model.predict_proba(X_predictable)

    #explain result with shap values
    explainer = shap.TreeExplainer(model['randomforestclassifier'])
    data_point = model['columntransformer'].transform(X_predictable)
    x_columns_names = model['columntransformer'].get_feature_names_out()
    data_point = pd.DataFrame(data_point, columns = x_columns_names)

    shap_values = explainer.shap_values(data_point)

    if income_prediction == '<=50K':
        prediction_prob = prediction_prob[0][0]
        shap_values = shap_values[0]
        expected_value = explainer.expected_value[0]
    else:
        prediction_prob = prediction_prob[0][1]
        shap_values = shap_values[1]
        expected_value = explainer.expected_value[1]

    # inverse transformation to original columns
    n_categories = [1, 1, 1, 1, 5, 1, 13, 1, 4, 1, 1, 1]
    new_shap_values = []
    for values in shap_values:
        #split shap values into a list for each feature
        values_split = np.split(values , np.cumsum(n_categories))
        
        #sum values within each list
        values_sum = [sum(l) for l in values_split]
        
        new_shap_values.append(values_sum)

    new_shap_values = np.array([new_shap_values[0][:-1]], dtype=object)

    shap.force_plot(expected_value, new_shap_values, X_predictable, matplotlib=True, show=False)
    pic_name = "2pic_name.png"
    matplotlib.pyplot.savefig(pic_name)



    return {'result': income_prediction,
            'prediction_prob': float(prediction_prob)*100,
            'pic_name': pic_name
            }



@app.get("/predict_induvidual_income/")
def predict_income_result(age: int, workclass:str, education: str, marital_status: str, occupation: str, relationship: str, 
                   ethnic_group: str, sex: str, country: str, capital_gain: int, capital_loss: int, hours_per_week: int):
    return predict_income(age, workclass, education, marital_status, occupation, relationship, ethnic_group, sex, country, capital_gain, capital_loss, hours_per_week)

# @app.get('/predict_with_img')
# def predict_with_img():
#     return 