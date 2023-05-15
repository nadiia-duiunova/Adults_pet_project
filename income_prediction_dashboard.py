import dash
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State
from dash import html, dcc
import dash_bootstrap_components as dbc
import requests
import shap
from PIL import Image
import plotly.express as px


#load data from souce
data = pd.read_csv("./clean_data.csv")
X = data.drop(columns=['Income', 'Education-Num'])
countries = pd.read_csv("./country.csv")

#get Labels for Categorical features 
#presonal info
sex_options = data['Sex'].unique()
country_options = countries['value']  # should be substituted with all 196 countries
ethnic_group_options = data['Ethnic group'].unique()

#family
marital_satus_options = ['Married', 'Single']
relationschip_options = data['Relationship'].unique()

#create app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


#create simple layout
app.layout = html.Div(
    children = [
        html.Div(
            children = [
                html.H1("Let's check your income"),
                html.P("Hi dear visitor! I'm happy to have you here and welcome to my little project.")
            ],
            className='header'
        ),
        html.Div(
            className = 'body',
            children = [
                html.Div(
                    className='project_description',
                    children = [
                        html.P('Here I am gonna add a project description. For that I kindly ask you to answer the following questions:')
                    ]
                ),
                html.Div(
                    className='questionnaire',
                    children = [
                        html.Div(
                            className='col_left',
                            children = [
                                html.Div(
                                className='personal_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Personal Info'),
                                        html.Label("Select your age"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'age_input',
                                            min=18, max=99, step=1, #this dataset can only predict income for adult people
                                            marks={18: '18',99: '99'},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            value = 35,
                                            included=False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Sex"),
                                        dcc.RadioItems(
                                            className='radio',
                                            id = 'sex_input',
                                            options = sex_options,
                                            inline = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Country"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'country_input',
                                            options = country_options, 
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your Ethnic Group"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'ethnic_group_input',
                                            options = ethnic_group_options,
                                            clearable = False
                                        )
                                    ]
                                ),
                                html.Div(
                                    className = 'family_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Family Info'),
                                        html.Br(),
                                        html.Label("Select your Marital Status"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'marital_status_input',
                                            options = marital_satus_options,
                                            searchable=False,
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Do you live with your family?"),
                                        dcc.RadioItems(
                                            className='radio',
                                            id = 'relationschip_input',
                                            options = [{'label': 'Yes', 'value': 'Family'},
                                                       {'label': 'No', 'value': 'Not-in-Family'}
                                                    ],
                                            inline = False
                                        )
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className='col_right',
                            children = [
                                html.Div(
                                    className = 'work_education_info',
                                    children = [
                                        html.H3('Education and profesional Info'),
                                        html.Label("Select your education"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'education_input',
                                            min = 0, max = 3, step = 1,
                                            marks={
                                                0: {'label': 'Undergraduated'},
                                                1: {'label': 'High school'},
                                                2: {'label': 'Some college'},
                                                3: {'label': 'University degree'}
                                            },
                                            value = 2, #default value is 'HS-grad' as it's the most popular value
                                            included=False
                                        ),
                                        html.Br(),
                                        html.Label("How many hours per week do you work?"),
                                        dcc.Slider(
                                            className='slider',
                                            id = 'hpw_input',
                                            min=0, max=100, step=1,
                                            marks = {0: '0', 40: '40', 100: '100'},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                            value = 40, #default value is 40 as it's the most popular value
                                            included=False
                                        ), 
                                        html.Br(),
                                        html.Label("Select your workclass"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'workclass_input',
                                            options = [
                                                {'label': 'Private', 'value': 'Private'},
                                                {'label': 'Unincorporated self employment', 'value': 'Self-emp-not-inc'},
                                                {'label': 'Incorporated self employment', 'value': 'Self-emp-inc'},
                                                {'label': 'Local government', 'value': 'Local-gov'},
                                                {'label': 'State government', 'value': 'State-gov'},
                                                {'label': 'Federal government', 'value': 'Federal-gov'},
                                                {'label': 'Without pay', 'value': 'Without-pay'}
                                            ],
                                            searchable=False,
                                            clearable = False
                                        ),
                                        html.Br(),
                                        html.Label("Select your occupation"),
                                        dcc.Dropdown(
                                            className='dropdown',
                                            id = 'occupation_input',
                                            options = [
                                                {'label': 'Profesional specialty', 'value': 'Prof-specialty'},
                                                {'label': 'Executional manager', 'value': 'Exec-managerial'},
                                                {'label': 'Administrative, clerical', 'value': 'Adm-clerical'},
                                                {'label': 'Sales', 'value': 'Sales'},
                                                {'label': 'Machine inspection', 'value': 'Machine-op-inspct'},
                                                {'label': 'Craft repair', 'value': 'Craft-repair'},
                                                {'label': 'Transportation', 'value': 'Transport-moving'},
                                                {'label': 'Handler, creaner', 'value': 'Handlers-cleaners'},
                                                {'label': 'Farming, fishing', 'value': 'Farming-fishing'},
                                                {'label': 'Technical support', 'value': 'Tech-support'},
                                                {'label': 'Protective service', 'value': 'Protective-serv'},
                                                {'label': 'Armed forces', 'value': 'Armed-Forces'},
                                                {'label': 'Private house service', 'value': 'Priv-house-serv'},
                                                {'label': 'Other services', 'value': 'Other-service'}
                                            ],
                                            searchable=False,
                                            clearable = False
                                        )
                                    ]
                                ),
                                html.Div(
                                    className = 'capital_operations_info',
                                    children = [
                                        html.Br(),
                                        html.H3('Capital operations Info'),
                                        html.P('*please try to calculate total sum of your capital operations. If in total you have gained the capital, put the total sum to the first input. If in total you have lost - put this value to the second input (without minus)',
                                               className='capital_explanation'),
                                        html.Label("Gained capital, USD"),
                                        html.Br(),
                                        dcc.Input(
                                            className = 'num_input',
                                            id = 'capital_gain_input',
                                            type="number",
                                            min=0, step=1,
                                            value = 0, #default value is 0 as it's the most popular value
                                            debounce=True
                                        ), 
                                        html.Br(),
                                        html.Br(),
                                        html.Label("Lost capital, USD"),
                                        html.Br(),
                                        dcc.Input(
                                            className = 'num_input',
                                            id = 'capital_loss_input',
                                            type="number",
                                            min=0, step=1,
                                            value = 0, #default value is 0 as it's the most popular value
                                            debounce=True
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='submit-button',
                    children=[dbc.Button('Submit', id='submit-btn', size = 'lg', color ='success', n_clicks=0)]
                ),
                html.Div(
                    className = 'output',
                    children = [
                        html.H3(id='output_text'),
                        html.Div(
                            children = [html.H5(id='explanation_text')],
                            className='explanation_text'
                        ),
                        html.Div(id='output_graph')
                    ]
                )
            ]
        ),
        html.Div(
            className = 'footer',
            children = [
                html.P([
                    'Developed by Nadiia Duiunonova in 2023 based on Adults dataset from ', 
                    html.A(children = ['USI'], href = 'https://archive.ics.uci.edu/ml/datasets/Adult', target="_blank", className='link')
                ]),
                dcc.Link(
                    className = 'link',
                    href='https://github.com/nadiia95/Adults_pet_project', 
                    children = [
                        html.Img(
                            alt = 'GitHub',
                            src="/assets/github.svg"
                        )
                    ]
                ),
                dcc.Link(
                    className = 'link',
                    href='https://www.linkedin.com/in/nadiia-duiunova/', 
                    children = [
                        html.Img(
                            alt = 'LinkedIn',
                            src="/assets/linkedin.png"
                        )
                    ]
                )
            ]
        )
    ]
)

@app.callback(Output('output_text', 'children'),
    Output('explanation_text', 'children'),
    Output('output_graph', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('age_input', 'value'),
    State('workclass_input', 'value'),
    State('education_input', 'value'),
    State('marital_status_input', 'value'),
    State('occupation_input', 'value'),
    State('relationschip_input', 'value'),
    State('ethnic_group_input', 'value'),
    State('sex_input', 'value'),
    State('country_input', 'value'),
    State('capital_gain_input', 'value'),
    State('capital_loss_input', 'value'),
    State('hpw_input', 'value')
)
def update_output_div(n_clicks, age_input, workclass_input, education_input, marital_satus_input, occupation_input, relationschip_input,
                      ethnic_group_input, sex_input, country_input, capital_gain_input, capital_loss_input, hpw_input):
    
    if n_clicks >0 :
        if workclass_input and marital_satus_input and occupation_input and relationschip_input and ethnic_group_input and sex_input and country_input:
            if workclass_input == 'Without-pay':
                return ['Unfortunatelly, we cannot predict your income if you do not earn money', '', '']
            else:
                response = requests.get(
                    'http://127.0.0.1:8000/predict_induvidual_income/',
                    params={
                        'age': age_input, 
                        'workclass': workclass_input, 
                        'education': education_input,
                        'marital_status': marital_satus_input,
                        'occupation': occupation_input,
                        'relationship': relationschip_input,
                        'ethnic_group': ethnic_group_input,
                        'sex': sex_input,
                        'country': country_input,
                        'capital_gain': capital_gain_input,
                        'capital_loss': capital_loss_input,
                        'hours_per_week': hpw_input
                    },
                    timeout=10
                )
                json_response = response.json()
                if json_response.get('result')=='<=50K':
                    income_group = '50K or below'
                    explanation = 'Why so? On the left part of the graph (red) you can find all the parameters, that interfere you from bigger money, to the right (blue) - those, which move you towards higher income. On the X axis tou can also find a so called Base line at the point of 0.75. When you get there, the prediction will turn to opposite direction'
                else:
                    income_group = 'above 50K'
                    explanation = 'Why so? First of all, congrats! \U0001F973 You are in the minority of people, who managed to do it over 50K, which is around 25%. On the left part of the graph (red) you can find all the parameters, that move you towards higher income, to the right (blue) - those, which'
                probability_of_group = round(float(json_response.get('prediction_prob')), 4)
            
                prediction_text = f'With the probability of {probability_of_group}% your income would be {income_group}.'

                img_path = json_response.get('pic_name')
                img = np.array(Image.open(img_path))

                fig = px.imshow(img, color_continuous_scale="gray")
                fig.update_layout(coloraxis_showscale=False)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)

                return prediction_text, explanation, dcc.Graph(figure=fig)
        else: 
            return ['Please fill all the fields and press Submit', '', '']
    else:
        return ['', '', '']


if __name__ == '__main__':
    app.run_server(debug=True)