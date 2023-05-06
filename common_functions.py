import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import scipy.sparse as sps

def get_data():
    adult_columns = [
        "Age",
        "Workclass",
        "final weight",
        "Education",
        "Education-Num",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Ethnic group",
        "Sex",
        "Capital Gain",
        "Capital Loss",
        "Hours per week",
        "Country",
        "Income",
    ]

    df = pd.read_csv("adult.data", header=None, names=adult_columns)
    df = df.replace(to_replace= ' ?', value = np.nan)

    TARGET = 'Income'

    X = df[['Age', 'Hours per week', 'Education', 'Capital Gain', 'Capital Loss']].copy()
    y = pd.DataFrame(df[TARGET])

    return X, y



def preprocess_data(data: pd.DataFrame, numerical_features_list: list, categorical_features_list: list, 
                    TARGET: str = 'Income', education: bool = True) -> pd.DataFrame:
    """Transform the data according to it's original format in order to feed it to the model.
    Parameters
    ----------
        data : pdandas.DataFrame 
            Dataframe with variables in columns and instances in rows, where data is represented in original data types.
        target : str
            Name of target variable
        numerical_features_list : list
            List of features, that have numerical format in original dataframe
        categorical_features_list : list
            List of features, that are represented as categories in original dataframe
    Returns
    -------
        preprocessed_data : pandas.DataFrame
            preprocessed data, ready to be fed to the model
    """
    X = data.drop(columns=[TARGET])
    y = list(data[TARGET])

    if education:
        columntransformer = ColumnTransformer(transformers = [
            ('ordinal', OrdinalEncoder(categories=[[' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',
                                            ' 12th',' HS-grad',' Some-college',' Assoc-voc',' Assoc-acdm', 
                                            ' Bachelors',' Masters',' Prof-school',' Doctorate']]),
                                    make_column_selector(pattern = 'Education')),
            ('stand scaler', StandardScaler(), numerical_features_list),
            ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
            remainder='drop')
    else:
        columntransformer = ColumnTransformer(transformers = [
            ('stand scaler', StandardScaler(), numerical_features_list),
            ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
            remainder='drop')

    X_trans = columntransformer.fit_transform(X)

    if sps.issparse(X_trans):
        X_trans = X_trans.toarray()

    x_columns_names = columntransformer.get_feature_names_out()
    X_trans = pd.DataFrame(X_trans, columns = x_columns_names)

    if education == False:
        X_trans = pd.merge(left=X_trans, right=pd.DataFrame(data["Education"]), left_index=True, right_index=True)

    y_trans = pd.DataFrame(data = y, index=range(0, len(y)), columns=[TARGET])
    y_trans[TARGET] = y_trans[TARGET].replace({' <=50K':0, ' >50K':1})

    preprocessed_data = pd.merge(left=y_trans, right=X_trans, left_index=True, right_index=True)


    return preprocessed_data



def cluster_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Cluster those cutegories, that make sence being clustered, like clustering countries into developed and developing

    Parameters
    ----------
        data : pandas.DataFrame
            Original dataframe with variables in columns and instances in rows   

    Returns
    -------
        data : pandas.DataFrame
            The same dataframe, but with some categories or some features clustered together
    """

    # cluster Workclass
    data['Workclass'] = data['Workclass'].replace({' Never-worked': ' Without-pay'})

    # cluster Marital status
    data.loc[
        lambda x: x["Marital Status"].isin([' Widowed', ' Separated', ' Married-spouse-absent', ' Never-married', ' Divorced']), "Marital Status"
    ] = "Single"

    data.loc[
        lambda x: x["Marital Status"].isin([' Married-AF-spouse', ' Married-civ-spouse']), "Marital Status"
    ] = "Married"

     # cluster Relationship
    data.loc[
        lambda x: x["Relationship"].isin([' Husband', ' Wife', ' Own-child']), "Relationship"
    ] = "Family"

    data.loc[
        lambda x: x["Relationship"].isin([' Not-in-family', ' Unmarried', ' Other-relative']), "Relationship"
    ] = "Not-in-Family"

    # cluster Countries
    data.loc[
        lambda x: x["Country"].isin([' Holand-Netherlands', ' Scotland', ' Italy', ' England', ' Ireland', ' Germany', ' Hong',  ' France', ' Taiwan', 
                                    ' Japan', ' Puerto-Rico', ' Canada', ' United-States']), "Country"
    ] = "Developed"

    data.loc[
        lambda x: x["Country"].isin([' Hungary', ' Greece', ' Portugal', ' Poland', ' Yugoslavia', ' Cambodia', ' Iran',  ' Philippines', ' Laos', ' Thailand', ' Vietnam', ' South', 
                                    ' China', ' India', ' Honduras', ' Outlying-US(Guam-USVI-etc)', ' Trinadad&Tobago', ' Ecuador',  ' Philippines', ' Nicaragua',
                                    ' Peru', ' Haiti', ' Columbia', ' Guatemala', ' Dominican-Republic', ' Jamaica',  ' Cuba', ' El-Salvador', ' Mexico']), "Country"
    ] = "Developing"

    return data

