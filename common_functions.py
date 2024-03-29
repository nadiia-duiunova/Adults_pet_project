import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import scipy.sparse as sps

def get_clean_data(url: str, drop_columns: list) -> pd.DataFrame:
    """ Downloads data from url and removes selected columns. Also removes all spaces before values of categorical featues
    
    Parameters
    ----------
        url: str
            Link to download data
        drop_columns: list
            List of columns, that have to be dropped from datadrame
    Returns
    -------
        df: pd.DataFrame: 
            Dataframe with initial data cleaning, including removement of missing data and spaces at the beginning of each categorical value.
    """

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

    df = pd.read_csv(url, header=None, names=adult_columns).apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.replace(to_replace= '?', value = np.nan)
    df = df.dropna(how='any').reset_index(drop=True)
    df = df.drop(columns=drop_columns)
    
    return df



def preprocess_data(data: pd.DataFrame, 
                    TARGET: str,
                    numerical_features_list: list, 
                    categorical_features_list: list,  
                    ordinal_feature: str = '', order_of_categories: list = []
                    ) -> pd.DataFrame:
    """Transform the data according to it's format in order to feed it to the model.
    
    Parameters
    ----------
        data : pdandas.DataFrame 
            Dataframe with variables in columns and instances in rows, where data is represented in original data types.
        TARGET : str
            Name of target variable
        numerical_features_list : list
            List of features, that have numerical format in original dataframe
        categorical_features_list : list
            List of features, that are represented as categories in original dataframe
        ordinal_feature: str
            This function can precess only 1 ordinal feature, will be optimized in future
        order_of_categories: list
            Here you have to provide the right ascending order of values of the ordinal feature as a list 
        
    !_NOTE_!
    --------
    If you process you ordinal feature separately, you will have to add it manually to the result of this function!!!

    Returns
    -------
        preprocessed_data : pandas.DataFrame
            Preprocessed data, ready to be fed to the model
    """

    X = data.drop(columns=[TARGET])
    y = list(data[TARGET])

    if ordinal_feature != '':
        if not order_of_categories:
            raise ValueError('order_of_categories cannot be empty')
        if len(order_of_categories) != len(data[ordinal_feature].unique()):
            raise ValueError('incorrect number of categories in order_of_categories')
        if numerical_features_list:
            if categorical_features_list:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('stand scaler', StandardScaler(), numerical_features_list),
                    ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                    remainder='drop')
            else:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('stand scaler', StandardScaler(), numerical_features_list)],
                    remainder='drop')
        else:
            if categorical_features_list:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature)),
                    ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                    remainder='drop')
            else:
                columntransformer = ColumnTransformer(transformers = [
                    ('ordinal', OrdinalEncoder(categories=[order_of_categories]),
                                            make_column_selector(pattern = ordinal_feature))],
                    remainder='drop')
    elif numerical_features_list:
        if categorical_features_list:
            columntransformer = ColumnTransformer(transformers = [
                ('stand scaler', StandardScaler(), numerical_features_list),
                ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
                remainder='drop')
        else:
            columntransformer = ColumnTransformer(transformers = [
            ('stand scaler', StandardScaler(), numerical_features_list)],
            remainder='drop')
    else:
        columntransformer = ColumnTransformer(transformers = [
            ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
            remainder='drop')

    X_trans = columntransformer.fit_transform(X)

    if sps.issparse(X_trans):
        X_trans = X_trans.toarray()

    x_columns_names = columntransformer.get_feature_names_out()
    X_trans = pd.DataFrame(X_trans, columns = x_columns_names)

    y_trans = pd.DataFrame(data = y, index=range(0, len(y)), columns=[TARGET])

    # for categorical target create a dictionary for substituting every category with a number and apply to target
    if all(isinstance(n, str) for n in y_trans[TARGET]):
        n_unique = len(y_trans[TARGET].unique())
        dict_of_values = {}
        for i in range(n_unique):
            key = y_trans[TARGET].unique()[i]
            dict_of_values[key] = i
            
        y_trans[TARGET] = y_trans[TARGET].replace(dict_of_values)

    # for numerical target - apply StandardScaler()
    else: 
        scaler = StandardScaler()
        y_trans[TARGET] = scaler.fit_transform(y_trans)

    preprocessed_data = pd.merge(left=y_trans, right=X_trans, left_index=True, right_index=True)


    return preprocessed_data

def cluster_education(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster Education values into 4 categories: Undergraduated, High school graduated, Some college and Above graduated

    Parameters
    -----------
        df: pd.DataFrame 
            Initial dataframe with original data in Education column

    Returns
    --------
        df: pd.DataFrame
            The same dataframe, as was inputed, but with clustered Education values
    """
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

    return df

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
    data['Workclass'] = data['Workclass'].replace({'Never-worked': 'Without-pay'})

    # cluster Marital status
    data['Marital Status'] = np.where(data['Marital Status'].isin(['Married-AF-spouse', 'Married-civ-spouse']), 'Married', 'Single')

    # cluster Relationship
    data['Relationship'] = np.where(data['Relationship'].isin(['Husband', 'Wife', 'Own-child']), 'Family', 'Not-in-Family')

    # cluster Countries
    data['Country'] = np.where(data['Country'].isin(['Hungary', 'Greece', 'Portugal', 'Poland', 'Holand-Netherlands', 'Scotland', 'Italy', 
                                                     'England', 'Ireland', 'Germany', 'Hong', 'France', 'Japan', 'Canada', 'United-States']
                                                    ), 'Developed', 'Developing')

    return data