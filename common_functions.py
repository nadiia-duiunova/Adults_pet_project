import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
import scipy.sparse as sps

def preprocess_data(dataframe: pd.DataFrame, numerical_features_list: list, categorical_features_list: list):
    X = dataframe.drop(['Income'], axis = 'columns')
    y = dataframe["Income"]

    columntransformer = ColumnTransformer(transformers = [
    ('ordinal', OrdinalEncoder(categories=[[' Preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',' 11th',
                                        ' 12th',' HS-grad',' Some-college',' Assoc-voc',' Assoc-acdm', 
                                        ' Bachelors',' Masters',' Prof-school',' Doctorate']]),
                                make_column_selector(pattern = 'Education')),
    ('stand scaler', StandardScaler(), numerical_features_list),
    ('onehot', OneHotEncoder(dtype='int', drop='first'), categorical_features_list)],
    remainder='drop')

    X_trans = columntransformer.fit_transform(X)

    if sps.issparse(X_trans):
        X_trans = X_trans.toarray()

    x_columns_names = columntransformer.get_feature_names_out()
    X_trans = pd.DataFrame(X_trans, columns = x_columns_names)

    y_train_df = pd.DataFrame(y)
    onehot = OneHotEncoder(dtype='int', drop='first')
    y_trans = onehot.fit_transform(y_train_df)
    y_column_name = onehot.get_feature_names_out()
    y_trans = pd.DataFrame.sparse.from_spmatrix(y_trans, columns=y_column_name)

    new_data = pd.merge(left=y_trans, right=X_trans, left_index=True, right_index=True)

    return new_data

def cluster_categorical(data):

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

