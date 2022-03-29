"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Program 8 Yellow taxi data
"""
from numpy import NaN
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from datetime import date

def import_data(file_name) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df = df[df['total_amount']>0]
    return df

# df = import_data('Yellow_taxi.csv')
# print(df)
def add_tip_time_features(df) -> pd.DataFrame:
    #3 new col
    #broken lmao
    percent_tip_data= 100*df['tip_amount']/(df['total_amount']-df['tip_amount'])
    df['percent_tip']= percent_tip_data
    #convert col to datetime
    dropoff= pd.to_datetime(df['tpep_dropoff_datetime'])
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])
    duration_data = dropoff-pickup
    df['duration']= duration_data
    #dayofweek_data= pickup.date.weekday()
    df['dayofweek'] = duration_data
    return df

def impute_numeric_cols(df, x_num_cols) -> pd.DataFrame:
    #Missing data in the columns x_num_cols are replaced with the 
    # median of the column. Returns a DataFrame containing only the imputed numerical columns from input df
    for col in x_num_cols:
        df[col]=df[col].replace(NaN,df[col].median())

    return df[x_num_cols]

def transform_numeric_cols(df_num, degree_num=2) -> pd.DataFrame:
    transformer = PolynomialFeatures(degree=degree_num,include_bias=False)
    X = transformer.fit_transform(df_num)
    return X

def fit_linear_regression(x_train, y_train):
    reg = LinearRegression().fit(x_train, y_train)
    return reg.intercept_,reg.coef_,reg

def predict_using_trained_model(mod, x, y):
    x_modded = mod.predict(x)
    return mean_squared_error(y,x_modded),r2_score(y,x_modded)