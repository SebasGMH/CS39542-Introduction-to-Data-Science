"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Program 8 Yellow taxi data
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import date

def import_data(file_name) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df = df[df['total_amount']>0]
    return df

# df = import_data('Yellow_taxi.csv')
# print(df)
def add_tip_time_features(df) -> pd.DataFrame:
    #3 new col
    percent_tip_data= 100*df['tip_amount']/(df['total_amount']-df['tip_amount'])
    df['percent_tip']= percent_tip_data
    #convert col to datetime
    dropoff= pd.to_datetime(df['tpep_dropoff_datetime'])
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])
    duration_data = dropoff-pickup
    df['duration']= duration_data
    dayofweek_data= pickup.weekday()
    df['dayofweek'] = dayofweek_data
    return df

def impute_numeric_cols(df, x_num_cols) -> pd.DataFrame:
    return

def transform_numeric_cols(df_num, degree_num=2) -> pd.DataFrame:
    return

def fit_linear_regression(x_train, y_train):
    return

def predict_using_trained_model(mod, x, y):
    return