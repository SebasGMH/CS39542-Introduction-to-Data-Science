"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Program 8 Yellow taxi data
Program 8: Logistic Taxis
"""
from numpy import NaN, true_divide
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datetime import date
import pickle

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
    df['duration']= (dropoff-pickup).dt.total_seconds()
    #dayofweek_data= pickup.date.weekday()
    df['dayofweek'] = pickup.dt.weekday
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

def add_boro(df, file_name) -> pd.DataFrame:
    df_=pd.read_csv(file_name,low_memory=False)
    #make dictionary with id and boroughs. make a list of the borughs correspondiung to pu and do
    # borough_dict=dict(zip(df_.LocationID, df_.borough))
    # df['PU_borough']=df['PULocationID'].map(borough_dict)
    # df['DO_borough']=df['DOLocationID'].map(borough_dict)
    # df=df.reset_index(drop=True)
    #using merge
    df_ = df_['LocationID','borough']
    df_.rename(columns={'LocationID':'PULocationID'})
    #do two merges for pu and do 
    df = df.merge(df_,how='left',on='PULocationID')
    df.rename(columns={'borough':'PU_borough'})
    df_.rename(columns={'PULocationID':'DOLocationID'})
    df = df.merge(df_,how='left',on='DOLocationID')
    df.rename(columns={'borough':'DO_borough'})
    return df

def add_flags(df) -> pd.DataFrame:
    #will attempt with brute force
    df.insert(0,'paid_toll',0)
    df.insert(0,'cross_boro',0)
    df['paid_toll'].loc[df['PU_borough'] != df['DO_borough']] = 1
    df['cross_boro'].loc[df['PU_borough'] != df['DO_borough']] = 1
    return df

def encode_categorical_col(col,prefix) -> pd.DataFrame:
    df = pd.get_dummies(col,prefix=prefix,columns=col,prefix_sep='')
    df = df.iloc[: , :-1]
    return df

def split_test_train(df, xes_col_names, y_col_name, test_size=0.25, random_state=12345):
    X_train, X_test, y_train, y_test = train_test_split(df[xes_col_names], df[y_col_name], test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def fit_logistic_regression(x_train, y_train,penalty='none',max_iter=1000) -> object:
    clf = LogisticRegression(penalty=penalty,solver = 'saga',max_iter=max_iter).fit(x_train, y_train)
    picklestring = pickle.dumps(clf)
    return picklestring

def predict_using_trained_model(mod_pkl, x, y):
    x_modded = mod_pkl.predict(x)
    return mean_squared_error(y,x_modded),r2_score(y,x_modded)