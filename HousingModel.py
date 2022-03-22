"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  skit learn documentation
Program 7: Housing model
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def make_df(housing_file, pop_file):
    house_df = pd.read_csv(housing_file)
    pop_df = pd.read_csv(pop_file)
    df = house_df.merge(pop_df, right_on='NTA Code', left_on='nta2010')
    df = df[df['total'].notnull()]
    df = df[df['Year']== 2010]
    df = df.drop(columns=['the_geom','boro','nta2010'])
    return df

# df = make_df('Housing_Database_by_NTA.csv','pop.csv')
# print(df)

def compute_lin_reg(x,y):
    reg = LinearRegression().fit(x, y)
    return reg.intercept_ , reg.coef_

def compute_boro_lr(df,xcol,ycol,boro=["All"]):
    if boro[0]=='ALL':
        return compute_lin_reg(df[xcol],df[ycol])
    df = df[df['Borough']==boro]
    return compute_lin_reg(df[xcol],df[ycol])

def MSE_loss(y_actual,y_estimate):
    MSE = np.square(np.subtract(y_actual,y_estimate)).mean()
    return MSE 

def RMSE(y_actual,y_estimate):
    return np.sqrt(MSE_loss(y_actual,y_estimate))

def compute_error(y_actual,y_estimate,loss_fnc):
    return loss_fnc(y_actual,y_estimate)