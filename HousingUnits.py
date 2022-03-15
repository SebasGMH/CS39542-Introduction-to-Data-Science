"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
Program 6: housing units
"""
import pandas as pd
def make_housing_df(file_name):
    df = pd.read_csv(file_name)
    df = df.dropna(subset=['total'])
    df= df.rename(columns={'nta2010':'NTA Code'})
    #print (df)
    return df
#make_housing_df('Housing_Database_by_NTA.csv')
def make_pop_df(file_name):
    df = pd.read_csv(file_name)
    df = df.loc[df['Year']==2010]
    return df

def combine_df(housing_df, pop_df, cols):
    df = housing_df.merge(pop_df, on='NTA Code')
    #df = df.loc[df[cols]]
    return df[cols]

def compute_density(df, col = 'Density'):
    col_vals = df['Population']/df['Shape__Area']
    df.insert(0, col,  col_vals)
    return df

def most_corr(df, y = 'total', xes = ['Population','Shape__Area','Density','comp2010']):
    greatest_corr = 0
    greatest_corr_col = ''
    for col in xes:
        if abs(df[col].corr(df[y])) > greatest_corr:
            greatest_corr = abs(df[col].corr(df[y],method='pearson'))
            greatest_corr_col = col
    return greatest_corr_col, greatest_corr

def convert_std_units(ser):
    s_mean = ser.mean()
    s_std  = ser.std()
    for index,value in ser.items():
        ser[index] = (value - s_mean)/(s_std)
    return ser