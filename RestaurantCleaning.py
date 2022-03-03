"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@hunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
"""
import pandas as pd
import re
#program 4

def make_insp_df(file_name):
    df = pd.read_csv(file_name)
    df = df[['CAMIS', 'DBA', 'BORO', 'PHONE', 'CUISINE DESCRIPTION', 'INSPECTION DATE', 'RECORD DATE', 'GRADE']]
    df = df[ df['GRADE'].notnull() ]
    return df
    
def clean_phone(phone_str):
    in_check = r'[0-9]{10}'
    if not re.search(in_check,phone_str):
        return None
    return

def convert_dates(df):
    return

def insp_day_of_week(insp):
    return

def days_since(insp_date, record_date):
    return

def group_df(df,categories=['INSP DAY','BORO']): 
    return