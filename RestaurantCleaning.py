"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
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
    
    #wanted this form to work
    #formated_phone='(012)-345-6789'
    #for n in range(9):
    #    num_replace = r'[string(n)]'
    #    formated_phone=re.sub(num_replace,phone_str[n],formated_phone)
    
    #ugly solution
    formated_phone = '('+phone_str[0]+phone_str[1]+phone_str[2]+')-'+phone_str[3]+phone_str[4]+phone_str[5]+'-'+phone_str[6]+phone_str[7]+phone_str[8]+phone_str[9]
    return formated_phone

def convert_dates(df):
    date_format= '%m/%d/%Y'
    df['INSPECTION DATE'] = pd.to_datetime(df['INSPECTION DATE'], format=date_format)
    df['RECORD DATE'] = pd.to_datetime(df['RECORD DATE'], format=date_format)
    return df

def insp_day_of_week(insp):
    if insp.year == 1900 and insp.month == 1 and insp.day== 1:
        return None
    return insp.weekday()

def days_since(insp_date, record_date):
    if insp_date.year == 1900 and insp_date.month == 1 and insp_date.day== 1:
        return None
    
    time_delta = (record_date - insp_date)
    total_seconds = time_delta.total_seconds()
    days = total_seconds/86400
    return int(days)
    
    #return (record_date - insp_date).Days

def group_df(df,categories=['INSP DAY','BORO']): 
    return (df.groupby(categories).size()).to_frame()