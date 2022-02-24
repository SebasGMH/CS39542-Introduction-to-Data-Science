"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@hunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
"""
import pandas as pd

def make_insp_df(file_name):
    #read in csv and keep 9 colums
    df = pd.read_csv(file_name)
    df = df[['CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'SCORE', 'GRADE', 'NTA']]
    #drop rows with null in score
    df = df[ df['SCORE'].notnull() ]
    return df
 
def predict_grade(num_violations):
    grade = ''
    if (num_violations < 14):
        grade = 'A'
    elif(num_violations < 28):
        grade = 'B'
    else:
        grade ='C'
    return grade

def grade2num(grade):
    num_grade= None
    if (grade=='A'):
        num_grade=4.0
    elif (grade == 'B'):
        num_grade=3.0
    elif (grade == 'C'):
        num_grade=2.0
    return num_grade

def make_nta_df(file_name):
     #read in csv and keep 2 colums
    df = pd.read_csv(file_name)
    df = df[['NTACode', 'NTAName']]
    return df

def compute_ave_grade(df,col):
    return

def neighborhood_grades(ave_df,nta_df):
    return