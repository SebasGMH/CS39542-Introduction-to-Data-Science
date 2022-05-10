"""
Name:  Giuliani Martinez Herrera
Email: giuliani.martinezherrer04@myhunter.cuny.edu
Resources:  Used python.org as a reminder of Python 3 print statements.
P13: EMS Queries
"""
import pandas  as pd
import pandasql as psql

"""
    :param: csv file
    :return: df with removed null values
"""
def make_df(file_name):
    df = pd.read_csv(file_name)
    #Drop rows with nulls in type description, incident date, incident time, borough name 
    df = df.dropna(subset=['TYP_DESC','INCIDENT_DATE','INCIDENT_TIME','BORO_NM'])
    return(df)

"""
    :param: formatted df
    :return: Selects and returns the BORO_NM from df
"""
def select_boro_column(df):
    q = 'select BORO_NM from df'
    return psql.sqldf(q)

"""
    :param: formated df , unformated boro name
    :return:Return all rows from the DataFrame, df, where the borough is boro_name
"""
def select_by_boro(df, boro_name):
    boro_name = boro_name.upper()
    q = f'select * from df where BORO_NM = "{boro_name}"'
    return psql.sqldf(q)

"""
    :param:formated df, unformated boro name
    :return:Returns the number of incidents from df, called in on New Year's Day (Jan 1, 2021) in the specified borough, boro_name
"""
def new_years_count(df, boro_name):
    boro_name = boro_name.upper()
    q = f'select COUNT(*) from df where INCIDENT_DATE = "01/01/2021" and BORO_NM = "{boro_name}"'
    return psql.sqldf(q)

"""
    :param:formated df
    :return:Returns the incident counts per radio code (TYP_DESC), sorted alphabetically by radio code (TYP_DESC)
"""
def incident_counts(df):
    q = 'select TYP_DESC,COUNT(*) from df group by TYP_DESC order by TYP_DESC ASC'
    return psql.sqldf(q)

"""
    :param:formated df, unformated boror name 
    :return:Returns the top 10 most commonly occurring incidence by radio code, and the number of incident occurrences, in specified borough.
"""
def top_10(df, boro_name):
    boro_name = boro_name.upper()
    q = f'select TYP_DESC,COUNT(*) from df where BORO_NM = "{boro_name}" group by TYP_DESC order by count(*) desc limit 10'
    return psql.sqldf(q)