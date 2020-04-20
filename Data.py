# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:38:36 2019
@author: Chexki
"""
# Sales Data Collection
#%%
import pyodbc
import pandas as pd
import sys
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf8')

#%%
def get_data():
    querystring_new = """ SQL QUERY HERE"""
    
    outputfile = 'output_sales_data.csv'
    
    conn = pyodbc.connect('DRIVER=SQL Server;SERVER=,PORT\\SQLExpress;DATABASE=;UID=;PWD=;TDS_Version=7.3;')
    cursor = conn.cursor()
    rows = cursor.execute(querystring_new).fetchall()
    col_names = [i[0] for i in cursor.description]
    
    df = pd.DataFrame((tuple(t) for t in rows)) 
    df.columns = col_names
    df = df.loc[:,~df.columns.duplicated()]
    cursor.close()
    conn.close()
    
    print('Query processed successfully.')
    return df
