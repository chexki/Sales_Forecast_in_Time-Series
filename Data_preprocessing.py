# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:38:36 2019

@author: chetanjawlae
"""
# Sales Forecasting Data Processing

#%%
# Imports
import pandas as pd
import numpy as np
import re

# FUNCTION IMPORTS
from Data import get_data
#%%
df = get_data()

# AVERAGE UNIT PRICE For Month and Year
df.UnitPrice = df.UnitPrice.astype(float)
df.DiscountAmount = df.DiscountAmount.astype(float)
df.MarginAmount = df.MarginAmount.astype(float)

df['final_amount'] = df.UnitPrice - df.DiscountAmount  - df.MarginAmount

contract_period = df.ContractTo - df.ContractFrom
df['contract_period_days'] = contract_period.apply(lambda x: re.findall(r'.*(?: d)',str(x))[0][:-1])
df['contract_period_days'] = df['contract_period_days'].astype(int)
df['contract_period_years'] =round(df['contract_period_days']/ 365)

#Correlation = df.DiscountAmount.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
#%%
df.dtypes
# PRICE
#df['units'] = 
df.final_amount = df.final_amount.astype(float)
df['per_unit_price'] = df.final_amount / df.contract_period_years

def Inf_treat():
    per_unit_price = []
    for a,b in zip(df.final_amount ,df.per_unit_price):
        if b == np.inf:
            per_unit_price.append(a)
        else:
            per_unit_price.append(b)
    return per_unit_price

df['per_unit_price'] = Inf_treat()

#%%
df.Business.value_counts()
df.ProductGroupName.value_counts()

#%%
def pass_data():
    return df