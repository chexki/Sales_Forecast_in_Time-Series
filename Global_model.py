# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:51:34 2019

@author: chetanjawlae
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import date
import calendar
import statsmodels.api as sm
import itertools
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#%%
from Data_preprocessing import pass_data
df  = pass_data()
#%%

def Process(Df_fin):
    InvoiceDate1 = Df_fin['InvoiceDate'].dt.year.astype('str') + '-' + Df_fin['InvoiceDate'].dt.month.astype('str') + '-' + Df_fin['InvoiceDate'].dt.day.astype('str') 
    Df_fin['InvoiceDate'] = pd.to_datetime(InvoiceDate1)
    Df_fin = Df_fin.sort_values('InvoiceDate')
    Df_fin = Df_fin[['InvoiceDate','final_amount']]
    Df_fin['Dates'] = Df_fin['InvoiceDate'].dt.to_period('M')
    Df_fin['Dates'] = Df_fin['Dates'].apply(lambda x : str(x) +'-'+'01')
    Df_fin['Dates'] = pd.to_datetime(Df_fin['Dates'])
    Df_fin = Df_fin.groupby('Dates')['final_amount'].sum().reset_index()
    Df_fin = Df_fin.set_index('Dates')
    return Df_fin

#%%
def forecast(Df_fin):
    y = Df_fin['final_amount'].resample('MS').mean()
    y =y.fillna(method='ffill')
# =============================================================================
#     y['2016':]
#     y.plot(figsize=(8, 5),linewidth=2.0)
#     plt.show()
#     from pylab import rcParams
#     Df_fin['figure.figsize'] = 18, 8
# =============================================================================
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# =============================================================================
#     fig = decomposition.plot()
#     plt.show() 
# =============================================================================
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    best_param = pd.DataFrame()
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                best_param1 = pd.DataFrame({'param': str(param),
                              'param_seasonal': str(param_seasonal),
                              'aic': results.aic},index = [0])
    
                best_param = best_param.append(best_param1)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    tuned_param  = best_param[best_param['aic']==best_param.aic.min()]
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order= eval(tuned_param.iloc[-1, 0]),
                                    seasonal_order= eval(tuned_param.iloc[-1, 1]),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print('SARIMAX')
    print(results.summary().tables[1])
# =============================================================================
#     results.plot_diagnostics(figsize=(16, 8))
#     plt.show()
# =============================================================================
    pred = results.get_prediction(start=pd.to_datetime('2019-04-01'), dynamic=False)
    pred_ci = pred.conf_int()
    
# =============================================================================
#     ax = y['2018':].plot(label='observed')
#     pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.6, figsize=(14, 7))
#     
#     ax.fill_between(pred_ci.index,
#                     pred_ci.iloc[:, 0],
#                     pred_ci.iloc[:, 1], color='k', alpha=.2)
#     
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Df_fin Final_amount')
#     plt.legend()
#     plt.show()
# =============================================================================

    #ERROR
    y_forecasted = pred.predicted_mean
    y_truth = y['2019-04-01':]
    
    # Compute the mean square error
    Df_fine = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of our forecasts is {}'.format(round(Df_fine, 2)))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(Df_fine), 2)))

    pred_uc = results.get_forecast(steps=6)
    pred_ci = pred_uc.conf_int()
    
# =============================================================================
#     ax = y.plot(label='observed', figsize=(14, 7))
#     pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
#     ax.fill_between(pred_ci.index,
#                     pred_ci.iloc[:, 0],
#                     pred_ci.iloc[:, 1], color='k', alpha=.25)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Df_fin Sales')
#     plt.legend()
#     plt.show()
# =============================================================================
    
    Date_list = y_truth.reset_index()['Dates'].tolist()
    Dates_all = Date_list + pred_ci.reset_index()['index'].tolist()
    list1 = [0] * len(Date_list)
    list2 = [0] * len(pred_ci)
    actual = y_truth.reset_index()['final_amount']
    lower_forecasted = list1 + pred_ci.reset_index()['lower final_amount'].tolist()
    upper_forecasted = list1 + pred_ci.reset_index()['upper final_amount'].tolist()
    mean_forecasted = pred.predicted_mean.tolist() + ((pred_ci.reset_index()['lower final_amount'] + pred_ci.reset_index()['upper final_amount']) /2).tolist()
    perc_error = np.array((actual/ mean_forecasted[:len(Date_list)])*100).tolist()
    actual_fin = actual.tolist() + list2
    perc_error_fin = perc_error + list2
    
    Results  = pd.DataFrame({'Dates':Dates_all,
                             'Actual Sales':actual_fin,
                              'Predicted Sales':mean_forecasted,
                              'Lower Predicted':lower_forecasted,
                              'Upper Predicted':upper_forecasted,
                              'Percent precision':perc_error_fin})
        
    return Results
#%%
# Current month range
on_date = pd.to_datetime(str(dt.datetime.now().year) + '-' + str(dt.datetime.now().month) + '-' + '01' + ' 00:00:00')
off_date = pd.to_datetime(dt.datetime(on_date.year,on_date.month,1) + dt.timedelta(days=calendar.monthrange(on_date.year,on_date.month)[1] - 1))

#%%
# =============================================================================
# FOR UNIVARIATE And NEW 
    
Products_total = ['List of Products']
flagship = ['List of Flagship Products']

####################################################################################################################################
forecast_df_new_flagship = pd.DataFrame()
# Actual

df_sub = df.loc[df['ProductGroupName'] == 'Some_Product']
Df_fin = Process(df_sub)
to_main = forecast(Df_fin)
to_main['Product'] = 'Some_Product'
forecast_df_new_flagship = forecast_df_new_flagship.append(to_main)

# Accuracy 
Accuracy_New_all = int(forecast_df_new_flagship[forecast_df_new_flagship['Percent precision'] != 0]['Percent precision'].sum() / forecast_df_new_flagship[forecast_df_new_flagship['Percent precision'] != 0].shape[0])
print(Accuracy_New_all)
Accuracy_New_current = int(forecast_df_new_flagship[forecast_df_new_flagship['Dates'] == on_date]['Percent precision'].sum() /forecast_df_new_flagship[forecast_df_new_flagship['Dates'] == on_date].shape[0])
print(Accuracy_New_current)


df_sub = df.loc[df['ProductGroupName'].isin(Products_total)]
Df_fin = Process(df_sub)
to_main = forecast(Df_fin)
to_main['Product'] = 'Products_Total'
forecast_df_new_flagship = forecast_df_new_flagship.append(to_main)

#####################################################################################################################################
#%%