
import pandas as pd
import numpy as np
from model.clean_data import date_asof


gateways = ['PSC', 'PTC', 'FDJ', 'J1','SoS']


#folder configurations
output_folder ='./output/{}/distribution_all.csv'
cost_folder = './model/data/Cost.csv'
complexity_folder = './model/data/data_complexity.csv'
model_folder = 'weibull_experiment_cleaned_dates_linear_complex'#'weibull_baysien'

comp =pd.read_csv(complexity_folder)
cost_dat = pd.read_csv(cost_folder)
cost_dat[['0', '1','2','3','4']] = cost_dat['Program'].str.split(' ', expand = True)

data = pd.read_csv(output_folder.format(model_folder))
data =data[~data.isComplete]
data = data[~ (data.gateway_start =='J1')]
data = data[~ (data.gateway_start =='SoS')]
test_cases_with_predictions = data.test_case[data.isPrediction].unique()
data =data[data.test_case.isin(test_cases_with_predictions)]
tcases = ['L460 2021MY 555e5 R9009','L461 2022MY 545e4 S9010A' ,'L551 2022MY AJ21-P4 235e3 C9162A','X391 2021MY 665e6 J9151',
          'X393 2022MY 544e4 H9211', 'X760 2020MY 335e3 M9023A','L663 2020MY 110 664e6 Y9079A','L560 2021MY I6+PHEV 355e3 T9122A','L560 2021MY I6+PHEV 355e3 T9122A']

data['dates_str'] = data.dates
# data['dates_str'] = data['dates_str'].replace(np.nan, 'NA')
data['survival_curve'] = data['survival_curve'].round(2)
data['pdf_curve'] = data['pdf_curve'].round(5)
data.dates = pd.to_datetime(data.dates,  format = '%Y/%m/%d', exact = True) #TODO check date format
data = data[data.gateway.isin(gateways)]