import pandas as pd


folder = 'weibull_experiment_cleaned_dates_linear_complex'
path = '.\\output\\{}\\{}'
path_read = path.format(folder,'distribution_all.csv')
path_out = path.format(folder,'J1_validation.csv')


data_planned_dates = pd.read_csv('.\\model\\data\\data_planned_dates.csv')



asof = pd.to_datetime('2018-11-07',format = '%Y-%m-%d')

data = pd.read_csv(path_read)
data = data[data.gateway=='J1'].set_index('test_case')
data.dates = pd.to_datetime(data.dates,format =  '%Y-%m-%d')

data = data[data.gateway =='J1']
data_actual_date =data.dates[data.planned_actual_date==1]
data_actual_date  = data_actual_date[data_actual_date<=asof]
data_expected_date =data.dates[data.expected_date==1]
data_expected_date = data_expected_date[data_expected_date.index.isin(data_actual_date.index)]

datedifference = (data_actual_date - data_expected_date).dt.days

data_planned_dates['J1'] = pd.to_datetime(data_planned_dates['J1'], format = '%Y-%m-%d')
data_planned_dates = data_planned_dates.set_index('Program Display Name')

data_planned_dates.J1[data_planned_dates.J1 <= '2018-11-07']
selected = data_planned_dates.J1[data_planned_dates.J1 <= '2018-11-07'].index
[i for i in selected if i in datedifference.index]


#### extract data
asof = pd.to_datetime('2018-11-07',format = '%Y-%m-%d')
data = pd.read_csv(path_read)

# data = data[data.gateway=='J1'].set_index('test_case')
data = data.set_index('test_case')
data.dates = pd.to_datetime(data.dates,format =  '%Y-%m-%d')

data['filter_dates'] = data['conservative_view'] + data['expected_date'] + data['planned_actual_date'] + data['cycle_plan_date'] +  data['aggressive_view']
data =data[data.filter_dates>0]
data = data[data.gateway.isin(['J1'])]

isComplete = data.isComplete.reset_index().drop_duplicates().set_index('test_case')

date_cols = ['conservative_view', 'planned_actual_date','aggressive_view', 'expected_date', 'cycle_plan_date']
validation_data = pd.DataFrame()

data_temp = data[data.gateway == 'J1']
for col in date_cols:
    validation_data[col] = data_temp.dates[data_temp[col] == 1]



validation_data['PSC_'+col] = pd.to_datetime(data_planned_dates['PSC'], format = '%Y-%m-%d')
validation_data['PTC_'+col] = pd.to_datetime(data_planned_dates['PTC'], format = '%Y-%m-%d')

validation_data['PTC_'+col] = validation_data['PTC_'+col].fillna(validation_data['PSC_'+col])


validation_data['prediction_period'] = (validation_data['planned_actual_date'] - validation_data['PTC_'+col]).dt.days
validation_data['actual_percentile']  =data['cumulative_curve'][data['planned_actual_date']==1]

validation_data = validation_data.merge(isComplete, left_index=True, right_index=True)
validation_data.to_csv(path_out)
validation_data  = validation_data[validation_data['planned_actual_date']<=asof]

validation_data['within_range'] = (validation_data['planned_actual_date'] >= validation_data['aggressive_view']) & (validation_data['planned_actual_date'] <= validation_data['conservative_view'])

(validation_data.planned_actual_date - validation_data.expected_date).dt.days.abs().mean()
validation_in_days = (validation_data.planned_actual_date - validation_data.expected_date).dt.days

print()