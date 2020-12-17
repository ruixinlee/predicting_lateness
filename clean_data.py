import pandas as pd
import numpy as np
from model.exploration import build_complexity_feature
from config.config import date_asof

input_folder = '.\\model\\data\\{}'


file_mapping = 'mapping_istrain.csv'
file_original_baselines = 'HimalayaReport_OriginalBaseline.txt'
file_planned_dates = 'HimalayaReport_PlannedDate.txt'

##output
output_complexity = 'data_complexity.csv'
output_original_baselines = 'data_original_baselines_dates.csv'
output_planned_dates = 'data_planned_dates.csv'

if __name__ == '__main__':

    data_mapping = pd.read_csv(input_folder.format(file_mapping))
    data_planned_dates = pd.read_csv(input_folder.format(file_planned_dates), delimiter= '|')
    data_original_baselines_dates = pd.read_csv(input_folder.format(file_original_baselines), delimiter= '|')


    col_program = 'Program Display Name'
    col_complexity = 'Nat Scale'
    cols = {'KO':'KO',
             'FC':'G0 FC',
             'PS':'G1 PS',
             'PSC': 'G2 PSC',
             'PTC': 'G3 PTC',
             'FDJ': 'G4 FDJ',
             'J1': 'G6 J1',
             'SoS': 'G7 SoS',
             #'Nat Scale' : 'Nat Scale'
             }


    ## filter out programmes that are suitable for training
    programmes = data_mapping[data_mapping.istrain ==1]
    data_planned_dates = data_planned_dates[data_planned_dates[col_program].isin(programmes[col_program])]
    data_original_baselines_dates = data_original_baselines_dates[data_original_baselines_dates[col_program].isin(programmes[col_program])]


    #replace N/A
    data_planned_dates = data_planned_dates.replace('N\A', np.nan)
    data_original_baselines_dates = data_original_baselines_dates.replace('N\A', np.nan)

    for c1,c2 in cols.items():
        data_planned_dates[c1] = data_planned_dates[c1].fillna(data_planned_dates[c2])
        data_original_baselines_dates[c1] = data_original_baselines_dates[c1].fillna(data_original_baselines_dates[c2])

        data_planned_dates[c1]  = pd.to_datetime(data_planned_dates[c1],format = '%d/%m/%Y')
        data_original_baselines_dates[c1]  = pd.to_datetime(data_original_baselines_dates[c1],format = '%d/%m/%Y')

    #set index
    data_planned_dates = data_planned_dates.set_index(col_program)
    data_original_baselines_dates = data_original_baselines_dates.set_index(col_program)

    #extract columns
    cols_req = list(cols.keys())
    data_complexity = data_planned_dates[['Nat Scale']]
    data_complexity = data_complexity.rename(columns = {'Nat Scale': 'Complexity'})
    data_complexity = build_complexity_feature(data_complexity)

    data_planned_dates = data_planned_dates[cols_req]
    data_original_baselines_dates = data_original_baselines_dates[cols_req]

    data_complexity = data_complexity.to_csv(input_folder.format('data_complexity.csv'))
    data_planned_dates = data_planned_dates.to_csv(input_folder.format('data_planned_dates.csv'))
    data_original_baselines_dates = data_original_baselines_dates.to_csv(input_folder.format('data_original_baselines_dates.csv'))

    print('test')