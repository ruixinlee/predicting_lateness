import model.Baysien as bays
import model.Baysien_Sampling as samps
import model.Baysien_Sampling_Viz as vizs
import pandas as pd
from config.config import output_folder



test =pd.read_csv('.\\model\\data\\data_planned_dates.csv')
test_cases = test['Program Display Name'].tolist()### for \weibull_experiment_cleaned_dates_linear_complex


if __name__ == '__main__':

    data_all = pd.DataFrame()
    for tc in test_cases:
        try:
            print(tc)
            bays.main([tc])
            exist = samps.main([tc])
            if exist:
                temp = vizs.main([tc])
                data_all = data_all.append(temp)
        except ValueError:
            continue
    data_all.to_csv(output_folder.format('distribution_all.csv'))
