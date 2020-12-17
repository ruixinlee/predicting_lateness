from model.clean_data import input_folder, output_original_baselines, output_planned_dates
from model.Baysien import read_data, find_current_gateways
from model.gaussian_process import *
import pickle
from config.config import predict_gateways,  col_complex, col_programme, output_folder,distribution_filename
import numpy as np
import pandas as pd

sample_size = sample_kwargs['draws']
nchain = sample_kwargs['chains']


def build_survival_function(survival_times2,y_mean, y_std ):
    survival_times = np.array(survival_times2)
    survival_times = survival_times.astype(int)
    # survival_times[survival_times > 365] = 365
    longest_survival = max(survival_times)
    shortest_survival = min([min(survival_times),0])
    t_plot = np.arange(shortest_survival, longest_survival, 1)
    weibull_pp_surv = (np.greater_equal
                       .outer(survival_times,
                              t_plot))

    weibull_pp_surv_mean = weibull_pp_surv.mean(axis=0)
    return t_plot,weibull_pp_surv_mean, survival_times


def get_planned_dates(output_planned_dates, test_case):
    data_planned_dates = pd.read_csv(input_folder.format(output_planned_dates)).set_index(col_programme)
    data_planned_dates = data_planned_dates[data_planned_dates.index.isin(test_case)]
    return data_planned_dates


def iscompleted_programme(output_planned_dates, test_case):
    data_planned_dates = get_planned_dates(output_planned_dates, test_case)
    current_gateway_data = find_current_gateways(data_planned_dates)
    first_gateway = [c for c in current_gateway_data.columns if current_gateway_data[c][0]==1]
    if len(first_gateway)==0:
        return True
    else:
        return False


def find_first_gateways(output_planned_dates, test_case):
    gateways = predict_gateways.copy()
    data_planned_dates = get_planned_dates(output_planned_dates,test_case)
    current_gateway_data = find_current_gateways(data_planned_dates)
    first_gateway = [c for c in current_gateway_data.columns if current_gateway_data[c][0]==1]
    if len(first_gateway) ==0:
        if data_planned_dates.PTC.isnull()[0]:
            gateways = ['PSC','PTC', 'FDJ', 'J1', 'SoS']
        else:
            gateways = ['PTC', 'FDJ', 'J1', 'SoS']
    else:
        pos = gateways.index(first_gateway[0]) -1
        gateways = [c for i,c in enumerate(gateways)if i>=pos]
    return(gateways)

def main(test_case):
    simulated_dates_all = pd.DataFrame()
    distribution_df_all = pd.DataFrame()
    gateways = find_first_gateways(output_planned_dates, test_case)

    data_original_baselines = pd.read_csv(input_folder.format(output_original_baselines)).set_index(
        col_programme)
    data_output_planned_dates = pd.read_csv(input_folder.format(output_planned_dates)).set_index(
        col_programme)

    data_original_baselines = data_original_baselines[data_original_baselines.index.isin(test_case)]
    data_output_planned_dates = data_output_planned_dates[data_output_planned_dates.index.isin(test_case)]

    gateways = [i for i in gateways if i in data_original_baselines.dropna(axis =1).columns.values]
    is_empty_baselines =data_original_baselines.isnull().sum().sum() == data_original_baselines.shape[1]

    if len(gateways)>1 and ~(is_empty_baselines):
        for i,_ in enumerate(gateways[:-1]):

            print(gateways[i])
            X_col = gateways[i]
            Y_col = gateways[i+1]
            data_delay_o = read_data(X_col, Y_col,train=False)
            data_delay_o = data_delay_o[data_delay_o.index.isin(test_case)]


            cycle_plan_X_date = pd.to_datetime(data_original_baselines[X_col],format = '%Y-%m-%d')
            cycle_plan_Y_date = pd.to_datetime(data_original_baselines[Y_col], format = '%Y-%m-%d')
            planned_Y_date = pd.to_datetime(data_output_planned_dates[Y_col], format = '%Y-%m-%d')
            planned_X_date = pd.to_datetime(data_output_planned_dates[X_col], format = '%Y-%m-%d')
            data = GP_model(data_delay_o,
                            X_col,
                            Y_col,
                            test_case,
                            resample=False)

            delay_trace = data['delay_trace']
            n_cols =  data['n_cols']
            y_mean = data['y_mean']
            y_std = data['y_std']
            X_pp = np.empty((nsim, n_cols))
            X_pp[:, 0] = 1
            X_pp[:, 2] = data_delay_o[col_complex]
            iscomplete = iscompleted_programme(output_planned_dates,test_case)
            past_gateway = False
            if i == 0:
                X_pp[:, 1] = data_delay_o[X_col]
                delay_start =data_delay_o[X_col]
                gateway_start =gateways[i]

                past_gateway = True


            elif (not iscomplete) and i==1:
                    # if not complete, then this is the current gateway
                    # if data_delay_o[X_col] <0 then we are still within given time
                    if (data_delay_o[X_col]<0)[0] or data_delay_o[X_col].isnull()[0]:
                        X_pp[:, 1] = survival_times
                        past_gateway = False
                    else:
                        X_pp[:, 1] = data_delay_o[X_col]
                        delay_start = data_delay_o[X_col]
                        gateway_start = gateways[i]
                        past_gateway = False
            else:
                X_pp[:, 1] = survival_times
                past_gateway = False


            if past_gateway:
                distribution_df = pd.DataFrame()
                distribution_df['dates'] = cycle_plan_X_date.values
                distribution_df['cycle_plan_date'] = 1
                distribution_df['isPrediction'] = False
                distribution_df['isComplete'] = iscomplete
                distribution_df['test_case'] = test_case[0]
                distribution_df['gateway'] = X_col
                distribution_df_all = distribution_df_all.append(distribution_df)

                distribution_df = pd.DataFrame()
                distribution_df['dates'] = planned_X_date.values
                distribution_df['planned_actual_date'] = 1
                distribution_df['isPrediction'] = False
                distribution_df['isComplete'] = iscomplete
                distribution_df['test_case'] = test_case[0]
                distribution_df['gateway'] = X_col
                distribution_df_all = distribution_df_all.append(distribution_df)

            indices = np.random.randint(0, sample_size*nchain, nsim)
            survival_times2 = []

            for i, ind in enumerate(indices):
                pointix, chainix = np.divmod(ind, nchain)
                points = delay_trace._straces[chainix].point(pointix)
                eta = points['beta'].dot(X_pp[i,:].T)
                s = points['s']
                # with pm.Model() as ppc:
                #     survival_dist = pm.Gumbel('suv', eta, s)
                survival_dist = np.random.gumbel(eta,s,1)[0]
                survival_times2.append(survival_dist)

            t_plot, weibull_pp_surv_mean, survival_times =  build_survival_function(survival_times2, y_mean, y_std)
            weibull_pp_curv_mean = 1- weibull_pp_surv_mean
            weibull_pp_pdf_mean = np.insert(np.diff(weibull_pp_curv_mean),0,0)


            ##output file
            test_cases = '-'.join(test_case)
            filename = distribution_filename.format(test_cases)

            simulated_dates = pd.DataFrame()
            simulated_dates['survival_days'] = survival_times
            simulated_dates['survival_days'] = pd.to_timedelta( simulated_dates['survival_days'], 'D')
            simulated_dates['cycle_plan_date'] = cycle_plan_Y_date[0]
            simulated_dates['simulated_completion_dates'] = simulated_dates['cycle_plan_date']  + simulated_dates['survival_days']
            simulated_dates['test_case'] = test_case[0]
            simulated_dates['gateway'] = Y_col

            distribution_df = pd.DataFrame()
            distribution_df['days'] = t_plot
            distribution_df['dates'] = cycle_plan_Y_date[0]
            distribution_df['dates'] = cycle_plan_Y_date[0] + pd.to_timedelta(distribution_df['days'], 'D')
            distribution_df['cycle_plan_date'] =  (distribution_df['dates'] ==cycle_plan_Y_date[0]).astype(int)

            distribution_df['planned_actual_date'] =  (distribution_df['dates'] ==planned_Y_date[0]).astype(int)

            distribution_df['survival_curve'] = weibull_pp_surv_mean
            distribution_df['cumulative_curve'] = weibull_pp_curv_mean
            distribution_df['pdf_curve'] =weibull_pp_pdf_mean
            if ~cycle_plan_Y_date.isnull()[0]:
                distribution_df['expected_date'] = (((cycle_plan_Y_date + pd.to_timedelta(survival_times.mean(), 'D'))).dt.date[0]== distribution_df['dates']).astype(int)
            else:
                distribution_df['expected_date']=0
            distribution_df['expected_days'] = survival_times.mean()

            distribution_df['test_case'] = test_case[0]
            distribution_df['gateway'] = Y_col
            distribution_df['conservative_view'] =0
            distribution_df['aggressive_view'] =0

            min_val = max(distribution_df.survival_curve.min(), 0.25)
            conservative_index =distribution_df.index[distribution_df.survival_curve <= 0.25][0]
            aggressive_index=distribution_df.index[distribution_df.survival_curve <= 0.75][0]
            distribution_df['conservative_view'].iloc[conservative_index] = 1
            distribution_df['aggressive_view'].iloc[aggressive_index] = 1
            distribution_df['gateway_start'] = gateway_start
            distribution_df['delay_start'] = delay_start[0]
            distribution_df['isComplete'] = iscomplete
            distribution_df['isPrediction'] = True
            distribution_df_all = distribution_df_all.append(distribution_df)


            simulated_dates_all = simulated_dates_all.append(simulated_dates)

        with open(output_folder.format(filename), 'wb') as buff:
            pickle.dump({'simualted_dates': simulated_dates_all,
                         'distribution_df': distribution_df_all},buff)

        return True
    else:
        return False

    print('complete')


if __name__ == '__main__':

    main([ 'L462 2021MY PT 344e4 D9180A'])
    print('test')
