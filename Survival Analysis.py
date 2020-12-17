from model.exploration import *
from model.utils import *
from scipy.stats import weibull_min
from scipy.optimize import minimize




def fit_weibull(surv_dist):
    surv_cdf = 1 - surv_dist
    x0 = [0.1,100,100]

    def fitweibull(x):
        x1, x2, x3 = x[0], x[1], x[2]
        print(x)
        wei_cdf = weibull_min.cdf(surv_cdf.index, c = x1, loc=x2, scale=x3)
        sum_abs_err = sum(abs(surv_cdf[surv_cdf.columns].values.flatten() - wei_cdf))
        return (sum_abs_err)


    result =minimize(fitweibull,x0, method = 'Nelder-Mead', tol =1e-20)
    print('test')
    return result.x


if __name__ == '__main__':

    plan_col = 'Planned'  # 'Planned' #
    delay_function = build_date_delay_feature
    complex_col = 'total_sum_complexity'

    # test_case = ['L462 2017MY 543e6 D9046', 'L460 2021MY 555e5 R9009']
    test_case_list = ['L462 2017MY 543e6 D9046', 'L551 2020MY 565e6 C9011A', 'L460 2021MY 555e5 R9009']
    # test_case_data['Actual_PTC_Gate'] = pd.to_datetime('2018-10-12')

    gateways_lists = [['Delay_PSC_Gate', 'Delay_PTC_Gate', 'Delay_FDJ_Gate', 'Delay_120_Gate', 'Delay_J1_Gate'],
                      ['Delay_PTC_Gate', 'Delay_FDJ_Gate', 'Delay_120_Gate', 'Delay_J1_Gate'],
                      ['Delay_PTC_Gate', 'Delay_FDJ_Gate', 'Delay_120_Gate', 'Delay_J1_Gate']
                      ]

    output_data = pd.DataFrame()
    for i,test_case in enumerate(test_case_list):
        outlier_case = ['X760 2018MY P2 245e4 M9022D',
                        'L494 2018MY SV SVR MCF 345A-HA ZS3007',
                        'X152 2019MY 125e3 E9104A',
                        'X152 2019MY P2 Eu6d-T 125e3 E9104B',
                        'X152 2019MY 123e2 E9145',
                        'X260LCN 2018MY AJ20-P4M 245e4 M9022H',
                        'X351 2017MY 111e1 J1396',
                        'X760 2019MY Eu6d-T 125e2 M9111A',
                        'X260LCN 2019MY 111e1 M7121B',
                        'X760LCN 2018MY 433e3 M7045',
                        'X152 2017MY 111e1 E9135','KP',
                        'L494 2019MY 246e4 S9132A']


        gateways = gateways_lists[i]

        data = import_data_from_bq(eng_bq_config)
        data = extract_data(data)
        data_comp = build_complexity_feature(data)[[complex_col]]
        data_test = data[data.index == test_case]
        if test_case == 'L460 2021MY 555e5 R9009':
            data_test['Actual_PTC_Gate'] = pd.to_datetime('2018-10-12')
        data_delay_test_o = delay_function(data_test, plan_col)
        data_delay_test_o = pd.merge(data_delay_test_o, data_comp, left_index=True,right_index=True, how = 'inner')
        data_delay_test = data_delay_test_o.copy(deep = True)

        print(data_test)
        print(data_delay_test)
        data_delay1_t= delay_function(data_test, plan_col)

        expected_delay = [data_delay_test[[gateways[0]]].values]
        test_complexity = data_delay_test[[complex_col]].values
        for gi, s_ in enumerate(gateways[:-1]):
            X_col = gateways[gi]
            Y_col = gateways[gi+1]

            data = data[~ (data.index == test_case)]
            data = data[~data.index.isin(outlier_case)]

            data_delay1 = delay_function(data, plan_col)  # data_delay1 = build_delay_feature(data, plan_col)
            data_delay_o = data_delay1[[X_col, Y_col]]
            data_delay_o = pd.merge(data_delay_o, data_comp, left_index=True,right_index=True, how = 'inner')
            data_delay_o = dropinfna(data_delay_o)
            data_delay_o['event'] = 1
            cph = CoxPHFitter()
            cph.fit(data_delay_o, duration_col=Y_col)
            output = cph.predict_survival_function(data_delay_test[[X_col,complex_col ]])
            output = output.rename(columns = {test_case: 'survival_probability'})
            # x_optim = fit_weibull(output[['survival_probability']])
            #
            # days = np.arange(output.index.min(),output.index.max())
            # extrapolated_cdf = weibull_min.cdf(days, c=x_optim[0], loc=x_optim[1], scale=x_optim[2])
            # extrapolated_surv = 1- extrapolated_cdf
            #
            # if Y_col == 'Delay_J1_Gate':
            #     print('test')
            # output = pd.DataFrame()
            # output['survival_probability'] = extrapolated_surv
            # output['delay_days'] = days


            expected_delay = cph.predict_expectation(data_delay_test[[X_col, complex_col]])
            data_delay_test[[Y_col]] = expected_delay
            print(f'{Y_col} : {expected_delay.round()}')
            output['Gateway'] = Y_col
            output['Program'] =test_case

            output_data = output_data.append(output)

    output_data.to_csv('../output/survival_analysis.csv')
    print('finish')


