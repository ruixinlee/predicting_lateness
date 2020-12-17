# from Utils.load_data import import_data_from_bq
from config.config import eng_bq_config
import pandas as pd
import seaborn as sns
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

from lifelines import CoxPHFitter
from scipy import stats



pd.set_option('display.max_columns',500)
pd.set_option('display.width', 150)



#TODO check all date time conversion

## some gate ways are ignored
order = OrderedDict({
         1: '{}_FC_Gate',
         2: '{}_PS_Gate',
         3: '{}_PSC_Gate',
         4: '{}_PTC_Gate',
         5: '{}_FDJ_Gate',
         6: '{}_120_Gate',
         7: '{}_J1_Gate',
         8: '{}_SoS_Gate'
        })

def dropinfna(data):
    data = data.replace([-np.inf, np.inf], np.nan)
    data = data.dropna()
    return data

def slice_orderDict(dict,start,end):
    keys = list(dict.keys())
    selected_keys = keys[start:end]
    return(OrderedDict({k:v for (k,v) in order.items() if k in selected_keys }))

def build_cols():

    actual_cols = [i.format('Actual') for i in list(order.values())]
    planned_cols = [i.format('Planned') for i in list(order.values())]
    engineering_cols = [i.format('Engineering') for i in list(order.values())]
    other_cols = ['Program_Display_Name', 'Program_Name', 'Program_Status', 'Complexity']

    return(actual_cols, planned_cols ,engineering_cols, other_cols)

def extract_data(data):
    actual_cols, planned_cols, engineering_cols, other_cols = build_cols()
    for k in actual_cols:
        data[k] = pd.to_datetime(data[k])

    for k in planned_cols:
        data[k] = pd.to_datetime(data[k])

    for k in engineering_cols:
        data[k] = pd.to_datetime(data[k])

    col_selected = actual_cols + planned_cols + engineering_cols+ other_cols
    data = data[col_selected]
    data = data.set_index('Program_Display_Name')
    return data

def drop_na_actual(data):
    actual_cols, _, _ = build_cols()
    return(data[~data[actual_cols].isnull().any(axis=1)]) # select all data with full dates

def build_cum_interval_feature(data,  actualorplanned=['Actual','Planned']):
    output = pd.DataFrame()
    data = build_interval_feature(data, actualorplanned)
    cols = [v.format('time2Actual') for k,v in order.items()][1:]
    data = data[cols]  #order data
    output = data.cumsum(axis = 0 )
    return output

def build_propotion_interval_feature(data,  actualorplanned=['Actual','Planned']):
    output = pd.DataFrame()
    data = build_interval_feature(data, actualorplanned)
    for acp in actualorplanned:
        cols = [v.format(acp) for v in order.items()]
        data = data[cols]
        data['total_time'] = data.sum(axis = 1)
        for k,v in order.items():
            col = v.format(acp)
            output[col+'_propotions'] = data[col]/data['total_time']
    return output

def build_complexity_feature(data):

    output = pd.DataFrame()
    data.Complexity = data.Complexity.str.replace('e','')
    data.Complexity = data.Complexity.str.replace('m','')
    strlen = len(data.Complexity[0])

    for i in range(strlen):
        output[f'c_{i}'] =  data.Complexity.str.slice(i,i+1)
        output[f'c_{i}'] = pd.to_numeric(output[f'c_{i}'])
        if i == 0:
            output['total_mult_complexity'] = output[f'c_{i}']
            output['total_sum_complexity'] = output[f'c_{i}']
        else:
            output['total_mult_complexity'] = output[f'c_{i}'] * output['total_mult_complexity']
            output['total_sum_complexity'] = output[f'c_{i}'] + output['total_sum_complexity']
    return output

def build_interval_feature(data, actualorplanned=['Actual','Planned'], sel_order = None):

    output = pd.DataFrame()
    if sel_order is not None:
        selected_order ={k:v for k,v in order.items() if v in sel_order}

    else:
        selected_order = order.copy()

    for i, (v,k) in enumerate(selected_order.items()):
        if i == 0:
            continue
        for stri in actualorplanned:
            pre = selected_order[list(slice_orderDict(selected_order,i-1,i))[0]].format(stri)
            post = selected_order[list(slice_orderDict(selected_order,i,i+1))[0]].format(stri)
            output[f'time2{post}'] =  find_gateway_interval(data, pre, post)

    return output

def build_date_delay_feature(data, plan):
    output = pd.DataFrame()
    for v,k in order.items():
        output[k.format('Delay')] = (data[k.format('Actual')] - data[k.format(plan)])/pd.Timedelta('1 days')
    return output

def build_total_delay_perc_feature(data, plan):
    output = pd.DataFrame()
    first_gate = '{}_FC_Gate'
    for v,k in order.items():
        tempbase =  (data[k.format(plan)]  - data[first_gate.format(plan)])/pd.Timedelta('1 days')
        output[k.format('Delay')] = (data[k.format('Actual')] - data[k.format(plan)])/pd.Timedelta('1 days')
        output[k.format('Delay')] = output[k.format('Delay')]/tempbase
    return output

def build_interval_delay_feature(data,timeline,isperc = True, sel_order = None):

    if sel_order is not None:
        selected_order ={k:v for k,v in order.items() if v in sel_order}

    else:
        selected_order = order.copy()

    data_actual_intervals = build_interval_feature(data,['Actual'], sel_order)
    data_planned_intervals = build_interval_feature(data,[timeline],sel_order)
    strings = f'time2{timeline}'
    output = pd.DataFrame()
    for i,(v,k) in enumerate(selected_order.items()):
        if i ==0:
            continue
        output[k.format('Delay_Inteval')] = (data_actual_intervals[k.format('time2Actual')] - data_planned_intervals[k.format(strings)])
        if isperc:
            output[k.format('Delay_Inteval')]=output[k.format('Delay_Inteval')]/data_planned_intervals[k.format(strings)]
    return output

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

def find_gateway_interval(data, col1, col2):
    output = (data[col2] - data[col1]) / pd.Timedelta('1 days')
    return output

def pairplot(data):
    g = sns.pairplot(data,kind = 'reg')
    #g.map_upper(plt.scatter, s=10)
    g.map_diag(sns.distplot, kde=True)
    #g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_upper(corrfunc)
    return(g)

if __name__ == '__main__':

    # data = import_data_from_bq(eng_bq_config)
    # data = drop_na_actual(data)
    data = extract_data(data)
    data_comp = build_complexity_feature(data)
    data_delay = build_interval_delay_feature(data, plan = 'Planned')
    data_actual_intervals = build_interval_feature(data,['Actual'])
    data_planned_intervals = build_interval_feature(data,['Planned'])
    data_engineering_interval = build_interval_feature(data,['Engineering'])
    data_cum = build_cum_interval_feature(data,['Actual'])
    data_interval_delay = build_interval_delay_feature(data,'Planned')
    # data_propotion = build_propotion_interval_feature(data,['Actual'])

    test = build_date_delay_feature(data)
    ## exclude potential outlier
    data = data.drop(test.index[test.Delay_PTC_Gate == test.Delay_PTC_Gate.max()])

    sns.pairplot(build_interval_feature(data,['Actual']).dropna())
    sns.pairplot(build_interval_feature(data,['Planned']).dropna())

    ##delay in dates
    sns_plot_product, sns_plot_product_ax = plt.subplots()
    data_delay = pd.concat([build_date_delay_feature(data), data_comp['total_sum_complexity']], axis=1)
    sns_plot_product = pairplot(data_delay.dropna())
    sns_plot_product.fig.suptitle('Correlations of Delay in Dates')
    sns_plot_product.savefig("..\\output\\Correlations of Delay in Dates.png")

    ## delay in interval
    sns_plot_product, sns_plot_product_ax = plt.subplots()
    data_delay = pd.concat([build_interval_delay_feature(data,'Planned'), data_comp['total_sum_complexity']], axis=1)
    sns_plot_product = pairplot(data_delay.dropna())

    sns_plot_product.fig.suptitle('Correlations of Delay in Interval (Days)')
    sns_plot_product.savefig("..\\output\\Correlations of Delay in Interval (Days) Against Planned.png")

    ## delay in perc
    sns_plot_product, sns_plot_product_ax = plt.subplots()
    data_delay = pd.concat([build_interval_delay_feature(data,'Planned', isperc = True), data_comp['total_sum_complexity']], axis=1)
    sns_plot_product = pairplot(data_delay.dropna())
    sns_plot_product.fig.suptitle('Correlations of Delay in Interval (percentage)')
    sns_plot_product.savefig("..\\output\\Correlations of Delay in Interval (percentage) Against Planned.png")


    ## delay in interval
    sns_plot_product, sns_plot_product_ax = plt.subplots()
    data_delay = pd.concat([build_interval_delay_feature(data,'Engineering'), data_comp['total_sum_complexity']], axis=1)
    sns_plot_product = pairplot(data_delay.dropna())

    sns_plot_product.fig.suptitle('Correlations of Delay in Interval (Days)')
    sns_plot_product.savefig("..\\output\\Correlations of Delay in Interval (Days) Against Engineering.png")

    ## delay in perc
    sns_plot_product, sns_plot_product_ax = plt.subplots()
    data_delay = pd.concat([build_interval_delay_feature(data,'Engineering', isperc = True), data_comp['total_sum_complexity']], axis=1)
    sns_plot_product = pairplot(data_delay.dropna())
    sns_plot_product.fig.suptitle('Correlations of Delay in Interval (percentage)')
    sns_plot_product.savefig("..\\output\\Correlations of Delay in Interval (percentage) Against Engineering.png")

    FDJ_J1 = find_gateway_interval(data,'Actual_J1_Gate','Actual_FDJ_Gate')
    PS_FDJ = find_gateway_interval(data,'Actual_J1_Gate','Actual_PS_Gate')
    P_FDJ_J1 = find_gateway_interval(data,'Planned_J1_Gate','Planned_FDJ_Gate')
    P_PS_FDJ = find_gateway_interval(data,'Planned_J1_Gate','Planned_PS_Gate')
    delay_PS_FDJ = PS_FDJ - P_PS_FDJ

    sns_plot_product, sns_plot_product_ax = plt.subplots()
    sns.scatterplot(FDJ_J1, data_comp.total_sum_complexity, ax=sns_plot_product_ax)
    sns_plot_product_ax.set_title('FDJ_J1 length vs sum of complexity')
    sns_plot_product_ax.set(xlabel='FDJ_J1 length in days', ylabel='sum of complexity scaling')
    sns_plot_product.savefig("..\\output\\FDJ_J1 length vs sum of complexity.png")

    sns_plot_product, sns_plot_product_ax = plt.subplots()
    delaydf = pd.DataFrame()
    delaydf['delay_PS_FDJ'] = delay_PS_FDJ
    delaydf['FDJ_J1'] = FDJ_J1
    delaydf = delaydf.dropna()
    sns.scatterplot(delaydf['FDJ_J1'],delaydf['delay_PS_FDJ'],  ax=sns_plot_product_ax)
    sns_plot_product_ax.set_title('FDJ_J1 length vs delay in delay_PS_FDJ')
    sns_plot_product_ax.set(xlabel='FDJ_J1 length in days', ylabel='delay_PS_FDJ')
    sns_plot_product.savefig("..\\output\\FDJ_J1 length vs delay in delay_PS_FDJ")



    sns_plot_product, sns_plot_product_ax = plt.subplots()
    sns.scatterplot(FDJ_J1, data_comp.total_mult_complexity, ax=sns_plot_product_ax)
    sns_plot_product_ax.set_title('FDJ_J1 length vs product of complexity')
    sns_plot_product_ax.set(xlabel='FDJ_J1 length in days', ylabel='product of complexity scaling')
    sns_plot_product.savefig("..\\output\\FDJ_J1 length vs product of complexity.png")


    sns_plot_product, sns_plot_product_ax = plt.subplots()
    sns.distplot(FDJ_J1.dropna())
    sns_plot_product_ax.set_title('FDJ_J1 length histogram')
    sns_plot_product_ax.set(xlabel='FDJ_J1 length in days')
    sns_plot_product.savefig("..\\output\\FDJ_J1 length histogram.png")

    ## l460
    data[data['Program_Display_Name'].str.split(expand = True)[0]=='L460'].sort_values(by = 'Actual_FC_Gate')

    regression_data = pd.DataFrame()
    regression_data['FDJ_J1'] =FDJ_J1
    regression_data['event'] = 1
    regression_data['sumcomp'] =data_comp.total_sum_complexity
    regression_data = regression_data.dropna()

    ## model
    cph = CoxPHFitter()
    cph.fit(regression_data, 'FDJ_J1', event_col='event')
    cph.print_summary()
    cph.predict_partial_hazard(regression_data[['sumcomp']])
    survival = cph.predict_survival_function(regression_data[['sumcomp']])
    survival = cph.predict_survival_function(np.array([[20]]))
