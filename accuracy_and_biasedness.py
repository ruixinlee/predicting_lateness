from model.exploration import *
import seaborn as sns
from matplotlib import  pyplot as plt


def series2df(measure_data):
    measure_data = measure_data.reset_index()
    measure_index = measure_data['index'].str.split('_',expand = True)
    measure_data['plan_type'] = measure_index[4]
    measure_data['gateway'] = measure_index[2]

    return measure_data



if __name__ == '__main__':

    data = import_data_from_bq(eng_bq_config)
    data = extract_data(data)

    p_data  = build_interval_delay_feature(data,'Planned',isperc=False)
    e_data = build_interval_delay_feature(data,'Engineering',isperc=False)


    e_total_data = build_interval_delay_feature(data, 'Engineering', isperc=False, sel_order=['{}_PS_Gate', '{}_J1_Gate'])
    p_total_data = build_interval_delay_feature(data, 'Planned', isperc=False, sel_order=['{}_PS_Gate', '{}_J1_Gate'])
    e_total_data  = e_total_data.rename(columns = {'Delay_Inteval_J1_Gate':'Delay_Inteval_PS-J1_Gate' })
    p_total_data  = p_total_data.rename(columns = {'Delay_Inteval_J1_Gate':'Delay_Inteval_PS-J1_Gate' })

    p_data = p_data.join(p_total_data, on = 'Program_Display_Name' )
    e_data = e_data.join(e_total_data, on = 'Program_Display_Name' )

    a_data = p_data.join(e_data, on='Program_Display_Name', how='outer', lsuffix='_Planned', rsuffix='_Engineering')
    order.update({9:'{}_PS-J1_Gate'})
    measures = pd.DataFrame()
    for i,(k,v) in enumerate(order.items()):
        ## doing a loop because we want to preserve as much data as possible for each pair
        if i == 0:
            continue
        col_p = v.format('Delay_Inteval') + '_Planned'
        col_e = v.format('Delay_Inteval') + '_Engineering'

        temp_data = a_data[[col_p, col_e]]
        temp_data = temp_data.dropna()

        mean_absolute_error = abs(temp_data).mean(axis = 0)
        mean_absolute_error = series2df(mean_absolute_error)
        mean_absolute_error['measure'] = 'mean absolute interval delay compare to plans'

        mean_error = temp_data.mean(axis = 0)
        mean_error = series2df(mean_error)
        mean_error['measure'] = 'mean interval delay compare to plans'

        median = temp_data.median(axis = 0)
        median = series2df(median)
        median['measure'] = 'median interval delay compare to plans'

        measures = measures.append(mean_absolute_error)
        measures = measures.append(mean_error)
        measures = measures.append(median)

    titles = ['mean absolute interval delay compare to plans', 'mean interval delay compare to plans', 'median interval delay compare to plans']
    gateway_order = {'PS':0, 'PSC':1,'PTC':2,'FDJ':3,'120':4,'J1':5,'SoS':6,'PS-J1':7}
    temp2 =pd.pivot_table(measures,values = 0, index = ['index', 'plan_type', 'gateway'], columns =['measure']).reset_index()
    temp2 ['gateway_num'] =temp2['gateway'].replace(gateway_order)
    temp2 = temp2.sort_values(by=['gateway_num', 'plan_type'])
   #g = sns.PairGrid(measures,y_vars=measures['index'], x_vars=measures['measure'])
    g = sns.PairGrid(temp2, y_vars=['gateway'], x_vars=titles, hue = 'plan_type', height=10, aspect=0.8)
    g = g.map(sns.stripplot, size=10, orient="h",linewidth=1, edgecolor="w", jitter = False)
    g = g.add_legend()
    g.set(xlim = (-170,170))
    g.fig.suptitle('Average Interval Delays against plans')


    for ax, title in zip(g.axes.flat, titles):
        # Set a different title for each axes
        ax.set(title=title)

        # Make the grid horizontal instead of vertical
        ax.xaxis.grid(True, which = 'major')
        ax.yaxis.grid(True)

    sns.despine(left=True, bottom=True)
    g.savefig('..\\output\\Average Interval Delays against plans.png')

    ##boxplot
    cols = a_data.columns
    box_data = pd.melt(a_data.reset_index(),id_vars='Program_Display_Name',value_vars=cols,value_name='interval_delay', var_name= 'interval_type')
    box_data = box_data.dropna()
    box_data_col = box_data['interval_type'].str.split('_',expand = True)
    box_data['plan_type'] = box_data_col[4]
    box_data['gateway'] = box_data_col[2]
    box_data['gateway_num'] = box_data['gateway'].replace(gateway_order)
    box_data = box_data.sort_values(by = ['gateway_num','plan_type'])
    g =sns.catplot(x ='gateway', y = 'interval_delay', hue = 'plan_type',data = box_data, kind = 'box', height=10, whis = 100000)
    g.fig.suptitle('Box plot of interval delay distributions')
    g.savefig('..\\output\\Box plot of interval delay distributions.png')


    ### kdeplot
    kde_data = box_data[['Program_Display_Name','interval_delay', 'plan_type', 'gateway']]
    kde_data = pd.pivot_table(kde_data,index=['Program_Display_Name','plan_type'],columns = ['gateway'],values='interval_delay').reset_index()
    kde_data =  kde_data[['plan_type','PS', 'PSC','PTC','FDJ','120','J1','SoS','PS-J1']]
    kde_data = kde_data.dropna()
    g = sns.PairGrid(kde_data, hue = 'plan_type')
    g = g.map_diag(sns.kdeplot)
    g = g.map_offdiag(sns.kdeplot, shade=True, shade_lowest=False)


    order.pop(9)
    date_delay = build_date_delay_feature(data, 'Planned')
    date_delay_melt = pd.melt(date_delay.reset_index(), id_vars='Program_Display_Name', value_vars=date_delay.columns.values, value_name='Date_delay_in_days',var_name='gateways')
    date_delay_melt = date_delay_melt.dropna()
    ax = sns.boxplot(x='gateways', y='Date_delay_in_days', data=date_delay_melt, whis=100000)
    ax = sns.swarmplot(x='gateways', y='Date_delay_in_days', data=date_delay_melt, color = '0.25')
    ax.set_title('date delays boxplot')
    print('test')
