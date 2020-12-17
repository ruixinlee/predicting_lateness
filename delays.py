from model.exploration import *


if __name__ == '__main__':
    ## sum of delay vs total complexity
    data = import_data_from_bq(bq_config)
    # data = drop_na_actual(data)
    data = extract_data(data)
    data_comp = build_complexity_feature(data)
    PS_PSC = find_gateway_interval(data,'Actual_PS_Gate','Actual_PSC_Gate')
    PSC_J1 = find_gateway_interval(data,'Actual_PSC_Gate','Actual_J1_Gate')
    P_PS_PSC = find_gateway_interval(data, 'Planned_PS_Gate', 'Planned_PSC_Gate')
    P_PSC_J1 = find_gateway_interval(data,'Planned_PSC_Gate','Planned_J1_Gate')

    delay_PS_PSC = PS_PSC - P_PS_PSC
    delay_PSC_J1 = PSC_J1 - P_PSC_J1

    sns_plot_product, sns_plot_product_ax = plt.subplots()
    delaydf = pd.DataFrame()
    delaydf['delay_PS_PSC'] = delay_PS_PSC
    delaydf['delay_PSC_J1'] = delay_PSC_J1
    delaydf = delaydf.dropna()
    sns.scatterplot(delaydf['delay_PS_PSC'],delaydf['delay_PSC_J1'],  ax=sns_plot_product_ax)
    sns_plot_product_ax.set_title('delay_PS_PSC vs delay_PSC_J1')
    sns_plot_product_ax.set(xlabel='delay_PS_PSC', ylabel='delay_PSC_J1')
    sns_plot_product.savefig("..\\output\\delay_PS_PSC vs delay_PSC_J1")

    print('test')