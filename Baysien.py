from model.gaussian_process import *
from model.scalers import *
from model.utils import *
from random import seed
from sklearn.neighbors import KernelDensity
from model.clean_data import input_folder, date_asof, output_complexity, output_original_baselines, output_planned_dates
from config.config import output_folder, gateways, col_complex, col_iscensored, col_programme


seed(12312312)


is_trained_resampled = True
is_resampled = True
date_asof_dt = pd.to_datetime(date_asof, format = '%Y-%m-%d')

def find_current_gateways(data_planned_dates):
    data_current_gateways = pd.DataFrame()
    gateways = ['FC', 'PS', 'PSC',  'PTC',  'FDJ',  'J1',  'SoS']
    for i,c in enumerate(gateways):
        if i==0:
            test = 1   #this is to test if first column already exist
        else:
            test = abs(data_current_gateways.sum(axis=1)-1)
        data_current_gateways[c] = (data_planned_dates[c] > date_asof).astype(int)*test
    return data_current_gateways

def read_data(X_col, Y_col, train = True):

    data_planned_dates =pd.read_csv(input_folder.format(output_planned_dates)).set_index(col_programme)
    data_original_baselines=pd.read_csv(input_folder.format(output_original_baselines)).set_index(col_programme)
    data_complexity=pd.read_csv(input_folder.format(output_complexity)).set_index(col_programme)

    #find current gateway
    data_current_gateways = find_current_gateways(data_planned_dates)

    #to date time
    data_planned_dates[X_col] = pd.to_datetime(data_planned_dates[X_col], format = '%Y-%m-%d')
    data_planned_dates[Y_col] = pd.to_datetime(data_planned_dates[Y_col], format =  '%Y-%m-%d')
    data_original_baselines[X_col] = pd.to_datetime(data_original_baselines[X_col], format =  '%Y-%m-%d')
    data_original_baselines[Y_col] = pd.to_datetime(data_original_baselines[Y_col], format = '%Y-%m-%d')

    #current gateway
    #TODO check if this logic is correct - does this always give the right gateway?
    if not train:
        current_programe = data_current_gateways.index[data_current_gateways[X_col]==1]
        data_planned_dates.at[current_programe,X_col]=pd.to_datetime(date_asof, format = '%Y-%m-%d')


    if train:
        #if X col hasn't finished then ignore Y column
        data_planned_dates[X_col][data_planned_dates[X_col]>date_asof_dt] = np.nan
        data_planned_dates[Y_col][data_planned_dates[X_col] > date_asof_dt] = np.nan  #if Xcol is later than Y col then ignore

    #if Y_col has dates later than current date then it is censored
    data_planned_dates[col_iscensored] =0
    data_planned_dates[col_iscensored] = (data_planned_dates[Y_col] > date_asof_dt ).astype(int)
    data_planned_dates[Y_col][data_planned_dates[Y_col] >  date_asof_dt ] =  date_asof_dt

    data_delay = pd.DataFrame()

    data_delay[col_complex] = data_complexity['total_sum_complexity']
    data_delay[X_col] = (data_planned_dates[X_col] - data_original_baselines[X_col]).dt.days
    data_delay[Y_col] = (data_planned_dates[Y_col] - data_original_baselines[Y_col]).dt.days
    data_delay[col_iscensored] = data_planned_dates[col_iscensored]
    data_delay.index = data_planned_dates.index
    return data_delay


def clean_data(data_delay,X_col,Y_col):
    data_delay = data_delay.dropna()
    # data_delay[X_col][data_delay[X_col]<0] = 0
    data_delay[Y_col][data_delay[Y_col]<0] = 0 # model will break if there is negative delay
    return data_delay


def x_simulation(x, nsim=1000):
    kde = KernelDensity(bandwidth=10).fit(x.values)
    samples = kde.sample(nsim, random_state=10123)
    plt.hist(x,density=True)
    plt.plot()
    return samples

def main(test_case):

    data_original_baselines = pd.read_csv(input_folder.format(output_original_baselines)).set_index(
        col_programme)
    data_original_baselines = data_original_baselines[data_original_baselines.index.isin(test_case)]

    gateways_new = [i for i in gateways if i in data_original_baselines.dropna(axis=1).columns.values]

    for i, _ in enumerate(gateways_new[:-1]):
        X_col = gateways_new[i]  # 'Delay_PSC_Gate'
        Y_col = gateways_new[i + 1]  # 'Delay_J1_Gate'
        print(Y_col)

        data_delay_o = read_data(X_col,Y_col, train=True)
        data_delay_o = clean_data(data_delay_o,X_col,Y_col)

        data_delay_o = data_delay_o[~data_delay_o.index.isin(test_case)]
        # data_delay_o = data_delay_o[(~(data_delay_o[X_col]==0) & ~(data_delay_o[Y_col]==0)) ] # not interested in the cases that are well behave

        fig, ax = render_mpl_table(data_delay_o.corr(), col_width=4.0)
        filename = f'correlations - {X_col}-{Y_col}.png'
        fig.savefig(output_folder.format(filename))

        data = GP_model(data_delay_o,
                        X_col,
                        Y_col,
                        test_case,
                        resample=is_trained_resampled)

        rhats = pd.DataFrame([pm.diagnostics.gelman_rubin(data['delay_trace'])])

        fig, ax = render_mpl_table(rhats, rounding=4, col_width=4.0)
        filename = f'rhats - {X_col}-{Y_col}.png'
        fig.savefig(output_folder.format(filename))



if __name__ == '__main__':
    test_case = ['L462 2017MY 543e6 D9046']
    main(test_case)