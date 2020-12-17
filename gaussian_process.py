import pymc3 as pm
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import  LinearRegression
from matplotlib import  pyplot as plt
import pickle
import time
import numpy as np
from theano import tensor as tt, shared
from config.config import output_folder, sample_kwargs, nsim


interval_length = 7 #TODO needs to be global


def gumble_sf(y,mu,sigma):
    return 1 - tt.exp(-tt.exp(-(y-mu)/sigma))


def GP_model(data_delay, X_col,Y_col,test_case, resample = False):
    test_cases = '-'.join(test_case)
    pickle_name = f'delay_model_train-{X_col}-{Y_col}-{test_cases}.p'
    pickle_path = output_folder.format(pickle_name)

    col_isCensored = 'isCensored?'
    col_X_intercept = 'intercept'
    col_complexity = 'Complexity'
    train_col = [X_col,col_complexity]
    seed = 192103

    if resample:

        n_rows, _ = data_delay.shape
        X = np.empty((n_rows,3))
        X[:,0] =1
        X[:,1:] = data_delay[train_col]
        n_cols = X.shape[1]
        y = data_delay[Y_col].values
        y_std = y
        #
        censored = (data_delay[col_isCensored]==1).values

        X_ = shared(X)
        censored_ = shared(censored)


        vague_sd_prior = 1000000
        with pm.Model() as delay_model:
            beta = pm.Normal('beta',0,vague_sd_prior,shape = n_cols)

            eta = beta.dot(X_.T)
            s = pm.HalfCauchy('s', 200)

            #for dead subjects (uncencored)
            y_obs = pm.Gumbel('y_obs',eta[~censored_],s, observed = y_std[~censored])
            y_cens = pm.Potential('y_cens',gumble_sf(y_std[censored],eta[censored_],s))

        with delay_model:

            # delay_trace = pm.sample(tune=tune_size, chains = 4, cores = 1, njobs=1)
            delay_trace = pm.sample(**sample_kwargs,  nuts_kwargs=dict(target_accept=.95))

        time.sleep(5)
        with open(pickle_path, 'wb') as buff:
            data = {'delay_model': delay_model, 'delay_trace': delay_trace, 'X_':X_,  'censored': censored_,'n_cols': n_cols, 'y_mean':y.mean(),'y_std':y.std()}
            pickle.dump(data, buff)

        #TODO why does using two chains not giving any convergence, while using 4 daoes?
        # doesn't work


    else:
        with open(pickle_path, 'rb') as buff:
            data = pickle.load(buff)


    return data
