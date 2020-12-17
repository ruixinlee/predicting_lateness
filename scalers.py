from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd

class NoScaler(BaseEstimator,TransformerMixin):
    def __init__(self, param = None):
        self.param = param

    def fit(self, X, y=None):
        return self

    def transform (self,X):
        return X.values

    def inverse_transform(self,X):
        return X.values

class data_scaler(BaseEstimator,TransformerMixin):
    def __init__(self, scaler):
        self.cols = None # cols has be to list of lists
        self.scaler = scaler
        self.scalerlist = None

    def fit(self,X,y=None):
        ## fit a scaler for each column
        self.cols = [i for i in X.columns]
        self.scalerlist =  {k:self.scaler() for k in self.cols}
        for k in self.cols:
            self.scalerlist[k].fit(X[[k]])
        return self

    def transform(self,X,y=None):
        ## transform specific columns in X
        output = pd.DataFrame(columns=X.columns)
        for k in X.columns:
            temp = self.scalerlist[k].transform(X[[k]])
            output[k] = temp.flatten()
        output.index = X.index
        return output

    def inverse_transform(self, X, y = None):
        output = pd.DataFrame(columns=X.columns)
        for k in X.columns:
            temp = self.scalerlist[k].inverse_transform(X[[k]])
            output[k] = temp.flatten()
        output.index = X.index

        return output
