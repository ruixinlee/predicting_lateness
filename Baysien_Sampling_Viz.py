import pandas as pd
import pickle
from config.config import distribution_filename,output_folder



def main(test_case):
    filename = distribution_filename.format(test_case[0])
    path = output_folder.format(filename)
    with open(path,'rb') as buff:
        data = pickle.load(buff)
    return(data['distribution_df'])

if __name__ == '__main__':

    data_all = pd.DataFrame()
    test_case = [
        'X351 2019MY 125e2 J9174A'
    ]

    for tc in test_case:
        temp = main([tc])
        data_all = data_all.append(temp)

    data_all.to_csv(output_folder.format('distribution_all.csv'))