import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('..\\input\\Time_to_J1_by_Scaling.csv')
data[0] =data['Natural_Scale'].astype(str).str[0]
data[1] =data['Natural_Scale'].astype(str).str[1]
data[2] =data['Natural_Scale'].astype(str).str[2]
data['sum'] = data['Electrical_Scale'] + data[0].astype(int) + data[1].astype(int) + data[2].astype(int)
data['product'] = data['Electrical_Scale'] * data[0].astype(int) * data[1].astype(int) * data[2].astype(int)
data_temp = data[data['Gateway'] == 'G1 PS']

sns_plot_sum, sns_plot_sum_ax = plt.subplots()
sns.scatterplot(data_temp.DaysFromGatewayToJobOne, data_temp['sum'], ax=sns_plot_sum_ax )
sns_plot_sum_ax.set_title('Sum of complexity scales vs days between PS to Job 1')
sns_plot_sum_ax.set(xlabel = 'days between PS to Job 1', ylabel ='sum of complexity scaling' )
sns_plot_sum.savefig("..\\output\\Sum of complexity scales vs PCDS days between PS to Job 1.png")

sns_plot_product, sns_plot_product_ax = plt.subplots()
sns.scatterplot(data_temp.DaysFromGatewayToJobOne, np.log(data_temp['product']), ax =sns_plot_product_ax)
sns_plot_product_ax.set_title('Product of complexity scale vs days between PS to Job 1')
sns_plot_product_ax.set(xlabel = 'days between PS to Job 1', ylabel ='product of complexity scaling' )
sns_plot_product.savefig("..\\output\\Product of complexity scales vs PCDS days between PS to Job 1.png")
print('test')