import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pdb


excel_path = 'database.xlsx'
database_neg = pd.read_excel(excel_path, sheet_name='阴性')
del database_neg['编号']
database_neg['阳性']= 0
database_pos = pd.read_excel(excel_path, sheet_name='阳性')
del database_pos['编号']
database_pos['阳性']= 1
# combine two tables
database = pd.concat([database_neg,database_pos],axis=0,ignore_index=True)
# change sex to one column
database = pd.concat([database, pd.get_dummies(database['性别'])], axis=1)
del database['性别']
del database['女']

Y = database['阳性'].values
predictorcols = ['男','年龄','身高cm','体重kg']
# x_label = database.columns.values
# x_label = np.delete(x_label,[3,13])
# delete 超气量、阳性

X1 = database[predictorcols]
X2 = sm.add_constant(X1)
X = X2.astype(float)
est = sm.OLS(Y, X)
est2 = est.fit()
print(est2.summary())
